"""Tests for the question-asking feature (QuestionService, tool, TUI modal).

Covers: headless rejection, parse_questions normalization, tool run output and
denied flag, the QuestionDialog keyboard flow (select / multi / custom / Esc),
and the app-level bridge (`_question_ask` unblocks on exit).
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.app import App

from opencode_py.question import (
    QuestionInfo,
    QuestionOption,
    QuestionRejectedError,
    QuestionService,
    parse_questions,
)
from opencode_py.tools.registry import Registry
from opencode_py.tools.question import tool as question_tool
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.question_dialog import QuestionDialog

SINGLE = [
    QuestionInfo(
        question="Pick a target",
        header="Target",
        options=[
            QuestionOption(label="Vercel", description="Edge network"),
            QuestionOption(label="Railway", description="Containers"),
        ],
    )
]

MULTI = [
    QuestionInfo(
        question="Pick a target",
        header="Target",
        options=[
            QuestionOption(label="Vercel", description="Edge network"),
            QuestionOption(label="Railway", description="Containers"),
        ],
    ),
    QuestionInfo(
        question="Extras?",
        header="Extras",
        multiple=True,
        custom=True,
        options=[QuestionOption(label="CI")],
    ),
]


# --------------------------------------------------------------------------
# QuestionService
# --------------------------------------------------------------------------

def test_ask_empty_returns_empty():
    svc = QuestionService()
    assert svc.ask([]) == []


def test_ask_headless_rejects():
    svc = QuestionService()
    with pytest.raises(QuestionRejectedError):
        svc.ask(SINGLE)


def test_ask_none_answer_rejects():
    svc = QuestionService(ask_callback=lambda qs: None)
    with pytest.raises(QuestionRejectedError):
        svc.ask(SINGLE)


def test_ask_returns_answers():
    svc = QuestionService(ask_callback=lambda qs: [["Vercel"]])
    assert svc.ask(SINGLE) == [["Vercel"]]


def test_parse_questions_normalizes():
    parsed = parse_questions(
        [
            {"question": "  Choose?  ", "options": [{"label": "a", "description": "d"}]},
            {"question": "", "options": []},  # dropped: no question text
            {"question": "Multi?", "multiple": True, "custom": False, "options": ["x"]},
            "not-a-dict",  # dropped
        ]
    )
    assert len(parsed) == 2
    assert parsed[0].question == "Choose?"
    assert parsed[0].header == "Choose?"
    assert parsed[0].options == [QuestionOption(label="a", description="d")]
    assert parsed[1].multiple is True
    assert parsed[1].custom is False
    assert parsed[1].options == [QuestionOption(label="x")]


def test_parse_questions_defaults():
    parsed = parse_questions([{"question": "q?", "options": []}])
    assert parsed[0].custom is True
    assert parsed[0].multiple is False


# --------------------------------------------------------------------------
# Question tool
# --------------------------------------------------------------------------

def test_tool_no_asker_denied():
    reg = Registry()
    t = question_tool(reg)
    res = t.run({"questions": [{"question": "q?", "options": [{"label": "a"}]}]})
    assert res.get("denied") is True
    assert res.get("error") is True


def test_tool_no_questions_error():
    reg = Registry()
    reg.question_asker = lambda qs: []
    t = question_tool(reg)
    assert t.run({})["error"] is True
    assert t.run({"questions": []})["error"] is True


def test_tool_formats_answers():
    reg = Registry()
    reg.question_asker = lambda qs: [["Vercel"]]
    t = question_tool(reg)
    res = t.run({"questions": [{"question": "Pick a target", "options": [{"label": "Vercel"}]}]})
    assert "error" not in res
    assert "Vercel" in res["output"]
    assert res["metadata"]["answers"] == [["Vercel"]]
    assert "Asked 1 question" in res["title"]


def test_tool_rejection_denied():
    reg = Registry()
    reg.question_asker = lambda qs: (_ for _ in ()).throw(
        QuestionRejectedError("nope")
    )
    t = question_tool(reg)
    res = t.run({"questions": [{"question": "q?", "options": [{"label": "a"}]}]})
    assert res["denied"] is True
    assert res["error"] is True
    assert "dismissed" in res["output"]


# --------------------------------------------------------------------------
# QuestionDialog (keyboard flow)
# --------------------------------------------------------------------------

class DialogHost(App):
    def __init__(self, questions, on_done=None) -> None:
        super().__init__()
        self._questions = questions
        self._on_done = on_done

    def on_mount(self) -> None:
        self.push_screen(QuestionDialog(self._questions, on_done=self._on_done))


async def test_dialog_single_select():
    holder: dict = {}
    app = DialogHost(SINGLE, on_done=lambda a: holder.update(a=a))
    async with app.run_test() as pilot:
        await pilot.press("enter")  # pick Vercel
        await pilot.pause()
    assert holder.get("a") == [["Vercel"]]


async def test_dialog_multi_flow():
    holder: dict = {}
    app = DialogHost(MULTI, on_done=lambda a: holder.update(a=a))
    async with app.run_test() as pilot:
        await pilot.press("enter")  # Vercel -> auto-advance to tab1
        await pilot.press("enter")  # toggle CI
        await pilot.press("tab")    # confirm tab
        await pilot.press("enter")  # submit
        await pilot.pause()
    assert holder.get("a") == [["Vercel"], ["CI"]]


async def test_dialog_numeric_pick():
    holder: dict = {}
    app = DialogHost(SINGLE, on_done=lambda a: holder.update(a=a))
    async with app.run_test() as pilot:
        await pilot.press("2")  # pick Railway
        await pilot.pause()
    assert holder.get("a") == [["Railway"]]


async def test_dialog_custom_answer():
    from textual.widgets import Input

    holder: dict = {}
    qs = [
        QuestionInfo(
            question="Model?",
            header="Model",
            custom=True,
            options=[QuestionOption(label="gpt-4o"), QuestionOption(label="claude")],
        )
    ]
    app = DialogHost(qs, on_done=lambda a: holder.update(a=a))
    async with app.run_test() as pilot:
        await pilot.press("j")
        await pilot.press("j")  # -> custom row
        await pilot.press("enter")  # edit mode
        dlg = app.screen
        inp = dlg.query_one("#question-input", Input)
        assert inp.has_focus
        for key in ("g", "r", "o"):
            await pilot.press(key)
        await pilot.pause()
        assert inp.value == "gro"
        await pilot.press("enter")  # submit custom
        await pilot.pause()
    assert holder.get("a") == [["gro"]]


async def test_dialog_escape_dismisses():
    holder: dict = {}
    app = DialogHost(SINGLE, on_done=lambda a: holder.update(a=a))
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
    assert holder.get("a") is None


async def test_dialog_tab_switches():
    app = DialogHost(MULTI)
    async with app.run_test() as pilot:
        dlg = app.screen
        await pilot.press("enter")  # Vercel -> tab1
        assert dlg._tab == 1
        await pilot.press("tab")  # -> confirm
        assert dlg._tab == 2
        await pilot.press("tab")  # wrap -> tab0
        assert dlg._tab == 0


# --------------------------------------------------------------------------
# App bridge: quitting unblocks a blocked question ask.
# --------------------------------------------------------------------------

async def test_question_ask_unblocks_on_exit():
    from opencode_py.question import QuestionInfo, QuestionOption

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        holder: dict = {}

        def worker() -> None:
            try:
                holder["result"] = app._question_ask(
                    [
                        QuestionInfo(
                            question="q?", header="H", options=[QuestionOption(label="a")]
                        )
                    ]
                )
            except QuestionRejectedError:
                holder["result"] = "rejected"

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        await asyncio.sleep(0.05)
        app._exit_requested.set()
        t.join(timeout=3)
        assert not t.is_alive(), "question ask hung after exit"
        assert holder.get("result") == "rejected"


async def test_question_service_wired_to_bridge():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        assert app.engine.question_service.ask_callback == app._question_ask
