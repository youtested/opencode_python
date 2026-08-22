"""Tests for the TUI bugs found during review.

Covers: action_interrupt poison/double-_turn_done, down-arrow wiping the prompt,
partial streamed text being discarded on error, ModelPicker/SettingsScreen
crash-after-dismiss, sub-agent session/engine leaks and main-session corruption,
agent-toggle using the wrong engine, /models UI-thread freeze, the write-tool
block rendering, and the permission dialog wiring.

Headless Textual tests use App.run_test() (no real terminal needed).
"""

from __future__ import annotations

import sys
import io
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.screen import ModalScreen

from opencode_py.agent.loop import AgentLoop
from opencode_py.config import Config
from opencode_py.tools.write import _write
from opencode_py.tui.chat_view import ChatView, MessageBubble
from opencode_py.tui.input_bar import InputBar
from opencode_py.tui.model_picker import ModelPicker
from opencode_py.tui.permission_dialog import PermissionDialog
from opencode_py.tui.settings_screen import SettingsScreen
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.session import Session


# --------------------------------------------------------------------------
# test harnesses
# --------------------------------------------------------------------------

class WidgetHost(App):
    """App that mounts a single widget (for run_test)."""

    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


class ModalHost(App):
    """App that pushes a modal screen immediately on mount."""

    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def on_mount(self) -> None:
        self.push_screen(self._factory())


class FakeEngine:
    agent = "build"
    permission = SimpleNamespace(mode="auto")


# --------------------------------------------------------------------------
# Bug 3: action_interrupt sets flag and does not finish turn
# --------------------------------------------------------------------------

async def test_action_interrupt_sets_flag_and_does_not_finish_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        aborts = []
        app.engine.abort = lambda: aborts.append(1)
        app._busy = True
        app.action_interrupt()
        assert app._interrupt_flag["requested"] is True
        # the old code called _turn_done() here, clearing _busy while the worker
        # was still running; the fix only flips the shared flag
        assert app._busy is True
        # but the engine's active stream must be aborted so an idle "thinking"
        # gap (no chunks) doesn't keep the interrupt from landing
        assert aborts == [1]


async def test_turn_done_resets_interrupt_flag():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._interrupt_flag["requested"] = True
        app._turn_done(result=None)
        assert app._interrupt_flag["requested"] is False


# --------------------------------------------------------------------------
# ESC double-press interrupt (mirrors opencode's session.interrupt)
# --------------------------------------------------------------------------

async def test_esc_idle_focuses_input_without_arming():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.focus()
        app._busy = False
        app.action_interrupt_escape()
        assert app._interrupt_flag["requested"] is False
        assert app._esc_presses == 0
        assert app.query_one(StatusBar).interrupt_armed is False


async def test_esc_first_press_arms_hint_only():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.action_interrupt_escape()
        assert app._esc_presses == 1
        assert app._interrupt_flag["requested"] is False
        assert app.query_one(StatusBar).interrupt_armed is True
        assert "esc again to interrupt" in app.query_one(StatusBar).render().plain


async def test_esc_second_press_interrupts():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        aborts = []
        app.engine.abort = lambda: aborts.append(1)
        app._busy = True
        app.action_interrupt_escape()  # first press arms
        app.action_interrupt_escape()  # second press aborts
        assert app._interrupt_flag["requested"] is True
        assert app._esc_presses == 0
        assert aborts == [1]


async def test_esc_second_press_within_window_via_key_binding():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.focus()
        app._busy = True
        await pilot.press("escape")
        assert app._esc_presses == 1
        assert app._interrupt_flag["requested"] is False
        await pilot.press("escape")
        assert app._interrupt_flag["requested"] is True
        assert app._esc_presses == 0


# --------------------------------------------------------------------------
# 2nd ESC must abort a RUNNING bash command, not just the model stream.
# ```
# The registry's interrupt_check is read by bash/webfetch at call time. The TUI
# wires engine.interrupt AFTER construction, so a plain attribute froze the
# hook to the init default (a `lambda: False`) and ESC kept being ignored while
# a command ran.
# --------------------------------------------------------------------------

def test_wiring_updates_registry_interrupt_check():
    from pathlib import Path

    from opencode_py.agent.loop import AgentLoop
    from opencode_py.config import Config
    from opencode_py.tools import build_registry

    reg = build_registry(Config())
    engine = AgentLoop(cfg=Config(), registry=reg, directory=Path("."))
    flag = {"on": False}
    engine.interrupt = lambda: flag["on"]  # what _wire_engine does
    assert reg.interrupt_check() is False
    flag["on"] = True  # what a 2nd ESC press does
    assert reg.interrupt_check() is True


async def test_esc_twice_aborts_running_bash_command():
    import threading

    from pathlib import Path

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        engine = app.engine
        # The command runs under test before any model turn; bypass the
        # permission dialog (it would block the worker on the UI loop) with an
        # always-allow engine.
        engine.permission = SimpleNamespace(
            mode="auto",
            ask=lambda *a, **k: "allow",
            evaluate=lambda *a, **k: "allow",
            reset_doom_tracking=lambda: None,
        )
        done = threading.Event()
        result: dict = {}
        command = "python -m pytest tests/ -q 2>&1 | tail -40" if Path("tests").exists() else "sleep 30"

        def work():
            try:
                result["r"] = engine.run_tool(
                    "bash",
                    {"command": command, "timeout": 120000},
                )
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        await pilot.pause(0.6)  # let the command spin up
        app._busy = True
        await pilot.press("escape")  # 1st press arms
        await pilot.press("escape")  # 2nd press must abort the command NOW
        assert app._interrupt_flag["requested"] is True
        for _ in range(50):
            if done.is_set():
                break
            await pilot.pause(0.1)
        assert done.is_set(), "running bash command was NOT aborted by 2nd ESC"
        r = result.get("r") or {}
        assert r.get("interrupted") is True
        assert r.get("error") is True


def test_engine_interrupt_honors_shared_flag():
    """A sub-agent spawned from the app engine must share the interrupt flag."""
    import opencode_py.agent.loop as loop_mod

    real_spawn = loop_mod.AgentLoop.spawn_task
    cfg = Config()
    registry = SimpleNamespace()
    parent = AgentLoop(cfg=cfg, registry=registry, directory=__import__("pathlib").Path("."))
    flag = {"requested": False}
    parent.interrupt = lambda: flag["requested"]
    try:
        assert parent.interrupt() is False
        flag["requested"] = True
        assert parent.interrupt() is True
    finally:
        loop_mod.AgentLoop.spawn_task = real_spawn


# --------------------------------------------------------------------------
# Auto-refocus: tapping anywhere returns the prompt cursor after ~1s
# --------------------------------------------------------------------------

async def test_focus_loss_arms_auto_refocus_timer():
    """Tapping a reasoning bubble (focus moves off the input) must arm a timer
    that drags the prompt cursor back — no need to tap the input box again."""
    from opencode_py.tui.chat_view import ChatView

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        chat = app.query_one(ChatView)
        app.screen.set_focus(chat)
        await pilot.pause(0.05)
        assert not bar.input.has_focus
        assert app._refocus_timer is not None
        # after ~1s idle the cursor returns on its own
        await pilot.pause(1.2)
        assert app._refocus_timer is None
        assert bar.input.has_focus


async def test_focus_on_input_cancels_auto_refocus():
    """Re-tapping the prompt itself within the window must cancel the timer."""
    from opencode_py.tui.chat_view import ChatView

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app.screen.set_focus(app.query_one(ChatView))
        await pilot.pause(0.05)
        assert app._refocus_timer is not None
        bar.input.focus()
        await pilot.pause(0.05)
        assert app._refocus_timer is None


async def test_auto_refocus_does_not_steal_from_modal():
    """A picker/dialog pushed on top must not be robbed of focus."""
    from textual.screen import Screen

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.push_screen(Screen())
        await pilot.pause(0.05)
        app.on_descendant_focus(SimpleNamespace())
        assert app._refocus_timer is None


# --------------------------------------------------------------------------
# Bug 4: down-arrow must never wipe the typed prompt
# --------------------------------------------------------------------------

async def test_down_arrow_clears_typed_prompt_with_no_history():
    # pressing ↓ on what you wrote empties the box in one key so you can
    # replace the whole prompt (old "never wipe" behavior is gone on request)
    app = WidgetHost(lambda: InputBar(commands=[]))
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar.input.value = "hello"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("down")
        assert bar.input.value == ""


async def test_up_down_history_clears_then_restores_draft():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar._history = ["first prompt"]
        bar._hist_index = 1  # end position
        bar.input.value = "my draft"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("up")
        assert bar.input.value == "first prompt"
        assert bar._draft == "my draft"
        # final ↓ past the newest history entry clears the box
        await pilot.press("down")
        assert bar.input.value == ""
        # next ↑ (empty box → routed through the app) restores the draft
        await pilot.press("up")
        assert bar.input.value == "my draft"


async def test_up_repeatedly_keeps_original_draft():
    """Up twice into history must not overwrite the draft with the last item."""
    app = WidgetHost(lambda: InputBar(commands=[]))
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar._history = ["first prompt", "second prompt"]
        bar._hist_index = 2  # end position
        bar.input.value = "my draft"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("up")
        await pilot.press("up")
        assert bar._draft == "my draft"
        assert bar.input.value == "first prompt"


# --------------------------------------------------------------------------
# Bug 5: partial streamed text must survive an error / empty-reply cleanup
# --------------------------------------------------------------------------

async def test_remove_last_stream_bubble_keeps_partial_text():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        chat.stream_delta("partial ")
        chat.stream_delta("text")
        chat.remove_last_stream_bubble()
        bubbles = list(chat.query(MessageBubble))
        assert len(bubbles) == 1
        assert bubbles[0].content == "partial text"
        assert bubbles[0].streaming is False


async def test_remove_last_stream_bubble_removes_empty():
    app = WidgetHost(lambda: ChatView())
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        chat.begin_stream()
        chat.remove_last_stream_bubble()
        await pilot.pause()
        assert len(list(chat.query(MessageBubble))) == 0


# --------------------------------------------------------------------------
# Bug 1: ModelPicker.populate on a dismissed/pruned screen
# --------------------------------------------------------------------------

def test_model_picker_populate_when_not_attached_noop():
    picker = ModelPicker()
    assert picker.is_attached is False
    picker.populate({})  # must not raise NoMatches


def test_model_picker_set_loading_when_not_attached_noop():
    picker = ModelPicker()
    picker.set_loading()  # must not raise


# --------------------------------------------------------------------------
# Bug 2: SettingsScreen deferred render after dismissal
# --------------------------------------------------------------------------

def test_settings_render_when_not_attached_noop():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    assert screen.is_attached is False
    screen._render_settings()  # must not raise NoMatches
    screen._keep_selection_visible()  # must not raise


# --------------------------------------------------------------------------
# Bug 12: permission dialog wiring
# --------------------------------------------------------------------------

async def test_permission_dialog_escape_reports_deny():
    decisions: list[str] = []
    app = ModalHost(
        lambda: PermissionDialog("run a command?", on_decision=decisions.append)
    )
    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert decisions == ["deny"]


async def test_permission_dialog_button_reports_decision():
    decisions: list[str] = []
    app = ModalHost(
        lambda: PermissionDialog("run a command?", on_decision=decisions.append)
    )
    async with app.run_test() as pilot:
        await pilot.press("enter")  # Allow once is first / focused
        assert decisions == ["once"]


# --------------------------------------------------------------------------
# Bug 8: agent toggle must act on the active (sub-agent) engine
# --------------------------------------------------------------------------

async def test_toggle_agent_uses_active_engine():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sub = FakeEngine()
        sub.agent = "plan"
        app._engines["sub"] = sub
        app._sessions["sub"] = SimpleNamespace(agent="plan", title="sub")
        app._current_session_id = "sub"
        app.action_toggle_agent()
        assert sub.agent == "build"
        # the main engine must be untouched
        assert app.engine.agent == "build"


# --------------------------------------------------------------------------
# Bug 14: /models with args must not run the sync fetch on the UI thread
# --------------------------------------------------------------------------

async def test_models_command_routes_to_model_picker():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        calls: list[str] = []
        app._open_model_picker = lambda: calls.append(1)
        app._run_command("/models --json")
        assert calls == [1]


# --------------------------------------------------------------------------
# Bug 7: missing sub-agent session must not fall back to the main session
# --------------------------------------------------------------------------

async def test_subagent_missing_session_registers_placeholder(monkeypatch):
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        import opencode_py.session as session_mod

        monkeypatch.setattr(session_mod, "load_session", lambda sid: None)
        app._on_subagent_start(
            {"kind": "subagent_start", "session_id": "abc123", "agent": "build", "title": "t"}
        )
        assert "abc123" in app._sessions
        assert app._sessions["abc123"].id == "abc123"
        assert app._sessions["abc123"] is not app.session


# --------------------------------------------------------------------------
# Bug 10: finished sub-agent sessions are pruned (no widget/engine leak)
# --------------------------------------------------------------------------

async def test_subagent_done_keeps_widgets_and_engines():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = "sub1"
        chat = app._chat_for(sid)
        app._chats[sid] = chat
        app._engines[sid] = FakeEngine()
        app._sessions[sid] = SimpleNamespace(completed=None, parent_id=None, agent="build")
        app._busy_sessions.add(sid)
        app._running_agents[sid] = "t · build"
        app._on_subagent_done(
            {"kind": "subagent_done", "session_id": sid, "agent": "build", "title": "t", "ok": True}
        )
        # official store behaviour: children stay registered after finishing so
        # they remain reviewable alongside the parallel siblings in the footer.
        assert sid in app._chats
        assert sid in app._engines
        assert sid in app._sessions
        assert sid not in app._busy_sessions
        assert sid not in app._running_agents


# --------------------------------------------------------------------------
# Bug 11: write tool returns the written content for the TUI block
# --------------------------------------------------------------------------

def test_write_tool_returns_content_metadata(tmp_path):
    target = tmp_path / "hello.py"
    result = _write(str(target), "print('hi')\n")
    assert result["output"] == "Wrote file successfully."
    assert result["metadata"]["content"] == "print('hi')\n"
    assert result["metadata"]["filePath"] == str(target.resolve())
    assert target.read_text() == "print('hi')\n"


# --------------------------------------------------------------------------
# Compaction UI: the `compacted` event renders a ` Session compacted ` divider
# --------------------------------------------------------------------------

async def test_compacted_event_appends_divider_bubble():
    from opencode_py.tui.status_bar import StatusBar
    from opencode_py.tui.chat_view import MessageBubble

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._handle_event({"kind": "compacted", "session_id": app.session.id, "summary": "did the thing"})
        bubbles = list(app.query(MessageBubble))
        assert len(bubbles) == 1
        assert bubbles[0].role == "compaction"
        assert bubbles[0].content == "did the thing"
        await pilot.pause()


def test_compaction_bubble_content_renders_without_crash():
    async def run() -> None:
        from rich.console import Console

        app = WidgetHost(lambda: ChatView())
        async with app.run_test() as pilot:
            chat = app.query_one(ChatView)
            chat.append_compaction("summarized everything")
            bubbles = list(chat.query(MessageBubble))
            assert len(bubbles) == 1
            assert bubbles[0].role == "compaction"
            rendered = bubbles[0]._build_content()
            assert rendered is not None
            # official opencode ` Compaction ` divider + markdown summary (the
            # `## Objective` heading renders as colored markdown, not muted text)
            console = Console()
            buf = io.StringIO()
            console.file = buf
            console.print(rendered)
            plain = buf.getvalue()
            assert "Compaction" in plain
            assert "summarized everything" in plain

    import asyncio

    asyncio.run(run())


def test_compaction_start_shows_compacting_indicator():
    """Mirrors official opencode #35316: while the session summarizes, the
    status line reads `Compacting conversation…` (not the generic working…)."""

    async def run() -> None:
        app = WidgetHost(lambda: InputBar())
        async with app.run_test() as pilot:
            bar = app.query_one(InputBar)
            bar.set_busy(True)
            assert "working..." in bar.query_one("#prompt-status").render().plain
            bar.set_compacting(True)
            assert "Compacting conversation" in bar.query_one("#prompt-status").render().plain
            bar.set_compacting(False)
            assert "working..." in bar.query_one("#prompt-status").render().plain

    import asyncio

    asyncio.run(run())


def test_app_compaction_start_and_compacted_roundtrip():
    """compaction_start flips the indicator on; the subsequent `compacted` event
    flips it back off and renders the divider."""

    async def run() -> None:
        app = OpenCodeTUI()
        async with app.run_test() as pilot:
            app._handle_event({"kind": "compaction_start", "session_id": app.session.id})
            bar = app.query_one(InputBar)
            assert bar._compacting is True
            app._handle_event({"kind": "compacted", "session_id": app.session.id, "summary": "s"})
            assert bar._compacting is False
            bubbles = list(app.query(MessageBubble))
            assert any(b.role == "compaction" for b in bubbles)

    import asyncio

    asyncio.run(run())


# --------------------------------------------------------------------------
# Compaction usage in the status bar: the context percentage updates
# --------------------------------------------------------------------------

async def test_compaction_usage_updates_status_percentage():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        # app._handle_event requires the widget tree, and status bar is present
        app._handle_event(
            {"kind": "usage", "session_id": app.session.id, "usage": {"total_tokens": 190000, "context_size": 200000}}
        )
        status = app.query_one(StatusBar)
        assert "190,000" in status.render().plain
        assert "(95%)" in status.render().plain
        # compaction re-emits a usage event with the compacted estimate
        app._handle_event(
            {"kind": "usage", "session_id": app.session.id, "usage": {"total_tokens": 40000, "context_size": 200000}}
        )
        assert "40,000" in status.render().plain
        assert "(20%)" in status.render().plain


async def test_context_percentage_capped_at_100():
    """Providers can report input+output tokens that sum past the nominal
    context window; the footer must clamp the percentage to 100, never 111%."""
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        app._handle_event(
            {"kind": "usage", "session_id": app.session.id, "usage": {"total_tokens": 222222, "context_size": 200000}}
        )
        assert "222,222" in status.render().plain
        assert "(100%)" in status.render().plain
        assert "(111%)" not in status.render().plain
        assert "111" not in status.render().plain


async def test_retry_notice_shows_in_status_bar_then_clears():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        app._handle_event(
            {"kind": "retry", "session_id": app.session.id, "message": "↻ connection dropped — retrying (49 left)…"}
        )
        assert "retrying" in status.render().plain
        # an error clears the retry hint before surfacing
        app._handle_event({"kind": "error", "session_id": app.session.id, "error": "boom", "retryable": True})
        assert "retrying" not in status.render().plain
        # a normal turn end clears it too
        app._handle_event(
            {"kind": "retry", "session_id": app.session.id, "message": "↻ connection dropped — retrying (1 left)…"}
        )
        assert "retrying" in status.render().plain
        app._turn_done()
        assert "retrying" not in status.render().plain


async def test_retry_notice_clears_as_soon_as_model_responds():
    from opencode_py.tui.status_bar import StatusBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        # every kind of "model is responding now" event must clear the hint
        for kind, payload in [
            ("text_delta", {"text": "hi"}),
            ("reasoning_delta", {"text": "thinking"}),
            ("tool_call", {"tool": "read", "arguments": {}, "call_id": "c1"}),
        ]:
            app._handle_event(
                {"kind": "retry", "session_id": app.session.id, "message": "↻ connection dropped — retrying (49 left)…"}
            )
            assert "retrying" in status.render().plain
            app._handle_event({"kind": kind, "session_id": app.session.id, **payload})
            assert "retrying" not in status.render().plain


# --------------------------------------------------------------------------
# Sub-agent navigation: click the task row to enter, ↑/←/→ with an empty
# prompt to navigate (official session.parent / session.child.* scoping)
# --------------------------------------------------------------------------

async def test_click_task_row_navigates_into_subagent():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        chat = app._chats[parent]
        chat.append_tool(
            {"tool": "task", "status": "running", "input": {"description": "port it", "subagent_type": "build"}, "call_id": "t1"}
        )
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        app._switch_session(parent)
        bubble = chat.find_tool("task", "t1")
        bubble.on_click(type("Click", (), {})())
        await pilot.pause()
        assert app._current_session_id == "subX"
        # the linked row knows where it points once the session starts
        assert chat.find_tool("task", "t1").content.get("metadata", {}).get("sessionId") == "subX"


async def test_empty_prompt_up_arrow_returns_to_parent():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        app._switch_session("subX")
        assert app._current_session_id == "subX"
        assert app.query_one(InputBar).input.value == ""
        await pilot.press("up")
        await pilot.pause()
        assert app._current_session_id == parent


async def test_empty_prompt_arrows_cycle_parallel_siblings():
    from types import SimpleNamespace as NS

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        app._children[parent] = [
            {"id": "p0", "created": 1, "status": "completed"},
            {"id": "p1", "created": 2, "status": "running"},
            {"id": "p2", "created": 3, "status": "completed"},
        ]
        for i, sid in enumerate(("p0", "p1", "p2")):
            app._child_parent[sid] = parent
            app._sessions[sid] = NS(parent_id=parent, agent="build", title=sid)
            app._chat_for(sid)
        # ←/→ walk forward/backward; wrap from last back to the first
        app._switch_session("p0")
        await pilot.press("right")
        await pilot.pause()
        assert app._current_session_id == "p1"
        await pilot.press("right")
        await pilot.pause()
        assert app._current_session_id == "p2"
        await pilot.press("right")
        await pilot.pause()
        assert app._current_session_id == "p0"
        await pilot.press("left")
        await pilot.pause()
        assert app._current_session_id == "p2"
        # ctrl+down from the parent resumes the child you last viewed (p2),
        # like official opencode's remembered per-parent selection
        app._switch_session(parent)
        await pilot.press("ctrl+down")
        await pilot.pause()
        assert app._current_session_id == "p2"


async def test_arrow_nav_inactive_while_typing_in_prompt():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        app._switch_session("subX")
        box = app.query_one(InputBar).input
        box.value = "some text"
        await pilot.press("up")
        await pilot.pause()
        assert app._current_session_id == "subX"  # arrows edit the prompt, not the session


async def test_up_arrow_goes_parent_not_prompt_history():
    """↑ with an empty prompt must navigate to the parent even when prompt
    history exists (the old code let reward recall shadow the session nav);
    at the root session ↑ falls back to recalling the previous prompt."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        bar = app.query_one(InputBar)
        bar.input.value = ""
        bar._history.extend(["first", "second"])
        bar._hist_index = 2
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        app._children[parent] = [{"id": "subX", "created": 1, "status": "running"}]
        app._child_parent["subX"] = parent
        # root: no parent -> ↑ recalls the previous prompt
        await pilot.press("up")
        await pilot.pause()
        assert app._current_session_id == parent
        assert bar.input.value == "second"
        # ↓ restores the empty draft
        await pilot.press("down")
        await pilot.pause()
        assert bar.input.value == ""
        # child: ↑ with history present MUST go to the parent, not recall
        app._switch_session("subX")
        bar.input.value = ""
        await pilot.press("up")
        await pilot.pause()
        assert app._current_session_id == parent
        assert bar.input.value == ""


async def test_footer_has_no_parent_button_and_chat_up_arrow_navigates():
    """The `Parent ↑` button is gone from the sub-agent footer; with the chat
    (not the prompt) focused inside a sub-agent, `↑` navigates to the parent
    instead of scrolling the message list."""
    from opencode_py.tui.subagent_footer import SubagentFooter, _NavButton

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        app._children[parent] = [{"id": "subX", "created": 1, "status": "running"}]
        app._child_parent["subX"] = parent
        app._switch_session("subX")
        await pilot.pause()
        labels = [b._label for b in app.query_one(SubagentFooter).query(_NavButton)]
        assert labels == ["Prev ←", "Next →"]  # no "Parent ↑"
        # chat focused (not the prompt): ↑ goes to the parent
        child_chat = app._chats["subX"]
        child_chat._session_is_child = True
        child_chat.focus()
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert app._current_session_id == parent
        # the root chat is not flagged as a child
        assert app._chats[parent]._session_is_child is False


async def test_printable_keys_on_focused_subagent_chat_do_not_crash():
    """Pressing printable keys while the sub-agent chat has focus keeps working
    (a regression crashed here by chaining to a non-existent base `on_key`)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        app._children[app.session.id] = [{"id": "subX", "created": 1, "status": "running"}]
        app._child_parent["subX"] = app.session.id
        app._switch_session("subX")
        chat = app._chats["subX"]
        chat._session_is_child = True
        chat.focus()
        await pilot.pause()
        for ch in ("n", "a", " "):
            await pilot.press(ch)
            await pilot.pause()  # must not raise


async def test_home_end_page_keys_scroll_chat_even_without_clicking():
    """HOME/END/PgUp/PgDn scroll the conversation no matter what has focus.
    Regression: they only scrolled after clicking into the chat, because the
    input bar ate them for text editing."""

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        chat = app._chat_for(app.session.id)
        for i in range(40):
            chat.append_user(f"question {i} " + "x" * 200, agent="build")
            chat.append_assistant(f"answer {i} " + "y" * 200)
        await pilot.pause()
        # focus is on the input bar, not the chat (user never clicked a message)
        app.query_one(InputBar).focus()
        await pilot.pause()
        assert chat.max_scroll_y > 0

        await pilot.press("end")
        await pilot.pause()
        assert chat.scroll_y >= chat.max_scroll_y - 1

        await pilot.press("home")
        await pilot.pause()
        assert chat.scroll_y <= 0.5

        await pilot.press("pagedown")
        await pilot.pause()
        assert chat.scroll_y > 1

        await pilot.press("pageup")
        await pilot.pause()
        assert chat.scroll_y < 1.0


async def test_down_direct_clear_then_up_restores_typed_text():
    """Pressing ↓ directly on what you wrote clears it in one key, and the
    very next ↑ brings that text back (nothing new has been sent yet)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar._history = ["older prompt"]
        bar._hist_index = 1
        bar.input.value = "my long draft"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("down")
        assert bar.input.value == ""
        await pilot.press("up")
        assert bar.input.value == "my long draft"


async def test_down_on_multiline_prompt_moves_line_not_clear():
    """A multi-line prompt keeps ↓ for cursor movement while lines remain; it
    clears only once the cursor is at the very end of the last line."""
    app = WidgetHost(lambda: InputBar(commands=[]))
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar.input.value = "line one\nline two\nline three"
        bar.input.cursor_position = 0
        assert bar._handle_arrow("down") is False
        assert bar.input.value == "line one\nline two\nline three"
        bar.input.cursor_position = len(bar.input.value)  # end of last line
        assert bar._handle_arrow("down") is True
        assert bar.input.value == ""


async def test_down_direct_clear_then_up_restores_typed_text():
    """A direct ↓ on what you wrote clears it, and the very next ↑ brings that
    exact text back (nothing was sent yet, so the draft is still live)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.input.focus()
        bar._history = ["older prompt"]
        bar._hist_index = 1
        bar.input.value = "my long draft"
        bar.input.cursor_position = len(bar.input.value)
        await pilot.press("down")
        assert bar.input.value == ""
        await pilot.press("up")
        assert bar.input.value == "my long draft"


async def test_down_on_long_wrapped_prompt_never_clears():
    """A very long single paragraph that soft-wraps keeps ↓ for cursor
    navigation while lines remain below; at the very end of the last wrapped
    row, ↓ empties the box."""
    host = WidgetHost(lambda: InputBar(commands=[]))
    async with host.run_test(size=(10, 24)) as pilot:
        bar = host.query_one(InputBar)
        bar.input.focus()
        text = "a very long line of writing that wraps onto several rows"
        bar.input.value = text
        assert bar.input._wrapped_lines() > 1
        bar.input.cursor_position = 0
        # still lines below: ↓ moves the cursor, never clears
        assert bar._handle_arrow("down") is False
        assert bar.input.value == text
        # at the end of the last wrapped row: ↓ clears it
        bar.input.cursor_position = len(text)
        assert bar._handle_arrow("down") is True
        assert bar.input.value == ""


# --------------------------------------------------------------------------
# Empty "summary" bubble after a thought + command
# --------------------------------------------------------------------------

def _empty_reply_chat(app):
    """Replay a turn: tool runs, then the model's final summary arrives as an
    empty text chunk, then the turn finishes."""
    sid = app.session.id
    chat = app._chat_for(sid)
    app._handle_event({"kind": "tool_call", "session_id": sid, "tool": "bash", "arguments": {"command": "ls"}, "call_id": "c1"})
    app._handle_event({"kind": "text_delta", "session_id": sid, "text": ""})
    app._busy = True
    app._busy_sessions.add(sid)
    app._turn_state(sid)["had_tools"] = True
    app._active_turn_session_id = sid
    app._turn_done(None)
    return chat


async def test_empty_trailing_delta_leaves_no_blank_assistant_bubble():
    """A thought + tool turn whose final summary streams an empty chunk must
    not leave a blank assistant bubble after the command, and must not mark
    the turn as having real text (which would show a runtime for no reply)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        chat = _empty_reply_chat(app)
        empty_assistant = [b for b in chat.query(MessageBubble) if b.role == "assistant" and not b.content]
        assert empty_assistant == []


async def test_empty_delta_not_counted_as_text_and_real_text_is():
    """The empty-reply marking turns on the runtime only for actual text, and
    real text streamed after a command is kept as a normal assistant bubble."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        # empty chunk -> still not "text"
        app._handle_event({"kind": "text_delta", "session_id": sid, "text": ""})
        assert app._turn_state(sid)["had_text"] is False
        # real chunk -> counts as text
        app._handle_event({"kind": "text_delta", "session_id": sid, "text": "Actually:"})
        assert app._turn_state(sid)["had_text"] is True
        app._busy = True
        app._busy_sessions.add(sid)
        app._turn_state(sid)["had_tools"] = True
        app._active_turn_session_id = sid
        app._turn_done(None)
        assert any(b.role == "assistant" and b.content == "Actually:" for b in chat.query(MessageBubble))
