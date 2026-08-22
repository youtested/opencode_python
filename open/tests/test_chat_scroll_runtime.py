"""Tests for chat scrolling (don't yank the user to the bottom while reading
history) and the per-turn runtime display (`▣ Build · model · 1m 12s`).
"""

from __future__ import annotations

import time

import pytest
from textual.app import App, ComposeResult

from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.chat_view import ChatView, MessageBubble
from opencode_py.tui.input_bar import InputBar, format_duration
from opencode_py.tui.markdown_renderer import render_markdown


# --------------------------------------------------------------------------
# Markdown render cache: repeat renders of the same (text, width) must be ~0ms
# --------------------------------------------------------------------------

def test_render_markdown_reuses_cached_renderable():
    render_markdown.cache_clear()
    text = "Hello **world** with a `code` span and [a link](https://x.dev)."
    first = render_markdown(text, width=80)
    hits0, misses0 = render_markdown.cache_info().hits, render_markdown.cache_info().misses
    second = render_markdown(text, width=80)
    info = render_markdown.cache_info()
    assert info.misses == misses0, "repeat render must be a cache hit, not a re-parse"
    assert info.hits == hits0 + 1
    assert second is first, "cached bubble must be reused, making redraws free"
    assert str(second) == str(first)


def test_render_markdown_cache_keyed_by_width():
    render_markdown.cache_clear()
    text = "- item\n- another item"
    narrow = render_markdown(text, width=40)
    wide = render_markdown(text, width=120)
    assert narrow is not wide
    assert render_markdown.cache_info().misses == 2
    # same width -> identical cached object again
    assert render_markdown(text, width=120) is wide


def test_render_markdown_redraw_loop_is_cached():
    render_markdown.cache_clear()
    text = "## Title\n\nground truth: " + "**b** " * 40 + "\n\n- a\n- b\n- c"
    render_markdown(text, width=100)  # warm
    hits_before = render_markdown.cache_info().hits
    for _ in range(500):  # spinner ticks / redraws: all cache hits
        render_markdown(text, width=100)
    hits_after = render_markdown.cache_info().hits
    assert hits_after - hits_before == 500


class WidgetHost(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


# --------------------------------------------------------------------------
# format_duration mirrors opencode's Locale.duration
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.1, "100ms"),
        (0.312, "312ms"),
        (12.5, "12.5s"),
        (72.3, "1m 12s"),
        (61.0, "1m 1s"),
        (3600.0, "1h 0m"),
        (3900.0, "1h 5m"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


# --------------------------------------------------------------------------
# Scroll: the chat must not yank the user to the bottom while reading history
# --------------------------------------------------------------------------

async def _scrolled_chat(pilot, chat) -> None:
    for i in range(30):
        chat.append_meta(f"line {i} " * 30)
    await pilot.pause(0.2)
    chat.focus()
    await pilot.press("pageup")  # user reads earlier history
    await pilot.pause(0.2)
    assert chat.scroll_y < chat.max_scroll_y


async def test_turn_done_shows_runtime_in_mode_line():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_state(app.session.id)["started"] = time.monotonic() - 72.3
        app._turn_state(app.session.id)["had_text"] = True
        app._turn_done()
        await pilot.pause()
        title = bar.query_one("#prompt-title").render()
        text = str(title)
        assert "1m 12s" in text
        assert "Build" in text


async def test_turn_done_no_runtime_without_started():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_state(app.session.id)["started"] = None
        app._turn_done()
        await pilot.pause()
        title = bar.query_one("#prompt-title").render()
        text = str(title)
        assert "·" not in text.replace("▣", "") or "m " not in text


# The runtime mirrors official opencode: it appears only on the final report
# (a real text answer), and it disappears the moment the model starts doing
# things again.

async def test_turn_done_no_runtime_for_tool_only_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_state(app.session.id)["started"] = time.monotonic() - 12.5
        app._turn_state(app.session.id)["had_tools"] = True
        app._turn_state(app.session.id)["had_text"] = False
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == ""


async def test_turn_done_no_runtime_for_error_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_state(app.session.id)["started"] = time.monotonic() - 12.5
        app._turn_state(app.session.id)["had_text"] = True
        app._turn_state(app.session.id)["had_error"] = True
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == ""


async def test_turn_done_no_runtime_for_interrupted_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_state(app.session.id)["started"] = time.monotonic() - 12.5
        app._turn_state(app.session.id)["had_text"] = True
        app._turn_state(app.session.id)["interrupted"] = True
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == ""


async def test_new_turn_clears_previous_runtime():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        app._turn_state(app.session.id)["started"] = time.monotonic() - 72.3
        app._turn_state(app.session.id)["had_text"] = True
        app._turn_done()
        await pilot.pause()
        assert bar.last_duration == "1m 12s"
        # the model starts doing things again -> the runtime disappears
        app._clear_last_duration()
        await pilot.pause()
        assert bar.last_duration == ""
        title = bar.query_one("#prompt-title").render()
        assert "1m 12s" not in str(title)


async def test_input_bar_set_last_duration_renders():
    app = WidgetHost(lambda: InputBar())
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        bar.set_header(agent="build", model="opencode/x", provider="opencode", permission_mode="auto")
        bar.set_last_duration("1m 12s")
        await pilot.pause()
        title = bar.query_one("#prompt-title").render()
        assert "1m 12s" in str(title)
