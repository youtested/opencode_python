"""Tests for the second TUI bug-fix round.

Covers: /undo re-entrant call_from_thread crash (A1), delta batching (A2),
tool_call finalizing the assistant bubble (A3), tool-only turns (A4),
busy-guarded slash commands, pruned-session clicks, tool_denied input rows,
the interrupted event, the permission-dialog exit hang, raw config-key
preservation, the model-picker context formatting, and InputBar history
navigation.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import threading
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from opencode_py.config import Config, save_config
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.chat_view import ChatView, MessageBubble, collapse_tool_output
from opencode_py.tui.input_bar import InputBar, PromptSubmitted
from opencode_py.tui.model_picker import ModelPicker
from opencode_py.tui.settings_screen import SettingsScreen


class FakeEngine:
    agent = "build"
    permission = type("P", (), {"mode": "auto"})()


class WidgetHost(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


async def _mounted_bubble(run: dict) -> MessageBubble:
    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.append_tool(run)
        bubbles = list(chat.query(MessageBubble))
        return bubbles[-1]


class PromptHost(App):
    """Records PromptSubmitted messages so tests can assert on the exact text
    the app would send for a prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputBar()

    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        self.submitted.append(event.value)


async def _tap(ta, *keys: str) -> None:
    """Dispatch key events straight into the textarea, back-to-back.

    A real terminal delivers a paste burst in a single read (microsecond gaps),
    but the test driver sleeps between keys — that would make even a genuine
    paste look like separate deliberate key presses. Dispatching directly keeps
    the inter-key gap ~0, matching a real paste.
    """
    from textual import events

    for k in keys:
        char = k if k.isalnum() else None
        await ta._on_key(events.Key(k, char))


async def test_multiline_paste_stays_one_prompt():
    """A multi-line paste arriving as raw key events (Termux IME paste, no
    bracketed-paste markers: each line break is an `enter` then `ctrl+j`) must
    be captured as newlines in the box, NOT submitted line-by-line."""
    app = PromptHost()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        ta = bar.input
        await _tap(ta, "l", "i", "n", "e", "1", "enter", "ctrl+j", "l", "i", "n", "e", "2")
        assert ta.text == "line1\nline2", "paste must insert one newline per line break"
        assert not app.submitted, "no line of the paste may be sent on its own"
        assert ta.text, "input must not be cleared by the paste"

        # a deliberate Enter afterwards submits the WHOLE block as ONE message
        ta._last_key_mono = 0.0
        await _tap(ta, "enter")
        await pilot.pause()
        assert app.submitted == ["line1\nline2"], "the whole pasted block must send as one prompt"
        assert ta.text == ""


async def test_enter_still_submits_after_normal_typing():
    """A normal Enter (arriving after the user stopped typing, not inside a
    paste burst) still submits the prompt as before."""
    app = PromptHost()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        ta = bar.input
        ta.text = "hello world"
        ta.move_cursor(ta.document.end)
        ta._last_key_mono = 0.0  # last key long ago -> not a paste
        await _tap(ta, "enter")
        await pilot.pause()
        assert app.submitted == ["hello world"]
        assert ta.text == ""


# --------------------------------------------------------------------------
# A1: /undo (and any command that makes the engine emit) must not crash the UI
# thread via a re-entrant call_from_thread.
# --------------------------------------------------------------------------

async def test_undo_command_from_ui_thread_does_not_crash():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        fd, path = tempfile.mkstemp()
        os.write(fd, b"new")
        os.close(fd)
        app.engine._undo_stack.append({"path": path, "original": b"old"})
        app._run_command("/undo")
        await pilot.pause()
        with open(path, "rb") as fh:
            assert fh.read() == b"old"
        os.unlink(path)


async def test_engine_event_from_ui_thread_handled_inline():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "hi"})
        app._flush_deltas()
        await pilot.pause()
        chat = app._chat_for(sid)
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert assistants and assistants[-1].content == "hi"


# --------------------------------------------------------------------------
# A2: deltas are batched into a single render instead of one per token.
# --------------------------------------------------------------------------

async def test_delta_batching_renders_once():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "a"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "b"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "c"})
        # buffered, not yet rendered to a bubble
        assert chat._stream_bubble is None
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 1
        assert assistants[0].content == "abc"


# --------------------------------------------------------------------------
# A3: a tool_call finalizes the assistant bubble; the next step's text must
# land in a fresh bubble (no merged text, no stale cursor).
# --------------------------------------------------------------------------

async def test_tool_call_finalizes_stream_and_new_text_is_new_bubble():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "Let me"})
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": " check"})
        app._on_engine_event(
            {
                "kind": "tool_call",
                "session_id": sid,
                "tool": "glob",
                "arguments": {"pattern": "*.py"},
                "call_id": "c1",
            }
        )
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        assistants = [b for b in bubbles if b.role == "assistant"]
        assert assistants and assistants[-1].content == "Let me check"
        assert assistants[-1].streaming is False
        tools = [b for b in bubbles if b.role == "tool"]
        assert tools and tools[-1].content.get("tool") == "glob"
        # a new tool-loop step's text must not merge into the previous bubble
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "Found"})
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 2
        assert assistants[-1].content == "Found"


async def test_prompt_promoted_clears_badge_and_next_text_is_fresh_bubble():
    """When the engine folds a queued prompt into the running turn (Session
    Drain), the TUI drops the ` QUEUED ` badge on that message and finalizes the
    previous reply — the folded prompt's answer streams into a NEW bubble, like
    opencode's 'pauses briefly, receives the chat, then reasons' flow."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        # first reply is streaming
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "working on it"})
        app._flush_deltas()
        await pilot.pause()
        # user queued a second prompt while busy
        queued = chat.append_user("immediately stop", queued=True)
        assert queued.queued is True
        assert chat.queued_count() == 1
        # engine promotes it into the same drain
        app._on_engine_event({"kind": "prompt_promoted", "session_id": sid, "text": "immediately stop"})
        await pilot.pause()
        assert chat.queued_count() == 0
        assert queued.queued is False
        # prior reply finalized, folded prompt's answer is a fresh bubble
        app._on_engine_event({"kind": "text_delta", "session_id": sid, "text": "stopping now"})
        app._flush_deltas()
        await pilot.pause()
        assistants = [b for b in chat.query(MessageBubble) if b.role == "assistant"]
        assert len(assistants) == 2
        assert assistants[0].content == "working on it"
        assert assistants[1].content == "stopping now"


async def test_reasoning_then_tool_call_does_not_leave_stream_bubble():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "reasoning_delta", "session_id": sid, "text": "think"})
        app._flush_deltas()
        app._on_engine_event(
            {
                "kind": "tool_call",
                "session_id": sid,
                "tool": "bash",
                "arguments": {"command": "ls"},
                "call_id": "c2",
            }
        )
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        # no empty assistant stream bubble lingering above the tool row
        assert not any(b.role == "assistant" and b.content == "" for b in bubbles)


async def test_reasoning_header_shows_thought_duration():
    """The collapsed reasoning header must show how long the model thought:
    `+ Thought for X.Xs` (mirrors opencode), once reasoning ends."""
    from rich.console import Console

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.stream_reasoning_delta("think carefully")
        await pilot.pause()
        bubble = chat.last_reasoning()
        assert bubble is not None
        assert bubble.streaming is True
        # while streaming it's the spinner; after end_reasoning the duration shows
        chat.end_reasoning()
        await pilot.pause()
        assert bubble.streaming is False
        assert bubble._thought_seconds is not None
        console = Console(width=120, record=True, file=io.StringIO())
        console.print(bubble._build_content())
        plain = console.export_text()
        assert "Thought for" in plain
        assert f"{bubble._thought_seconds:.1f}s" in plain


async def test_eager_thinking_bubble_appears_immediately():
    """Pressing Enter must mount an immediate `Thinking...` bubble, before the
    provider sends anything back (the eager placeholder)."""
    from rich.console import Console

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.begin_thinking()
        await pilot.pause()
        bubble = chat.last_reasoning()
        assert bubble is not None
        assert bubble.streaming is True
        assert bubble.content == ""
        # the spinner + Thinking label is visible right away
        console = Console(width=120, record=True, file=io.StringIO())
        console.print(bubble._build_content())
        assert "Thinking" in console.export_text()
        # if no reasoning ever arrives, end_reasoning drops the placeholder
        chat.end_reasoning()
        await pilot.pause()
        assert list(chat.query(MessageBubble)) == []
        assert chat.last_reasoning() is None


async def test_eager_thinking_receives_real_reasoning():
    """A reasoning delta must stream into the eager placeholder bubble (single
    bubble, not a second one), and end up as a `Thought for X.Xs` header."""
    from rich.console import Console

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.begin_thinking()
        chat.stream_reasoning_delta("here is my plan")
        await pilot.pause()
        bubbles = list(chat.query(MessageBubble))
        assert len(bubbles) == 1
        assert bubbles[0].role == "reasoning"
        assert bubbles[0].content == "here is my plan"
        chat.end_reasoning()
        await pilot.pause()
        console = Console(width=120, record=True, file=io.StringIO())
        console.print(bubbles[0]._build_content())
        plain = console.export_text()
        assert "Thought for" in plain


# --------------------------------------------------------------------------
# A4: a tool-only turn must not claim "no reply from the model".
# --------------------------------------------------------------------------

async def test_tool_only_turn_does_not_report_no_reply():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        app._turn_state(sid)["had_tools"] = True
        app._turn_done(result=None)
        chat = app._chat_for(sid)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert not any("no reply from the model" in str(m) for m in metas)


async def test_turn_done_still_reports_no_reply_when_nothing_happened():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._turn_done(result=None)
        chat = app._chat_for(app.session.id)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("no reply from the model" in str(m) for m in metas)


# --------------------------------------------------------------------------
# Busy guard: mutating slash commands are blocked while a turn runs.
# --------------------------------------------------------------------------

async def test_busy_blocks_mutating_command():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        app.on_prompt_submitted(PromptSubmitted("/undo"))
        chat = app._chat_for(app.session.id)
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("still working" in str(m) for m in metas)


async def test_busy_allows_safe_command():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._busy = True
        ran: list[str] = []
        app._run_command = lambda line: ran.append(line)
        app.on_prompt_submitted(PromptSubmitted("/help"))
        assert ran == ["/help"]


# --------------------------------------------------------------------------
# Pruned sub-agent: clicking its task row must not open an empty chat wired to
# the main engine.
# --------------------------------------------------------------------------

async def test_switch_to_pruned_session_does_not_switch():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._pruned.add("dead")
        app._switch_session("dead")
        assert app._current_session_id == app.session.id


async def test_subagent_done_keeps_reviewable_session():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = "sub1"
        chat = app._chat_for(sid)
        app._chats[sid] = chat
        app._engines[sid] = FakeEngine()
        app._sessions[sid] = type("S", (), {"completed": None, "parent_id": None, "agent": "build"})()
        app._busy_sessions.add(sid)
        app._running_agents[sid] = "t · build"
        app._on_subagent_done(
            {"kind": "subagent_done", "session_id": sid, "agent": "build", "title": "t", "ok": True}
        )
        # official store behaviour: finished children stay registered so their
        # task rows remain clickable and the footer's (2 of N) count persists
        assert sid in app._sessions
        assert sid in app._chats
        assert sid not in app._busy_sessions
        assert sid not in app._running_agents


# --------------------------------------------------------------------------
# tool_denied must render a row with the tool input even without a prior
# tool_call event.
# --------------------------------------------------------------------------

async def test_tool_denied_appends_input_row():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event(
            {
                "kind": "tool_denied",
                "session_id": sid,
                "tool": "write",
                "reason": "rejected by user",
                "call_id": "c9",
                "input": {"filePath": "x.py"},
            }
        )
        await pilot.pause()
        tools = [b for b in chat.query(MessageBubble) if b.role == "tool"]
        assert tools
        assert tools[-1].content.get("tool") == "write"
        assert tools[-1].content.get("input") == {"filePath": "x.py"}
        assert tools[-1].content.get("output") == "rejected by user"


# --------------------------------------------------------------------------
# Running tools show the actual action (like opencode's InlineToolRow), not a
# generic "~ Preparing..." placeholder.
# --------------------------------------------------------------------------

async def test_running_tool_shows_action_not_placeholder():
    cases = [
        ({"tool": "edit", "input": {"filePath": "/x.py"}, "call_id": "c1"}, "← Edit /x.py"),
        ({"tool": "write", "input": {"filePath": "/y.py"}, "call_id": "c2"}, "← Write /y.py"),
        ({"tool": "glob", "input": {"pattern": "**/*.py", "path": "src"}, "call_id": "c3"}, '✱ Glob "**/*.py" in src'),
        ({"tool": "grep", "input": {"pattern": "def foo", "path": "src"}, "call_id": "c4"}, '✱ Grep "def foo" in src'),
        ({"tool": "webfetch", "input": {"url": "https://a.com"}, "call_id": "c5"}, "% WebFetch https://a.com"),
    ]
    for run, expected in cases:
        bubble = await _mounted_bubble({**run, "status": "running"})
        content = bubble._build_content()
        plain = content.plain if hasattr(content, "plain") else str(content)
        assert "~ " not in plain, (run["tool"], plain)
        assert expected in plain, (run["tool"], plain)


async def test_pending_tool_still_shows_placeholder_when_input_unknown():
    bubble = await _mounted_bubble(
        {"tool": "edit", "status": "running", "input": {}, "call_id": "c9"}
    )
    content = bubble._build_content()
    plain = content.plain if hasattr(content, "plain") else str(content)
    assert "~ Preparing edit..." in plain


# --------------------------------------------------------------------------
# Streaming performance: a long in-flight assistant message renders as fast
# plain text (avoiding O(n^2) markdown re-parses on every flush); the final
# message still gets the full markdown render once streaming ends.
# --------------------------------------------------------------------------

async def test_streaming_long_message_uses_fast_plain_render():
    def flattened(renderable) -> str:
        if hasattr(renderable, "plain"):
            return renderable.plain
        if hasattr(renderable, "renderables"):
            return "".join(flattened(r) for r in renderable.renderables)
        return ""

    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.begin_stream()
        chat.stream_delta("**bold** text " * 400)  # ~5200 chars > threshold
        bubble = chat._stream_bubble
        plain = flattened(bubble._build_content())
        assert "**bold**" in plain  # raw markers -> not markdown-rendered
        chat.end_stream()
        plain = flattened(bubble._build_content())
        assert "**bold**" not in plain  # final render is full markdown
        assert "bold" in plain


# --------------------------------------------------------------------------
# Unlocked rotation: when the engine fails over to a backup lane the model
# shown under the input box must switch automatically (deepseek -> nemotron).
# --------------------------------------------------------------------------

async def test_rotated_event_updates_header_model():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.cfg.model = "deepseek-v4-flash-free"
        app.cfg.provider = "opencode"
        # The header reflects the ENGINE lane, not cfg, once the background
        # warm-up finishes — sync the engine too so the first assertion is
        # deterministic regardless of whether the warm thread has landed yet.
        eng = app._engines.get(app.session.id)
        if eng is not None:
            eng.model_id = "deepseek-v4-flash-free"
            eng.provider_id = "opencode"
        app._update_header()
        bar = app.query_one(InputBar)
        assert "deepseek-v4-flash-free" in bar.model
        app._on_engine_event(
            {
                "kind": "rotated",
                "session_id": app.session.id,
                "provider": "openrouter",
                "model": "nemotron-nano",
                "reason": "timeout",
            }
        )
        await pilot.pause()
        bar = app.query_one(InputBar)
        assert bar.model == "nemotron-nano"
        assert bar.provider == "openrouter"


# --------------------------------------------------------------------------
# The interrupted event is surfaced instead of silently dropped.
# --------------------------------------------------------------------------

async def test_interrupted_event_shows_meta_and_marks_turn():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sid = app.session.id
        chat = app._chat_for(sid)
        app._on_engine_event({"kind": "interrupted", "session_id": sid})
        await pilot.pause()
        assert app._turn_state(sid)["interrupted"] is True
        metas = [b.content for b in chat.query(MessageBubble) if b.role == "meta"]
        assert any("Interrupted" in str(m) for m in metas)


# --------------------------------------------------------------------------
# Permission dialog: quitting the app must unblock the engine thread quickly.
# --------------------------------------------------------------------------

async def test_permission_ask_unblocks_on_exit():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        holder: dict[str, str] = {}

        def worker() -> None:
            holder["result"] = app._permission_ask("run this command?", [])

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        await asyncio.sleep(0.05)
        app._exit_requested.set()
        t.join(timeout=3)
        assert not t.is_alive(), "permission ask hung after exit"
        assert holder.get("result") == "reject"


# --------------------------------------------------------------------------
# Config: save_config must preserve unknown raw keys (mcpServers/plugins/tools).
# --------------------------------------------------------------------------

def test_save_config_preserves_raw_keys(tmp_path):
    cfg = Config.from_dict(
        {
            "model": "opencode/foo",
            "mcpServers": {"local": {"command": "npx"}},
            "plugins": ["@opencode/plugin-ts"],
            "tools": {"bash": {"deny": "*"}},
        },
        Path("."),
    )
    p = tmp_path / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["model"] == "opencode/foo"
    assert data["mcpServers"] == {"local": {"command": "npx"}}
    assert data["plugins"] == ["@opencode/plugin-ts"]
    assert data["tools"] == {"bash": {"deny": "*"}}


def test_save_config_known_keys_override_raw():
    cfg = Config.from_dict({"model": "opencode/old", "theme": "solarized"}, Path("."))
    cfg.theme = "opencode"
    p = Path(tempfile.mkdtemp()) / "opencode.json"
    save_config(cfg, path=p)
    data = json.loads(p.read_text())
    assert data["theme"] == "opencode"


# --------------------------------------------------------------------------
# Model picker: "128k"-style context strings must not crash int().
# --------------------------------------------------------------------------

def test_format_context_handles_k_and_junk():
    from opencode_py.tui.model_picker import _format_context

    assert _format_context(128000) == "128,000"
    assert _format_context("128k") == "128,000"
    assert _format_context("1m") == "1,000,000"
    assert _format_context("junk") == "junk"
    assert _format_context(None) == "?"
    assert _format_context(0) == "0"


# --------------------------------------------------------------------------
# Chat view: failed tools surface an error line; long write output collapses.
# --------------------------------------------------------------------------

async def test_error_line_shows_failed_tool_error():
    b = await _mounted_bubble({"tool": "read", "status": "error", "error": "No such file"})
    err = b._error_line(b.content)
    assert err is not None and "No such file" in str(err)


async def test_error_line_hidden_for_denial():
    b = await _mounted_bubble({"tool": "read", "status": "error", "output": "user dismissed"})
    assert b._error_line(b.content) is None


def test_write_render_collapses_long_content():
    long = "\n".join(f"line {i}" for i in range(200))
    collapsed = collapse_tool_output(long, 10, 10 * 80)
    assert collapsed["overflow"] is True
    assert "line 199" not in collapsed["output"]
    short = collapse_tool_output("tiny", 10, 10 * 80)
    assert short["overflow"] is False
    assert short["output"] == "tiny"


async def test_write_tool_block_uses_metadata_content():
    b = await _mounted_bubble(
        {
            "tool": "write",
            "status": "completed",
            "input": {"filePath": "x.py"},
            "metadata": {"content": "print('hi')\n"},
        }
    )
    assert b._tool_block() is True


# --------------------------------------------------------------------------
# Settings: the "small model" picker must not retarget the app engine.
# --------------------------------------------------------------------------

def test_small_model_row_does_not_propagate():
    screen = SettingsScreen(cfg=Config(), engine=FakeEngine(), auth=None)
    rows = screen._build_rows()
    model_row = next(r for r in rows if r.label == "model")
    small_row = next(r for r in rows if r.label == "small model")
    assert model_row.propagate is True
    assert small_row.propagate is False


# --------------------------------------------------------------------------
# InputBar history: repeated Up must not clobber the typed draft, Down must
# restore it, and Down/Up with no history must never wipe the input.
# --------------------------------------------------------------------------

async def _mounted_bar(history: list[str], pilot) -> InputBar:
    bar = pilot.app.query_one(InputBar)
    bar._history = list(history)
    bar._hist_index = len(bar._history)
    return bar


async def test_repeated_up_preserves_typed_draft():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["c1", "c2"], pilot)
        bar.input.value = "my draft"
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c2"
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c1"
        assert bar._draft == "my draft"


async def test_down_clears_box_then_up_restores_draft():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["c1", "c2"], pilot)
        bar.input.value = "my draft"
        bar._handle_arrow("up")
        bar._handle_arrow("up")
        assert bar._handle_arrow("down") is True
        assert bar.input.value == "c2"
        # final ↓ past the newest history entry clears the box…
        assert bar._handle_arrow("down") is True
        assert bar.input.value == ""
        assert bar._hist_index == 2
        # …and the next ↑ brings back what we were typing
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "my draft"
        # another ↑ walks into the recent history
        assert bar._handle_arrow("up") is True
        assert bar.input.value == "c2"


async def test_down_clears_typed_input_with_no_history():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar([], pilot)
        bar.input.value = "typed text"
        assert bar._handle_arrow("down") is True
        assert bar.input.value == ""


async def test_up_with_no_history_keeps_typed_input():
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar([], pilot)
        bar.input.value = "typed text"
        assert bar._handle_arrow("up") is False
        assert bar.input.value == "typed text"


# --------------------------------------------------------------------------
# Model picker: providers as headers with models underneath, free first,
# and a search box that filters the rendered rows.
# --------------------------------------------------------------------------

class PickerHost(App):
    """App that mounts a ModelPicker directly (so populate can be inspected)."""

    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


class NavPicker(ModelPicker):
    """ModelPicker with the network worker disabled so tests are deterministic."""

    def _start_worker(self) -> None:
        pass


class PickerPushHost(App):
    """App that pushes a NavPicker via push_screen (loads its CSS) and can
    capture the chosen "provider/model" on dismiss."""

    def __init__(self) -> None:
        super().__init__()
        self._picker = NavPicker()
        self.choice = None

    def on_mount(self) -> None:
        self.push_screen(self._picker, self._on_choice)

    def _on_choice(self, choice: str | None) -> None:
        self.choice = choice


def test_picker_row_label_free_and_current():
    from opencode_py.tui.model_picker import _model_row_label

    row = _model_row_label("opencode/deepseek-v4-flash-free", {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash", "free": True}, "opencode/deepseek-v4-flash-free")
    text = row.render().plain
    assert "FREE" in text
    assert "DeepSeek V4 Flash" in text
    # current model marked with the bullet
    assert "\u25cf" in text


def test_picker_row_label_free_sort_key():
    from opencode_py.tui.model_picker import _model_row_label

    free_row = _model_row_label("p/a", {"id": "a", "free": True}, "")
    paid_row = _model_row_label("p/b", {"id": "b", "free": False}, "")
    assert "FREE" in free_row.render().plain
    assert "FREE" not in paid_row.render().plain


async def test_picker_renders_providers_with_models_and_free_first():
    from opencode_py.tui.model_picker import ModelPicker

    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "openai": [{"id": "gpt-4o", "name": "GPT-4o", "free": False}],
                "groq": [{"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "free": True}],
            }
        )
        await pilot.pause()
        lv = picker.query_one("#models-list")
        rendered = [str(item.children[0].render().plain) for item in lv.query("ListItem")]
        # both provider headers present
        assert any("OpenAI" in r for r in rendered)
        assert any("Groq" in r for r in rendered)
        # free provider's models appear before paid provider's models
        groq_idx = next(i for i, r in enumerate(rendered) if "Groq" in r)
        openai_idx = next(i for i, r in enumerate(rendered) if "OpenAI" in r)
        assert groq_idx < openai_idx
        # free model carries the FREE tag
        assert any("Llama 3.3 70B" in r and "FREE" in r for r in rendered)


async def test_zen_section_groups_free_first_then_family():
    # OpenCode Zen mixes many upstream vendors; free models must come first
    # under a "Free" sub-group, then non-free models grouped by upstream family.
    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "opencode": [
                    {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol", "free": False},
                    {"id": "claude-opus-4-5", "name": "Claude Opus 4.5", "free": False},
                    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash", "free": True},
                    {"id": "big-pickle", "name": "Big Pickle", "free": True},
                    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "free": False},
                ]
            }
        )
        await pilot.pause()
        lv = picker.query_one("#models-list")
        rendered = [str(item.children[0].render().plain) for item in lv.query("ListItem")]
        # header first
        assert rendered[0] == "  OpenCode Zen"
        # free sub-group comes before any family sub-group...
        free_idx = rendered.index("   Free")
        gpt_idx = rendered.index("   OpenAI")
        anthropic_idx = rendered.index("   Anthropic")
        google_idx = rendered.index("   Google")
        assert free_idx < min(gpt_idx, anthropic_idx, google_idx)
        # ...and free models sit right under "Free"
        assert rendered[free_idx + 1] == "   Big Pickle FREE"
        assert rendered[free_idx + 2] == "   DeepSeek V4 Flash FREE"
        # family sub-groups sorted alphabetically by family name
        assert anthropic_idx < google_idx < gpt_idx


async def test_zen_family_maps_upstream_vendor():
    from opencode_py.tui.model_picker import _zen_family

    assert _zen_family("gpt-5.6-sol") == "OpenAI"
    assert _zen_family("claude-opus-4-5") == "Anthropic"
    assert _zen_family("gemini-3.6-flash") == "Google"
    assert _zen_family("kimi-k3") == "Moonshot"
    assert _zen_family("grok-4.6") == "xAI"
    assert _zen_family("qwen3.6-plus") == "Alibaba"
    assert _zen_family("glm-5.2") == "Zhipu AI"
    assert _zen_family("pizza-bot") == "Other"


async def test_picker_search_filters_models():
    from opencode_py.tui.model_picker import ModelPicker
    from textual.widgets import Input

    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "openai": [
                    {"id": "gpt-4o", "name": "GPT-4o", "free": False},
                    {"id": "gpt-4o-mini", "name": "GPT-4o mini", "free": False},
                ]
            }
        )
        await pilot.pause()
        box = picker.query_one("#models-search", Input)
        box.value = "mini"
        picker._populate_list()
        await pilot.pause()
        lv = picker.query_one("#models-list")
        rendered = [str(item.children[0].render().plain) for item in lv.query("ListItem")]
        assert any("GPT-4o mini" in r for r in rendered)
        assert not any(r.endswith("GPT-4o") for r in rendered if "mini" not in r)
        # header remains so the provider context is visible
        assert any("OpenAI" in r for r in rendered)


# --------------------------------------------------------------------------
# Regression: the picker's list-builder must not be named `_render`, which
# would shadow Textual's internal Widget._render() (returns the widget's
# Visual). When the screen gets flagged dirty, Textual calls `_render()` on it;
# a wrong return type crashes the compositor with
# "AttributeError: 'NoneType' object has no attribute 'render_strips'".
# --------------------------------------------------------------------------

def test_picker_does_not_shadow_textual_render():
    from opencode_py.tui.model_picker import ModelPicker

    assert ModelPicker._render.__qualname__ == "Widget._render"
    # the method that builds the list rows must exist under the renamed id
    assert hasattr(ModelPicker, "_populate_list")


async def test_picker_survives_compositor_rerender():
    from opencode_py.tui.model_picker import ModelPicker

    picker = ModelPicker()
    host = PickerHost(lambda: picker)
    async with host.run_test() as pilot:
        await pilot.pause()
        picker.populate(
            {
                "openai": [{"id": "gpt-4o", "name": "GPT-4o", "free": False}],
            }
        )
        await pilot.pause()
        # force the screen (a widget itself) to re-render its own content as
        # the compositor does after any layout/style invalidation
        for _ in range(3):
            picker.refresh()
            await pilot.pause()
        lv = picker.query_one("#models-list")
        # header + model row + "Add custom provider" section (its header + row)
        assert len(list(lv.query("ListItem"))) == 4


# --------------------------------------------------------------------------
# Ctrl+M opens the model picker. On most terminals Ctrl+M sends the same
# byte as Enter, so an empty Enter must NOT open the picker (it does nothing);
# a non-empty Enter still submits instead.
# --------------------------------------------------------------------------

async def _run_models_press(keys: list[str]) -> str:
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        for key in keys:
            await pilot.press(key)
            await pilot.pause()
        return type(app.screen).__name__


async def test_ctrl_m_opens_model_picker():
    assert (await _run_models_press(["ctrl+m"])) == "ModelPicker"


async def test_empty_enter_does_not_open_model_picker():
    # a real terminal delivers Ctrl+M as the Enter byte (\r), which Textual
    # normalizes to "enter" -- an empty Enter must be swallowed, not treated
    # as a request to open the models list
    assert (await _run_models_press(["enter"])) != "ModelPicker"


async def test_typed_enter_submits_not_opens_picker():
    from opencode_py.tui.input_bar import InputBar

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).input.value = "hello world"
        await pilot.press("enter")
        await pilot.pause()
        assert type(app.screen).__name__ != "ModelPicker"


# --------------------------------------------------------------------------
# Regression: the picker's CSS used to be a module-level constant that was
# never attached to the class, so none of the styling applied (providers and
# models rendered the same default color). Verify the CSS is loaded and the
# key rules (purple provider headers, borderless search bar) take effect.
# --------------------------------------------------------------------------

def test_picker_css_is_attached_to_class():
    from opencode_py.tui.model_picker import ModelPicker

    assert hasattr(ModelPicker, "CSS") and ModelPicker.CSS.strip()
    assert "#models-search" in ModelPicker.CSS
    assert "group-header" in ModelPicker.CSS


async def test_picker_header_and_search_styles_apply():
    # push_screen is what triggers Textual's `_load_screen_css`, i.e. the
    # exact path the real app uses to attach the picker's CSS
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(
            {"groq": [{"id": "llama-3.3-70b-versatile", "name": "L", "free": True}]}
        )
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        header = list(lv.query("ListItem"))[0].children[0]
        assert header.styles.color.rgb == (157, 124, 216)  # #9d7cd8 purple
        box = host._picker.query_one("#models-search")
        assert box.styles.border.top[0] in ("none", "")
        # header row carries the title and the esc hint
        header_row = host._picker.query_one("#models-header")
        assert [c.render().plain for c in header_row.children] == ["Models", "esc"]


# --------------------------------------------------------------------------
# Navigation: the search box keeps focus (like opentui's dialog filter) while
# Up/Down move the highlight across the models (skipping provider headers) and
# Enter selects the highlighted model and dismisses with "provider/model".
# --------------------------------------------------------------------------

PICKER_DATA = {
    "groq": [{"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "free": True}],
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o", "free": False},
        {"id": "gpt-4o-mini", "name": "GPT-4o mini", "free": False},
    ],
}


async def test_picker_arrows_move_between_model_rows():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        # rows: 0 Groq header, 1 llama, 2 OpenAI header, 3 gpt-4o, 4 gpt-4o-mini
        assert lv.index == 1  # first model row, not the header
        await pilot.press("down")
        await pilot.pause()
        assert lv.index == 3  # skips the OpenAI header
        await pilot.press("down")
        await pilot.pause()
        assert lv.index == 4
        await pilot.press("down")
        await pilot.pause()
        assert lv.index == 4  # clamped at the last model
        await pilot.press("up")
        await pilot.pause()
        assert lv.index == 3


async def test_picker_enter_selects_and_dismisses():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        await pilot.press("down")  # llama -> gpt-4o
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert host.choice == "openai/gpt-4o"


async def test_picker_refresh_rebuild_does_not_double_dismiss():
    """Refresh/search fires a spurious ListView.Selected from programmatic
    `lv.index` assignment; a later real pick (or Esc) must not call dismiss
    again. Previously the second dismiss hit pop_screen with a single-element
    stack and raised ScreenStackError (CRASH: the picker "selecting thing
    disappeared" then the whole app threw)."""
    import textual.app as _textual_app

    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        # simulate the refresh path: rebuild the list (sets lv.index, which
        # would fire a Selected while `_rebuilding`), then select a row
        host._picker.action_refresh_models()
        await pilot.pause()
        host._picker._populate_list()
        await pilot.pause()
        # a genuinely queued Selected fired against the rebuilt list, pointing
        # at whatever the rebuild re-highlighted (the first model row)
        lv = host._picker.query_one("#models-list")
        host._picker.on_list_view_selected(_FakeSelected(lv.index))
        await pilot.pause()
        await pilot.pause()
        assert host.choice == "groq/llama-3.3-70b-versatile"
        # pressing escape afterwards must be a no-op, not a second dismiss crash
        await pilot.press("escape")
        await pilot.pause()
        host._picker._close(None)
        await pilot.pause()
        assert host.choice == "groq/llama-3.3-70b-versatile"


class _FakeSelected:
    """Stand-in for ListView.Selected carrying an index."""

    def __init__(self, index):
        self.index = index
        self.item = None


async def test_picker_double_choose_is_idempotent():
    """Two selection events racing (a click + a queued Enter) must both resolve
    to ONE dismiss — the second call is a no-op rather than popping a screen
    off an already-empty stack."""
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        # simulate a click selecting row 2 (gpt-4o) followed by a queued Enter
        host._picker.on_list_view_selected(_FakeSelected(3))
        host._picker._choose_current()
        await pilot.pause()
        await pilot.pause()
        assert host.choice == "openai/gpt-4o"


async def test_picker_search_filter_enter_selects():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        host._picker.populate(PICKER_DATA)
        await pilot.pause()
        box = host._picker.query_one("#models-search")
        box.value = "gpt-4o-mini"
        await pilot.pause()
        lv = host._picker.query_one("#models-list")
        # only OpenAI header + gpt-4o-mini remain: the model row is index 1
        assert lv.index == 1
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert host.choice == "openai/gpt-4o-mini"


async def test_picker_escape_dismisses_without_choice():
    host = PickerPushHost()
    async with host.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert host.choice is None


# -- regression: MessageBubble constructor dropped its own state ------------
# The dead-code bug moved the can_focus / queued / streaming initialization below
# a `return {}` in `_diff_opts`, so constructor args were silently ignored.

def test_message_bubble_constructor_applies_queued_streaming_focus():
    b = MessageBubble("reasoning", "**Title**\n\nbody", queued=True, streaming=True)
    assert b.can_focus is True
    assert b.queued is True
    assert b.streaming is True


def test_message_bubble_only_reasoning_is_focusable():
    assert MessageBubble("reasoning", "").can_focus is True
    assert MessageBubble("assistant", "hi").can_focus is False
    assert MessageBubble("user", "hi").can_focus is False


async def test_cleared_unsent_prompt_never_resurfaces_after_submit():
    """Pressing ↓ clears the whole prompt in one keystroke; that cleared,
    unsent text is not part of history, and once a NEW prompt is actually sent,
    the old text can never come back via the draft on the next ↑."""
    host = WidgetHost(lambda: InputBar())
    async with host.run_test() as pilot:
        bar = await _mounted_bar(["sent one", "sent two"], pilot)
        bar.input.value = "UNSENT draft"
        bar._handle_arrow("up")  # capture the draft, show "sent two"
        bar._handle_arrow("up")  # show "sent one"
        bar._handle_arrow("down")  # show "sent two"
        bar._handle_arrow("down")  # final ↓ clears the box
        assert bar.input.value == ""
        # send a brand-new prompt: the replaced text must be forgotten
        bar.on_prompt_submitted(PromptSubmitted("fresh send"))
        assert bar._draft == ""
        assert bar.input.value == ""
        # the next ↑↑ shows only what was really submitted, newest first
        bar._handle_arrow("up")
        assert bar.input.value == "fresh send"
        bar._handle_arrow("up")
        assert bar.input.value == "sent two"


# --------------------------------------------------------------------------
# Resume round-trip: the full conversation must reach disk, unchanged.
# --------------------------------------------------------------------------

class _TextRotation:
    """Synchronous fake provider: returns text deltas, records nothing else."""

    def __init__(self, text: str) -> None:
        self.text = text

    def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
        from opencode_py.providers.base import ProviderEvent

        on_event(ProviderEvent(kind="text_delta", text=self.text))
        return "opencode", "deepseek-v4-flash-free"


async def _drive_resumed(app, pilot, rotation, prompt: str = "hello again") -> None:
    app._engines["res1"].rotation = rotation
    app.on_prompt_submitted(PromptSubmitted(prompt))
    for _ in range(300):
        await pilot.pause()
        if not app._busy:
            break


async def test_resume_round_trips_full_conversation(monkeypatch, tmp_path):
    """Regression: a resumed session continues on disk with EVERY previous
    message — the saved body must equal the in-memory history after the turn,
    not a truncated/stale snapshot."""
    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    base = [
        {"role": "user", "content": "old q1"},
        {"role": "assistant", "content": "old a1"},
        {"role": "user", "content": "old q2"},
        {"role": "assistant", "content": "old a2"},
    ]
    session_mod.save_session(session_mod.Session({"id": "res1", "title": "old", "created": 1, "messages": list(base)}))

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._resume_session("res1")
        await _drive_resumed(app, pilot, _TextRotation("new reply"))
        await _drive_resumed(app, pilot, _TextRotation("second reply"), prompt="again")

        in_mem = app._sessions["res1"].messages
        # turns themselves don't write; only an exit/close save-all flushes
        app._save_all_live_sessions()
        disk = json.loads((tmp_path / "res1.json").read_text(encoding="utf-8"))["messages"]
        assert disk == in_mem, "disk must match the in-memory history after a save-all"
        roles = [m["role"] for m in disk]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
        assert [m.get("content") for m in disk if m.get("content")] == [
            "old q1", "old a1", "old q2", "old a2", "hello again", "new reply", "again", "second reply",
        ]


async def test_resume_save_never_injects_placeholder_tool_rows(monkeypatch, tmp_path):
    """Regression: save_session used to inject synthetic missing-tool-result
    messages into the persisted transcript. The disk body must stay the real
    conversation — an orphaned tool call is preserved, not replaced by a fake
    result."""
    import json
    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    orphaned = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "real output"},
    ]
    session_mod.save_session(session_mod.Session({"id": "res1", "title": "old", "created": 1, "messages": list(orphaned)}))

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._resume_session("res1")
        await _drive_resumed(app, pilot, _TextRotation("done"))
        disk = json.loads((tmp_path / "res1.json").read_text(encoding="utf-8"))["messages"]
        texts = [m.get("content") for m in disk]
        assert "[interrupted — tool result missing]" not in texts, "no fake placeholder rows on disk"
        # the orphaned tool call and every real message survived
        assert any(m.get("tool_call_id") == "c1" and m.get("content") == "real output" for m in disk)


async def test_on_unmount_saves_idle_resumed_session(monkeypatch, tmp_path):
    """Regression: tokens that landed in the engine's live history but haven't
    been flushed by an autosave/turn-done save must be persisted on unmount —
    otherwise a graceful close right after streaming leaves the picker's copy
    a turn behind and a restart shows an incomplete conversation."""
    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    session_mod.save_session(session_mod.Session({"id": "res1", "title": "old", "created": 1, "messages": [
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
    ]}))

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._resume_session("res1")
        # stream a reply straight into the engine's live history with NO
        # autosave/turn-done in between (the exact gap on_unmount closes)
        engine = app._engines["res1"]
        engine._history.append({"role": "assistant", "content": "just-streamed tokens"})
        app.on_unmount()
        disk = json.loads((tmp_path / "res1.json").read_text(encoding="utf-8"))["messages"]
        contents = [m.get("content") for m in disk if m.get("content")]
        assert "just-streamed tokens" in contents, "unmount must flush un-autosaved history"
        assert "old q" in contents and "old a" in contents, "full conversation must stay intact"


async def test_welcome_logo_shows_on_first_open_and_hides_on_conversation():
    """The opencode ASCII logo banner appears on the very first launch (empty
    chat) and disappears the moment a conversation starts (first message or
    stream). Cleared (deleted) sessions get a fresh logo again."""
    from opencode_py.tui.chat_view import OPENCODE_LOGO

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatView)
        # first open: logo is mounted and the chat is in the "welcome" state
        assert chat.welcome_empty(), "welcome screen should show on first launch"
        logo = chat.query_one(".chat-welcome-logo")
        assert str(logo.content) == OPENCODE_LOGO.rstrip(), "exact opencode logo text"
        assert len(chat.query(MessageBubble)) == 0, "no messages on first open"

        # typing the first message starts the conversation -> logo disappears
        chat.append_user("hello")
        await pilot.pause()
        assert not chat.welcome_empty(), "logo must vanish once the conversation starts"
        assert not chat.query(".chat-welcome-logo"), "logo widget must be unmounted"
        assert len(chat.query(MessageBubble)) == 1, "user message remains after logo hides"

        # clearing the workspace (session deleted) is NOT a fresh first open —
        # the logo belongs to the app's initial launch only, so it stays gone
        chat.clear()
        await pilot.pause()
        assert not chat.welcome_empty(), "logo must not return after a workspace reset"


async def test_toggle_agent_rebuilds_plan_permissions():
    """Switching the agent (Ctrl+T / `/agent`) must rebuild the permission
    engine: plan mode force-denies bash/write/edit/apply_patch, so a plan agent
    can't execute a mutating tool call the model happens to emit. Without the
    rebuild the engine keeps the build-agent "allow" rules for those tools."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        engine = app.engine
        assert engine.agent == "build"
        assert engine.permission.evaluate("bash", "ls -la") == "allow"

        app._set_agent("plan")
        assert engine.agent == "plan"
        for tool in ("bash", "write", "edit", "apply_patch"):
            assert engine.permission.evaluate(tool, "*") == "deny", f"{tool} must be denied in plan"
        for tool in ("read", "glob", "grep", "webfetch"):
            assert engine.permission.evaluate(tool, "*") == "allow", f"{tool} must stay allowed in plan"

        app._set_agent("build")
        assert engine.agent == "build"
        assert engine.permission.evaluate("bash", "ls -la") == "allow"


# --------------------------------------------------------------------------
# B5: no ghost sessions — a conversation must be started before anything is
# persisted, and leftover titled-but-empty files are hidden from the picker.
# --------------------------------------------------------------------------


async def test_picker_hides_persisted_empty_sessions(monkeypatch, tmp_path):
    """A persisted session with a title but ZERO messages (the phantom
    "hello world" leftovers) must not appear in the /sessions picker; a real
    conversation must."""
    import os
    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))

    ghost = session_mod.new_session(directory=str(tmp_path), title="hello world")
    session_mod.save_session(ghost)
    real = session_mod.new_session(directory=str(tmp_path), title="real chat")
    real.messages = [{"role": "user", "content": "hi"}]
    session_mod.save_session(real)

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        captured = []
        app.push_screen = lambda screen, callback=None, *a, **k: captured.append(screen)
        app.action_sessions()
        await pilot.pause()
        assert captured, "picker should have been pushed"
        titles = [s.get("title") for s in captured[0].sessions if s.get("id")]
        assert "hello world" not in titles, "title-only empty session must be hidden"
        assert "real chat" in titles, "real conversation must be listed"


async def test_start_turn_writes_nothing_until_save_condition(monkeypatch, tmp_path):
    """Starting a conversation must NOT write a file immediately. A session
    file is only created by an actual save condition: the 1s crash-safety
    autosave tick, the exit/close save-all, or the Termux-close signal."""
    from unittest import mock

    from opencode_py import session as session_mod
    from opencode_py.agent.loop import AgentLoop
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        engine = app.engine
        dummy = mock.Mock()
        dummy.provider_id = ""
        dummy.model_id = ""
        dummy.error = ""
        dummy.text = ""
        dummy.reasoning = ""
        dummy.tool_calls_made = 0
        dummy.usage = None
        with mock.patch.object(AgentLoop, "run_turn", return_value=dummy):
            app._start_turn(app.session.id, "hello world", engine)
        path = tmp_path / (app.session.id + ".json")
        assert not path.exists(), (
            "starting a turn must NOT persist anything — only exit/close or a "
            "crash-autosave tick writes a file"
        )
        # let the worker finish so _turn_done clears busy/autosave cleanly
        await pilot.pause()


async def test_save_all_live_sessions_persists_only_conversations(monkeypatch, tmp_path):
    """The exit/close save-all persists every live session that has a
    conversation and never writes (or re-writes) an empty one."""
    import json as _json

    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.session.messages = [{"role": "user", "content": "hi"}]
        app._save_all_live_sessions()
        path = tmp_path / (app.session.id + ".json")
        assert path.exists(), "exit/close save-all must write a conversation"
        body = _json.loads(path.read_text(encoding="utf-8"))
        assert body["messages"] == [{"role": "user", "content": "hi"}]

        # strip the conversation back out: the save-all must not (re)write it
        app.session.messages = []
        if path.exists():
            path.unlink()
        app._save_all_live_sessions()
        assert not path.exists(), "an empty session must never be (re)written"


async def test_untouched_launch_leaves_no_session_file(monkeypatch, tmp_path):
    """Opening and closing the TUI without any conversation must not write a
    session file (previously an empty titled file was created eagerly)."""
    import os
    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    app = OpenCodeTUI()
    async with app.run_test():
        pass
    files = [f for f in os.listdir(tmp_path) if f.endswith(".json") and not f.endswith(".bak")]
    assert files == [], f"no conversation -> no session file, got {files}"


# --------------------------------------------------------------------------
# B6: /sessions popup — Save button persists the current session, and Select
# mode allows batch-deleting several sessions at once.
# --------------------------------------------------------------------------


async def test_session_save_button_persists_current_session(monkeypatch, tmp_path):
    """The Save action must durably write the session the user is currently
    in, including its live history."""
    import json as _json
    from opencode_py import session as session_mod
    from opencode_py.tui.app import OpenCodeTUI

    monkeypatch.setattr(session_mod.GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.session.messages = [{"role": "user", "content": "hi"}]
        assert app._save_current_session() is True
        path = tmp_path / (app.session.id + ".json")
        assert path.exists(), "save button must write the current session"
        body = _json.loads(path.read_text(encoding="utf-8"))
        assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_session_save_button_wiring():
    """Clicking the Save button inside the popup calls the app's save hook."""
    from unittest import mock

    from opencode_py.tui.app import OpenCodeTUI

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        captured = []
        app.push_screen = lambda screen, callback=None, *a, **k: captured.append(screen)
        app._save_current_session = mock.Mock(return_value=True)
        app.action_sessions()
        await pilot.pause()
        assert captured, "picker should be pushed"
        screen = captured[0]
        screen.action_save()
        app._save_current_session.assert_called_once()


async def test_session_list_select_and_batch_delete():
    """Select mode toggles checkboxes; deleting removes every selected session
    through the app's on_delete hook (busy sessions are kept and reported)."""
    from opencode_py.tui.app import OpenCodeTUI

    deleted: list[str] = []
    busy_id = "busy"

    def on_delete(session_id: str) -> bool:
        if session_id == busy_id:
            return False  # protected (running) — can't delete
        deleted.append(session_id)
        return True

    sessions = [
        {"id": "a", "title": "one", "agent": "build", "created": 1786000000},
        {"id": "b", "title": "two", "agent": "build", "created": 1786000001},
        {"id": busy_id, "title": "running", "agent": "build", "created": 1786000002},
    ]
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        from opencode_py.tui.session_list import SessionList

        app.push_screen(
            SessionList(sessions, on_delete=on_delete)
        )
        await pilot.pause()
        picker = app.screen
        # enter select mode and mark two sessions
        await picker.action_toggle_select()
        assert picker.select_mode is True
        picker.selected.update({"a", busy_id})
        await picker._after_batch_delete(("batch", True))
        assert sorted(deleted) == ["a"], "only the deletable selection is removed"
        remaining = {s["id"] for s in picker.sessions}
        assert "a" not in remaining
        assert "b" in remaining
        assert busy_id in remaining, "a busy session must stay listed"
        assert picker.selected == {busy_id}, "the failed id stays selected"


async def test_session_select_mode_enter_selects_multiple_without_dismissing():
    """Bug: enter was consumed by ListView and dismissed/resumed instead of
    toggling the checkbox. In select mode, enter must keep selecting rows and
    never close the popup."""
    from opencode_py.tui.app import OpenCodeTUI
    from opencode_py.tui.session_list import SessionList
    from textual.widgets import OptionList

    sessions = [
        {"id": "a", "title": "one", "agent": "build", "created": 1786000002},
        {"id": "b", "title": "two", "agent": "build", "created": 1786000001},
        {"id": "c", "title": "three", "agent": "build", "created": 1786000000},
    ]
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.push_screen(SessionList(sessions, on_delete=lambda sid: True))
        await pilot.pause()
        picker = app.screen
        await picker.action_toggle_select()
        ol = picker.query_one("#session-list", OptionList)
        ol.highlighted = 1  # session "one"
        await pilot.press("enter")  # toggle "one"
        await pilot.press("down")   # highlight "two"
        await pilot.press("enter")  # toggle "two"
        await pilot.pause()
        assert picker.selected == {"a", "b"}, picker.selected
        assert app.screen is picker, "enter must not dismiss in select mode"
