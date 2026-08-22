"""Parallel sub-agent TUI (mirrors the official opencode Subagent component,
SubagentFooter and session.child.* navigation).

Covers: the `Build Task — …` task-row rendering with the live current tool /
toolcall count / completion detail, the `view subagents` hint, the (2 of N)
footer with usage, the children registry surviving subagent_done, the parent/
prev/next/first session routing, and the completion-summary metadata.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.chat_view import (
    ChatView,
    MessageBubble,
    format_completed_subagent_detail,
    format_subagent_retry,
    format_subagent_title,
    format_subagent_toolcalls,
)
from opencode_py.tui.subagent_footer import NavRequested, SubagentFooter


class WidgetHost(App):
    def __init__(self, factory) -> None:
        super().__init__()
        self._factory = factory

    def compose(self) -> ComposeResult:
        yield self._factory()


def _plain(renderable) -> str:
    if hasattr(renderable, "plain"):
        return renderable.plain
    if hasattr(renderable, "renderables"):
        return "".join(_plain(r) for r in renderable.renderables)
    return ""


async def _mounted_bubble(run: dict) -> MessageBubble:
    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.append_tool(run)
        return list(chat.query(MessageBubble))[-1]


# --------------------------------------------------------------------------
# formatSubagent* helpers mirror the official functions literally.
# --------------------------------------------------------------------------

def test_format_subagent_title():
    assert format_subagent_title("build", "fix the login bug") == "Build Task — fix the login bug"
    assert (
        format_subagent_title("plan", "review the API", background=True)
        == "Plan Task (background) — review the API"
    )


def test_format_subagent_helpers():
    assert format_subagent_toolcalls(1) == "1 toolcall"
    assert format_subagent_toolcalls(4) == "4 toolcalls"
    assert format_subagent_retry(2, "provider down") == "Retrying (attempt 2) · provider down"
    assert format_completed_subagent_detail(3, "12.5s") == "3 toolcalls · 12.5s"
    assert format_completed_subagent_detail(0, "12.5s") == "12.5s"


# --------------------------------------------------------------------------
# Task row rendering mirrors the official Subagent component.
# --------------------------------------------------------------------------

async def test_task_row_running_shows_title_and_spinner():
    bubble = await _mounted_bubble(
        {"tool": "task", "status": "running", "input": {"description": "fix login", "subagent_type": "build"}, "call_id": "t1"}
    )
    plain = _plain(bubble._build_content())
    assert "Build Task — fix login" in plain
    assert "Delegating" not in plain


async def test_task_row_completed_shows_icon_toolcalls_duration_and_hint():
    run = {
        "tool": "task",
        "status": "completed",
        "input": {"description": "fix login", "subagent_type": "build"},
        "metadata": {"sessionId": "sub9", "toolcalls": 3, "duration": "1m 12s"},
        "call_id": "t1",
    }
    host = WidgetHost(lambda: ChatView())
    async with host.run_test() as pilot:
        chat = host.query_one(ChatView)
        chat.append_tool(run)
        bubble = list(chat.query(MessageBubble))[-1]
        plain = _plain(bubble._build_content())
        assert "✓ Build Task — fix login" in plain
        assert "↳ 3 toolcalls · 1m 12s" in plain
        assert "view subagents" in plain


async def test_task_row_running_shows_live_current_tool_from_child_chat():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app._chats[app.session.id]
        child = app._chat_for("subA")
        parent.append_tool(
            {"tool": "task", "status": "running", "input": {"description": "port it", "subagent_type": "build"}, "call_id": "t1"}
        )
        bubble = parent.find_tool("task", "t1")
        bubble.set_tool_metadata("sessionId", "subA")
        # the sub-agent starts a bash command — the parent row must now show
        # `↳ Bash npm run test` (opencode reads the child session's parts)
        child.append_tool(
            {"tool": "bash", "status": "running", "input": {"command": "npm run test"}, "call_id": "c1"}
        )
        plain = _plain(bubble._build_content())
        assert "↳ Bash npm run test" in plain


async def test_task_row_running_counts_toolcalls_without_named_current_tool():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app._chats[app.session.id]
        child = app._chat_for("subB")
        parent.append_tool(
            {"tool": "task", "status": "running", "input": {"description": "port it", "subagent_type": "build"}, "call_id": "t1"}
        )
        bubble = parent.find_tool("task", "t1")
        bubble.set_tool_metadata("sessionId", "subB")
        child.append_tool({"tool": "todowrite", "status": "completed", "input": {}, "call_id": "c1"})
        child.append_tool({"tool": "todowrite", "status": "running", "input": {}, "call_id": "c2"})
        plain = _plain(bubble._build_content())
        assert "↳ 2 toolcalls" in plain


async def test_task_row_retry_line_and_error_color():
    bubble = await _mounted_bubble(
        {
            "tool": "task",
            "status": "running",
            "input": {"description": "fix", "subagent_type": "build"},
            "metadata": {"sessionId": "sub9", "retry": {"attempt": 2, "message": "provider down"}},
            "call_id": "t1",
        }
    )
    plain = _plain(bubble._build_content())
    assert "Retrying (attempt 2) · provider down" in plain


# --------------------------------------------------------------------------
# SubagentFooter: `Build (2 of 4)` + usage + position among parallel children.
# --------------------------------------------------------------------------

async def test_subagent_footer_renders_label_index_total_and_usage():
    host = WidgetHost(lambda: SubagentFooter())
    async with host.run_test() as pilot:
        footer = host.query_one(SubagentFooter)
        footer.show(label="Build", index=2, total=4, usage={"total_tokens": 12345, "context_size": 200000})
        info = footer.query_one("#subagent-info")
        plain = info.render().plain
        assert "Build" in plain
        assert "(2 of 4)" in plain
        assert "12,345 (6%)" in plain


async def test_subagent_footer_hidden_by_default():
    host = WidgetHost(lambda: SubagentFooter())
    async with host.run_test() as pilot:
        footer = host.query_one(SubagentFooter)
        assert not footer.display
        footer.hide()
        assert not footer.display


# --------------------------------------------------------------------------
# App-level routing: children registry, task-row linking, (2 of N) footer,
# parent/prev/next/first-children navigation.
# --------------------------------------------------------------------------

def _register_children(app: OpenCodeTUI, parent: str, count: int) -> list[str]:
    sids = []
    for i in range(count):
        sid = f"c{i}"
        sids.append(sid)
        app._chat_for(sid)
        app._sessions[sid] = SimpleNamespace(parent_id=parent, agent="build", title=f"t{i}")
        app._children.setdefault(parent, []).append(
            {"id": sid, "title": f"t{i}", "agent": "build", "created": 100 + i, "status": "completed"}
        )
        app._child_parent[sid] = parent
    return sids


async def test_subagent_start_registers_child_and_links_task_row():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app._chats[app.session.id]
        parent.append_tool(
            {"tool": "task", "status": "running", "input": {"description": "port it", "subagent_type": "build"}, "call_id": "t1"}
        )
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subX", "agent": "build", "title": "port it"})
        assert app._child_parent["subX"] == app.session.id
        records = app._children[app.session.id]
        assert any(r["id"] == "subX" for r in records)
        bubble = parent.find_task("subX")
        assert bubble is not None
        assert bubble.content["metadata"]["sessionId"] == "subX"
        assert app.session.id in app._chats


async def test_subagent_done_keeps_sibling_count():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        _register_children(app, app.session.id, 2)
        app._children[app.session.id].append(
            {"id": "subY", "title": "t", "agent": "build", "created": 999, "status": "running"}
        )
        app._child_parent["subY"] = app.session.id
        app._on_subagent_done({"kind": "subagent_done", "session_id": "subY", "agent": "build", "title": "t", "ok": True})
        records = app._children[app.session.id]
        assert len(records) == 3
        status = next(r["status"] for r in records if r["id"] == "subY")
        assert status == "completed"


async def test_footer_shows_current_position_among_siblings():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sids = _register_children(app, app.session.id, 3)
        footer = app.query_one(SubagentFooter)
        app._current_session_id = sids[1]
        app._update_footer()
        assert footer.display
        plain = footer.query_one("#subagent-info").render().plain
        assert "(2 of 3)" in plain
        # back on the parent: hidden
        app._current_session_id = app.session.id
        app._update_footer()
        assert not footer.display


async def test_prev_next_parent_first_child_navigation():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        child_chats = _register_children(app, parent, 3)
        app._sessions[parent] = SimpleNamespace(parent_id=None, agent="build", title="main")

        app._current_session_id = child_chats[0]
        app._go_next()
        assert app._current_session_id == child_chats[1]
        app._go_next()
        assert app._current_session_id == child_chats[2]
        app._go_next()
        assert app._current_session_id == child_chats[0]  # wraps
        app._go_prev()
        assert app._current_session_id == child_chats[2]
        app._go_parent()
        assert app._current_session_id == parent
        # ctrl+down resumes the child you were last viewing, not always the first
        app._go_first_child()
        assert app._current_session_id == child_chats[2]
        # with no prior selection it falls back to the first child
        app._last_selection.pop(parent, None)
        app._go_parent()
        app._go_first_child()
        assert app._current_session_id == child_chats[0]


async def test_nav_requested_message_routes_to_go_methods():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        sids = _register_children(app, parent, 3)
        app._sessions[parent] = SimpleNamespace(parent_id=None, agent="build", title="main")
        app._current_session_id = sids[0]
        app.post_message(NavRequested("next"))
        await pilot.pause()
        assert app._current_session_id == sids[1]


async def test_session_nav_active_false_while_typing():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        bar = app.query_one("InputBar")
        assert app._session_nav_active() is True
        bar.input.value = "hello"
        assert app._session_nav_active() is False
        bar.input.value = ""
        assert app._session_nav_active() is True


async def test_finalize_task_row_adds_toolcalls_and_duration():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app._chats[app.session.id]
        child = app._chat_for("subC")
        child.append_tool({"tool": "bash", "status": "completed", "input": {"command": "echo hi"}, "call_id": "c1"})
        parent.append_tool(
            {"tool": "task", "status": "running", "input": {"description": "port", "subagent_type": "build"}, "call_id": "t1"}
        )
        bubble = parent.find_tool("task", "t1")
        bubble.set_tool_metadata("sessionId", "subC")
        app._task_start["subC"] = time.monotonic() - 5
        # the real engine path: tool_complete updates the row AND finalizes it
        app._on_engine_event(
            {
                "kind": "tool_complete",
                "session_id": app.session.id,
                "tool": "task",
                "status": "completed",
                "call_id": "t1",
                "input": {"description": "port", "subagent_type": "build"},
                "metadata": {"sessionId": "subC"},
            }
        )
        await pilot.pause()
        meta = bubble.content["metadata"]
        assert meta["toolcalls"] == 1
        assert "5.0s" in meta["duration"]
        plain = _plain(bubble._build_content())
        assert "✓ Build Task — port" in plain
        assert "↳ 1 toolcall · 5.0s" in plain

async def test_subagent_chat_shows_parent_directive_at_very_top():
    """Opening a sub-agent shows the parent's instruction (the task prompt) as
    the first message of its chat, like official opencode's directive block."""
    from opencode_py.tui.chat_view import MessageBubble

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        app._on_subagent_start(
            {
                "kind": "subagent_start",
                "session_id": "subD",
                "agent": "build",
                "title": "port it",
                "prompt": "Refactor the login module to use JWT.",
            }
        )
        app._switch_session("subD")
        chat = app._chats["subD"]
        bubbles = list(chat.query(MessageBubble))
        # the directive is the first (top) message
        assert bubbles[0].directive == "port it"
        assert "Refactor the login module to use JWT." in str(bubbles[0]._message)
        plain = _plain(bubbles[0]._build_content())
        assert "Directive" in plain
        assert "port it" in plain
        # no streamed content was inserted before it
        assert len(bubbles) == 1


async def test_parallel_task_rows_link_to_correct_session_by_call_id():
    """With parallel sub-agents the subagent_start events arrive concurrently;
    each running task row must link to the session it ACTUALLY spawned (matched
    by the tool call id), regardless of arrival order. Regression: reverse
    order used to swap the links, so clicking the still-working row opened the
    already-done agent instead."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        chat = app._chats[parent]
        chat.append_tool({"tool": "task", "status": "running", "input": {"description": "Agent A", "subagent_type": "build"}, "call_id": "cA"})
        chat.append_tool({"tool": "task", "status": "running", "input": {"description": "Agent B", "subagent_type": "build"}, "call_id": "cB"})
        # subagent_start arrives in the OPPOSITE order of the task rows (B first)
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subB", "agent": "build", "title": "Agent B", "call_id": "cB"})
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subA", "agent": "build", "title": "Agent A", "call_id": "cA"})
        assert chat.find_tool("task", "cA").content["metadata"]["sessionId"] == "subA"
        assert chat.find_tool("task", "cB").content["metadata"]["sessionId"] == "subB"
        # clicking each row goes to its own session
        chat.find_tool("task", "cB").on_click(type("Click", (), {})())
        await pilot.pause()
        assert app._current_session_id == "subB"
        chat.find_tool("task", "cA").on_click(type("Click", (), {})())
        await pilot.pause()
        assert app._current_session_id == "subA"


async def test_selected_agent_is_marked_on_parent_rows():
    """The task row of the sub-agent you're viewing is highlighted; back at the
    parent the last-opened child stays highlighted (official's active agent)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        chat = app._chats[parent]
        chat.append_tool({"tool": "task", "status": "running", "input": {"description": "Agent A", "subagent_type": "build"}, "call_id": "cA"})
        chat.append_tool({"tool": "task", "status": "running", "input": {"description": "Agent B", "subagent_type": "build"}, "call_id": "cB"})
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subA", "agent": "build", "title": "Agent A", "call_id": "cA"})
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subB", "agent": "build", "title": "Agent B", "call_id": "cB"})
        app._switch_session("subA")
        assert chat.find_tool("task", "cA").selected is True
        assert chat.find_tool("task", "cB").selected is False
        # back at the parent: the last-selected child stays marked
        app._switch_session(parent)
        assert chat.find_tool("task", "cA").selected is True
        assert chat.find_tool("task", "cB").selected is False


async def test_ctrl_down_resumes_last_viewed_subagent():
    """Returning to the parent and pressing ctrl+down goes back to the child
    you were last viewing (official remembers the selection per parent)."""
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        chat = app._chats[parent]
        chat.append_tool({"tool": "task", "status": "running", "input": {"description": "Agent A", "subagent_type": "build"}, "call_id": "cA"})
        chat.append_tool({"tool": "task", "status": "running", "input": {"description": "Agent B", "subagent_type": "build"}, "call_id": "cB"})
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subA", "agent": "build", "title": "Agent A", "call_id": "cA"})
        app._on_subagent_start({"kind": "subagent_start", "session_id": "subB", "agent": "build", "title": "Agent B", "call_id": "cB"})
        # view B, go back to parent, ctrl+down -> resumes B (not the first A)
        app._switch_session("subB")
        app._switch_session(parent)
        assert app._current_session_id == parent
        await pilot.press("ctrl+down")
        await pilot.pause()
        assert app._current_session_id == "subB"


async def test_task_row_reopens_persisted_subagent_after_restart(tmp_path, monkeypatch):
    """Regression: a completed sub-agent's task row clicked in a fresh app run
    (its session is only on disk, filtered out of the picker) used to switch to
    an EMPTY chat. It must load the saved history and navigate into it."""
    from opencode_py.globals import Path as GPath
    from opencode_py.session import new_session, save_session

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    sub_id = "subpersisted"
    sub = new_session(directory=str(tmp_path), provider="opencode", model="m")
    sub.id = sub_id
    sub.parent_id = "someparent"
    sub.title = "Sub agent"
    sub.messages = [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "here are the files"},
    ]
    save_session(sub)

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        parent = app.session.id
        chat = app._chats[parent]
        chat.append_tool(
            {"tool": "task", "status": "completed",
             "input": {"description": "Sub agent", "subagent_type": "build"},
             "output": "reply", "call_id": "c1",
             "metadata": {"sessionId": sub_id}}
        )
        assert sub_id not in app._sessions
        chat.find_tool("task", "c1").on_click(type("Click", (), {})())
        await pilot.pause()
        assert app._current_session_id == sub_id
        # the reopened chat rendered the saved conversation, not a blank view
        reopened = app._chats[sub_id]
        from opencode_py.tui.chat_view import MessageBubble

        contents = " | ".join(
            str(b.content) for b in reopened.query(MessageBubble) if b.role in ("user", "assistant")
        )
        assert "here are the files" in contents
        # the child relationship is restored from the persisted parent_id
        assert app._parent_of(sub_id) == "someparent"
