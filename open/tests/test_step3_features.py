"""Step 3 features: opencode-style session picker (Today/Yesterday groups,
arrow navigation, resume), session→Markdown export, and /export command."""

from __future__ import annotations

import datetime

from textual.app import App
from textual.widgets import ListView, OptionList

from opencode_py.commands import CommandContext, build_registry
from opencode_py.config import Config
from opencode_py.session import (
    Session,
    group_sessions,
    new_session,
    save_session,
    session_to_markdown,
)
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.session_list import (
    ConfirmDeleteDialog,
    RenameDialog,
    SessionList,
)


def _ts(offset_days: float) -> float:
    return (
        datetime.datetime.now() - datetime.timedelta(days=offset_days)
    ).timestamp()


# ---------------------------------------------------------------------------
# group_sessions: Today / Yesterday / older-by-day, newest-first
# ---------------------------------------------------------------------------

def test_group_sessions_buckets_today_yesterday_older():
    older = _ts(9)
    yesterday = _ts(1.2)
    today_early = _ts(0)
    today_late = _ts(-0.01)  # a few minutes in the future vs the first stamp
    older_label = datetime.datetime.fromtimestamp(older).strftime("%A, %B %d, %Y")
    sessions = [
        {"id": "old", "created": older, "title": "old"},
        {"id": "yday", "created": yesterday, "title": "yday"},
        {"id": "today-early", "created": today_early, "title": "early"},
        {"id": "today-late", "created": today_late, "title": "late"},
    ]
    groups = group_sessions(sessions)
    labels = [label for label, _ in groups]
    assert labels[0] == "Today"
    assert labels[1] == "Yesterday"
    assert labels[2] == older_label
    assert [s["id"] for s in groups[0][1]] == ["today-late", "today-early"]
    assert [s["id"] for s in groups[1][1]] == ["yday"]


def test_group_sessions_accepts_session_objects():
    older = _ts(5)
    now = _ts(0)
    sessions = [
        Session({"id": "a", "created": older, "title": "old"}),
        Session({"id": "b", "created": now, "title": "new"}),
    ]
    groups = group_sessions(sessions)
    assert groups[0][0] == "Today"
    assert [s.id for s in groups[0][1]] == ["b"]


def test_group_sessions_drops_missing_created():
    groups = group_sessions([{"id": "x", "title": "x"}])
    assert groups[0][1][0]["id"] == "x"


# ---------------------------------------------------------------------------
# session_to_markdown
# ---------------------------------------------------------------------------

def test_session_to_markdown_includes_tools_and_skips_system():
    sess = Session(
        {
            "id": "testid",
            "title": "Build it",
            "provider": "opencode",
            "model": "deepseek-v4-flash-free",
            "agent": "build",
            "messages": [
                {"role": "system", "content": "you are an agent"},
                {"role": "user", "content": "make a file"},
                {
                    "role": "assistant",
                    "content": "on it",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "write", "arguments": '{"path": "a.txt"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "write", "content": "wrote a.txt"},
                {"role": "assistant", "content": "done"},
                {"role": "compaction", "content": "summarized earlier"},
            ],
        }
    )
    md = session_to_markdown(sess)
    assert md.startswith("# Build it")
    assert "deepseek-v4-flash-free" in md
    assert "make a file" in md
    assert "**write**" in md
    assert "a.txt" in md
    assert "wrote a.txt" in md
    assert "done" in md
    assert "Compaction" in md
    assert "you are an agent" not in md


def test_session_to_markdown_serializes_pretty_arguments():
    sess = Session(
        {
            "id": "x",
            "title": "t",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "a.txt"},
            ],
        }
    )
    md = session_to_markdown(sess)
    assert '"cmd": "ls"' in md
    assert "a.txt" in md


# ---------------------------------------------------------------------------
# SessionList TUI: arrows navigate; Enter resumes; headers are not targets
# ---------------------------------------------------------------------------

class SessionListHost(App):
    def __init__(self, sessions: list[dict], current: str = ""):
        super().__init__()
        self.dismissed = "no-dismiss"
        self._dlg = SessionList(sessions, current=current)

    def on_mount(self):
        self.push_screen(self._dlg, self.dismissed_set)

    def dismissed_set(self, value):
        self.dismissed = value


def _sessions():
    return [
        {"id": "old", "title": "Old one", "agent": "build", "created": _ts(9), "status": ""},
        {"id": "now", "title": "Today one", "agent": "plan", "created": _ts(0), "status": ""},
    ]


async def test_session_list_groups_and_highlights_current():
    app = SessionListHost(_sessions(), current="now")
    async with app.run_test() as pilot:
        ol = app._dlg.query_one("#session-list", OptionList)
        rows = app._dlg._rows()
        assert "Today" in rows[0][1]  # newest group comes first
        # the current session's row is highlighted on open
        assert ol.highlighted_option.id == "now"
        # section headers are disabled separator options
        assert ol._options[0].disabled is True


async def test_session_list_arrow_navigation():
    app = SessionListHost(_sessions(), current="")
    async with app.run_test() as pilot:
        ol = app._dlg.query_one("#session-list", OptionList)
        first = ol.highlighted
        assert first == 1  # the first selectable row (the Today header is idx 0)
        await pilot.press("down")
        # headers are disabled separators now, so Down jumps over them to the
        # next real session (the older-day group's header sits between them)
        assert ol.highlighted == 3  # index 2 is the disabled "Tuesday" header
        assert ol._options[ol.highlighted].disabled is False
        await pilot.press("up")
        assert ol.highlighted == 1  # back over the header to "now"


async def test_session_list_enter_resumes_selected():
    app = SessionListHost(_sessions(), current="")
    async with app.run_test() as pilot:
        idx = _row_index(app._dlg, "old")
        ol = app._dlg.query_one("#session-list", OptionList)
        ol.highlighted = idx
        await pilot.press("enter")
        assert app.dismissed == "old"


async def test_session_list_enter_on_header_does_not_dismiss():
    app = SessionListHost(_sessions(), current="")
    async with app.run_test() as pilot:
        ol = app._dlg.query_one("#session-list", OptionList)
        ol.highlighted = 0  # the first section header (disabled)
        await pilot.press("enter")
        assert app.dismissed == "no-dismiss"


async def test_session_list_escape_closes():
    app = SessionListHost(_sessions(), current="")
    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert app.dismissed is None


async def test_session_list_escapes_markup_in_titles():
    """Titles are user text and may contain rich markup (`[/]`, `[b]bold[/b]`).
    They must render literally — unescaped markup would raise MarkupError and
    take the whole picker list (headers included) down with it."""
    from rich.text import Text

    app = SessionListHost(
        [
            {
                "id": "x",
                "title": "Fix [/] panic [in] kernel",
                "agent": "build",
                "created": _ts(0),
                "status": "",
            }
        ],
        current="",
    )
    async with app.run_test() as pilot:
        ol = app._dlg.query_one("#session-list", OptionList)
        # the options were built without raising MarkupError; rendering each
        # prompt to plain text must show the title literally
        joined = " ".join(Text.from_markup(o.prompt).plain for o in ol._options)
        assert "Fix [/] panic [in] kernel" in joined
        assert "Today" in joined  # the section header still renders above the row


class RenameDeleteHost(App):
    """SessionList wired to app-side rename/delete callbacks, so the tests can
    assert the callbacks run and the list view stays consistent."""

    def __init__(self, sessions, current="", del_ok=True):
        super().__init__()
        self.dismissed = "no-dismiss"
        self.renames = []
        self.deletes = []
        self._del_ok = del_ok
        self._dlg = SessionList(
            sessions,
            current=current,
            on_rename=self._ren,
            on_delete=self._del,
        )

    def _ren(self, sid, title):
        self.renames.append((sid, title))
        return None

    def _del(self, sid):
        self.deletes.append(sid)
        return self._del_ok

    def on_mount(self):
        self.push_screen(self._dlg, lambda v: setattr(self, "dismissed", v))


def _row_index(dlg, sid):
    rows = dlg._rows()
    return next(i for i, (rid, _) in enumerate(rows) if rid == sid)


async def test_session_list_rename_via_ctrl_n():
    app = RenameDeleteHost(_sessions(), current="now")
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "now")
        await pilot.press("ctrl+n")
        assert isinstance(app.screen, RenameDialog)
        app.screen.query_one("#rename-input").value = "  New name  "
        await pilot.press("enter")
        await pilot.pause()
        # stripped title flows through the on_rename callback
        assert app.renames == [("now", "New name")]
        # back on the list, row renamed, highlight restored to it
        assert isinstance(app.screen, SessionList)
        row = app._dlg._rows()[_row_index(app._dlg, "now")][1]
        assert "New name" in row
        assert app._dlg.query_one("#session-list", OptionList).highlighted_option.id == "now"


async def test_session_list_rename_escape_cancels():
    app = RenameDeleteHost(_sessions(), current="")
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "old")
        await pilot.press("ctrl+n")
        await pilot.press("escape")
        await pilot.pause()
        assert app.renames == []
        assert isinstance(app.screen, SessionList)
        assert "Old one" in app._dlg._rows()[_row_index(app._dlg, "old")][1]


async def test_session_list_rename_on_header_is_noop():
    app = RenameDeleteHost(_sessions(), current="")
    async with app.run_test() as pilot:
        app._dlg.query_one("#session-list", OptionList).highlighted = 0
        await pilot.press("ctrl+n")
        assert app.renames == []
        assert isinstance(app.screen, SessionList)


async def test_session_list_delete_confirms_and_removes():
    app = RenameDeleteHost(_sessions(), current="now")
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "old")
        await pilot.press("ctrl+d")
        assert isinstance(app.screen, ConfirmDeleteDialog)
        app.screen.query_one("#del-yes").press()
        await pilot.pause()
        assert app.deletes == ["old"]
        ids = [rid for rid, _ in app._dlg._rows() if rid]
        assert "old" not in ids
        assert "now" in ids  # the rest survive


async def test_session_list_delete_cancel_keeps_row():
    app = RenameDeleteHost(_sessions(), current="")
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "old")
        await pilot.press("ctrl+d", "escape")
        await pilot.pause()
        assert app.deletes == []
        assert isinstance(app.screen, SessionList)
        assert "old" in [rid for rid, _ in app._dlg._rows() if rid]


async def test_session_list_delete_via_y_key():
    app = RenameDeleteHost(_sessions(), current="")
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "old")
        await pilot.press("ctrl+d", "y")
        await pilot.pause()
        assert app.deletes == ["old"]
        assert isinstance(app.screen, SessionList)


async def test_session_list_delete_active_session_delivers_confirm():
    """The active session is deletable too — it just goes through the same
    confirmation popup as everything else."""
    app = RenameDeleteHost(_sessions(), current="now")
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "now")
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteDialog)  # confirm still shown
        app.screen.query_one("#del-yes").press()
        await pilot.pause()
        assert app.deletes == ["now"]
        assert isinstance(app.screen, SessionList)
        assert "now" not in [rid for rid, _ in app._dlg._rows() if rid]


async def test_session_list_active_row_marked_with_green_dot():
    """The active session's row carries a green • marker so it's obvious which
    one is live before a rename/delete."""
    app = RenameDeleteHost(_sessions(), current="now")
    async with app.run_test() as pilot:
        row = [t for rid, t in app._dlg._rows() if rid == "now"][0]
        other = [t for rid, t in app._dlg._rows() if rid == "old"][0]
        assert "•" in row
        assert "•" not in other


async def test_session_list_long_title_truncated_not_panned():
    """Long titles are truncated to one line (OptionList can't slide a single
    row past the viewport — the old per-row sideways panning was removed for
    speed). Nothing is clipped sideways; the list has no horizontal scroll."""
    app = SessionListHost(
        [
            {"id": "x", "title": "T" * 200, "agent": "build", "created": _ts(0), "status": ""},
            {"id": "y", "title": "Short", "agent": "build", "created": _ts(1), "status": ""},
        ],
        current="",
    )
    async with app.run_test() as pilot:
        ol = app._dlg.query_one("#session-list", OptionList)
        prompts = {o.id: o.prompt for o in ol._options}
        assert "…" in prompts["x"], "overlong titles must be truncated to one line"
        assert "…" not in prompts["y"]
        assert ol.max_scroll_x == 0, "no horizontal panning anymore"


async def test_session_list_delete_rejected_on_error():
    app = RenameDeleteHost(_sessions(), current="", del_ok=False)
    async with app.run_test() as pilot:
        dlg = app._dlg
        dlg.query_one("#session-list", OptionList).highlighted = _row_index(dlg, "old")
        await pilot.press("ctrl+d")
        app.screen.query_one("#del-yes").press()
        await pilot.pause()
        assert app.deletes == ["old"]  # callback was reached
        # but it failed, so the row stays
        assert "old" in [rid for rid, _ in app._dlg._rows() if rid]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# App-side deletion: the main session resets to a fresh workspace; a deleted
# non-main current session hops back to the main session.
# ---------------------------------------------------------------------------

async def test_delete_main_session_resets_workspace():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        old_id = app.session.id
        app._main_chat.append_user("hello")
        ok = app._delete_session(old_id)
        await pilot.pause()
        assert ok is True
        assert app.session.id != old_id
        assert app._current_session_id == app.session.id
        assert app._sessions.get(app.session.id) is app.session
        assert app._chats.get(app.session.id) is app._main_chat
        assert not list(app._main_chat.children)  # chat was emptied for the fresh session


async def test_delete_non_main_current_session_switches_back():
    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        sess = new_session(directory=".", provider="opencode", model="m")
        app._sessions[sess.id] = sess
        app._engines[sess.id] = app.engine
        chat = app._chat_for(sess.id)
        app._current_session_id = sess.id
        ok = app._delete_session(sess.id)
        await pilot.pause()
        assert ok is True
        assert app._current_session_id == app.session.id  # hopped back to main
        assert sess.id not in app._sessions
        assert sess.id not in app._chats
        assert chat is not None  # it had a chat that got removed from the dict


# ---------------------------------------------------------------------------
# Resume plumbing
# ---------------------------------------------------------------------------

async def test_resume_live_session_switches():
    app = OpenCodeTUI()
    async with app.run_test():
        sid2 = "live2"
        app._sessions[sid2] = type("S", (), {"title": "Two", "agent": "build", "created": None})()
        app._engines[sid2] = app.engine
        app._chats[sid2] = app._chat_for(sid2)
        app._resume_session(sid2)
        assert app._current_session_id == sid2


async def test_resume_persisted_session_rebuilds_engine(monkeypatch):
    app = OpenCodeTUI()
    from opencode_py import session as session_mod

    sess = Session(
        {
            "id": "res1",
            "title": "Old talk",
            "agent": "build",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "on it",
                    "reasoning_content": "let me think about this",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "write", "arguments": '{"filePath": "a.txt", "content": "x"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "write", "content": "wrote a.txt"},
                {"role": "assistant", "content": "done"},
            ],
        }
    )
    monkeypatch.setattr(session_mod, "load_session", lambda sid: sess if sid == "res1" else None)
    async with app.run_test():
        app._resume_session("res1")
        assert app._current_session_id == "res1"
        assert app._engines["res1"].get_history() == sess.messages
        chat = app._chat_for("res1")
        assert len(list(chat.children)) == 5  # user + thought + tool row + 2 assistant
        roles = [getattr(c, "role", "") for c in chat.children]
        assert roles == ["user", "reasoning", "assistant", "tool", "assistant"]
        # the tool row must carry a dict input, not the raw JSON string
        tool = next(c for c in chat.children if getattr(c, "role", "") == "tool")
        assert tool._message["input"] == {"filePath": "a.txt", "content": "x"}
        assert tool._message["status"] == "completed"  # gray-block frame key


async def test_resume_bash_history_renders_gray_block(monkeypatch):
    """A replayed bash result renders as a block (panel background), exactly
    like the live TUI — requires status "completed", not "done"."""
    from opencode_py import session as session_mod

    app = OpenCodeTUI()
    sess = Session(
        {
            "id": "res3",
            "title": "bashy",
            "messages": [
                {"role": "user", "content": "run it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "a.txt\nb.txt"},
            ],
        }
    )
    monkeypatch.setattr(session_mod, "load_session", lambda sid: sess if sid == "res3" else None)
    async with app.run_test():
        app._resume_session("res3")
        chat = app._chat_for("res3")
        tool = next(c for c in chat.children if getattr(c, "role", "") == "tool")
        assert tool._message["status"] == "completed"
        assert tool._tool_block() is True
        from opencode_py.tui.theme import get_theme

        assert tool.styles.background.hex == get_theme("opencode").c("background_panel")


async def test_resume_todowrite_history_parses_todos(monkeypatch):
    """todowrite results are JSON text in history; metadata.todos is rebuilt so
    the result still renders as its colored todo block."""
    from opencode_py import session as session_mod

    app = OpenCodeTUI()
    sess = Session(
        {
            "id": "res4",
            "title": "todos",
            "messages": [
                {"role": "user", "content": "plan"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "todowrite", "arguments": "{}"}}
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "todowrite",
                    "content": '[{"content": "build", "status": "in_progress", "priority": "high"}]',
                },
            ],
        }
    )
    monkeypatch.setattr(session_mod, "load_session", lambda sid: sess if sid == "res4" else None)
    async with app.run_test():
        app._resume_session("res4")
        chat = app._chat_for("res4")
        tool = next(c for c in chat.children if getattr(c, "role", "") == "tool")
        assert tool._message["metadata"]["todos"] == [
            {"content": "build", "status": "in_progress", "priority": "high"}
        ]


async def test_resume_rejects_malformed_tool_arguments(monkeypatch):
    """History whose tool arguments aren't JSON must not crash rendering."""
    app = OpenCodeTUI()
    from opencode_py import session as session_mod

    sess = Session(
        {
            "id": "res2",
            "title": "messy",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "bash", "arguments": "not-json{"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "oops"},
            ],
        }
    )
    monkeypatch.setattr(session_mod, "load_session", lambda sid: sess if sid == "res2" else None)
    async with app.run_test():
        app._resume_session("res2")
        assert app._current_session_id == "res2"
        chat = app._chat_for("res2")
        assert len(list(chat.children)) == 1  # just the tool row, no crash


async def test_resume_missing_session_does_not_crash(monkeypatch):
    app = OpenCodeTUI()
    from opencode_py import session as session_mod

    monkeypatch.setattr(session_mod, "load_session", lambda sid: None)
    async with app.run_test():
        app._resume_session("nope")
        assert app._current_session_id == app.session.id


async def test_prompt_sets_session_title():
    """The TUI names the session from its first real (non-command) message."""
    from opencode_py.tui.input_bar import PromptSubmitted

    app = OpenCodeTUI()
    async with app.run_test():
        app._busy = True  # skip the worker; title is set before the busy check
        app.on_prompt_submitted(PromptSubmitted("Fix the wifi driver"))
        assert app.session.title == "Fix the wifi driver"
        # commands must NOT become titles (a plain non-picker command)
        app._busy = True
        app.on_prompt_submitted(PromptSubmitted("/help"))
        assert app.session.title == "Fix the wifi driver"


async def test_slash_sessions_opens_the_picker():
    """`/sessions` must open the same opencode-style picker as Ctrl+R, not the
    plain text `_sessions` listing."""
    from opencode_py.tui.session_list import SessionList

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app._run_command("/sessions")
        await pilot.pause()
        assert isinstance(app.screen, SessionList)


async def test_session_picker_filters_empty_and_subagent_sessions(monkeypatch, tmp_path):
    """Persisted noise (empty placeholders, sub-agent sessions) stays out of
    the opencode-style picker; message-bearing sessions stay in."""
    from opencode_py.globals import Path as GPath
    from opencode_py.tui.session_list import SessionList

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    save_session(
        Session(
            {
                "id": "good",
                "title": "Real talk",
                "created": _ts(0),
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    )
    save_session(Session({"id": "empty", "title": "", "created": _ts(0), "messages": []}))
    save_session(
        Session(
            {
                "id": "untitled",
                "title": "",
                "created": _ts(0),
                "messages": [
                    {"role": "user", "content": "Bring back the windows"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
        )
    )
    save_session(
        Session(
            {
                "id": "sub",
                "title": "sub",
                "parent_id": "good",
                "created": _ts(0),
                "messages": [],
            }
        )
    )

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.action_sessions()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionList)
        rows = {rid: text for rid, text in screen._rows() if rid}
        assert "good" in rows
        assert "empty" not in rows
        assert "sub" not in rows
        # untitled-but-message-bearing sessions show a title derived from
        # their first user message
        assert "Bring back the windows" in rows["untitled"]


async def test_session_picker_shows_older_sessions_beyond_newest_placeholders(monkeypatch, tmp_path):
    """A burst of fresh empty placeholder launches must not push real (older)
    conversations out of the picker — filter before capping."""
    from opencode_py.globals import Path as GPath
    from opencode_py.tui.session_list import SessionList

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    # 150 empty placeholders created AFTER the real conversation (newest first)
    for i in range(150):
        save_session(
            Session({"id": f"ph{i}", "title": "", "created": _ts(-i * 0.001), "messages": []})
        )
    save_session(
        Session(
            {
                "id": "real",
                "title": "The real conversation",
                "created": _ts(2),
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    app = OpenCodeTUI()
    async with app.run_test() as pilot:
        app.action_sessions()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionList)
        ids = [rid for rid, _ in screen._rows() if rid]
        assert "real" in ids
        assert "ph0" not in ids


# ---------------------------------------------------------------------------
# /export command
# ---------------------------------------------------------------------------

def test_export_command_writes_markdown(tmp_path, monkeypatch):
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "sessions_dir", staticmethod(lambda: tmp_path))
    monkeypatch.chdir(tmp_path)
    sess = Session(
        {
            "id": "abc123",
            "title": "Export me",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }
    )
    save_session(sess)

    reg = build_registry()
    cmd = reg.get("export")
    assert cmd is not None
    out: list[str] = []

    class FakeEngine:
        session_id = "abc123"

    ctx = CommandContext(config=Config(), auth=None, engine=FakeEngine())
    ctx.reply = out.append
    cmd.handler(ctx, "")

    path = tmp_path / "opencode-session-abc123.md"
    assert path.exists()
    assert "Export me" in path.read_text(encoding="utf-8")
    assert any("opencode-session-abc123.md" in line for line in out)


# ---------------------------------------------------------------------------
# resumed session reply routing (engine events must carry their session id)
# ---------------------------------------------------------------------------

def _make_resume_session(sid: str, title: str) -> Session:
    return Session(
        {
            "id": sid,
            "title": title,
            "agent": "build",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "done"},
            ],
        }
    )


class _ReplyRotation:
    def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
        from opencode_py.providers.base import ProviderEvent

        on_event(ProviderEvent(kind="text_delta", text="reply ok"))
        return "opencode", "deepseek-v4-flash-free"


class _BoomRotation:
    def stream(self, messages, tools, on_event, on_notice=None, **kwargs):
        from opencode_py.providers.base import ProviderError

        raise ProviderError("connection refused")


async def _drive_turn(app, pilot, rotation) -> None:
    from opencode_py.tui.input_bar import PromptSubmitted

    app._engines["res1"].rotation = rotation
    app.on_prompt_submitted(PromptSubmitted("hello again"))
    for _ in range(80):
        await pilot.pause()
        if not app._busy:
            break


async def test_resumed_reply_renders_in_resumed_chat_not_main(monkeypatch):
    """A reply from a resumed session's engine must land in THAT session's chat.
    Regression: engine events carried no session_id, so every event fell back to
    the main session id and the reply rendered into the hidden main chat."""
    from opencode_py import session as session_mod

    app = OpenCodeTUI()
    sess = _make_resume_session("res1", "Old talk")
    monkeypatch.setattr(session_mod, "load_session", lambda sid: sess if sid == "res1" else None)
    async with app.run_test() as pilot:
        app._resume_session("res1")
        await _drive_turn(app, pilot, _ReplyRotation())
        main_roles = [getattr(c, "role", "") for c in app._chat_for(app.session.id).children]
        res_roles = [getattr(c, "role", "") for c in app._chat_for("res1").children]
        assert "assistant" in res_roles  # the reply went to the resumed chat
        assert "assistant" not in main_roles  # and NOT to the main chat


async def test_resumed_error_renders_in_resumed_chat_not_main(monkeypatch):
    """A provider error from a resumed session's engine must surface in THAT
    session's chat. Regression: errors were routed to the main session chat."""
    from opencode_py import session as session_mod

    app = OpenCodeTUI()
    sess = _make_resume_session("res1", "Old talk")
    monkeypatch.setattr(session_mod, "load_session", lambda sid: sess if sid == "res1" else None)
    async with app.run_test() as pilot:
        app._resume_session("res1")
        await _drive_turn(app, pilot, _BoomRotation())
        res_roles = [getattr(c, "role", "") for c in app._chat_for("res1").children]
        main_roles = [getattr(c, "role", "") for c in app._chat_for(app.session.id).children]
        assert "meta" in res_roles  # the ⚠ error landed in the resumed chat
        assert "meta" not in main_roles

def test_group_sessions_never_mixes_years():
    """older-day labels must include the year so sessions from different years
    that share the same month+day+weekday (recurring every 5/6/11 years) can't
    silently merge into one group — e.g. Aug 17 2020 and Aug 17 2026 are both
    Mondays and used to both show as "Monday, August 17"."""
    from opencode_py.session import group_sessions

    aug17_2026 = datetime.datetime(2026, 8, 17, 10, 0).timestamp()
    aug17_2020 = datetime.datetime(2020, 8, 17, 10, 0).timestamp()
    aug20 = datetime.datetime(2026, 8, 20, 10, 0).timestamp()  # today
    def mk(ts, ident):
        return {"id": ident, "created": ts, "title": ident}
    groups = group_sessions([mk(aug20, "t"), mk(aug17_2020, "old"), mk(aug17_2026, "mid")])
    labels = [label for label, _ in groups]
    assert labels[0] == "Today"
    assert "Monday, August 17, 2026" in labels
    assert "Monday, August 17, 2020" in labels
    ids = {s["id"] for _, items in groups for s in items}
    assert ids == {"t", "old", "mid"}
    by_label = {label: [s["id"] for s in items] for label, items in groups}
    assert by_label["Monday, August 17, 2026"] == ["mid"]
    assert by_label["Monday, August 17, 2020"] == ["old"]
