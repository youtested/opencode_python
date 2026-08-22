"""opencode_py Textual TUI app.

Mirrors opencode's session screen: header status bar (agent/model/provider/
permission), scrollable chat with live tool blocks + diff rendering, and a
prompt input bar. The engine runs in a worker thread; events are bridged to
the UI via call_from_thread.
"""

from __future__ import annotations

import asyncio
import copy
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical

from ..config import load_config
from .chat_view import ChatView, MessageBubble
from .input_bar import (
    AgentToggleRequested,
    CommandSelected,
    InputBar,
    ModelsRequested,
    PromptSubmitted,
    RotationLockToggled,
    SessionNavRequested,
    SessionsRequested,
    format_duration,
)
from .status_bar import StatusBar
from .subagent_footer import NavRequested, SubagentFooter

if TYPE_CHECKING:
    from ..agent.loop import AgentLoop
    from ..config import Config
    from ..question import QuestionInfo


def _prewarm_heavy_deps() -> None:
    """Import the engine chain + provider internals on a background thread so
    they don't sit on the first-turn path (best effort — lazy imports re-run
    normally if any of this fails)."""
    try:
        import opencode_py.agent.loop  # noqa: F401
        import opencode_py.agent.compaction  # noqa: F401
        import opencode_py.commands  # noqa: F401
        import opencode_py.tools  # noqa: F401
        import opencode_py.providers.zen  # noqa: F401
        import opencode_py.providers.openai_compat  # noqa: F401
        import opencode_py.providers.anthropic  # noqa: F401
    except Exception:  # pragma: no cover - best-effort warm-up
        pass


# Read-only slash commands allowed while a turn is running. Everything else
# (undo/clear/compact/agent/model/theme/...) mutates engine state or the running
# turn and is blocked until the current request finishes.
_SAFE_WHILE_BUSY = {
    "help",
    "config",
    "permissions",
    "sessions",
    "ls",
    "models",
    "review",
    "connect",
    "resume",
}


class OpenCodeTUI(App):
    TITLE = "opencode"
    SUB_TITLE = "opencode_py"

    ENABLE_COMMAND_PALETTE = False  # ctrl+p is bound to Settings instead

    CSS = """
    Screen {
        background: #0a0a0a;
        color: #eeeeee;
    }
    #root {
        layout: vertical;
        height: 1fr;
    }
    #chat-stack {
        layout: vertical;
        height: 1fr;
    }
    ChatView {
        width: 100%;
        height: 1fr;
        padding: 0 2;
        background: #0a0a0a;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    .chat-welcome-logo {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    SubagentFooter {
        width: 100%;
        height: auto;
        background: #141414;
        padding: 0 2;
    }
    #subagent-info {
        width: 1fr;
        height: 1;
        padding: 0 1 0 2;
    }
    #subagent-nav {
        height: 1;
        padding: 0 1;
        background: #141414;
    }
    .subagent-gap {
        height: 1;
    }
    InputBar {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: #0a0a0a;
    }
    .prompt-frame {
        height: auto;
        background: #0a0a0a;
    }
    #prompt-accent {
        width: 1;
        height: 3;
        background: #fab283;
    }
    .prompt-body {
        width: 1fr;
        height: auto;
        padding: 0 0 0 1;
    }
    #prompt-input {
        width: 1fr;
        background: #1e1e1e;
        color: #eeeeee;
        border: none;
        outline: none;
        padding: 0 1;
        height: 3;
        min-height: 3;
        content-align: left middle;
    }
    #prompt-input:focus {
        border: none;
        outline: none;
    }
    #prompt-title {
        width: 1fr;
        height: 1;
        margin-top: 1;
        margin-bottom: 1;
        padding: 0 1;
        color: #808080;
    }
    #prompt-meta {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: #808080;
    }
    #prompt-status {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: #808080;
    }
    #prompt-status.hidden {
        display: none;
    }
    #suggestions {
        height: auto;
        max-height: 10;
        padding: 0 1;
        background: #1e1e1e;
        color: #eeeeee;
        border: round #9d7cd8;
        margin: 0 1 1 2;
    }
    #suggestions.hidden {
        display: none;
    }
    StatusBar {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: #0a0a0a;
        color: #808080;
    }
    .cmd-popup {
        width: 74;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    .cmd-popup.settings {
        width: 74;
        max-height: 90%;
    }
    .cmd-popup-title {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $surface;
    }
    .settings-scroll {
        height: 40;
        min-height: 4;
        max-height: 70%;
        border: none;
    }
    .settings-body {
        padding: 1 2;
        color: $text;
    }
    .settings-edit {
        margin: 0 2;
        height: 3;
    }
    .settings-hint {
        dock: bottom;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    .cmd-popup-actions {
        height: 3;
        align: center middle;
        background: $surface;
        padding: 0 2;
    }
    .cmd-popup.session-popup {
        max-height: 90%;
    }
    #session-list {
        height: auto;
        max-height: 60vh;
        border: none;
        padding: 0;
        scrollbar-size-horizontal: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+r", "resume", "Resume"),
        Binding("ctrl+t", "toggle_agent", "Switch agent"),
        Binding("ctrl+a", "sessions", "Sessions"),
        Binding("ctrl+m", "models", "Models"),
        Binding("escape", "interrupt_escape", "Interrupt (press twice)"),
        Binding("ctrl+p", "settings", "Settings"),
        Binding("ctrl+s", "settings", "Settings"),
        Binding("ctrl+shift+e", "toggle_thought", "Expand/collapse thought"),
        # session routing between parallel sub-agents (opencode's
        # session.parent / session.child.next / session.child.previous /
        # session.child.first). Non-priority so the prompt keeps the arrow keys
        # for cursor movement while it is non-empty.
        Binding("up", "fd_parent", "Parent session", priority=False),
        Binding("left", "fd_prev", "Previous subagent", priority=False),
        Binding("right", "fd_next", "Next subagent", priority=False),
        Binding("ctrl+down", "fd_first", "View subagents"),
        # HOME/END/PgUp/PgDn always scroll the conversation, no matter where
        # focus is (the input box normally eats them for text editing). Prior
        # bindings win over the focused widget, so they work right out of the
        # gate — no need to click a message to hand focus to the chat first.
        Binding("home", "chat_home", "Scroll to top", priority=True),
        Binding("end", "chat_end", "Scroll to bottom", priority=True),
        Binding("pageup", "chat_page_up", "Scroll page up", priority=True),
        Binding("pagedown", "chat_page_down", "Scroll page down", priority=True),
    ]

    def __init__(
        self,
        cfg: Config | None = None,
        engine: AgentLoop | None = None,
        directory: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg or load_config()
        self.directory = directory or Path.cwd()
        # Engine-chain imports (agent.loop, tools, commands, session, auth) are
        # deferred out of module scope so `import opencode_py.tui` (and thus
        # first paint) doesn't pay ~0.4s before the app even exists. The engine
        # is built on demand: a background thread warms it right after on_mount
        # for the real app, and any synchronous first access (tests, sub-agent
        # spawn) builds it inline. `engine=` still injects a prebuilt engine.
        self._engine: AgentLoop | None = engine
        self._engine_lock = threading.Lock()
        if engine is not None:
            self._wire_engine(engine)
        from ..globals import Path as GPath
        from ..auth import Auth

        self.auth = Auth(auth_file=GPath.auth_file())
        # Shared interrupt flag: Ctrl+C flips this and every engine (main + any
        # spawned sub-agents, which capture the same callable by reference) sees
        # it at the next iteration and stops. It is reset once the turn ends so
        # future turns are not aborted.
        self._interrupt_flag = {"requested": False}
        from ..commands import build_registry as build_command_registry

        self.command_registry = build_command_registry()
        from ..session import new_session

        self.session = new_session(
            directory=str(self.directory),
            provider=self.cfg.provider,
            model=self.cfg.model,
            agent=self.cfg.default_agent or "build",
        )
        if engine is not None:
            engine.session_id = self.session.id
        self._chats: dict[str, ChatView] = {}
        self._sessions: dict[str, Any] = {self.session.id: self.session}
        self._engines: dict[str, AgentLoop] = {}
        if engine is not None:
            self._engines[self.session.id] = engine
        self._main_engine: AgentLoop | None = engine
        self._current_session_id = self.session.id
        self._active_turn_session_id = self.session.id
        self._busy = False
        self._busy_sessions: set[str] = set()
        self._running_agents: dict[str, str] = {}
        # Per-session turn bookkeeping. A single global set of had_text/
        # had_reasoning/... flags is WRONG: a sub-agent's events (keyed by the
        # child session id) or a prompt submitted on another session while this
        # one streams would overwrite the active turn's flags, so _turn_done
        # could finalize the wrong bubble / show the wrong runtime. Every
        # event handler writes to the SESSION's slot; _turn_done reads the
        # slot of the turn that actually finished.
        self._turn: dict[str, dict[str, Any]] = {}
        self._esc_presses = 0
        self._esc_timer: Any = None
        # Auto-refocus timer: dragging the prompt cursor back after the focus
        # lands somewhere else (e.g. a tapped reasoning bubble), so typing keeps
        # working without having to tap the input box again.
        self._refocus_timer: Any = None
        self._main_screen: Any = None
        self._pruned: set[str] = set()
        # Live sub-agent family tree (mirrors the official store): parent_id ->
        # ordered list of child records {id, title, agent, created, status}.
        # Kept even after a child finishes so `(2 of N)` counts stay correct and
        # finished children stay reviewable.
        self._children: dict[str, list[dict[str, Any]]] = {}
        self._child_parent: dict[str, str] = {}
        # Last sub-agent viewed under each parent, so returning to a parent and
        # pressing ctrl+down resumes that same agent (official opencode keeps
        # your previous sub-agent selection per parent).
        self._last_selection: dict[str, str] = {}
        self._task_start: dict[str, float] = {}
        self._usage: dict[str, dict[str, int]] = {}
        # delta batching: text/reasoning deltas are queued (non-blocking) and
        # flushed on a short timer so a fast stream isn't re-rendered per token.
        self._pending: dict[str, dict[str, list[str]]] = {}
        self._delta_timer: Any = None
        # Periodic autosave while a turn streams: the engine keeps the in-flight
        # assistant reply live in its history, so this persists the conversation
        # up to the very last token. If Termux/Android kills the app suddenly
        # (no graceful exit), the session file is at most a few seconds stale
        # and the picker resumes where the user left off.
        self._autosave_timer: Any = None
        self._exit_requested = threading.Event()
        # Invalidates any in-flight streaming autosave when its turn ends or the
        # app saves-all (exit/teardown), so a stale worker can never overwrite a
        # newer durable copy. See _autosave_in_flight.
        self._autosave_generation = 0
        self._autosave_thread: threading.Thread | None = None

    @property
    def engine(self) -> AgentLoop:
        """The main engine, built lazily on first access.

        Building happens synchronously here (so tests / synchronous handlers see
        a fully-wired engine), and a background thread warmed it after mount for
        the real app — whichever builds first, the lock ensures one instance.
        """
        self._ensure_engine()
        assert self._main_engine is not None
        return self._main_engine

    def _wire_engine(self, engine: AgentLoop) -> None:
        engine.on_event = self._on_engine_event
        engine.interrupt = self._interrupt_requested
        # Permission "ask" mode: bridge the engine thread to a modal dialog.
        # Sub-agents share the same PermissionEngine instance, so one hook works
        # for all sessions.
        engine.permission.ask_callback = self._permission_ask
        # Question "ask" mode: bridge the engine thread's question.ask to a
        # modal dialog, mirroring the official TUI's question popup. Sub-agents
        # share the same QuestionService instance, so one hook works for all.
        engine.question_service.ask_callback = self._question_ask

    def _ensure_engine(self) -> None:
        if self._main_engine is not None:
            return
        with self._engine_lock:
            if self._main_engine is not None:
                return
            from ..agent.loop import AgentLoop
            from ..tools import build_registry as build_tool_registry

            engine = AgentLoop(
                cfg=self.cfg,
                registry=build_tool_registry(self.cfg),
                directory=self.directory,
                auth=self.auth,
                agent=self.cfg.default_agent or "build",
            )
            self._wire_engine(engine)
            engine.session_id = self.session.id
            self._main_engine = engine
            self._engines.setdefault(self.session.id, engine)

    def _warm_engine(self) -> None:
        """Background engine build launched after mount: keeps first paint fast
        while the (~0.4s) engine-chain import runs off the UI thread."""
        try:
            self._ensure_engine()
        except Exception as e:
            # A failed warm-up must not kill the app's message loop; the engine
            # is rebuilt synchronously on first real use anyway.
            sys.stderr.write(f"[tui] engine warm-up failed: {e}\n")
            return
        # Pre-warm the per-model context/output lookups (and their lazy provider
        # imports: zen.py / openai_compat ~180ms) so the FIRST turn isn't held
        # up by a one-time lookup on the request path.
        try:
            from ..providers import model_context_size, model_output_limit

            model_context_size(self.cfg.provider, self.cfg.model, auth=self.auth)
            model_output_limit(self.cfg.provider, self.cfg.model)
        except Exception:
            pass
        try:
            self.call_from_thread(self._update_header)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="chat-stack"):
                yield ChatView()
            yield SubagentFooter()
            yield InputBar(
                commands=[
                    {"name": c.name, "description": c.description}
                    for c in self.command_registry.list()
                    if not c.hidden
                ]
            )
            yield StatusBar()

    def on_mount(self) -> None:
        self._thread_id = threading.get_ident()
        self._main_screen = self.screen
        self._update_header()
        status = self.query_one(StatusBar)
        status.set_directory(str(self.directory))
        self._main_chat = self.query_one(ChatView)
        self._chats[self.session.id] = self._main_chat
        self._footer = self.query_one(SubagentFooter)
        # First open: show the opencode logo banner until the first message
        # starts a real conversation (opencode shows its logo on the launch
        # screen, then it disappears once you begin typing/chatting).
        self._main_chat.show_logo()
        self.query_one(InputBar).focus()
        if self._main_engine is None:
            # First paint first: the engine chain imports ~0.4s of heavy modules
            # (agent.loop, tools, commands, …) that don't touch the widgets on
            # screen. Build that on a background thread so the frame is up while
            # it warms; _update_header shows cfg defaults until it's ready.
            threading.Thread(target=self._warm_engine, daemon=True).start()

    # -- session routing --------------------------------------------------
    def _chat_for(self, session_id: str) -> ChatView:
        """Chat view for a session, creating (hidden) one on first use so a
        spawned sub-agent has a live, switchable conversation."""
        chat = self._chats.get(session_id)
        if chat is not None:
            return chat
        chat = ChatView()
        self._chats[session_id] = chat
        try:
            self.query_one("#chat-stack", Vertical).mount(chat, after=self._main_chat)
        except Exception:
            pass
        chat.display = "none"
        return chat

    def _active_engine(self) -> AgentLoop:
        return self._engines.get(self._current_session_id, self.engine)

    def _active_session(self) -> Any:
        return self._sessions.get(self._current_session_id, self.session)

    def action_sessions(self) -> None:
        """Ctrl+A / `/sessions`: open the opencode-style picker (Today /
        Yesterday / older sections). Live sessions surface first (running
        sub-agents marked), then persisted sessions from disk."""
        from ..session import list_sessions, suggested_title
        from .session_list import SessionList

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _collect(sid: str, title: str, agent: str, created: float | None, status: str) -> None:
            if sid in seen:
                return
            seen.add(sid)
            rows.append(
                {
                    "id": sid,
                    "title": title,
                    "agent": agent,
                    "created": created,
                    "status": status,
                }
            )

        for sid, sess in self._sessions.items():
            status = "running" if sid in self._busy_sessions else ""
            _collect(sid, sess.title, sess.agent, getattr(sess, "created", None), status)
        seen_count = 0
        for sess in list_sessions():
            if getattr(sess, "parent_id", None):
                continue  # sub-agent sessions are transient, not resume targets
            if not getattr(sess, "has_messages", True):
                continue  # opened but never chatted — not a session to resume
            _collect(sess.id, sess.title or suggested_title(sess), sess.agent, sess.created, "")
            seen_count += 1
            if seen_count >= 200:
                break
        self.push_screen(
            SessionList(
                rows,
                current=self._current_session_id,
                on_rename=self._rename_session,
                on_delete=self._delete_session,
                on_save=self._save_current_session,
            ),
            self._on_session_picked,
        )

    def action_models(self) -> None:
        """Ctrl+M: launch the model picker (same as `/models`)."""
        self._open_model_picker()

    def _on_session_picked(self, choice: str | None) -> None:
        if choice:
            self._resume_session(choice)
        try:
            self.query_one(InputBar).input.focus()
        except Exception:
            pass

    def _switch_session(self, session_id: str) -> None:
        if session_id == self._current_session_id:
            return
        if session_id in self._pruned:
            # a finished sub-agent session was pruned (no engine/history left);
            # opening it would create an empty chat wired to the wrong engine
            self.notify("That sub-agent session is finished and closed.")
            self.query_one(InputBar).focus()
            return
        old = self._chats[self._current_session_id]
        self._current_session_id = session_id
        new = self._chat_for(session_id)
        old.display = "none"
        new.display = "block"
        # While a sub-agent chat has focus, ↑/←/→ route to session navigation
        # (parent / siblings) instead of scrolling the message list.
        new._session_is_child = bool(self._parent_of(session_id))
        # Remember the last sub-agent viewed under its parent so returning and
        # pressing ctrl+down resumes it (official keeps the selection).
        parent_of = self._parent_of(session_id)
        if parent_of:
            self._last_selection[parent_of] = session_id
        sess = self._sessions.get(session_id)
        if sess and sess.title:
            self.notify(f"Session: {sess.title}", markup=False)
        self._update_header()
        self._update_footer()
        # The footer's context hint (`12,345 (6%)`) must follow the session
        # now on screen — otherwise the previous session's token counter stays
        # painted on the status bar after a switch/resume.
        try:
            self.query_one(StatusBar).set_usage(self._usage.get(session_id) or {})
        except Exception:
            pass
        self.query_one(InputBar).focus()

    def on_open_task_session(self, event: Any) -> None:
        sid = getattr(event, "sid", None)
        if not sid:
            return
        # A finished sub-agent from a previous run is only persisted on disk
        # (it's filtered out of the picker); reopening its task row must load
        # that history instead of switching to an empty chat.
        if sid not in self._sessions and sid not in self._pruned:
            from ..session import load_session

            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
                chat = self._chat_for(sid)
                self._render_history(chat, sess.messages)
        self._switch_session(sid)

    # -- sub-agent navigation (official session.child.*) ------------------
    def _sibling_records(self, parent_id: str) -> list[dict[str, Any]]:
        """Every sub-agent the parent spawned, oldest first (official numbers
        ``(2 of 4)`` by creation time across ALL of the parent's children)."""
        return sorted(self._children.get(parent_id, []), key=lambda c: c.get("created") or 0)

    def _session_nav_active(self) -> bool:
        """Arrow-key session routing must not fight the prompt's cursor/history
        keys: while the user is typing a non-empty prompt the arrows belong to
        the input (official's input scope wins over the session scope)."""
        from .input_bar import PromptTextArea

        focused = self.focused
        if isinstance(focused, PromptTextArea):
            return not focused.text.strip()
        return True

    def _parent_of(self, session_id: str) -> str | None:
        """The parent session id of a session: from the live children registry
        (authoritative while a sub-agent is/was running) or the saved parent_id."""
        parent = self._child_parent.get(session_id)
        if parent:
            return parent
        sess = self._sessions.get(session_id)
        return getattr(sess, "parent_id", None)

    def _go_parent(self) -> bool:
        parent_id = self._parent_of(self._current_session_id)
        if parent_id:
            self._switch_session(parent_id)
            return True
        return False

    def _go_prev(self) -> None:
        self._move_sibling(-1)

    def _go_next(self) -> None:
        self._move_sibling(1)

    def _move_sibling(self, direction: int) -> bool:
        parent_id = self._parent_of(self._current_session_id)
        if not parent_id:
            return False
        siblings = self._sibling_records(parent_id)
        if len(siblings) <= 1:
            return False
        try:
            index = next(i for i, c in enumerate(siblings) if c.get("id") == self._current_session_id)
        except StopIteration:
            return False
        target = (index + direction) % len(siblings)
        self._switch_session(siblings[target]["id"])
        return True

    def _go_first_child(self) -> None:
        children = self._sibling_records(self._current_session_id)
        if not children:
            return
        # resume the child you were last viewing under this parent, like
        # official opencode; fall back to the first child.
        last = self._last_selection.get(self._current_session_id)
        target = next((c["id"] for c in children if c.get("id") == last), None)
        self._switch_session(target or children[0]["id"])

    def action_fd_parent(self) -> None:
        if self._session_nav_active():
            self._go_parent()

    def action_fd_prev(self) -> None:
        if self._session_nav_active():
            self._go_prev()

    def action_fd_next(self) -> None:
        if self._session_nav_active():
            self._go_next()

    def action_fd_first(self) -> None:
        self._go_first_child()

    def _chat_scroll_target(self) -> ChatView | None:
        """The chat of the session currently on screen (used by HOME/END/PgUp/
        PgDn so they scroll the conversation no matter what widget has focus)."""
        return self._chats.get(self._current_session_id)

    def action_chat_home(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_home(animate=False)

    def action_chat_end(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_end(animate=False)

    def action_chat_page_up(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_page_up(animate=False)

    def action_chat_page_down(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_page_down(animate=False)

    def on_nav_requested(self, event: NavRequested) -> None:
        if event.action == "parent":
            self._go_parent()
        elif event.action == "prev":
            self._go_prev()
        elif event.action == "next":
            self._go_next()

    def on_session_nav_requested(self, event: SessionNavRequested) -> None:
        """Arrow keys pressed with an empty prompt (posted from the input bar).
        `↑` parent (or previous prompt when there is no parent), `←`/`→` cycle
        the parallel sub-agent siblings, `↓` recalls the next prompt / draft."""
        key = event.direction
        if key == "up":
            bar = self.query_one(InputBar)
            if (
                bar.input.value == ""
                and bar._hist_index == len(bar._history)
                and bar._draft
            ):
                # the final ↓ cleared the box: the next ↑ restores what we were
                # typing instead of session-navigating away
                bar.input.value = bar._draft
                bar.input.cursor_position = len(bar._draft)
                return
            if not self._go_parent():
                bar.recall_history("up")
        elif key == "down":
            self.query_one(InputBar).recall_history("down")
        elif key == "left":
            self._go_prev()
        elif key == "right":
            self._go_next()

    def _update_footer(self) -> None:
        """Show the subagent footer only while viewing a child session
        (opencode's SubagentFooter: `Build (2 of 4)` + usage + parent/prev/next)."""
        footer = getattr(self, "_footer", None)
        if footer is None:
            return
        current = self._current_session_id
        self._mark_selected_task()
        parent_id = self._parent_of(current)
        if not parent_id:
            footer.hide()
            return
        sess = self._sessions.get(current)
        siblings = self._sibling_records(parent_id)
        if not siblings:
            footer.hide()
            return
        index = next(
            (i for i, c in enumerate(siblings) if c.get("id") == current),
            0,
        )
        footer.show(
            label=str(getattr(sess, "agent", "") or "build").title(),
            index=index + 1,
            total=len(siblings),
            usage=self._usage.get(current),
        )

    def _mark_selected_task(self) -> None:
        """Highlight which sub-agent is currently selected on the parent's task
        rows: the child being viewed, or — while sitting at the parent — the
        child you last opened (official opencode marks the active sub-agent)."""
        current = self._current_session_id
        parent_id = self._parent_of(current)
        if parent_id:
            target_sid, parent_chat_id = current, parent_id
        else:
            target_sid = self._last_selection.get(current)
            parent_chat_id = current
        for chat in self._chats.values():
            try:
                for bubble in chat.query(MessageBubble):
                    if bubble.role != "tool" or not isinstance(bubble.content, dict) or bubble.content.get("tool") != "task":
                        continue
                    meta = bubble.content.get("metadata") or {}
                    bubble.selected = str(meta.get("sessionId") or "") == target_sid
            except Exception:
                pass

    def _resume_session(self, session_id: str) -> None:
        """Switch to a live session or load a persisted one (engine + chat
        rebuilt around its saved history) so the conversation can continue."""
        if session_id in self._pruned:
            self.notify("That sub-agent session is finished and closed.")
            return
        if session_id in self._sessions:
            self._switch_session(session_id)
            return
        from ..session import load_session, suggested_title
        from ..agent.loop import AgentLoop
        from ..tools import build_registry as build_tool_registry

        sess = load_session(session_id)
        if sess is None:
            self.notify("Session not found.")
            return
        if not sess.title:
            sess.title = suggested_title(sess)
        engine = AgentLoop(
            cfg=self.cfg,
            registry=build_tool_registry(self.cfg),
            directory=Path(sess.directory) if sess.directory else self.directory,
            auth=self.auth,
            agent=sess.agent or "build",
            provider_id=sess.provider or self.cfg.provider,
            model_id=sess.model or self.cfg.model,
            session_id=session_id,
        )
        engine.on_event = self._on_engine_event
        engine.interrupt = self._interrupt_requested
        engine.permission.ask_callback = self._permission_ask
        engine.question_service.ask_callback = self._question_ask
        engine.set_history(sess.messages)
        self._engines[session_id] = engine
        self._sessions[session_id] = sess
        chat = self._chat_for(session_id)
        chat.display = "none"
        self._render_history(chat, sess.messages)
        self._switch_session(session_id)
        self.notify(f"Resumed: {sess.title or session_id}", markup=False)

    def _rename_session(self, session_id: str, title: str) -> str | None:
        """Persist a rename; returns an error message or None on success."""
        from ..session import load_session, save_session

        try:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = load_session(session_id)
                if sess is None:
                    return "Session not found."
            sess.title = title
            # Renames only stick to disk when the session actually has a
            # conversation — a never-chatted scratch session must not be
            # materialised into a file just because it was titled.
            if getattr(sess, "messages", None):
                save_session(sess)
        except Exception as e:
            return f"Rename failed: {e}"
        return None

    def _save_current_session(self) -> bool:
        """Save the session the user is currently in, right now (the Save
        button in the /sessions popup). Includes the engine's live history so
        the durable copy matches what's on screen."""
        from ..session import save_session

        sid = self._current_session_id
        sess = self._sessions.get(sid) or self.session
        engine = self._engines.get(sid)
        try:
            if engine is not None:
                history = engine.get_history()
                if history:
                    sess.messages = history
            if not getattr(sess, "messages", None):
                self.notify("Nothing to save — this session has no conversation.")
                return False
            save_session(sess)
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error", markup=False)
            return False
        self.notify(f"Session saved: {sess.title or '(untitled)'}", markup=False)
        return True

    def _delete_session(self, session_id: str) -> bool:
        """Delete a session from disk (and its live registrations). Any session
        is deletable — deleting the one you're in resets the workspace to a
        brand-new session so a later save doesn't resurrect the deleted file.
        A session mid-turn is still protected (it owns a running engine)."""
        from ..session import delete_session, new_session

        if session_id in self._busy_sessions:
            return False  # never drop a running turn out from under it
        if session_id == self.session.id:
            # deleting the workspace itself -> start a fresh session in place
            old_id = self.session.id
            self.session = new_session(
                directory=str(self.directory),
                provider=self.cfg.provider,
                model=self.cfg.model,
                agent=self.engine.agent,
            )
            self.engine.session_id = self.session.id
            try:
                self.engine.set_history([])
            except Exception:
                pass
            self._chats.pop(old_id, None)
            self._sessions.pop(old_id, None)
            self._engines.pop(old_id, None)
            self._turn.pop(old_id, None)
            self.engine.clear_prompts()
            self._chats[self.session.id] = self._main_chat
            self._sessions[self.session.id] = self.session
            self._engines[self.session.id] = self.engine
            self._current_session_id = self.session.id
            self._active_turn_session_id = self.session.id
            self._pruned.discard(old_id)
            self._main_chat.clear()
            self._update_header()
            # Do NOT persist the fresh replacement session here: saving an empty
            # session on every delete is how 0-message ghost files accumulate.
            # It only gets its own file once the user actually chats and one of
            # the save conditions fires (crash-safety autosave / exit / close).
            self.notify("Session deleted — starting a new one.")
            delete_session(old_id)
            return True
        # a non-main session (resumed, sub-agent, …)
        was_live = session_id in self._sessions
        was_current = session_id == self._current_session_id
        if was_current:
            self._switch_session(self.session.id)  # hop back to a live session
        # drop the deleted session from the sub-agent family tree
        self._children.pop(session_id, None)  # its own children first
        for pid, records in list(self._children.items()):
            self._children[pid] = [r for r in records if r.get("id") != session_id]
        self._child_parent.pop(session_id, None)
        self._task_start.pop(session_id, None)
        self._usage.pop(session_id, None)
        self._sessions.pop(session_id, None)
        self._engines.pop(session_id, None)
        chat = self._chats.pop(session_id, None)
        self._turn.pop(session_id, None)
        self._pruned.add(session_id)
        self._running_agents.pop(session_id, None)
        self._refresh_running_agents()
        if chat is not None:
            try:
                chat.remove()
            except Exception:
                pass
        if was_current:
            self.notify("Session deleted.")
        # a live (possibly never-persisted) session was removed above; a
        # purely-persisted one is gone once its file is deleted
        return True if was_live else delete_session(session_id)

    def _render_history(self, chat: ChatView, messages: list[dict[str, Any]]) -> None:
        """Replay saved messages into a ChatView (tool calls paired with their
        results so each run reads as one row). Tool `arguments` arrive as JSON
        strings in history but the renderers expect dicts, so parse them.

        Mounting one bubble costs ~6ms of Textual widget machinery, so a long
        saved session is filled in progressively (first chunk synchronous, the
        rest queued on the event loop) instead of freezing the UI for seconds.
        """
        import json as _json

        def _args(raw: Any) -> dict[str, Any]:
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    parsed = _json.loads(raw)
                except _json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            return {}

        def _reasoning_text(raw: Any) -> str:
            if isinstance(raw, str):
                return raw
            if isinstance(raw, list):
                parts = []
                for p in raw:
                    if isinstance(p, dict):
                        parts.append(str(p.get("text") or p.get("content") or ""))
                    elif isinstance(p, str):
                        parts.append(p)
                return "\n".join(p for p in parts if p)
            return str(raw or "")

        tasks: list[Callable[[], None]] = []  # one deferred bubble-mount per message
        pending: list[tuple[str, str, dict[str, Any]]] = []  # (call_id, toolname, input)
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                continue
            if role == "compaction" or msg.get("compaction"):
                text = str(content or "")
                tasks.append(lambda t=text: chat.append_compaction(t))
                continue
            if role == "user":
                text = str(content or "")
                agent = str(msg.get("agent") or "build")
                tasks.append(lambda t=text, a=agent: chat.append_user(t, agent=a))
                continue
            if role == "assistant":
                reasoning = msg.get("reasoning_content")
                if reasoning:
                    rtext = _reasoning_text(reasoning)
                    tasks.append(lambda t=rtext: chat.append_reasoning(t))
                if content:
                    ctext = str(content)
                    tasks.append(lambda t=ctext: chat.append_assistant(t))
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    pending.append(
                        (call.get("id", ""), fn.get("name", "tool"), _args(fn.get("arguments")))
                    )
                continue
            if role == "tool":
                call_id = msg.get("tool_call_id") or ""
                tool = msg.get("name") or "tool"
                tool_input: dict[str, Any] = {}
                for j, (cid, name, args) in enumerate(pending):
                    if cid == call_id:
                        tool, tool_input = name, args
                        pending.pop(j)
                        break
                else:
                    if pending:
                        # call_id missing (out-of-order results / poisoned save):
                        # match the OLDEST pending call for the SAME tool name so
                        # parallel tool results don't get swapped between
                        # different tools; only fall back to the oldest call when
                        # no same-name call is still waiting. Use the pending
                        # call's own id/name/args so the row stays accurate.
                        idx = 0
                        for j, (cid2, name2, args2) in enumerate(pending):
                            if name2 == tool:
                                idx = j
                                break
                        call_id, tool, tool_input = pending.pop(idx)
                metadata: dict[str, Any] = {}
                if tool == "todowrite":
                    # history stores the todos as JSON text; the renderer reads
                    # them from metadata.todos
                    try:
                        parsed = _json.loads(str(content or ""))
                        if isinstance(parsed, list):
                            metadata["todos"] = parsed
                    except _json.JSONDecodeError:
                        pass
                payload: dict[str, Any] = {
                    "id": call_id,
                    "tool": tool,
                    "input": tool_input,
                    # "completed" (not "done"): tool renderers + the gray
                    # block frame key off this exact value
                    "status": "completed",
                    "done": True,
                    "output": content or "",
                    "metadata": metadata,
                }
                tasks.append(lambda p=payload: chat.append_tool(p))
                continue
            if content:
                text = str(content)
                tasks.append(lambda t=text: chat.append_meta(t))

        # A session cut off mid-tool-run (interrupt / sudden kill / old
        # poisoned save) ends with an assistant tool_calls message whose results
        # never arrived. Render those as interrupted rows so resuming shows
        # exactly where the turn stopped instead of silently dropping them.
        for call_id, name, tool_input in pending:
            payload = {
                "id": call_id,
                "tool": name,
                "input": tool_input,
                "status": "interrupted",
                "done": False,
                "output": "",
                "metadata": {},
            }
            tasks.append(lambda p=payload: chat.append_tool(p))

        def _mount(i: int) -> None:
            step = 15
            for t in tasks[i : i + step]:
                t()
            nxt = i + step
            if nxt < len(tasks):
                self.call_later(lambda n=nxt: _mount(n))

        _mount(0)

    # -- running agents ---------------------------------------------------
    def _refresh_running_agents(self) -> None:
        """Show the launched sub-agents in the status line above the prompt,
        like opencode's `Delegating...` indicator (transient, no sidebar)."""
        try:
            bar = self.query_one(InputBar)
        except Exception:
            return
        bar.set_running_agents(list(self._running_agents.values()))

    # -- engine event bridge ---------------------------------------------
    def _on_engine_event(self, event: dict[str, Any]) -> None:
        # Called from the engine thread; hop to the UI thread.
        if getattr(self, "_thread_id", None) == threading.get_ident():
            # Already on the UI thread (e.g. a /command handler that makes the
            # engine emit an event, like /undo): call_from_thread would raise
            # RuntimeError, so handle inline instead.
            self._handle_event(event)
            return
        kind = event.get("kind")
        if kind in ("text_delta", "reasoning_delta"):
            # Don't block the engine stream on the UI render: queue the delta
            # (non-blocking) and let the flush timer batch it into one render.
            # FIFO scheduling keeps ordering with the blocking events below.
            self._schedule_async(event)
            return
        try:
            self.call_from_thread(self._handle_event, event)
        except RuntimeError:
            # app shutting down / loop gone — drop the event safely
            pass

    def _schedule_async(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_handle(event), loop)
        except (RuntimeError, Exception):
            pass

    async def _async_handle(self, event: dict[str, Any]) -> None:
        try:
            with self._context():
                self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A delta-render failure must never vanish into a dropped future
            # (``run_coroutine_threadsafe`` swallows coroutine exceptions and
            # would leave the screen silently frozen mid-stream). Log it so the
            # bug is actually visible in the console.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[tui] async event handler error: {e!r}\n")

    def _queue_delta(self, session_id: str, text: str, kind: str) -> None:
        if not text:
            # drop empty deltas entirely — an empty text chunk must not create
            # an assistant bubble that never gets filled
            return
        ui_thread = getattr(self, "_thread_id", None)
        if ui_thread is not None and threading.get_ident() != ui_thread:
            # Single-writer invariant: _pending and _delta_timer are only
            # touched by the UI thread (deltas arrive via _schedule_async on
            # the app loop; flushes run on the flush timer). If a future caller
            # ever queues off-thread, marshal to the UI thread instead of
            # racing the timer/swap instead of corrupting the buffer.
            try:
                self.call_from_thread(self._queue_delta, session_id, text, kind)
            except RuntimeError:
                pass
            return
        buf = self._pending.setdefault(session_id, {"text": [], "reasoning": []})
        buf[kind].append(text)
        if self._delta_timer is None:
            self._delta_timer = self.set_timer(0.03, self._flush_deltas)

    def _cancel_delta_timer(self) -> None:
        if self._delta_timer is not None:
            try:
                self._delta_timer.stop()
            except Exception:
                pass
            self._delta_timer = None

    def _flush_deltas(self) -> None:
        """Render any buffered text/reasoning deltas (one render per batch)."""
        self._cancel_delta_timer()
        pending = self._pending
        self._pending = {}
        for session_id, buf in pending.items():
            chat = self._chat_for(session_id)
            for t in buf.get("reasoning") or []:
                if t:
                    chat.stream_reasoning_delta(t)
            if buf.get("text"):
                chat.end_reasoning()
                for t in buf["text"]:
                    if t:
                        chat.stream_delta(t)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        session_id = event.get("session_id") or self.session.id
        chat = self._chat_for(session_id)
        status = self.query_one(StatusBar)
        if kind == "step":
            pass
        elif kind == "prompt_promoted":
            # the engine folded the oldest queued prompt into the running turn
            # (opencode's Session Drain) — drop its ` QUEUED ` badge, finalize
            # the previous reply's bubble, and let the next text stream in as a
            # fresh response to the promoted prompt.
            # the previous reply's text may still be sitting in the delta buffer
            # (deltas are batched on a 30ms flush timer) — empty it BEFORE
            # remove_last_stream_bubble decides whether the stream bubble is
            # "empty", otherwise the trailing bubble can be dropped while its
            # text is still un-rendered.
            self._flush_deltas()
            chat.promote_next_queued()
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
        elif kind == "retry":
            status.set_retry_message(event.get("message", "↻ retrying…"))
            self._on_retry_event(session_id, event)
        elif kind == "error":
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            self._show_error(event.get("error", "unknown error"), retryable=bool(event.get("retryable")), session_id=session_id)
        elif kind == "text_delta":
            # an empty trailing chunk (a failed summary) is real output only
            # if it actually carries text — otherwise it would light up the
            # runtime line and spawn a blank assistant bubble
            text = event.get("text") or ""
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            if text:
                self._turn_state(session_id)["had_text"] = True
            self._queue_delta(session_id, text, "text")
        elif kind == "reasoning_delta":
            text = event.get("text") or ""
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            if text:
                self._turn_state(session_id)["had_reasoning"] = True
            self._queue_delta(session_id, text, "reasoning")
        elif kind == "tool_call":
            # the model is now responding/acting — drop any stale retry hint
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            # render any buffered text first so the tool row lands below it
            self._flush_deltas()
            tool = event.get("tool", "?")
            # the current thought/assistant stream is over — finalize it so a
            # multi-step tool loop doesn't merge every step's text into one
            # bubble or leave a stale ▍ cursor
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            self._turn_state(session_id)["had_tools"] = True
            chat.append_tool(
                {
                    "tool": tool,
                    "status": "running",
                    "input": event.get("arguments", {}),
                    "call_id": event.get("call_id", ""),
                }
            )
        elif kind == "tool_start":
            tool_run = {
                "tool": event.get("tool", "?"),
                "status": "running",
                "input": event.get("input", {}),
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(tool_run):
                # No exact call_id match (e.g. the tool_call event never
                # arrived for this session) — append the row instead of leaving
                # a stale placeholder or updating the wrong tool.
                chat.append_tool(tool_run)
        elif kind == "tool_complete":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error" if event.get("status") == "error" else "completed",
                "input": event.get("input", {}),
                "output": event.get("output", ""),
                "metadata": event.get("metadata", {}),
                "call_id": event.get("call_id", ""),
            }
            chat.update_tool_bubble(run)
            if event.get("tool") == "task":
                # enrich the task row with the child's runtime + toolcall count
                # (official's `↳ 3 toolcalls · 12.5s` completion detail).
                self._finalize_task_row(chat, run)
        elif kind == "tool_denied":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error",
                "input": event.get("input", {}),
                "output": event.get("reason") or "permission denied",
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(run):
                # No preceding tool_call (e.g. a permission denial) — show the
                # denied row so the rejected action is visible.
                chat.append_tool(run)
        elif kind == "interrupted":
            self._flush_deltas()
            self._turn_state(session_id)["interrupted"] = True
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            chat.end_stream()
            chat.append_meta("⏹ Interrupted")
        elif kind == "usage":
            self._usage[session_id] = event.get("usage") or {}
            if session_id == self._current_session_id:
                status.set_usage(event.get("usage") or {})
        elif kind == "compaction_start":
            # official opencode: show `Compacting conversation…` with a spinner
            # while the session summarizes to recover/avoid context overflow.
            # Only when the compacting session is the one on screen: a
            # sub-agent (task tool) compacting in the background must NOT flash
            # the main bar's spinner — its summary bubble goes to the sub-agent's
            # own (hidden) chat, so the result would otherwise never appear and
            # just look like the current conversation vanished.
            if session_id == self._current_session_id:
                try:
                    self.query_one(InputBar).set_compacting(True)
                except Exception:
                    pass
            # Kick off the live `▸ Compacted summary` bubble (hidden chats too —
            # they get the final divider on switch).
            chat.begin_compaction_stream()
        elif kind == "summary_delta":
            # Stream the anchored summary into the compaction bubble live so the
            # user watches it being written rather than waiting on a spinner.
            text = event.get("text") or ""
            if text:
                chat.stream_compaction_delta(text)
        elif kind == "compacted":
            # opencode renders a ` Session compacted ` divider when the session
            # summarizes to recover/avoid context overflow. Mirror that: flush
            # pending deltas, end the current reasoning bubble, then finalize
            # the live summary bubble into the divider.
            self._flush_deltas()
            if session_id == self._current_session_id:
                try:
                    self.query_one(InputBar).set_compacting(False)
                except Exception:
                    pass
            chat.end_compaction_stream(event.get("summary") or "")
            self._turn_state(session_id)["had_tools"] = True
        elif kind == "rotated":
            # Show the failover popup ONLY when rotation is unlocked. When the
            # selected model is locked the lane can't change, so a "model
            # changed" toast would be noise (or a lie) — suppress it.
            eng = self._engines.get(session_id) or self.engine
            provider = event.get("provider", "?")
            model = event.get("model", "?")
            # reflect the lane that actually answered in the mode line under
            # the input box (deepseek -> nemotron) as soon as it switches.
            eng.provider_id = provider
            eng.model_id = model
            self._update_header()
            if not getattr(eng, "rotation_locked", False):
                reason = event.get("reason") or "provider error"
                self.notify(
                    f"Now using {model} · {provider}\n{reason}",
                    title="Model changed",
                )
        elif kind == "subagent_start":
            self._on_subagent_start(event)
        elif kind == "subagent_done":
            self._on_subagent_done(event)

    def _on_subagent_start(self, event: dict[str, Any]) -> None:
        from ..session import load_session

        sid = event.get("session_id") or ""
        if not sid:
            return
        if sid not in self._sessions:
            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
            else:
                # register a placeholder so the fallbacks
                # (`_sessions.get(sid) or self.session`) never route a sub-agent
                # turn's history onto the main session file.
                from ..session import Session
                self._sessions[sid] = Session({"id": sid}, directory=str(self.directory))
        sub = self.engine.find_subagent(sid)
        if sub is not None:
            self._engines[sid] = sub
        self._busy_sessions.add(sid)
        chat = self._chat_for(sid)
        # Show the parent's instruction at the very top of the sub-agent's
        # chat, exactly like official opencode renders the task directive as
        # the session's first message.
        prompt = event.get("prompt") or ""
        if prompt and not chat.children:
            chat.append_directive(prompt, title=event.get("title") or "")
        self._chats[sid] = chat
        title = event.get("title") or "sub-agent"
        agent = event.get("agent") or "build"
        sess = self._sessions.get(sid)
        parent_id = getattr(sess, "parent_id", None) or self._active_turn_session_id
        if parent_id:
            # register the child in the parent's family tree (numbered siblings
            # for the footer's `(2 of N)`), and attach its id to the parent
            # chat's running task row so it renders live state / is clickable.
            self._child_parent[sid] = parent_id
            records = self._children.setdefault(parent_id, [])
            if not any(c.get("id") == sid for c in records):
                records.append(
                    {
                        "id": sid,
                        "title": title,
                        "agent": agent,
                        "created": time.time(),
                        "status": "running",
                    }
                )
            self._link_task_row(sid, parent_id, call_id=event.get("call_id") or "")
        self._running_agents[sid] = f"{title} · {agent}"
        self._refresh_running_agents()
        self._update_footer()
        self.notify(f"Sub-agent started: {title}", markup=False)

    def _on_subagent_done(self, event: dict[str, Any]) -> None:
        sid = event.get("session_id") or ""
        sess = self._sessions.get(sid)
        if sess is not None:
            sess.completed = time.time()
            # Keep the in-memory session's messages in sync with what the
            # sub-agent engine actually produced (spawn_task saves to disk, but
            # the app-side object was last loaded with the empty placeholder).
            sub = self._engines.get(sid)
            if sub is not None:
                try:
                    sess.messages = sub.get_history()
                except Exception:
                    pass
        self._busy_sessions.discard(sid)
        chat = self._chats.get(sid)
        if chat is not None:
            chat.end_reasoning()
            # drop an empty streaming cursor if the sub-agent replied with no text
            chat.remove_last_stream_bubble()
            chat.end_stream()
        title = event.get("title") or "sub-agent"
        ok = event.get("ok", True)
        self._running_agents.pop(sid, None)
        self._refresh_running_agents()
        # Keep the finished sub-agent session registered (like the official
        # store) so its task row stays clickable, its chat stays reviewable,
        # and the parent's `(2 of N)` footer count keeps them in the total.
        record = None
        for r in self._children.get(self._child_parent.get(sid, ""), []):
            if r.get("id") == sid:
                record = r
                break
        if record is not None:
            record["status"] = "completed" if ok else "failed"
            record["completed"] = time.time()
        self._update_footer()
        if ok:
            self.notify(f"Sub-agent done: {title}", markup=False)
        else:
            self.notify(f"Sub-agent failed: {title}", severity="error", markup=False)

    # -- task-row enrichment (official Subagent completion detail) --------
    def _link_task_row(self, sid: str, parent_id: str, call_id: str = "") -> None:
        """Attach a sub-agent's session id + start time to the exact task row
        that spawned it (the parent chat's row with the matching tool call id).

        With parallel sub-agents the events arrive concurrently, so "newest
        unattached running row" is wrong under the hood — the call id makes the
        mapping unambiguous (official opencode keys every sub-agent by its task
        call). A missing id keeps the old fallback for compatibility."""
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        if call_id:
            bubble = chat.find_tool("task", call_id)
            if bubble is not None and bubble.content.get("status") == "running" and not (bubble.content.get("metadata") or {}).get("sessionId"):
                bubble.set_tool_metadata("sessionId", sid)
                self._task_start[sid] = time.monotonic()
                return
        for child in reversed(list(chat.query(MessageBubble))):
            if child.role != "tool" or child.content.get("tool") != "task":
                continue
            meta = child.content.get("metadata") or {}
            if child.content.get("status") == "running" and not meta.get("sessionId"):
                child.set_tool_metadata("sessionId", sid)
                self._task_start[sid] = time.monotonic()
                break

    def _finalize_task_row(self, chat: ChatView, run: dict[str, Any]) -> None:
        """Write the completed task row's runtime + toolcall count (the
        `↳ 3 toolcalls · 12.5s` line under a finished sub-agent)."""
        sid = (run.get("metadata") or {}).get("sessionId")
        if not sid:
            return
        bubble = chat.find_task(sid)
        if bubble is None:
            return
        start = self._task_start.pop(sid, None)
        if start is not None:
            bubble.set_tool_metadata("duration", format_duration(time.monotonic() - start))
        child = self._chats.get(sid)
        toolcalls = 0
        if child is not None:
            toolcalls = len([r for r in child.tool_runs() if r.get("status") == "completed"])
        bubble.set_tool_metadata("toolcalls", toolcalls)

    def _on_retry_event(self, session_id: str, event: dict[str, Any]) -> None:
        """A sub-agent's provider lane is retrying — paint its parent task row
        error-red with `↳ Retrying (attempt N) · …` (official Subagent retry)."""
        parent_id = self._child_parent.get(session_id)
        if not parent_id:
            return
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        bubble = chat.find_task(session_id)
        if bubble is not None:
            bubble.set_tool_metadata(
                "retry",
                {"attempt": event.get("attempt", 1), "message": event.get("message", "")},
            )

    def _clear_task_retry(self, session_id: str) -> None:
        """The sub-agent made progress again — remove the retry decoration."""
        parent_id = self._child_parent.get(session_id)
        if not parent_id:
            return
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        bubble = chat.find_task(session_id)
        if bubble is not None:
            bubble.pop_tool_metadata("retry")

    # -- prompt handling -------------------------------------------------
    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        value = event.value
        if not value.strip():
            return
        sid = self._current_session_id
        chat = self._chat_for(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        # always show what the user typed, then route it
        bubble = chat.append_user(value, agent=engine.agent)
        if value.lstrip().startswith("/"):
            self._run_command(value.lstrip())
            return
        # name the session from its first real message (opencode behaviour)
        if not getattr(session, "title", ""):
            session.title = value.strip()[:60]
        if self._busy:
            # opencode's queue-and-promote: never drop a prompt typed while the
            # agent is busy. It goes into the ENGINE's FIFO (thread-safe, one
            # per session) and its bubble shows the ` QUEUED ` badge. The next
            # provider-turn boundary of the SAME running turn folds it in
            # (run_turn drains the queue), so there is no fresh-turn gap —
            # exactly how opencode keeps one Session Drain going.
            bubble.queued = True
            depth = engine.queue_prompt(value)
            self.notify(f"Queued ({depth}) — will run in the current turn")
            return
        self._start_turn(sid, value, engine)

    def _autosave_in_flight(self) -> None:
        """Periodic crash-safety save of the running turn's session.

        The engine keeps the in-flight assistant reply live in get_history()
        as the stream grows, so this captures the very last conversation. Runs
        on the app thread via a timer; a sudden process kill can only lose the
        tokens streamed since the previous tick.

        The durable write (temp file + fsync + atomic rename + .bak replica +
        index) can take 10-25ms on this phone's flash, so it is done on a
        background thread — never on the UI thread where it would hitch the
        render every 2s mid-stream. The thread never mutates the live Session
        (snapshot copy) and its write is guarded by an autosave generation so
        it can't race a newer turn/exit save.
        """
        if not self._busy:
            return
        if self._autosave_thread is not None and self._autosave_thread.is_alive():
            return  # a slower disk can't pile up saves
        try:
            sid = self._active_turn_session_id
            engine = self._engines.get(sid)
            sess = self._sessions.get(sid)
            if engine is not None and sess is not None:
                history = engine.get_history()
                # Only overwrite with a real conversation: the worker may not
                # have appended the prompt yet when the first tick fires, and
                # saving the empty history would destroy the durable copy that
                # _start_turn just wrote.
                if not history:
                    return
                generation = self._autosave_generation
                self._autosave_thread = threading.Thread(
                    target=self._autosave_write,
                    args=(sid, history, generation),
                    daemon=True,
                )
                self._autosave_thread.start()
        except Exception:
            pass

    def _autosave_write(self, sid: str, history: list[dict[str, Any]], generation: int) -> None:
        """Worker thread body: persist a snapshot WITHOUT mutating the live
        Session shared with the UI thread. Skipped via `should_write` if the
        autosave generation moved on (turn ended / exit save-all started)."""
        try:
            from ..session import save_session

            sess = self._sessions.get(sid)
            if sess is None:
                return
            snapshot = copy.copy(sess)
            snapshot.messages = list(history)
            save_session(
                snapshot,
                should_write=lambda: self._autosave_generation == generation,
            )
        except Exception:
            pass

    def _cancel_autosave(self) -> None:
        self._autosave_generation += 1  # invalidate any in-flight streaming autosave
        if self._autosave_timer is not None:
            try:
                self._autosave_timer.stop()
            except Exception:
                pass
            self._autosave_timer = None

    def _turn_state(self, sid: str) -> dict[str, Any]:
        """The per-session turn bookkeeping slot (created on first touch)."""
        st = self._turn.get(sid)
        if st is None:
            st = {
                "had_text": False,
                "had_reasoning": False,
                "had_error": False,
                "had_tools": False,
                "interrupted": False,
                "started": None,
            }
            self._turn[sid] = st
        return st

    def _start_turn(self, sid: str, value: str, engine: AgentLoop) -> None:
        """Start a model turn in a worker thread for the initial prompt.

        Prompts submitted while this turn runs are queued on the engine and
        folded into the SAME turn (one Session Drain) at the next provider-turn
        boundary; this is only the drain's starting point.
        """
        chat = self._chat_for(sid)
        status = self.query_one(StatusBar)
        # show an eager Thinking… bubble immediately (before the first token
        # arrives) so the UI reacts to Enter instead of sitting silent
        chat.begin_thinking()
        status.set_streaming(True)
        try:
            bar = self.query_one(InputBar)
            bar.set_busy(True)
        except Exception:
            pass
        st = self._turn_state(sid)
        st.update(
            had_text=False,
            had_reasoning=False,
            had_error=False,
            had_tools=False,
            interrupted=False,
        )
        st["started"] = time.monotonic()
        # the previous turn's runtime disappears the moment the model starts
        # working again (official opencode shows it only on the final report)
        self._clear_last_duration()
        self._busy = True
        self._busy_sessions.add(sid)
        self._active_turn_session_id = sid
        # Persistence policy: starting a turn NEVER writes a file. A session
        # is only durably stored once one of the save conditions fires —
        # (1) the 2s crash-safety autosave tick while a real conversation
        # streams (phone dies / reboot mid-turn -> at most the last 2 seconds
        # are lost), (2) the final save-all on exit (ctrl+q / /exit), or (3) the
        # SIGTERM/SIGHUP handler when Termux is closed. Merely starting a turn
        # must never mint an empty or half-built session file.
        if self._autosave_timer is None:
            self._autosave_timer = self.set_interval(2.0, self._autosave_in_flight)

        def run():
            result = None
            try:
                result = engine.run_turn(value)
            except Exception as e:  # never let a worker crash silently
                self.call_from_thread(self._show_error, f"{type(e).__name__}: {e}", False, sid)
            finally:
                # Pass THIS worker's sid: the global _active_turn_session_id can
                # already point at a newer turn (another session's prompt or a
                # promoted queued prompt), and finalizing the wrong session's
                # chat would clear the wrong flags and promote the wrong queue.
                self.call_from_thread(self._turn_done, result, sid)

        self.run_worker(run, thread=True)

    def _show_error(self, message: str, retryable: bool = False, session_id: str | None = None) -> None:
        sid = session_id or self.session.id
        self._turn_state(sid)["had_error"] = True
        chat = self._chat_for(sid)
        self._flush_deltas()
        chat.end_reasoning()
        chat.remove_last_stream_bubble()
        chat.append_meta(f"⚠ {message}")
        hint = " Retry, or check /connect for a model/API key." if retryable else ""
        self.notify(f"error: {message}{hint}", severity="error")
        chat.end_stream()
        # Reset the visual streaming state here too, not only in _turn_done. The
        # worker's `finally` normally calls _turn_done and clears these, but if
        # that call_from_thread ever fails (e.g. app unmount mid-error) the UI
        # must not stay stuck on an "streaming" indicator on the error path.
        self._streaming_visual_reset()

    def _clear_last_duration(self) -> None:
        """Hide the previous turn's runtime on the mode line.

        Official opencode shows the runtime (`▣ Build · model · 1m 12s`) only
        while the final report is displayed; it disappears as soon as the model
        starts doing things again (a new turn begins working / running tools).
        """
        try:
            self.query_one(InputBar).set_last_duration("")
        except Exception:
            pass

    def _streaming_visual_reset(self) -> None:
        """Defensively clear the streaming-indicator UI state (status bar +
        input bar). Does NOT touch _busy/_busy_sessions: those belong to the
        worker thread and are owned by _turn_done's finally, so resetting them
        here could race an active turn on another session."""
        try:
            self.query_one(StatusBar).set_streaming(False)
        except Exception:
            pass
        try:
            self.query_one(InputBar).set_busy(False)
        except Exception:
            pass

    def _turn_done(self, result: Any = None, sid: str | None = None) -> None:
        self._interrupt_flag["requested"] = False
        self._cancel_autosave()
        # Never trust the global _active_turn_session_id here: it can already
        # point at a NEWER turn (another session's prompt, or the promoted
        # queued prompt of a just-finished drain). This worker finalizes the
        # session it actually ran — passed in by _start_turn's run().
        sid = sid or self._active_turn_session_id
        st = self._turn_state(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        if result is not None and result.provider_id:
            # reflect the lane/model that actually answered (e.g. openrouter)
            engine.provider_id = result.provider_id
            engine.model_id = result.model_id or engine.model_id
            engine.rebuild_rotation()
            self._update_header()
        chat = self._chat_for(sid)
        status = self.query_one(StatusBar)
        status.set_retry_message("")
        self._flush_deltas()
        chat.end_reasoning()
        if not st["had_text"] and not st["had_reasoning"] and not st["had_error"] and not st["had_tools"] and not st["interrupted"]:
            # provider returned nothing (no text, no reasoning, no tool call,
            # no error) — drop the empty streaming cursor bubble before
            # end_stream clears its pointer.
            chat.remove_last_stream_bubble()
            chat.append_meta(
                "(no reply from the model — check your connection and /connect "
                "for a working model, or switch rotation in /config)"
            )
            self.notify("No reply from the model.", severity="warning")
        chat.end_stream()
        if st["had_text"] and not st["had_error"]:
            # the mode line lives fixed above the prompt box now, not in the chat
            self._update_header()
        # show the finished turn's runtime (`▣ Build · model · 1m 12s`) like
        # opencode's per-message footer, but ONLY on the final report. Official
        # opencode computes the duration when the message finished with a real
        # text answer (`finish` not tool-calls), so a tool-only, errored, or
        # interrupted turn shows no runtime — it appears again on the next
        # turn's last report.
        started = st["started"]
        st["started"] = None
        if started is not None:
            elapsed = time.monotonic() - started
            if st["had_text"] and not st["had_error"] and not st["interrupted"]:
                try:
                    from .input_bar import format_duration

                    self.query_one(InputBar).set_last_duration(format_duration(elapsed))
                except Exception:
                    pass
        status.set_streaming(False)
        self._busy = False
        self._busy_sessions.discard(sid)
        try:
            self.query_one(InputBar).set_busy(False)
        except Exception:
            pass
        # Keep the in-memory snapshot fresh (a later exit/close save reads it)
        # but persist NOTHING at turn end: per the save policy a conversation
        # file is only written on exit / Termux close / a crash mid-turn (the
        # 2s autosave tick already snapshotted it while streaming).
        session.messages = engine.get_history()
        # this turn's bookkeeping slot is fully consumed — drop it (each future
        # turn / session / sub-agent owns its own slot, so nothing leaks across)
        interrupted = st["interrupted"]
        self._turn.pop(sid, None)
        self._disarm_interrupt_escape()
        # opencode's queue-and-promote fallback: normally the ENGINE folds
        # queued prompts into the running turn (one Session Drain) — but a
        # prompt submitted in the tiny window after the turn's last boundary
        # but before it returns sits in the engine queue. Promote it as the
        # start of the next drain so nothing is ever left stuck "queued".
        # An interrupted turn stops the drain for real: the remaining queue
        # stays queued (the ` QUEUED ` badges keep showing) and only runs when
        # the next prompt starts a fresh drain.
        if not interrupted and engine.prompt_pending() > 0:
            value = engine.pop_prompt()
            self._chat_for(sid).promote_next_queued()
            self.notify("Running queued request")
            if value:
                self._start_turn(sid, value, engine)
                return
        self.query_one(InputBar).focus()

    # -- command handling -------------------------------------------------
    def _run_command(self, line: str) -> None:
        from ..commands import handle_command
        from ..commands import CommandContext

        # /models is a full-screen, live model list. The bare form is already
        # intercepted by the command popup; with arguments it would fall through
        # to the sync fetch_zen_models() in commands.py and freeze the UI thread,
        # so route it to the picker (which fetches off-thread) instead.
        # /sessions opens the same opencode-style picker as Ctrl+R (the plain
        # /sessions command only prints a text list).
        stripped = line.strip()
        name = (
            stripped[1:].split(maxsplit=1)[0]
            if stripped.startswith("/") and len(stripped) > 1
            else ""
        )
        if name == "models":
            self._open_model_picker()
            return
        if name == "sessions":
            self.action_sessions()
            return
        # Mutating commands must not run mid-turn: they'd race the running
        # engine (e.g. /undo popping the undo stack the worker is appending to).
        if self._busy and name not in _SAFE_WHILE_BUSY:
            self._chat_for(self._current_session_id).append_meta(
                "⏳ still working on the previous request…"
            )
            self.notify("Busy — finish or interrupt the running request first (Ctrl+C)")
            return

        engine = self._active_engine()
        session = self._active_session()

        def reply(text: str) -> None:
            # persistent chat output + a short toast; /models & friends must
            # not vanish into a transient notification
            self._chat_for(self._current_session_id).append_meta(text)
            self.notify(text.splitlines()[0][:60] if text else "", timeout=3, markup=False)

        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=session,
            engine=engine,
            worktree=str(self.directory),
            reply=reply,
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            connect=self._open_connect,
            registry=self.command_registry,
        )
        handle_command(self.command_registry, ctx, line)
        self._update_header()

    def _preview_command(self, name: str) -> str:
        """Run a read-only command with a collecting reply and return its output."""
        from ..commands import CommandContext, handle_command

        collected: list[str] = []
        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=self._active_session(),
            engine=self._active_engine(),
            worktree=str(self.directory),
            reply=collected.append,
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            connect=self._open_connect,
            registry=self.command_registry,
        )
        handle_command(self.command_registry, ctx, f"/{name}")
        return "\n".join(collected)

    def on_command_selected(self, event: CommandSelected) -> None:
        # /models is the full-screen, live-updating provider model list
        if event.name == "models":
            self._open_model_picker()
            return
        if event.name == "sessions":
            self.action_sessions()
            return
        cmd = self.command_registry.get(event.name)
        content: str | None = None
        if cmd is not None and cmd.preview:
            content = self._preview_command(event.name).strip() or cmd.description

        def done(result: str | None) -> None:
            bar = self.query_one(InputBar)
            bar.input.focus()
            if result == "run":
                self._run_command(f"/{event.name}")
            else:
                # Esc = back: put the command back in the input, cursor at the end
                bar.input.value = f"/{event.name}"
                bar.input.cursor_position = len(bar.input.value)

        from .command_popup import CommandPopup

        self.push_screen(
            CommandPopup(event.name, event.description, content=content),
            done,
        )

    def _open_model_picker(self) -> None:
        def on_picked(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception:
                pass
            if not choice:
                return
            provider, _, model = choice.partition("/")
            if not provider or not model:
                return
            self.cfg.provider = provider
            self.cfg.model = model
            engine = self._active_engine()
            engine.provider_id = provider
            engine.model_id = model
            engine.rebuild_rotation()
            self.notify(f"Model set to {provider}/{model}")
            self._update_header()
            from ..config import save_config

            save_config(self.cfg)

        from .model_picker import ModelPicker

        self.push_screen(
            ModelPicker(current=self.cfg.model, cfg=self.cfg, auth=self.auth),
            on_picked,
        )

    def _open_connect(self, provider: str = "") -> None:
        from .connect_screen import ConnectScreen

        self.app.push_screen(
            ConnectScreen(auth=self.auth, on_connected=self._on_connected, initial=provider),
            self._on_connect_dismissed,
        )

    def _on_connected(self, provider_id: str) -> None:
        self.notify(f"Saved API key for {provider_id}.")
        self._update_header()

    def _on_connect_dismissed(self, result: str | None) -> None:
        if result:
            self.notify(f"Connected {result}.")

    # -- actions ---------------------------------------------------------
    def on_agent_toggle_requested(self, event: AgentToggleRequested) -> None:
        self.action_toggle_agent()

    def on_sessions_requested(self, event: Any) -> None:
        self.action_sessions()

    def on_models_requested(self, event: Any) -> None:
        self.action_models()

    def _interrupt_requested(self) -> bool:
        return bool(self._interrupt_flag["requested"])

    def _interrupt_engines(self) -> None:
        """Force-close every engine's active provider stream.

        Flipping the interrupt flag alone only stops the turn at the next
        per-chunk check; an idle "thinking" gap (no chunks arriving) would keep
        the stream blocked until the model emits. Closing the stream makes the
        blocked read surface the interrupt immediately."""
        engines = [self.engine]
        engines.extend(self._engines.values())
        seen: set[int] = set()
        for engine in engines:
            if id(engine) in seen:
                continue
            seen.add(id(engine))
            try:
                engine.abort()
            except Exception:
                pass

    def action_interrupt(self) -> None:
        # Flipping the shared flag makes run_turn stop at its next iteration
        # check (loop.py). The worker thread's `finally` then calls _turn_done,
        # which resets the flag and clears _busy — we must NOT call _turn_done
        # here, or the worker would finish concurrently and double-complete.
        if self._busy:
            self.notify("Interrupting...")
            self._interrupt_flag["requested"] = True
            self._interrupt_engines()
            self._disarm_interrupt_escape()

    def action_interrupt_escape(self) -> None:
        """ESC with opencode's double-press contract (`session.interrupt`).

        Mirrors the official TUI exactly: a press counter, a 5s window that
        resets it, and the actual abort only on the SECOND press within that
        window. First press arms the `esc again to interrupt` hint in the
        footer; the second press sends the abort — in Python terms, it flips
        the shared interrupt flag that every engine (main + sub-agents) checks
        at each stream chunk / tool boundary, ending the turn as interrupted
        (opencode's `MessageAbortedError` -> "interrupted"). Idle sessions just
        move focus back to the prompt.
        """
        self._cancel_esc_timer()
        if not self._busy:
            self.query_one(InputBar).focus()
            return
        # opencode: `setStore("interrupt", store.interrupt + 1)` on every press
        self._esc_presses += 1
        self._esc_timer = self.set_timer(5.0, self._disarm_interrupt_escape)
        if self._esc_presses >= 2:
            # opencode: `sdk.client.session.abort({ sessionID })` + reset
            self._interrupt_flag["requested"] = True
            self._interrupt_engines()
            self._disarm_interrupt_escape()
            self.notify("Interrupting...")
            return
        self._arm_interrupt_escape(armed=True)

    def _arm_interrupt_escape(self, armed: bool) -> None:
        if armed:
            self._esc_presses = 1
        else:
            self._esc_presses = 0
            self._cancel_esc_timer()
        try:
            self.query_one(StatusBar).set_interrupt_armed(armed)
        except Exception:
            pass

    def _cancel_esc_timer(self) -> None:
        if self._esc_timer is not None:
            self._esc_timer.stop()
            self._esc_timer = None

    # -- auto-refocus -------------------------------------------------------
    def _is_main_screen_active(self) -> bool:
        """True when the normal chat screen is on top (no modal / app exiting)."""
        try:
            return self._main_screen is not None and self.screen is self._main_screen
        except Exception:
            return False

    def on_descendant_focus(self, event: Any) -> None:
        """Drag the prompt cursor back ~1s after focus leaves the input box.

        Tapping a reasoning bubble (to expand it) or the chat area on a phone
        steals focus, so the blinking cursor disappears and typing stops. Any
        focus change outside the prompt arms a short timer; if the user touches
        nothing else in the meantime, the cursor comes back by itself."""
        if self._refocus_timer is not None:
            self._refocus_timer.stop()
            self._refocus_timer = None
        if not self._is_main_screen_active():
            # a picker / dialog on top, or the app is shutting down — never
            # steal focus or arm timers
            return
        try:
            if self.query_one(InputBar).input.has_focus:
                return
        except Exception:
            return
        self._refocus_timer = self.set_timer(1.0, self._refocus_prompt)

    def _refocus_prompt(self) -> None:
        self._refocus_timer = None
        if not self._is_main_screen_active():
            return
        try:
            self.query_one(InputBar).input.focus()
        except Exception:
            pass

    def _disarm_interrupt_escape(self) -> None:
        self._arm_interrupt_escape(False)

    # -- permission dialog (engine thread -> UI) --------------------------
    def _permission_ask(self, description: str, always_patterns: list[str]) -> str:
        """Bridge the engine thread's permission.ask to a modal dialog.

        Runs on the engine worker thread. Pushes the dialog on the UI thread via
        call_from_thread (which blocks until the push returns), then waits for the
        user's decision. Returns "once" / "always" / "reject".
        """
        outcome: dict[str, str] = {}
        decided = threading.Event()

        def on_decision(decision: str) -> None:
            outcome["decision"] = decision
            decided.set()

        try:
            self.call_from_thread(
                self._show_permission_dialog, description, on_decision
            )
        except Exception:
            return "reject"
        # The dialog stays open until the user answers; polling the shutdown
        # event means quitting the app unblocks the engine thread immediately
        # instead of hanging for the timeout below.
        while not self._exit_requested.is_set():
            if decided.wait(timeout=0.5):
                break
        return outcome.get("decision", "reject")

    def _show_permission_dialog(
        self, description: str, on_decision: Any
    ) -> None:
        if not self.is_attached:
            on_decision("deny")
            return
        from .permission_dialog import PermissionDialog

        self.push_screen(PermissionDialog(description, on_decision=on_decision))

    # -- question dialog (engine thread -> UI) ----------------------------
    def _question_ask(self, questions: list[QuestionInfo]) -> list[list[str]]:
        """Bridge the engine thread's question.ask to a modal dialog.

        Runs on the engine worker thread, mirroring ``_permission_ask``.
        Returns the answers (list of list[str], one per question) or raises
        QuestionRejectedError when the user dismisses / the app is quitting.
        """
        from ..question import QuestionInfo, QuestionRejectedError

        result: dict[str, Any] = {}
        answered = threading.Event()

        def on_done(answers: list[list[str]] | None) -> None:
            result["answers"] = answers
            answered.set()

        try:
            self.call_from_thread(
                self._show_question_dialog, questions, on_done
            )
        except Exception:
            raise QuestionRejectedError("no UI to ask the user") from None
        while not self._exit_requested.is_set():
            if answered.wait(timeout=0.5):
                break
        answers = result.get("answers")
        if answers is None:
            raise QuestionRejectedError("user dismissed the question")
        return answers

    def _show_question_dialog(
        self, questions: list[QuestionInfo], on_done: Any
    ) -> None:
        if not self.is_attached:
            on_done(None)
            return
        from .question_dialog import QuestionDialog

        self.push_screen(QuestionDialog(questions, on_done=on_done))

    def _save_all_live_sessions(self) -> None:
        """Final persist of every LIVE session that actually has a conversation.

        This is one of the ONLY durable-write paths (besides the in-flight
        crash-safety autosave tick). It runs when the user quits (ctrl+q /
        /exit), when the TUI tears down, and when Termux closes the app via
        SIGTERM/SIGHUP. Sessions with no conversation are skipped, so an
        untouched scratch session never leaves a file behind.
        """
        from ..session import save_session

        # Newer than any in-flight autosave: a worker that hasn't written yet
        # must not overwrite these final bodies with its (older) snapshot.
        self._autosave_generation += 1
        for sid, sess in list(self._sessions.items()):
            engine = self._engines.get(sid)
            history = None
            if engine is not None:
                try:
                    history = engine.get_history()
                except Exception:
                    history = None
            if not history:
                history = list(getattr(sess, "messages", None) or [])
            if not history:
                continue
            sess.messages = history
            try:
                save_session(sess)
            except Exception:
                pass

    def on_exit_app(self) -> None:
        """Fired when the app quits (ctrl+q / /exit).

        Unblocks any engine thread waiting on a permission dialog and persists
        the final state of every live session that has a conversation.
        """
        self._exit_requested.set()
        try:
            self._save_all_live_sessions()
        except Exception:
            pass

    def on_unmount(self) -> None:
        """Final persist as the TUI tears down (same policy as on_exit_app)."""
        try:
            self._save_all_live_sessions()
        except Exception:
            pass

    def action_resume(self) -> None:
        """Ctrl+R: open the session picker to continue a past conversation."""
        self.action_sessions()

    def action_toggle_agent(self) -> None:
        engine = self._active_engine()
        self._set_agent("plan" if engine.agent != "plan" else "build")

    def action_settings(self) -> None:
        from .settings_screen import SettingsScreen

        self.push_screen(
            SettingsScreen(
                cfg=self.cfg,
                engine=self.engine,
                auth=self.auth,
                session=self.session,
                on_model_change=self._set_model,
            ),
            self._on_settings_done,
        )

    def _on_settings_done(self, result: Any) -> None:
        self.query_one(InputBar).focus()

    def _on_model_picked(self, model: str | None) -> None:
        if model:
            self._set_model(model)

    def action_focus_input(self) -> None:
        self.query_one(InputBar).focus()

    def action_toggle_thought(self) -> None:
        chat = self._chat_for(self._current_session_id)
        chat.toggle_last_reasoning()
        self.query_one(InputBar).focus()

    def _set_model(self, model: str) -> None:
        self.cfg.model = model
        engine = self._active_engine()
        engine.model_id = model
        engine.rebuild_rotation()
        self.notify(f"Model set to opencode/{model}")
        self._update_header()

    def on_rotation_lock_toggled(self, event: RotationLockToggled) -> None:
        """Clicking the model dot in the meta row pins/unpins the selected model."""
        engine = self._active_engine()
        engine.rotation_locked = not engine.rotation_locked
        self.cfg.rotation_lock = engine.rotation_locked
        if engine.rotation_locked:
            self.notify(
                f"Rotation locked — staying on {engine.model_id} "
                "(rate limits/hard errors will surface, not switch)"
            )
        else:
            self.notify("Rotation unlocked — will fail over to backup lanes on errors")
        try:
            from ..config import save_config

            save_config(self.cfg)
        except Exception:
            pass
        self._update_header()

    def _set_agent(self, agent: str) -> None:
        engine = self._active_engine()
        if engine.agent == agent:
            self.notify(f"Agent: {agent}")
            self._update_header()
            return
        engine.agent = agent
        # The PermissionEngine is built from the agent's permission defaults at
        # construction time (build vs plan differ: plan force-denies
        # bash/write/edit/apply_patch). Switching agents mid-session must
        # rebuild it, otherwise those deny rules never take effect and a plan
        # agent can still execute a mutating tool call the model emits (e.g. a
        # bash command). Sub-agents share this same engine (spawn passes
        # permission_engine=self.permission), so one rebuild covers them too.
        from ..permission import PermissionEngine, merge_permissions

        engine.permission = PermissionEngine.from_config(
            merge_permissions(self.cfg.permission, agent),
            mode=getattr(engine.permission, "mode", "auto"),
        )
        engine.permission.ask_callback = self._permission_ask
        sess = self._sessions.get(self._current_session_id)
        if sess is not None:
            sess.agent = agent
        self.notify(f"Agent: {agent}")
        self._update_header()

    def _update_header(self) -> None:
        status = self.query_one(StatusBar)
        # The engine may not be warmed yet (first paint happens before the
        # background engine build finishes) — fall back to cfg values rather
        # than forcing the ~0.4s import on the UI thread mid-mount.
        engine = self._engines.get(self._current_session_id) or self._main_engine
        if engine is None:
            header = {
                "agent": self.cfg.default_agent or "build",
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                "permission_mode": "auto",
                "rotation_locked": False,
            }
        else:
            header = {
                "agent": engine.agent,
                # reflect the model/provider that actually answered: rotation can
                # fail over to a backup lane (e.g. deepseek -> nemotron) while
                # cfg.model keeps the user's configured base model.
                "model": getattr(engine, "model_id", "") or self.cfg.model,
                "provider": getattr(engine, "provider_id", "") or self.cfg.provider,
                "permission_mode": engine.permission.mode,
                "rotation_locked": getattr(engine, "rotation_locked", False),
            }
        status.set_header(**header)
        try:
            bar = self.query_one(InputBar)
        except Exception:
            return
        if hasattr(bar, "set_header"):
            bar.set_header(**header)


def run_tui(cfg: Config | None = None, directory: Path | None = None) -> None:
    import os
    import signal

    # The engine chain (agent.loop, tools, commands) and the provider internals
    # (zen model list, OpenAI-compat SSE layer) are ~0.6s of lazy imports that
    # aren't needed until the first Enter. Warm them on a background thread NOW
    # so they overlap app.run()'s one-time compose/layout/first-paint work and
    # the first prompt responds immediately instead of waiting for the chain.
    # (This runs from run_tui, not at module import, so it can't race the app's
    # own imports on the main thread.)
    threading.Thread(target=_prewarm_heavy_deps, daemon=True).start()

    app = OpenCodeTUI(cfg=cfg, directory=directory)

    def _close_save_all(signum: int, frame: Any) -> None:
        """Termux-close / kill save (SIGTERM, SIGHUP).

        ``on_exit_app``/``on_unmount`` only run on a graceful exit. When the
        user closes Termux the process gets a signal instead, so save every
        live conversation synchronously here (best effort), then re-raise the
        default signal so shutdown stays immediate.
        """
        try:
            app._save_all_live_sessions()
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, _close_save_all)
    except (AttributeError, ValueError):  # pragma: no cover - non-unix
        pass
    try:
        signal.signal(signal.SIGHUP, _close_save_all)
    except (AttributeError, ValueError):  # pragma: no cover - non-unix
        pass
    try:
        app.run()
    finally:
        # Release MCP server processes (and any other engine resources) so a
        # server started for this session isn't left dangling after exit.
        engine = getattr(app, "_main_engine", None)
        if engine is not None:
            close = getattr(engine, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # pragma: no cover - best effort on exit
                    pass


if __name__ == "__main__":
    run_tui()
