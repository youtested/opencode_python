"""Main agentic loop.

Flow per turn:
  1. build messages (system + trimmed history + latest user turn + agent reminder)
  2. stream model -> emit text/reasoning/tool_call events to on_event
  3. if tool calls: permission check -> run tool -> append tool result; loop again
  4. cap iterations (safety) and honor interrupt

build agent: full tools. plan agent: edit/write/bash denied by permissions.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..globals import resolve_worktree
from ..permission import PermissionEngine, merge_permissions
from ..providers import ContextOverflowError, ProviderError, RateLimitError, StreamInterrupted, build_rotation
from ..session import new_session, save_session
from ..tools import glob as glob_mod
from ..tools.registry import Registry
from . import compaction as compact_mod
from . import messages as msg_mod
from . import parse as parse_mod
from . import system as system_mod

MAX_STEPS = 50
MAX_UNDO = 20


def _missing_directories(path: Path) -> list[Path]:
    """Return the chain of directories (deepest-first) that don't exist yet,
    walking from `path` upward. Used to undo mkdir(parents=True) side effects."""
    missing: list[Path] = []
    p = path
    while str(p) not in ("", ".") and not p.exists():
        missing.append(p)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return missing


@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    tool_calls_made: int = 0
    usage: dict[str, int] | None = None
    provider_id: str = ""
    model_id: str = ""
    error: str = ""


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


class AgentLoop:
    def __init__(
        self,
        *,
        cfg: Config,
        registry: Registry,
        directory: Path,
        provider=None,
        auth=None,
        permission_engine: PermissionEngine | None = None,
        question_service: Any = None,
        on_event: Callable[[dict], None] | None = None,
        agent: str = "build",
        provider_id: str = "",
        model_id: str = "",
        interrupt: Callable[[], bool] | None = None,
        session_id: str | None = None,
        provider_factory: Callable[[], Any] | None = None,
        ask_lock: threading.Lock | None = None,
    ):
        self.cfg = cfg
        self.registry = registry
        self.directory = directory
        self.worktree = resolve_worktree(directory)
        self.auth = auth
        self.agent = agent
        self._prev_agent: str = agent
        self._turn_agent: str = agent
        # Fall back to the configured provider/model so the active lane is the
        # source of truth (a hardcoded "opencode" default would resolve the
        # wrong context window / output limit for any other provider).
        self.provider_id = provider_id or cfg.provider
        self.model_id = model_id or cfg.model
        # `interrupt` is a property: assigning it also re-points the registry's
        # `interrupt_check` hook (which bash/webfetch read at call time). The
        # TUI wires the live callback AFTER construction (`_wire_engine`), so a
        # plain attribute would leave the tools frozen on the init default and
        # ESC/Ctrl+C could never abort a running command.
        self.interrupt = interrupt or (lambda: False)
        self._session_id = session_id or uuid.uuid4().hex
        # factory used to build rotations for spawned sub-agents (override in
        # tests); default matches the parent's own rotation construction.
        self.provider_factory = provider_factory or (lambda: build_rotation(cfg, auth, self._session_id))

        self.rotation = provider or build_rotation(cfg, auth, self._session_id)
        # When True the selected model is pinned: rate limits / hard failures
        # surface instead of failing over to another lane (the TUI lock dot).
        self.rotation_locked = bool(getattr(cfg, "rotation_lock", False))

        self.permission = permission_engine or PermissionEngine.from_config(
            merge_permissions(cfg.permission, agent),
            mode="auto",
        )
        self.on_event = on_event
        self._history: list[dict] = []
        self._call_seq = 0
        self._pending_calls: list[dict] = []
        # FIFO of prompts submitted while a turn was running. Mirroring
        # opencode, these are consumed INSIDE the running turn (a single
        # "Session Drain") at the next provider-turn boundary — the turn keeps
        # working straight into the next prompt instead of ending and starting
        # a fresh turn.
        self._prompt_queue: list[str] = []
        self._prompt_lock = threading.Lock()
        # Serializes permission/question prompts when tool calls run in
        # parallel: only one UI modal may be shown at a time, the rest of the
        # workers wait their turn instead of stacking dialogs.
        self._ask_lock = ask_lock or threading.Lock()
        # Per-thread id of the tool call currently spawning a sub-agent. With
        # parallel execution each worker thread sets its own call id, so a
        # subagent_start event can be linked back to the exact tool row that
        # launched it (official opencode keys sub-agents by their task call).
        self._spawn_ctx = threading.local()
        self._undo_stack: list[dict] = []
        self._usage_total: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._compaction_summary: str = ""
        self.subagents: dict[str, "AgentLoop"] = {}
        # the task tool looks this up lazily so sub-agents can nest
        self.registry.task_spawner = self.spawn_task
        # Registry.reads `interrupt_check` at call time so ESC/Ctrl+C aborts a
        # running command/fetch. The `interrupt` property keeps it in sync
        # whenever the TUI wires the live callback after construction.
        self.registry.interrupt_check = self.interrupt
        # the question tool asks the user through this service (TUI attaches an
        # ask_callback that surfaces a modal and blocks until answered). Built
        # after permissions so the per-agent "question" rule is already loaded.
        from ..question import QuestionService

        self.question_service = question_service or QuestionService()
        self.registry.question_asker = self.question_service.ask

    @property
    def interrupt(self) -> Callable[[], bool]:
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value: Callable[[], bool]) -> None:
        self._interrupt = value
        registry = getattr(self, "registry", None)
        if registry is not None:
            try:
                registry.interrupt_check = value
            except Exception:
                pass

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Bind a session id, keeping the rotation's Zen identity in sync.

        The TUI assigns the persisted session id after construction; the
        rotation's `x-opencode-session` must follow so every provider built
        for the current conversation reports the same id.
        """
        self._session_id = value
        rotation = getattr(self, "rotation", None)
        if rotation is not None and getattr(rotation, "session_id", None) is not None:
            rotation.session_id = value

    def find_subagent(self, session_id: str) -> "AgentLoop | None":
        """Depth-first search for a sub-agent by session id (nested included)."""
        stack = [self]
        while stack:
            loop = stack.pop()
            if loop.session_id == session_id:
                return loop
            stack.extend(loop.subagents.values())
        return None

    # -- event plumbing --------------------------------------------------
    def _emit(self, kind: str, **kwargs: Any) -> None:
        event = {"kind": kind, "session_id": self._session_id, **kwargs}
        if self.on_event:
            self.on_event(event)

    # -- prompt queue (opencode's queue-and-promote, in-turn drain) -------
    def queue_prompt(self, text: str) -> int:
        """Add a prompt submitted while a turn is running.

        It sits in the FIFO until the running turn reaches its next
        provider-turn boundary, then the engine folds it into the SAME turn
        (opencode's Session Drain). Returns the queue depth after adding.
        """
        with self._prompt_lock:
            self._prompt_queue.append(text)
            return len(self._prompt_queue)

    def prompt_pending(self) -> int:
        with self._prompt_lock:
            return len(self._prompt_queue)

    def prompt_peek(self) -> str:
        with self._prompt_lock:
            return self._prompt_queue[0] if self._prompt_queue else ""

    def pop_prompt(self) -> str | None:
        """Pop the oldest queued prompt (used by the TUI for the leftover after a
        drain ends; run_turn itself uses ``_next_prompt`` internally)."""
        return self._next_prompt()

    def _next_prompt(self) -> str | None:
        """Pop the oldest queued prompt (engine thread, run_turn)."""
        with self._prompt_lock:
            if not self._prompt_queue:
                return None
            return self._prompt_queue.pop(0)

    def clear_prompts(self) -> None:
        with self._prompt_lock:
            self._prompt_queue.clear()

    def rebuild_rotation(self) -> None:
        """Rebuild the failover lanes from the current config.

        The rotation is built once at startup, so picking a different model or
        provider at runtime would otherwise keep using the old lanes. Call this
        whenever `cfg.model` / `cfg.provider` change so the next turn uses the
        newly picked model.
        """
        self.rotation = build_rotation(self.cfg, self.auth, self._session_id)

    def _emit_tool(self, kind: str, tool: str, **kwargs: Any) -> None:
        self._emit(kind, tool=tool, **kwargs)

    # -- permission ------------------------------------------------------
    def check_permission(self, tool: str, input_value: str, display: str, call_id: str = "", arguments: dict | None = None, permission: str | None = None, action: str | None = None) -> bool:
        permission_name = permission or tool
        if tool in ("write", "edit", "apply_patch") and permission is None:
            permission_name = "edit"
        if action is None:
            action = self.permission.evaluate(permission_name, input_value)
        if action == "allow":
            return True
        kwargs: dict[str, Any] = {"reason": "denied by permission", "call_id": call_id}
        if action == "deny":
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
            return False
        # ask
        # "Always" scope: the specific permission+input being approved, NOT a
        # universal "*" — one approval granted to `bash npm install` must not
        # silently auto-approve every future ask for any tool/command.
        always_patterns: list[str] = [f"{permission_name} {input_value}".strip()]
        # Cross-thread safe: when a step runs several tools in parallel this
        # lock guarantees only one permission/question modal is visible at a
        # time (the TUI shows exactly one dialog at once).
        with self._ask_lock:
            allowed = self.permission.ask(display, always_patterns)
        if not allowed:
            kwargs = {"reason": "rejected by user", "call_id": call_id}
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
        return allowed

    # -- tools ------------------------------------------------------------
    def run_tool(self, name: str, arguments: dict, call_id: str = "") -> dict[str, Any]:
        tool = self.registry.get(name)
        if tool is None:
            return {"output": f"Unknown tool: {name}", "error": True}
        # Relative paths are resolved against the session directory (git
        # worktree root), NOT Path.cwd(): launching opencode from a
        # subdirectory would otherwise make read/write/edit/bash act on the
        # wrong files while the undo snapshots use self.directory. Normalize
        # here so every component agrees. `resolved_path` keeps the canonical
        # target used both for permission matching and worktree confinement.
        resolved_path = None
        for key in ("filePath", "path", "workdir"):
            if key in arguments and isinstance(arguments[key], str) and arguments[key]:
                p = Path(arguments[key])
                if not p.is_absolute():
                    p = self.directory / p
                p = p.resolve()
                arguments[key] = str(p)
                resolved_path = str(p)
                break

        # Permission match value: path-based tools match their rules (e.g.
        # "*.env" -> ask) against the RESOLVED PATH, not the JSON blob of all
        # arguments (which ends in `}` and never matches a path pattern). task
        # matches on the sub-agent type, mirroring opencode. bash matches on the
        # COMMAND STRING — rules like {"bash": {"npm install *": "deny"}} must
        # actually fire, and the JSON dump of arguments would never match any
        # command pattern (it's "{...}", not "npm install ...").
        input_value = json.dumps(arguments, sort_keys=True)
        if name == "task":
            # opencode's task input id is the sub-agent type, so a plan
            # agent's `task: {grant: deny}` rule actually matches.
            input_value = str(arguments.get("subagent_type", "build"))
        elif name == "bash":
            input_value = str(arguments.get("command", ""))
        elif resolved_path is not None and name in ("read", "write", "edit", "apply_patch", "grep", "glob"):
            input_value = resolved_path

        permission = name
        if name in ("write", "edit", "apply_patch"):
            permission = "edit"
        # Worktree confinement: a path-based tool targeting a file OUTSIDE the
        # canonical worktree root goes through the `external_directory`
        # permission (default: ask). The rule existed in the config defaults
        # but was never evaluated, so every absolute path was silently allowed.
        # NOTE: external_directory is an ADDITIONAL gate, NOT a replacement for
        # the tool's own permission — otherwise a plan agent's edit/write "deny"
        # could be bypassed by pointing the tool at an absolute path outside the
        # worktree. Combine both with the strictest action (deny > ask > allow)
        # and gate exactly once, so there is a single permission dialog.
        is_external = (
            resolved_path is not None
            and name in ("read", "write", "edit", "apply_patch", "grep", "glob")
            and not Path(resolved_path).is_relative_to(self.worktree)
        )
        if is_external:
            combined = [
                self.permission.evaluate(permission, input_value),
                self.permission.evaluate("external_directory", input_value),
            ]
            action: str | None = "deny" if "deny" in combined else ("ask" if "ask" in combined else "allow")
        else:
            action = None

        display = f"{name} {input_value[:120]}"

        if not self.check_permission(name, input_value, display, call_id=call_id, arguments=arguments, permission=permission, action=action):
            return {
                "output": f"Permission denied for {name}. Tell the user what to do differently.",
                "error": True,
                "denied": True,
            }

        mutates = name in ("edit", "write", "apply_patch")
        file_path = arguments.get("filePath")
        snapshot: bytes | None = None
        snapshot_path: Path | None = None
        created_dirs: list[Path] = []
        if mutates and file_path:
            p = Path(file_path)
            if not p.is_absolute():
                p = self.directory / p
            snapshot_path = p
            snapshot = p.read_bytes() if p.exists() else None
            # record which parent directories would be newly created by the
            # write's mkdir(parents=True) so undo can clean them back up.
            if snapshot is None:
                created_dirs = _missing_directories(p.parent)

        # The glob tool resolves a missing `path` against this thread's
        # session worktree (each worker thread gets its own), so parallel
        # globs never fall back to the process CWD.
        glob_mod.set_worktree(self.worktree)

        self._emit_tool("tool_start", name, input=arguments, status="running", call_id=call_id)
        try:
            if name == "task":
                # let spawn_task tag its subagent_start event with this call id
                self._spawn_ctx.call_id = call_id
            result = tool.run(arguments)
        except Exception as e:
            result = {"output": f"{name} failed: {e}", "error": True}
        if mutates and snapshot_path is not None:
            self._undo_stack.append(
                {
                    "path": str(snapshot_path),
                    "original": snapshot,
                    "dirs": [str(d) for d in created_dirs],
                }
            )
            if len(self._undo_stack) > MAX_UNDO:
                self._undo_stack.pop(0)
        self._emit_tool(
            "tool_complete",
            name,
            input=arguments,
            status="error" if result.get("error") else "completed",
            output=result.get("output", ""),
            metadata=result.get("metadata", {}),
            call_id=call_id,
        )
        return result

    # -- sub-agents -------------------------------------------------------
    def close(self) -> None:
        """Release OS resources held by this engine (MCP server processes).

        Called when an engine (or a finished sub-agent) is done; the parent
        keeps its own refcount on shared servers, so a child's close only
        drops the reference the sub-agent acquired.
        """
        registry = getattr(self, "registry", None)
        close = getattr(registry, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    def abort(self) -> None:
        """Force-close the active provider stream(s) for this engine and every
        sub-agent it spawned.

        Used right after the shared interrupt flag is flipped so a blocked
        stream read wakes up immediately — even during an idle "thinking" gap
        where no chunk arrives to trigger the per-chunk interrupt check."""
        rotation = getattr(self, "rotation", None)
        if getattr(rotation, "abort", None) is not None:
            try:
                rotation.abort()
            except Exception:
                pass
        # Force-close any webfetch responses this engine has in flight, so a
        # blocked HTTP read surfaces the interrupt right away instead of letting
        # the fetch run to its timeout after the user pressed ESC twice.
        registry = getattr(self, "registry", None)
        abort_fetches = getattr(registry, "abort_fetches", None)
        if callable(abort_fetches):
            try:
                abort_fetches()
            except Exception:
                pass
        subagents = getattr(self, "subagents", {}) or {}
        for sub in tuple(subagents.values()):
            try:
                sub.abort()
            except Exception:
                pass

    def _subagent_bridge(self, sub_id: str) -> Callable[[dict], None]:
        """Forward a sub-agent's events to our own on_event, tagged with the
        sub-session id so the UI can route them to the right chat view.

        Nested sub-agents already carry their own `session_id` (tagged by the
        deeper bridge); keep that id so a grandchild's events reach its own chat
        instead of being re-tagged with the direct child's id.
        """

        def forward(event: dict[str, Any]) -> None:
            kind = event.get("kind", "")
            sid = event.get("session_id") or sub_id
            payload = {k: v for k, v in event.items() if k not in ("kind", "session_id")}
            self._emit(kind, session_id=sid, **payload)

        return forward

    def spawn_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a sub-agent in its own session; returns the sub-agent's reply.

        The sub-agent gets its own history/session but shares the parent's
        config, auth, permission engine, directory, and interrupt flag. Events
        stream out tagged with the sub-session id. Nested `task` calls work:
        sub-agents build their own registry, which lazily resolves the same hook.
        """
        from ..tools import build_registry

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return {"output": "task: no prompt provided.", "error": True}
        description = str(arguments.get("description", "")).strip() or "sub-agent"
        subagent_type = str(arguments.get("subagent_type", "")).strip() or "build"

        # A read-only plan agent must not spawn a build sub-agent: the child
        # would inherit read-write tools. Force the child to stay read-only.
        if self.agent == "plan" and subagent_type != "plan":
            subagent_type = "plan"

        sub_session = new_session(
            directory=str(self.directory),
            provider=self.provider_id or self.cfg.provider,
            model=self.model_id or self.cfg.model,
            agent=subagent_type,
            title=description,
            parent_id=self.session_id,
        )
        # No save here: the sub-agent session has no conversation yet and must
        # not mint an empty file. It is persisted only once it has content (the
        # completion saves below), or by the app's exit/close save-all.

        sub = AgentLoop(
            cfg=self.cfg,
            registry=build_registry(self.cfg),
            directory=self.directory,
            provider=self.provider_factory(),
            auth=self.auth,
            permission_engine=self.permission,
            question_service=self.question_service,
            on_event=self._subagent_bridge(sub_session.id),
            agent=subagent_type,
            provider_id=self.provider_id,
            model_id=self.model_id,
            interrupt=self.interrupt,
            session_id=sub_session.id,
            provider_factory=self.provider_factory,
            ask_lock=self._ask_lock,
        )
        self.subagents[sub_session.id] = sub
        self._emit(
            "subagent_start",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            prompt=prompt,
            call_id=getattr(self._spawn_ctx, "call_id", None) or "",
        )

        try:
            result = sub.run_turn(prompt)
        except Exception as e:
            # never leave the sub-agent session dangling: persist and report done
            sub_session.messages = sub.get_history()
            sub_session.completed = time.time()
            try:
                if sub_session.messages:
                    save_session(sub_session)
            except Exception:
                pass
            sub.close()
            self._emit(
                "subagent_done",
                session_id=sub_session.id,
                agent=subagent_type,
                title=description,
                ok=False,
            )
            return {
                "output": f"sub-agent failed: {e}",
                "error": True,
                "metadata": {
                    "sessionId": sub_session.id,
                    "title": description,
                    "status": "error",
                },
            }
        sub_session.messages = sub.get_history()
        sub_session.completed = time.time()
        try:
            if sub_session.messages:
                save_session(sub_session)
        except Exception:
            pass
        sub.close()

        self._emit(
            "subagent_done",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            ok=not result.error,
        )
        text = result.text or result.error or "(no reply from sub-agent)"
        return {
            "output": text,
            "error": bool(result.error and not result.text),
            "metadata": {
                "sessionId": sub_session.id,
                "title": description,
                "status": "error" if result.error else "completed",
            },
        }

    _PARALLEL_TOOL_WORKERS = 6
    _MUTATING_TOOLS = ("edit", "write", "apply_patch")

    def _run_tools_parallel(self, prepared):
        """Run every tool call of a step concurrently and return their results
        in the original call order.

        `prepared` is a list of ``(call, name, arguments)`` triples. Reads and
        other independent calls run concurrently on worker threads — `task` calls
        therefore spawn and run their sub-agents at the same time (official
        opencode's parallel agent fan-out). MUTATING tools (edit/write/apply_patch)
        are kept strictly sequential, in call order, so two writes to the same
        file can't interleave their read-modify-write and their undo snapshots
        can't race. Permission prompts stay serialized via ``_ask_lock`` so the
        TUI never shows stacked dialogs.
        """
        from concurrent.futures import ThreadPoolExecutor

        def execute(index: int, call: dict, name: str, arguments: dict):
            if self.interrupt():
                return index, call, name, {"output": "(interrupted)", "error": True, "stopped": True}
            try:
                tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
            except Exception as e:  # run_tool already guards, but be safe
                tool_result = {"output": f"{name} failed: {e}", "error": True}
            return index, call, name, tool_result

        ordered: list[tuple[dict, str, dict] | None] = [None] * len(prepared)
        with ThreadPoolExecutor(max_workers=min(len(prepared), self._PARALLEL_TOOL_WORKERS)) as pool:
            futures = {}
            for i, (call, name, arguments) in enumerate(prepared):
                if name not in self._MUTATING_TOOLS:
                    futures[pool.submit(execute, i, call, name, arguments)] = i
            # Mutating calls execute on this thread one at a time, in original
            # order (strictly serialized; never concurrent with one another).
            for i, (call, name, arguments) in enumerate(prepared):
                if name in self._MUTATING_TOOLS:
                    _, _, _, tool_result = execute(i, call, name, arguments)
                    ordered[i] = (call, name, tool_result)
            for fut in futures:
                index, call, name, tool_result = fut.result()
                ordered[index] = (call, name, tool_result)
        return [entry for entry in ordered if entry is not None]

    # -- main turn --------------------------------------------------------
    def run_turn(self, user_text: str) -> TurnResult:
        result = TurnResult()
        reset = getattr(self.permission, "reset_doom_tracking", None)
        if reset:
            reset()

        self._history.append({"role": "user", "content": user_text})

        # agent reminder (plan/build-switch)
        reminder = system_mod.agent_reminder(self.agent, self._was_plan())
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
        )

        history = list(self._history)
        messages = msg_mod.build_messages(history=history[:-1], user_text=user_text, reminder=reminder)
        messages = self._prepend_system(messages, system_prompt)

        tools = self._active_tool_schemas()

        # The selected model's real window drives both compaction and the
        # trimming safety net, so this works for ANY model — not just the
        # bundled free ones. An unknown window (0) falls back to the configured
        # hard budget and lets the post-overflow recovery path compact.
        ctx = self._model_context_size()
        output_limit = self._model_output_limit()
        usable = compact_mod.usable_context(ctx, output_limit) if ctx > 0 else 0

        # Proactive compaction (mirrors upstream `compactIfNeeded`): estimate the
        # request about to be sent — system + messages + tools — and compact it
        # before it ever reaches the provider's length limit. This runs on the
        # UNTRIMMED history so compaction (which preserves the conversation via
        # the anchored summary) gets first pick; the trim below is only a
        # last-resort safety net that may drop old turns. The estimate alone can
        # undercount tool-heavy conversations, so ALSO honor the actual usage
        # reported by the provider for the last completion (persists across
        # turns on the loop) — mirrors opencode's `lastFinished.tokens` check.
        if self.cfg.compaction_enabled and usable > 0:
            overflow = compact_mod.is_overflow(ctx, compact_mod.estimate_request(messages, tools), output_limit)
            if not overflow and self._usage_total:
                actual = self._usage_total.get("total_tokens") or (
                    self._usage_total.get("input_tokens", 0) + self._usage_total.get("output_tokens", 0)
                )
                overflow = actual >= usable
            if overflow:
                compacted = self._compact_context(system_prompt)
                if compacted is not None:
                    messages = compacted

        # Last-resort cap: never send past the model's usable window (or its raw
        # context when the reserve can't be sized, or the configured budget when
        # the window is unknown). After compaction this is normally a no-op.
        trim_budget = usable if usable > 0 else (ctx if ctx > 0 else self.cfg.context_budget)
        messages = msg_mod.trim_history(messages, trim_budget)

        for step in range(MAX_STEPS):
            if self.interrupt():
                self._emit("interrupted")
                break

            # opencode's queue-and-promote: if a prompt was queued while a
            # provider turn ran, fold it into this SAME drain at the next
            # provider-turn boundary — the generation "pauses for a beat", the
            # chat enters context as a continuation, then the model reasons
            # about it (keep working / stop, whatever it asks). Mirror the
            # official runLoop, which only exits when the last assistant's
            # parent IS the last user message, so a promoted prompt keeps the
            # drain going. Only after the first turn (step>0) — the initial
            # prompt is already in `messages`, and folding before the first
            # stream would merge two user messages.
            if step > 0:
                queued = self._next_prompt()
                if queued is not None:
                    self._history.append({"role": "user", "content": queued})
                    self._emit("prompt_promoted", text=queued)
                    messages = list(self._history)

            self._emit("step", step=step)
            # stream
            self._stream(messages, tools, result, system_prompt)

            if result.error:
                break

            # collect tool calls
            if self._pending_calls:
                calls = self._pending_calls
                self._pending_calls = []
                # Drop degenerate calls (missing name); a model that only emits
                # empty tool calls would otherwise spin a silent 50-step loop.
                calls = [c for c in calls if c.get("name")]
                if not calls:
                    result.error = "model produced an invalid tool call (missing name)"
                    self._emit("error", error=result.error)
                    break
            else:
                # The model replied with text but made no tool call — a
                # complete answer for the current prompt. If a prompt was
                # queued meanwhile, don't exit: the loop-top fold on the next
                # iteration promotes it into this same drain (the stream "pauses
                # a beat", the chat enters context, then the model reasons).
                if self.prompt_pending() == 0:
                    break
                continue

            # Some models emit tool calls without an id. Assign a stable
            # fallback (must match assistant_message_from_calls) so the
            # assistant declaration and the following tool-result messages use
            # the same id; otherwise strict OpenAI-compatible backends reject
            # with "insufficient tool messages following tool_calls". The
            # counter keeps ids unique across every step and turn so the UI can
            # reliably match tool rows by call_id (per-step indices collide and
            # cause duplicate/incorrect tool rows).
            for call in calls:
                if not call.get("id"):
                    self._call_seq += 1
                    call["id"] = f"call_{self._call_seq}"

            # append assistant message with calls to history — keep the model's
            # own text and reasoning in the same message so the next tool-loop
            # request is a faithful replay (reasoning models lose thread when
            # their previous message is stored empty).
            self._history.append(
                parse_mod.assistant_message_from_calls(
                    calls,
                    reasoning=result.reasoning,
                    content=result.text,
                )
            )
            messages = list(self._history)

            # Execute the step's tool calls. A step with several calls runs them
            # concurrently (official opencode launches sub-agents in parallel
            # when the model emits multiple `task` calls in one reply). Each
            # call runs on its own worker; results come back on this thread and
            # history is written in the ORIGINAL call order so the
            # assistant_message -> tool_result pairing stays valid for replay.
            # No early-return may happen between committing the assistant
            # message and writing its tool results: strict backends reject a
            # request whose assistant tool_calls are not all answered by tool
            # messages ("insufficient tool messages following tool_calls"), and
            # the poisoned history would then be persisted and re-sent forever.
            # An interrupt here instead yields placeholder "(interrupted)"
            # results so the pair always closes.
            prepared = []
            for call in calls:
                name = call.get("name", "")
                try:
                    arguments = parse_mod.parse_arguments(call.get("arguments", "{}"))
                except Exception:
                    arguments = {"arguments": call.get("arguments", "{}")}
                prepared.append((call, name, arguments))

            for call, name, arguments in prepared:
                self._emit_tool("tool_call", name, arguments=arguments, call_id=call.get("id", ""))

            if len(prepared) > 1:
                outcomes = self._run_tools_parallel(prepared)
            else:
                outcomes = []
                for call, name, arguments in prepared:
                    if self.interrupt():
                        outcomes.append(
                            (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                        )
                        break
                    tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
                    outcomes.append((call, name, tool_result))

            for call, name, tool_result in outcomes:
                result.tool_calls_made += 1
                self._history.append(
                    parse_mod.tool_result_message(
                        call.get("id", ""),
                        name,
                        tool_result.get("output", ""),
                        error=bool(tool_result.get("error")),
                    )
                )
            messages = list(self._history)

            if self.interrupt():
                self._emit("interrupted")
                return result

            # Auto-compaction between steps (mirrors upstream opencode): after a
            # step completes, compact as soon as the ACTUAL provider-reported
            # usage fills the usable window — not only when the request is
            # estimated to overflow at the start of a turn. Tool loops grow the
            # context faster than the turn-start estimate predicts, so without
            # this a long session runs to 100% and stalls. Uses the SELECTED
            # model's window, so it works for any provider/model lane.
            if self.cfg.compaction_enabled:
                compacted = self._maybe_compact_by_usage(system_prompt)
                if compacted is not None:
                    messages = compacted

        if result.error:
            self._history.append({"role": "user", "content": f"[system] {result.error}"})
        # Track agent state for build-switch detection
        self._prev_agent = self.agent
        return result

    # -- streaming --------------------------------------------------------
    def _stream(self, messages, tools, result: TurnResult, system_prompt: str) -> None:
        self._pending_calls = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # The assistant reply being streamed is kept LIVE in self._history as it
        # grows, so the app's periodic autosave (get_history) persists the exact
        # conversation up to the last token even if the app is killed suddenly
        # mid-stream — the session resumes where the user left off.
        live_assistant: dict | None = None

        def _ensure_live_assistant() -> dict:
            nonlocal live_assistant
            if live_assistant is None:
                live_assistant = {"role": "assistant", "content": "", "reasoning_content": ""}
                self._history.append(live_assistant)
            return live_assistant

        def _drop_live_assistant() -> None:
            nonlocal live_assistant
            if live_assistant is None:
                return
            idx = next((i for i, m in enumerate(self._history) if m is live_assistant), None)
            if idx is not None:
                self._history.pop(idx)
            live_assistant = None

        # Tool-loop requests are rebuilt from raw history (which never holds the
        # system prompt); re-prepend it so every request is well-formed.
        if not messages or messages[0].get("role") != "system":
            messages = self._prepend_system(messages, system_prompt)
        # An interrupted/force-killed turn can leave the request with an
        # assistant message declaring tool_calls but no following tool results;
        # strict backends reject that payload. Repair the local request copy so
        # the provider always sees a well-formed conversation.
        messages = msg_mod.repair_tool_pairs(messages)

        def on_event(evt) -> None:
            kind = evt.kind
            if kind == "text_delta":
                text_parts.append(evt.text)
                _ensure_live_assistant()["content"] += evt.text
                self._emit("text_delta", text=evt.text)
            elif kind == "reasoning_delta":
                reasoning_parts.append(evt.text)
                _ensure_live_assistant()["reasoning_content"] += evt.text
                self._emit("reasoning_delta", text=evt.text)
            elif kind == "tool_call":
                for tc in evt.tool_calls or []:
                    self._pending_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            elif kind == "usage":
                u = evt.usage
                # Context-window usage = the last completion's input (full
                # conversation context) + its output — mirrors opencode's
                # input+output+reasoning+cache. Summing across streams would
                # double-count history on multi-step tool loops.
                self._usage_total["input_tokens"] = u.input_tokens
                self._usage_total["output_tokens"] = u.output_tokens
                self._usage_total["total_tokens"] = u.input_tokens + u.output_tokens
                result.usage = dict(self._usage_total)
                usage_evt = dict(self._usage_total)
                ctx = self._model_context_size()
                if ctx:
                    result.usage["context_size"] = ctx
                    usage_evt["context_size"] = ctx
                self._emit("usage", usage=usage_evt)
            elif kind == "error":
                result.error = evt.error
                self._emit("error", error=evt.error)
            elif kind == "done":
                pass

        try:
            # Automatic retry for transient failures (streaming timeouts, 5xx,
            # overload, short rate limits) — mirrors upstream opencode's retry
            # policy, which waits out transient errors with exponential backoff
            # instead of surfacing them. Mirrors the official behavior:
            #   BASE_DELAY_MS=500, factor 2, jitter ±20%, cap 10s,
            #   honoring Retry-After when the provider sends it.
            # Events stay buffered inside rotation until a lane completes, so a
            # failed attempt emits nothing visible and any retry is clean.
            # Smart resume: if the model was mid-operation (it already ran
            # tools this turn), a plain re-send would replay every tool call
            # from scratch. Instead nudge the SAME model to continue; if
            # nothing happened yet, re-send the prompt as-is.
            for attempt in range(self.cfg.auto_retry_count + 1):
                try:
                    provider_id, model_id = self.rotation.stream(
                        self._retry_messages(messages, result, attempt),
                        tools,
                        on_event,
                        on_notice=self._on_rotated,
                        is_interrupted=self.interrupt,
                        locked=self.rotation_locked,
                    )
                    break
                except ContextOverflowError:
                    # has its own compaction recovery path below — never retry
                    raise
                except RateLimitError as e:
                    # A real 429/quota wall. Retry ONLY when the provider tells
                    # us exactly how long to wait (Retry-After); otherwise (e.g.
                    # Zen's FreeUsageLimitError, which sends no retry-after) the
                    # quota is gone and retrying 50× would be pure noise — it is
                    # NOT a dropped connection. Surface it right away so the
                    # user sees "rate limit" instead of "↻ connection dropped".
                    last_error = e
                    if not self.cfg.auto_retry or e.retry_after is None or attempt >= self.cfg.auto_retry_count:
                        raise
                    if self.interrupt():
                        raise
                    remaining = self.cfg.auto_retry_count - attempt
                    self._emit(
                        "retry",
                        attempt=attempt + 1,
                        total=self.cfg.auto_retry_count,
                        message=f"↻ rate limited — retrying ({remaining} left)…",
                    )
                    time.sleep(self._retry_delay(e, attempt))
                except ProviderError as e:
                    last_error = e
                    if not self.cfg.auto_retry or not e.retryable or attempt >= self.cfg.auto_retry_count:
                        raise
                    if self.interrupt():
                        raise
                    # surface a transient footer hint so the retry isn't silent
                    remaining = self.cfg.auto_retry_count - attempt
                    self._emit(
                        "retry",
                        attempt=attempt + 1,
                        total=self.cfg.auto_retry_count,
                        message=f"↻ connection dropped — retrying ({remaining} left)…",
                    )
                    time.sleep(self._retry_delay(e, attempt))
            else:
                # loop exhausted without success (the `break` never ran)
                raise last_error
            result.provider_id = provider_id
            result.model_id = model_id
        except ContextOverflowError as e:
            # The history overflowed the model's window even after budget
            # trimming (estimates are cheap). Recover by summarizing the
            # conversation into an anchored summary and keeping the recent tail
            # verbatim (upstream opencode's compaction), then retry once. This
            # fixes the turn instead of surfacing a hard error mid-conversation.
            if self.cfg.compaction_enabled:
                compacted = self._compact_context(system_prompt)
            else:
                compacted = None
            if compacted is not None:
                messages = compacted
                try:
                    provider_id, model_id = self.rotation.stream(
                        messages, tools, on_event, on_notice=self._on_rotated, is_interrupted=self.interrupt,
                        locked=self.rotation_locked,
                    )
                    result.provider_id = provider_id
                    result.model_id = model_id
                    result.error = ""
                except ContextOverflowError as e2:
                    result.error = f"context overflow (even after compaction): {e2}"
                    self._emit("error", error=result.error, retryable=True)
                except ProviderError as e2:
                    result.error = str(e2)
                    self._emit("error", error=result.error, retryable=bool(e2.retryable))
                except StreamInterrupted:
                    raise
                except Exception as e2:
                    result.error = str(e2)
                    self._emit("error", error=result.error)
                if result.error:
                    pass  # surfaced above
            else:
                result.error = f"context overflow: {e}"
                self._emit("error", error=result.error, retryable=True)
        except RateLimitError as e:
            result.error = f"rate limit: {e}"
            self._emit("error", error=result.error, retryable=True)
        except ProviderError as e:
            result.error = str(e)
            self._emit("error", error=result.error, retryable=bool(e.retryable))
        except StreamInterrupted:
            # User aborted mid-stream (Esc pressed twice / Ctrl+C): NOT an
            # error — end the turn as interrupted. Partial text already
            # streamed stays on screen, like upstream opencode's abort.
            self._emit("interrupted")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result.error = str(e)
            self._emit("error", error=result.error)

        result.text = "".join(text_parts)
        result.reasoning = "".join(reasoning_parts)
        if self._pending_calls:
            # run_turn builds ONE assistant message carrying text + reasoning +
            # tool_calls from result.text/result.reasoning — drop the live
            # text-only message first so one assistant turn isn't split (strict
            # backends and reasoning models lose thread on the next step).
            _drop_live_assistant()
        elif result.text:
            # keep the live message (it already holds every streamed delta);
            # sync it to the final buffers in case no delta path weirdly fired
            if live_assistant is None:
                _ensure_live_assistant()
            live_assistant["content"] = result.text
            live_assistant["reasoning_content"] = result.reasoning
        elif result.reasoning:
            # reasoning-only reply: replay it faithfully as what the model
            # actually returned — empty content plus a separate reasoning
            # signal. Storing the thinking also as `content` produces a message
            # that ever after disagrees with the model's real output, which
            # strict thinking-mode backends reject on the next request
            # ("reasoning_content must be passed back").
            if live_assistant is None:
                _ensure_live_assistant()
            live_assistant["content"] = ""
            live_assistant["reasoning_content"] = result.reasoning
        elif result.error:
            # nothing streamed — don't leave an empty assistant message behind
            _drop_live_assistant()
        else:
            # keep role alternation valid and avoid an empty-content message
            self._history.append(
                {"role": "assistant", "content": "(no response)", "reasoning_content": ""}
            )

    def undo_last(self) -> str:
        """Revert the most recent edit/write tool call (file-level snapshot)."""
        if not self._undo_stack:
            return "Nothing to undo."
        entry = self._undo_stack.pop()
        path = Path(entry["path"])
        try:
            if entry["original"] is None:
                if path.exists():
                    path.unlink()
                dirs = entry.get("dirs") or []
                for d in reversed(dirs):
                    d = Path(d)
                    # only remove directories we created and that are now empty
                    try:
                        if d.is_dir() and not any(d.iterdir()):
                            d.rmdir()
                    except OSError:
                        pass
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry["original"])
        except OSError as e:
            return f"Undo failed for {path}: {e}"
        self._emit("undo", path=str(path))
        return f"Reverted {path}."

    def _on_rotated(self, provider_id: str, model_id: str, reason: str = "") -> None:
        """A failover lane succeeded; announce it (with the switch reason)."""
        self._emit("rotated", provider=provider_id, model=model_id, reason=reason)

    def _model_context_size(self) -> int:
        """Context-window size of the active lane (0 when unknown)."""
        from ..providers import model_context_size

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_context_size(pid, mid, auth=self.auth)

    def _model_output_limit(self) -> int:
        """Max output tokens of the active lane (0 when unknown)."""
        from ..providers import model_output_limit

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_output_limit(pid, mid)

    # continuation nudges sent to the SAME model when it stops mid-operation.
    # A plain re-send would replay every tool call from scratch; nudging lets
    # the model continue where it left off. Never rotates to another lane.
    _RETRY_NUDGES = (
        "keep going",
        "don't stop, keep going",
        "continue working, don't stop",
        "keep going, finish the task",
    )

    def _retry_messages(self, messages: list, result: TurnResult, attempt: int) -> list:
        """Rebuild the request for a retry on the same model.

        - If the turn already made progress (tools executed), the model was
          mid-operation: don't replay from scratch — append a short continuation
          nudge so it picks up where it was cut off. Each retry uses the next
          nudge in the list.
        - After every nudge is used, fall back to re-sending the full context
          unchanged: a fresh continuation beats piling on more "keep going"
          noise (which reasoning models read as a mid-thought interrupt and
          answer with nothing).
        - If nothing has happened yet, re-send the original prompt unchanged.
        """
        nudges = self._RETRY_NUDGES
        if result.tool_calls_made > 0 and attempt > 0:
            if attempt <= len(nudges):
                nudge = nudges[(attempt - 1) % len(nudges)]
                return [*messages, {"role": "user", "content": nudge}]
            return list(messages)
        return list(messages)

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Backoff before a retry, mirroring upstream opencode's retry policy.

        Honors ``Retry-After`` milliseconds when the provider sent them
        (capped at the 10s ceiling), otherwise exponential backoff with jitter:
        ``500ms * 2^attempt`` ±20%, capped at 10s.
        """
        retry_after = getattr(error, "retry_after", None)
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), 10.0)
            except (TypeError, ValueError):
                pass
        base = 0.5 * (2**attempt)
        cap = min(base, 10.0)
        seed = hash(repr(error)) & 31
        return round(cap * (0.8 + (seed / 31) * 0.4), 3)

    def _active_tool_schemas(self) -> list[dict]:
        schemas = self.registry.schemas()
        if self.agent == "plan":
            schemas = [
                s
                for s in schemas
                if s["function"]["name"] not in ("bash", "write", "edit", "apply_patch")
            ]
        return schemas

    def _was_plan(self) -> bool:
        return self._prev_agent == "plan"

    def _prepend_system(self, messages: list[dict], system_prompt: str) -> list[dict]:
        return [{"role": "system", "content": system_prompt}] + messages

    # -- compaction --------------------------------------------------------
    def force_compact(self) -> str:
        """Manually compact the conversation now (the `/compact` command).

        Mirrors upstream opencode: `/compact` runs the same AI compaction as the
        automatic overflow path. Returns the anchored summary ("" when nothing
        was compacted). Raises on provider failure.
        """
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
        )
        messages = self._compact_context(system_prompt)
        if messages is None:
            return ""
        return self._compaction_summary or ""

    def _maybe_compact_by_usage(self, system_prompt: str) -> list[dict] | None:
        """Compact after a step when actual usage has filled the usable window.

        Mirrors upstream opencode's `compaction.isOverflow({tokens, model})`,
        which runs after every completed step: once the provider-reported token
        count (input + output of the last completion) reaches the usable window
        (context minus the reserve), the conversation is summarized so the next
        request never hits the provider's hard length limit. The window comes
        from the SELECTED model, so this works for any provider/model lane.
        """
        ctx = self._model_context_size()
        if ctx <= 0:
            return None
        output_limit = self._model_output_limit()
        usable = compact_mod.usable_context(ctx, output_limit)
        if usable <= 0:
            return None
        count = self._usage_total.get("total_tokens") or (
            self._usage_total.get("input_tokens", 0) + self._usage_total.get("output_tokens", 0)
        )
        if count < usable:
            return None
        return self._compact_context(system_prompt)

    def _compact_context(self, system_prompt: str) -> list[dict] | None:
        """Summarize the conversation into an anchored summary and continue.

        Mirrors upstream opencode's compaction: split the history into a head
        (to summarize) and a recent tail (kept verbatim), ask a model to write
        an anchored summary, then rebuild the request from summary + tail so the
        turn continues instead of erroring out. Returns the rebuilt request
        messages, or None if compaction can't run (no history / model failure).
        """
        if not self.cfg.compaction_enabled:
            return None
        history = [m for m in self._history if not m.get("compaction")]
        if len(history) < 4:
            return None
        # Size the preserved tail from the active lane's context window (the
        # model that is actually answering), not a hardcoded 200k assumption.
        ctx = self._model_context_size()
        head, tail = compact_mod.select_tail(
            history,
            tail_turns=self.cfg.compaction_tail_turns,
            context=ctx,
            output_limit=self._model_output_limit(),
        )
        if not head:
            return None
        prompt = compact_mod.summarize_conversation_prompt(head, tail)
        summary_texts: list[str] = []

        # Emit BEFORE the summary stream so the TUI can show the official
        # opencode "Compacting conversation…" indicator while the model works.
        self._emit("compaction_start", reason="auto")

        def on_summary(evt) -> None:
            if evt.kind == "text_delta":
                summary_texts.append(evt.text)
                # Stream the summary live so the TUI shows it being written
                # instead of a static "Compacting…" spinner that dumps the whole
                # block at the end.
                self._emit("summary_delta", text=evt.text)

        try:
            self.rotation.stream(
                [{"role": "user", "content": prompt}],
                [],
                on_summary,
                on_notice=self._on_rotated,
                locked=self.rotation_locked,
            )
        except Exception as e:
            # `compaction_start` was already emitted; a failure must still emit a
            # terminal event or the TUI's "Compacting…" spinner never clears
            # (the InputBar only resets on `compacted`). Report it there so both
            # the state clears and the user sees what went wrong.
            self._emit("compacted", summary=f"Compaction failed: {e}")
            return None
        summary = "".join(summary_texts).strip()
        if not summary:
            self._emit("compacted", summary="Compaction produced no summary")
            return None
        self._compaction_summary = summary
        # Upstream opencode emits the compaction checkpoint as a USER message
        # (`<conversation-checkpoint>` in to-llm-message.ts), NOT an assistant
        # message. Fabricating an assistant message here would be replayed to the
        # API in thinking mode without a `reasoning_content`, which strict
        # gateways reject with "reasoning_content in thinking mode must be passed
        # back". The tail already starts with a user turn, so fold the summary
        # into that first message to keep role alternation intact (a bare
        # `user,user` sequence is rejected by Anthropic).
        summary_text = f"[Summary of earlier conversation]\n{summary}"
        if tail and tail[0].get("role") == "user":
            tail0 = dict(tail[0])
            content = tail0.get("content")
            if isinstance(content, str):
                tail0["content"] = f"{summary_text}\n\n{content}"
            elif isinstance(content, list):
                # Multimodal user turn: keep the original parts verbatim but
                # PREPEND the summary as a text part — the old code replaced
                # the list unchanged, quietly losing the entire anchored
                # summary (the conversation history evaporated on the next
                # compaction/overflow).
                tail0["content"] = [{"type": "text", "text": summary_text}] + list(content)
            tail0["compaction"] = True
            summary_msg = tail0
            new_history = [summary_msg] + tail[1:]
        else:
            summary_msg = {
                "role": "user",
                "content": summary_text,
                "compaction": True,
            }
            new_history = [summary_msg] + tail
        self._history = new_history
        self._emit("compacted", summary=summary)
        # Recompute the context estimate so the TUI's `12,345 (6%)` reflects
        # the compacted conversation, not the pre-summary size that triggered
        # the overflow (mirrors opencode recomputing tokens after compaction).
        total = sum(compact_mod.estimate_tokens(compact_mod.msg_text(m)) for m in [{"role": "system", "content": system_prompt}] + list(self._history))
        self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
        usage_evt = dict(self._usage_total)
        if ctx:
            usage_evt["context_size"] = ctx
        self._emit("usage", usage=usage_evt)
        messages = self._prepend_system(list(self._history), system_prompt)
        return messages

    # -- session glue -----------------------------------------------------
    def set_history(self, history: list[dict]) -> None:
        self._history = list(history)
        if not history:
            self._usage_total = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    def get_history(self) -> list[dict]:
        return list(self._history)

    def add_placeholder_tool_message(self, output: str) -> None:
        self._history.append({"role": "assistant", "content": output})
