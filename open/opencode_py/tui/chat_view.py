"""Chat view: scrollable message list mirroring opencode's session screen.

Rendering mirrors opencode's TUI (packages/tui/src/routes/session/index.tsx):

  - User messages are a full-width block with a single left border strip in the
    agent accent color, a `backgroundPanel` fill and padding (no title), plus an
    optional ` QUEUED ` badge when the turn hasn't started yet.
  - Assistant text flows as plain markdown indented from the left with a block
    cursor (▍) while streaming, then a muted `▣ Build · model` mode line.
  - Reasoning streams as a spinner `Thinking...` and collapses to a clickable
    `+ Thought: <title>` line (opencode's hide mode); clicking toggles the body.
  - Tools render as compact inline rows (`{icon} {label}`, spinner while running)
    or, for tools that produce a result block (bash output, edit diff, todos,
    questions, apply-patch), a subtle left-bordered block on the panel background.
    Per-tool rendering mirrors opencode: Read shows `↳ Loaded <file>`, Glob/Grep
    show `(N matches)`, etc.
"""

from __future__ import annotations

import re
import time
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from .theme import get_theme
from .markdown_renderer import render_markdown
from .diff_renderer import render_diff
from .input_bar import SessionNavRequested

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.08  # opencode uses 80ms

# ASCII-art opencode wordmark shown on the very first launch, before any
# conversation exists (mirrors the official opencode banner; see the Logo
# component / logo.ts in packages/tui). Left column spells OPEN (rendered in
# muted gray), right column spells CODE (rendered in white + bold).
_LOGO_OPEN = [
    "                   ",
    "█▀▀█ █▀▀█ █▀▀█ █▀▀█",
    "█__█ █__█ █^^^ █  █",
    "▀▀▀▀ █▀▀▀ ▀▀▀▀ ▀  ▀",
]
_LOGO_CODE = [
    "             ▄     ",
    "█▀▀▀ █▀▀█ █▀▀█ █▀▀█",
    "█___ █__█ █__█ █^^^",
    "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀",
]
_LOGO_MARKS = {"_": " ", "^": "▀", "~": "▀", ",": "▄"}


def _logo_expand(line: str) -> str:
    """Render the official logo's shadow marks ('_', '^', '~', ',') as plain
    block/space characters."""
    return "".join(_LOGO_MARKS.get(ch, ch) for ch in line)


def _tint(hex_color: str, overlay: str, alpha: float) -> str:
    """Port of the official tint(): blend `overlay` into `hex_color` by alpha."""
    def _rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    base, ov = _rgb(hex_color), _rgb(overlay)
    mixed = tuple(round(b + (o - b) * alpha) for b, o in zip(base, ov))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _logo_char(char: str, fg: str, bold: bool, shadow: str) -> tuple[str, str]:
    """Map one raw logo cell to (display char, rich style). Mirrors the official
    Logo component: '_' is a shadow-filled space, '^' an upper-half block on a
    shadow background, '~'/',' dimmed shadow blocks; everything else is plain."""
    b = "bold " if bold else ""
    if char == "_":
        return " ", f"{b}on {shadow}"
    if char == "^":
        return "▀", f"{b}{fg} on {shadow}"
    if char == "~":
        return "▀", f"{b}{shadow}"
    if char == ",":
        return "▄", f"{b}{shadow}"
    return char, f"{b}{fg}"


OPENCODE_LOGO = "\n".join(
    _logo_expand(lo) + " " + _logo_expand(rc) for lo, rc in zip(_LOGO_OPEN, _LOGO_CODE)
)


def opencode_logo_text() -> Text:
    """The launch logo as a two-tone renderable mirroring the official Logo
    component: OPEN in muted gray, CODE in white/bold, with 25% shadow tints
    on the interior fills and dimmed partial blocks."""
    theme = get_theme("opencode")
    gray, white = theme.c("text_muted"), theme.c("text")
    shadow_open = _tint(theme.c("background"), gray, 0.25)
    shadow_code = _tint(theme.c("background"), white, 0.25)
    out = Text()
    for i, (lo, rc) in enumerate(zip(_LOGO_OPEN, _LOGO_CODE)):
        if i:
            out.append("\n")
        for ch in lo:
            txt, style = _logo_char(ch, gray, False, shadow_open)
            out.append(txt, style=style)
        out.append(" ")
        for ch in rc:
            txt, style = _logo_char(ch, white, True, shadow_code)
            out.append(txt, style=style)
    return out


class OpenTaskSession(Message):
    """A completed task tool row was clicked; open that sub-session."""

    def __init__(self, sid: str) -> None:
        super().__init__()
        self.sid = sid

# tool -> icon. Mirrors opencode's InlineTool usage.
TOOL_ICONS = {
    "bash": "$",
    "shell": "$",
    "execute": "$",
    "read": "→",
    "write": "←",
    "glob": "✱",
    "grep": "✱",
    "webfetch": "%",
    "webfetch_many": "%",
    "websearch": "◈",
    "edit": "←",
    "apply_patch": "%",
    "todowrite": "⚙",
    "task": "│",
    "question": "→",
    "skill": "→",
    "mcp": "⊙",
    "notify": "·",
}
# tool -> label used in inline rows / titles.
TOOL_NAMES = {
    "bash": "Shell",
    "shell": "Shell",
    "execute": "Execute",
    "read": "Read",
    "write": "Write",
    "glob": "Glob",
    "grep": "Grep",
    "webfetch": "WebFetch",
    "webfetch_many": "WebFetch Batch",
    "websearch": "WebSearch",
    "edit": "Edit",
    "apply_patch": "Apply Patch",
    "todowrite": "TodoWrite",
    "task": "Task",
}

# tools that have a dedicated renderer in opencode (everything else is "generic").
_TOOL_DISPLAYS = {
    "bash",
    "glob",
    "read",
    "grep",
    "webfetch",
    "websearch",
    "write",
    "edit",
    "task",
    "apply_patch",
    "todowrite",
    "question",
    "skill",
    "execute",
}


def _plain(content: Any, width: int | None = None) -> RenderableType:
    """Plain flowing markdown with no surrounding box (assistant text)."""
    return render_markdown(str(content), width=width)


def _render_diff(
    diff_text: str,
    filepath: str = "",
    width: int | None = None,
    opts: dict[str, Any] | None = None,
) -> RenderableType:
    """Render a unified diff the way opencode's `<diff>` edit block does.

    Includes a line-number gutter, +/- signs and syntax-highlighted content
    (all matched to the official opencode dark theme). ``opts`` mirrors the
    official diff config: ``diff_style`` (``"split"``/``"stacked"``),
    ``diff_wrap_mode`` (``"word"``/``"none"``) and ``suppress_backgrounds``.
    """
    opts = opts or {}
    style = opts.get("diff_style", "split")
    # opencode chooses split only when the terminal is wider than 120 cols and
    # the diff_style config has not forced "stacked".
    view = "auto" if style == "split" else "unified"
    return render_diff(
        diff_text,
        filename=filepath,
        view=view,
        width=width or 0,
        wrap=opts.get("diff_wrap_mode", "word"),
        suppress_backgrounds=opts.get("suppress_backgrounds", False),
    )


def collapse_tool_output(output: str, max_lines: int, max_chars: int) -> dict:
    """Mirror opencode's collapse-tool-output: cap lines and chars with '…'."""
    lines = output.split("\n")
    if len(lines) <= max_lines and len(output) <= max_chars:
        return {"output": output, "overflow": False}
    preview = "\n".join(lines[:max_lines])
    if len(preview) > max_chars:
        return {"output": preview[: max(0, max_chars - 1)] + "…", "overflow": True}
    return {"output": "\n".join(lines[:max_lines] + ["…"]), "overflow": True}


def _format_input(inp: dict[str, Any], omit: tuple[str, ...] = ()) -> str:
    """Mirror opencode's `input()` helper: `[key=value, key2=value2]`."""
    parts = []
    for key, value in inp.items():
        if key in omit:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    if not parts:
        return ""
    return "[" + ", ".join(parts) + "]"


def format_subagent_title(agent: str, description: str, background: bool = False) -> str:
    """Mirror opencode's ``formatSubagentTitle``:
    ``Build Task — fix the login bug`` (+ ``(background)`` for background agents)."""
    label = str(agent).title() if str(agent).strip() else "General"
    return f"{label} Task{' (background)' if background else ''} — {description}"


def format_subagent_toolcalls(count: int) -> str:
    """Mirror opencode's ``formatSubagentToolcalls``: `1 toolcall` / `3 toolcalls`."""
    return f"{count} toolcall{'s' if count != 1 else ''}"


def format_subagent_retry(attempt: int, message: str) -> str:
    """Mirror opencode's ``formatSubagentRetry``:
    ``Retrying (attempt 2) · <message>``."""
    return f"Retrying (attempt {attempt}) · {message}"


def format_completed_subagent_detail(toolcalls: int, duration: str) -> str:
    """Mirror opencode's ``formatCompletedSubagentDetail`:
    ``3 toolcalls · 12.5s`` (just the duration when no tools ran)."""
    if toolcalls == 0:
        return duration
    if not duration:
        return format_subagent_toolcalls(toolcalls)
    return f"{format_subagent_toolcalls(toolcalls)} · {duration}"


# tool -> display name for the live `↳ Bash npm run test` line under a running
# task row (mirrors opencode's `titlecase(tool)` in the Subagent component).
_TOOL_TITLES = {
    "bash": "Bash",
    "shell": "Bash",
    "execute": "Execute",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "apply_patch": "Patch",
    "glob": "Glob",
    "grep": "Grep",
    "webfetch": "WebFetch",
    "webfetch_many": "WebFetch",
    "websearch": "WebSearch",
    "todowrite": "TodoWrite",
    "task": "Task",
    "question": "Ask",
    "skill": "Skill",
    "mcp": "MCP",
    "notify": "Notify",
}


def tool_title(run: dict[str, Any]) -> str:
    """The title bit of a tool run (opencode's ``state.title``): Bash -> the
    command, Read/Edit/Write -> the file path, Glob/Grep -> the pattern, etc."""
    tool = str(run.get("tool", ""))
    inp = run.get("input") or {}
    if tool in ("bash", "shell", "execute"):
        return str(inp.get("command", "")).strip()
    if tool in ("read", "write", "edit"):
        return str(inp.get("filePath", "")).strip()
    if tool == "apply_patch":
        return str(inp.get("command", "") or inp.get("filePath", "")).strip()
    if tool in ("glob", "grep"):
        pattern = str(inp.get("pattern", "")).strip()
        path = str(inp.get("path", "")).strip()
        title = pattern
        if pattern and path:
            title += f" in {path}"
        return title
    if tool in ("webfetch", "webfetch_many"):
        return str(inp.get("url", "")).strip()
    if tool == "websearch":
        return str(inp.get("query", "")).strip()
    return ""


def tool_label(run: dict[str, Any]) -> str:
    """Full `↳ Bash npm run test` style label for a tool run."""
    tool = str(run.get("tool", "?"))
    title = tool_title(run)
    base = _TOOL_TITLES.get(tool, tool)
    return f"{base} {title}".strip() if title else base


def reasoning_summary(text: str) -> dict:
    """Mirror opencode's reasoningSummary: extract a bold `**Title**` block."""
    content = str(text).strip()
    match = re.match(r"^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)", content)
    if not match:
        return {"title": None, "body": content}
    return {"title": match.group(1).strip(), "body": content[match.end():].strip()}


def _tool_display(tool: str) -> str:
    return tool if tool in _TOOL_DISPLAYS else "generic"


class MessageBubble(Static):
    """One chat element (user, assistant, reasoning, mode-line, meta) or a tool run."""

    queued: reactive = reactive(False)  # user message waiting for the turn to start
    streaming: reactive = reactive(False)
    expanded: reactive = reactive(False)  # reasoning body collapsed/expanded
    selected: reactive = reactive(False)  # this task row is the active sub-agent

    def __init__(
        self,
        role: str,
        content: Any = "",
        agent: str = "build",
        queued: bool = False,
        streaming: bool = False,
        directive: str = "",
        **kwargs: Any,
    ) -> None:
        self.role = role
        self.agent = agent
        self._message = content
        # Streaming deltas accumulate here and are joined once per flush
        # (append-only: `"".join(parts)` is O(n) per flush instead of an
        # O(n²) `content + text` rebuild of the growing string).
        self._stream_parts: list[str] = []
        # Optional header for the sub-agent "directive" block: the text of the
        # parent's instruction shown at the very top of a sub-agent's chat.
        self.directive = directive
        self._spinner = 0
        self._timer: Any = None
        self._thought_started: float | None = None
        self._thought_seconds: float | None = None
        super().__init__("", **kwargs)
        self.can_focus = role == "reasoning"
        self.set_reactive(MessageBubble.queued, queued)
        self.set_reactive(MessageBubble.streaming, streaming)
        self._refresh()

    def _diff_opts(self) -> dict[str, Any]:
        try:
            cfg = self.app.cfg
        except Exception:
            cfg = None
        if cfg is not None:
            try:
                return {
                    "diff_style": getattr(cfg, "diff_style", "split"),
                    "diff_wrap_mode": getattr(cfg, "diff_wrap_mode", "word"),
                    "suppress_backgrounds": getattr(cfg, "suppress_backgrounds", False),
                }
            except Exception:
                pass
        return {}

    def watch_queued(self, value: bool) -> None:
        self._refresh()

    def watch_streaming(self, value: bool) -> None:
        self._refresh()

    def watch_expanded(self, value: bool) -> None:
        self._refresh()

    def watch_selected(self, value: bool) -> None:
        self._refresh()

    @property
    def content(self) -> Any:
        """Raw payload (assistant text or tool-run dict)."""
        return self._message

    # -- content ----------------------------------------------------------
    def _build_content(self) -> RenderableType:
        theme = get_theme("opencode")
        if self.role == "user":
            inner: list[RenderableType] = []
            if self.directive:
                header = Text()
                header.append("▣ ", style=theme.agent_color("build"))
                header.append(self.directive, style=f"bold {theme.c('text')}")
                header.append(" · ", style=theme.c("text_muted"))
                header.append("Directive", style=theme.c("text_muted"))
                inner.append(header)
            inner.append(Text(str(self._message), style=theme.c("text")))
            if self.queued:
                color = theme.agent_color(self.agent)
                inner.append(Text(" QUEUED ", style=f"bold on {color}"))
            return Group(*inner)
        if self.role == "assistant":
            width = self.size.width if self.size else None
            text = str(self._message)
            if self.streaming:
                # Plain wrapped text for the ENTIRE streaming phase. Parsing the
                # whole accumulated markdown document on every delta flush was
                # O(reply-length²) and caused visible layout jumps mid-reply.
                # Streaming is append-only plain text; the final message
                # re-renders once as full markdown in end_stream.
                group: RenderableType = Text(text, style=theme.c("text"))
            else:
                group = _plain(self._message, width=width)
            if self.streaming:
                return Group(group, Text("▍", style=theme.c("primary")))
            return group
        if self.role == "reasoning":
            return self._build_reasoning()
        if self.role == "assistant_mode":
            # e.g. `▣ Build · model`
            t = Text()
            t.append("▣ ", style=theme.agent_color(self.agent))
            t.append(self.agent.title(), style=theme.c("text"))
            if self._message:
                t.append(f" · {self._message}", style=theme.c("text_muted"))
            return t
        if self.role == "meta":
            return Text(str(self._message), style=theme.c("text_muted"))
        if self.role == "compaction":
            if self.streaming:
                # Live compaction: show the anchored summary as it streams in
                # (plain text + cursor), then re-render as the ` Compaction `
                # divider + markdown in end_compaction_stream.
                theme = get_theme("opencode")
                header = Text("▸ Compacted summary", style=theme.c("warning"))
                body = str(self._message or "")
                if body:
                    return Group(header, Text(body, style=theme.c("text")), Text("▍", style=theme.c("primary")))
                return Group(header, Text("▍", style=theme.c("primary")))
            return self._build_compaction()
        return self._render_tool(self._message) if isinstance(self._message, dict) else _plain(self._message, width=self.size.width if self.size else None)

    def _build_compaction(self) -> RenderableType:
        """Centered ` Compaction ` divider (opencode's compaction part,
        i18n `ui.messagePart.compaction`) with the anchored summary beneath —
        rendered as markdown so the headings (`## Objective`, `## Important
        Details`, `## Work State`, `## Next Move`, `## Relevant Files`) get the
        same colored styling as the official opencode summary message."""
        theme = get_theme("opencode")
        width = self.size.width if self.size else 80
        title = " Compaction "
        n = max(1, (width - len(title)) // 2)
        divider = Text("─" * n, style=theme.c("border_active"))
        divider.append(title, style=theme.c("border_active"))
        divider.append("─" * (width - len(title) - n), style=theme.c("border_active"))
        parts: list[RenderableType] = [divider]
        summary = str(self._message or "").strip()
        if summary:
            parts.append(render_markdown(summary, width=width))
        return Group(*parts)

    # -- reasoning (mirrors opencode's ReasoningPart, thinking mode "hide") --
    def _build_reasoning(self) -> RenderableType:
        """Collapsed by default: `+ Thought: <title>`. Streaming shows a spinner
        with `Thinking...`; clicking the header toggles the muted markdown body
        (also while the thought is still streaming)."""
        theme = get_theme("opencode")
        summary = reasoning_summary(self._message)
        title = summary["title"]
        body = summary["body"]
        prefix = "- " if self.expanded else "+ "

        if self.streaming:
            header = Text(
                f"{prefix}{SPINNER_FRAMES[self._spinner]} Thinking" + (f": {title}" if title else ""),
                style=theme.c("warning"),
            )
        else:
            header = Text(prefix, style=theme.c("warning"))
            header.append("Thought", style=theme.c("warning"))
            if self._thought_seconds is not None:
                header.append(f" for {self._thought_seconds:.1f}s", style=theme.c("warning"))
            if title:
                header.append(f": {title}", style=theme.c("warning"))
        parts: list[RenderableType] = [header]
        if self.expanded and body:
            parts.append(Text(body, style=theme.c("text_muted")))
        return Group(*parts)

    # -- tool rendering (mirrors opencode's per-tool components) ----------
    def _render_tool(self, tool_run: dict[str, Any]) -> RenderableType:
        display = _tool_display(tool_run.get("tool", "?"))
        fn = getattr(self, f"_render_{display}", None)
        if fn is None:
            fn = self._render_generic
        return fn(tool_run)

    def _status(self, run: dict[str, Any]) -> str:
        return run.get("status", "pending")

    def _error(self, run: dict[str, Any]) -> str:
        return run.get("error") or ""

    def _denied(self, run: dict[str, Any]) -> bool:
        # permission denials may arrive as an error field or as output text
        err = run.get("error") or (run.get("output") or "")
        return any(
            m in err
            for m in (
                "QuestionRejectedError",
                "rejected permission",
                "permission denied",
                "denied by permission",
                "user dismissed",
                "specified a rule",
            )
        )

    def _failed(self, run: dict[str, Any]) -> bool:
        """Mirror opencode's InlineTool.failed: a real error that isn't a denial."""
        if self._denied(run):
            return False
        if run.get("status") == "error":
            return True
        return bool(run.get("error"))

    def _inline(
        self,
        icon: str,
        pending: str,
        label: str,
        *,
        spinner: bool = False,
        complete: bool | str | None = None,
        color: str | None = None,
    ) -> RenderableType:
        """Mirror opencode's InlineToolRow: ~ pending / spinner running / icon label.

        While running, once the tool input is known (complete truthy) we show the
        actual action (`← Edit /x.py`) instead of `~ Preparing edit...`, matching
        opencode. Colors mirror opencode: running -> primary, completed -> textMuted,
        failed -> error (red), denied -> strikethrough. ``color`` overrides the
        running/completed color (opencode passes ``theme.error`` for a retrying
        sub-agent row).
        """
        theme = get_theme("opencode")
        status = self._status(self._message)
        denied = self._denied(self._message)
        failed = self._failed(self._message)
        done = status in ("completed", "error") if complete is None else bool(complete)

        if status == "running":
            if spinner:
                return Text(f"{SPINNER_FRAMES[self._spinner]} {label}", style=color or theme.c("primary"))
            if complete:
                return Text(f"{icon} {label}", style=color or theme.c("primary"))
            return Text(f"~ {pending}", style=theme.c("text_muted"))
        if not done:
            return Text(f"~ {pending}", style=theme.c("text_muted"))

        main = color or (theme.c("error") if failed else theme.c("text_muted"))
        style = f"{main} strike" if denied else main
        return Text(f"{icon} {label}", style=style)

    def _error_line(self, run: dict[str, Any]) -> RenderableType | None:
        """A red error line for a failed tool (None when not a real error)."""
        if not self._failed(run) or self._denied(run):
            return None
        err = (self._error(run) or "").strip()
        if not err:
            return None
        return Text(err, style=get_theme("opencode").c("error"))

    def _render_bash(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        command = str((run.get("input") or {}).get("command", ""))
        output = (run.get("output") or "").strip()
        workdir = str((run.get("input") or {}).get("workdir") or "")
        if status == "running":
            return Text(f"{SPINNER_FRAMES[self._spinner]} {command}", style=theme.c("text"))
        if output:
            lines: list[RenderableType] = []
            if workdir and workdir != ".":
                lines.append(Text(f"# Running in {workdir}", style=theme.c("text_muted")))
            lines.append(Text(f"$ {command}", style=theme.c("text")))
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            lines.append(Text(output_text, style=theme.c("text")))
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                lines.append(Text(label, style=theme.c("text_muted")))
            return Group(*lines)
        return self._inline("$", "Writing command...", command)

    def _render_read(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        filepath = str((run.get("input") or {}).get("filePath", ""))
        loaded = (run.get("metadata") or {}).get("loaded") or []
        extra = _format_input((run.get("input") or {}), omit=("filePath",))
        row = self._inline(
            "→",
            "Reading file...",
            f"Read {filepath}{extra}",
            spinner=status == "running",
            complete=bool(filepath or extra),
        )
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if status == "completed" and loaded:
            sub = [
                Text(f"↳ Loaded {p}", style=theme.c("text_muted"))
                for p in (loaded if isinstance(loaded, list) else [loaded])
            ]
            return Group(row, *sub)
        return row

    def _render_glob(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("count")
        label = f'Glob "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Finding files...", label, complete=bool(pattern or path))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_grep(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("matches")
        label = f'Grep "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Searching content...", label, complete=bool(pattern or path))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_webfetch(self, run: dict[str, Any]) -> RenderableType:
        url = str((run.get("input") or {}).get("url", ""))
        row = self._inline("%", "Fetching from the web...", f"WebFetch {url}", complete=bool(url))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_websearch(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        meta = run.get("metadata") or {}
        provider = meta.get("provider", "")
        query = str(inp.get("query", ""))
        count = meta.get("numResults")
        label = f'{provider + " " if provider else ""}"{query}"'
        if isinstance(count, int):
            label += f" ({count} results)"
        row = self._inline("◈", "Searching web...", label, complete=bool(query))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row

    def _render_write(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        filepath = str((run.get("input") or {}).get("filePath", ""))
        written = (run.get("metadata") or {}).get("content")
        if self._status(run) == "completed" and written:
            content = str(written)
            collapsed = collapse_tool_output(content, 10, 10 * 80)
            output = content if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# Wrote {filepath}", style=theme.c("text_muted")),
                Text(output, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        return self._inline("←", "Preparing write...", f"Write {filepath}", complete=bool(filepath))

    def _render_edit(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        filepath = str((run.get("input") or {}).get("filePath", ""))
        diff = (run.get("metadata") or {}).get("diff")
        if self._status(run) == "completed" and diff:
            return Group(
                Text(f"← Edit {filepath}", style=theme.c("text_muted")),
                _render_diff(diff, filepath, self.size.width if self.size else None, self._diff_opts()),
            )
        err = self._error_line(run)
        if err is not None:
            replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
            return Group(
                self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}", complete=bool(filepath or replace_all)),
                err,
            )
        replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
        return self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}", complete=bool(filepath or replace_all))

    def _render_apply_patch(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        files = (run.get("metadata") or {}).get("files") or []
        if self._status(run) == "completed" and files:
            lines: list[RenderableType] = []
            for f in files if isinstance(files, list) else [files]:
                if not isinstance(f, dict):
                    continue
                rel = f.get("relativePath", "")
                title = f"← Patched {rel}"
                if f.get("type") == "delete":
                    title = f"# Deleted {rel}"
                elif f.get("type") == "add":
                    title = f"# Created {rel}"
                lines.append(Text(title, style=theme.c("text_muted")))
                patch = f.get("patch")
                if patch:
                    fpath = f.get("filePath") or f.get("relativePath") or ""
                    lines.append(_render_diff(patch, fpath, self.size.width if self.size else None, self._diff_opts()))
            return Group(*lines)
        return self._inline("%", "Preparing patch...", "Patch", complete=True)

    def _render_todowrite(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        todos = (run.get("metadata") or {}).get("todos") or (run.get("input") or {}).get("todos") or []
        if self._status(run) == "completed" and todos:
            lines: list[RenderableType] = [Text("# Todos", style=theme.c("text_muted"))]
            for todo in todos if isinstance(todos, list) else [todos]:
                if not isinstance(todo, dict):
                    continue
                status = todo.get("status", "pending")
                mark = "✓" if status == "completed" else ("•" if status == "in_progress" else " ")
                color = theme.c("warning") if status == "in_progress" else theme.c("text_muted")
                lines.append(Text(f"[{mark}] {todo.get('content', '')}", style=color))
            return Group(*lines)
        return self._inline("⚙", "Updating todos...", "Updating todos...", complete=True)

    def _render_question(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        questions = (run.get("input") or {}).get("questions") or []
        answers = (run.get("metadata") or {}).get("answers")
        count = len(questions) if isinstance(questions, list) else 0
        if self._status(run) == "completed" and answers is not None:
            lines: list[RenderableType] = [Text("# Questions", style=theme.c("text_muted"))]
            for i, q in enumerate(questions if isinstance(questions, list) else []):
                if not isinstance(q, dict):
                    continue
                lines.append(Text(str(q.get("question", "")), style=theme.c("text_muted")))
                ans = answers[i] if isinstance(answers, list) and i < len(answers) else None
                lines.append(
                    Text(
                        ", ".join(ans) if isinstance(ans, list) else str(ans) if ans else "(no answer)",
                        style=theme.c("text"),
                    )
                )
            return Group(*lines)
        return self._inline("→", "Asking questions...", f"Asked {count} question{'s' if count != 1 else ''}", complete=True)

    def _render_task(self, run: dict[str, Any]) -> RenderableType:
        """Render a `task` tool row mirroring opencode's Subagent component
        (packages/tui/src/routes/session/index.tsx):

          running   : `│ Build Task — fix login` (spinner) + live `↳ Bash …`
          completed : `✓ Build Task — fix login` then `↳ 3 toolcalls · 12.5s`
          retrying  : the whole row turns error-red with `↳ Retrying (attempt ·)`
          hint      : a persistent `↓ ctrl+down view subagents` line

        Clicking the row opens the sub-agent's own session (see on_click).
        """
        theme = get_theme("opencode")
        status = self._status(run)
        inp = run.get("input") or {}
        meta = run.get("metadata") or {}
        agent_label = str(inp.get("subagent_type", "")).strip() or "build"
        description = str(inp.get("description", "")).strip()
        if not description:
            description = str(meta.get("title") or "")
        background = bool(meta.get("background"))
        sid = str(meta.get("sessionId") or "")
        retry = meta.get("retry") or {}
        title = format_subagent_title(agent_label, description, background)
        icon = "✓" if status == "completed" else "│"
        # the active sub-agent (the one selected / being viewed) gets an accent
        # marker on its row, mirroring opencode's highlighted current agent.
        if self.selected:
            icon = f"▸ {icon}"

        sub: list[RenderableType] = []
        if status == "running":
            runs = self._task_child_runs(sid)
            # official `current()`: the last running/completed child tool that
            # has a real `state.title` (`↳ Bash npm run test`); otherwise just
            # the toolcall count.
            current = next(
                (tool_label(r) for r in reversed(runs) if r.get("status") in ("running", "completed") and tool_title(r)),
                None,
            )
            if current:
                sub.append(Text(f"↳ {current}", style=theme.c("text_muted")))
            elif runs:
                sub.append(Text(f"↳ {format_subagent_toolcalls(len(runs))}", style=theme.c("text_muted")))
        elif status == "completed":
            toolcalls = meta.get("toolcalls")
            if toolcalls is None:
                toolcalls = len([r for r in self._task_child_runs(sid) if r.get("status") == "completed"])
            duration = str(meta.get("duration") or "")
            detail = format_completed_subagent_detail(toolcalls or 0, duration)
            if detail:
                sub.append(Text(f"↳ {detail}", style=theme.c("text_muted")))
        elif status == "error" and sid:
            err = (run.get("error") or "").strip()
            if err:
                sub.append(Text(f"↳ {err}", style=theme.c("error")))

        lines: list[RenderableType] = [
            self._inline(
                icon,
                "Delegating...",
                title,
                spinner=status == "running",
                complete=bool(description),
                color=theme.c("error") if retry.get("attempt") else None,
            )
        ]
        if retry.get("attempt"):
            lines.append(
                Text(
                    f"↳ {format_subagent_retry(int(retry['attempt']), str(retry.get('message') or ''))}",
                    style=theme.c("error"),
                )
            )
        lines += sub
        if sid and self._is_last_task_row():
            lines.append(Text("↓  ctrl+down  view subagents", style=theme.c("text_muted")))
        return Group(*lines)

    # -- sub-agent task row live state ------------------------------------
    def _task_parent_chat(self) -> ChatView | None:
        """The ChatView this task row is mounted in (the parent session's chat)."""
        parent = self.parent
        if isinstance(parent, ChatView):
            return parent
        return None

    def _is_last_task_row(self) -> bool:
        """True when this is the newest task row in the parent chat — the row
        under which opencode draws the `view subagents` hint (once, per message)."""
        chat = self._task_parent_chat()
        if chat is None:
            return False
        latest: MessageBubble | None = None
        try:
            nodes = list(chat.walk_children(MessageBubble))
        except Exception:
            nodes = [n for n in getattr(chat, "children", ()) if isinstance(n, MessageBubble)]
        for child in nodes:
            if child.role == "tool" and isinstance(child.content, dict) and child.content.get("tool") == "task":
                latest = child
        return latest is self

    def _task_child_chat(self, session_id: str) -> ChatView | None:
        """The sub-agent's own ChatView (opencode reads the child session's parts
        via ``sync.data.part[msg.id]``; here the child's tool bubbles are the
        equivalent live store). None when it isn't registered / was pruned."""
        if not session_id:
            return None
        try:
            app = self.app
        except Exception:
            # WidgetHost / off-pump tests, or the app is already torn down —
            # there's simply no child ChatView to read live state from.
            return None
        chats = getattr(app, "_chats", None)
        if not isinstance(chats, dict):
            return None
        return chats.get(session_id)

    def _task_child_runs(self, session_id: str) -> list[dict[str, Any]]:
        chat = self._task_child_chat(session_id)
        if chat is None:
            return []
        return [
            r
            for r in chat.tool_runs()
            if r.get("status") in ("running", "completed", "error")
        ]

    def _render_execute(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        status = self._status(run)
        icon = "✓" if status == "completed" else "│"
        row = self._inline(icon, "execute", "execute", spinner=status == "running")
        calls = (run.get("metadata") or {}).get("toolCalls") or []
        sub = []
        for c in calls if isinstance(calls, list) else []:
            if not isinstance(c, dict):
                continue
            name = c.get("tool", "")
            args = _format_input(c.get("input") or {})
            failed = " (failed)" if c.get("status") == "error" else ""
            sub.append(Text(f"↳ {name}{args}{failed}", style=theme.c("text_muted")))
        return Group(row, *sub) if sub else row

    def _render_skill(self, run: dict[str, Any]) -> RenderableType:
        name = str((run.get("input") or {}).get("name", ""))
        return self._inline("→", "Loading skill...", f'Skill "{name}"', complete=bool(name))

    def _render_generic(self, run: dict[str, Any]) -> RenderableType:
        theme = get_theme("opencode")
        tool = run.get("tool", "?")
        output = (run.get("output") or "").strip()
        if self._status(run) == "completed" and output:
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# {tool} {_format_input(run.get('input') or {})}".strip(), style=theme.c("text_muted")),
                Text(output_text, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        label = f"{tool} {_format_input(run.get('input') or {})}".strip()
        return self._inline("⚙", "Running...", label, complete=bool(label))

    # -- frame (border / background / padding) ----------------------------
    def _apply_frame(self) -> None:
        theme = get_theme("opencode")
        st = self.styles
        if self.role == "user":
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.agent_color(self.agent))
            st.padding = (1, 1, 1, 2)
        elif self.role == "tool" and self._tool_block():
            # opencode BlockTool: left border in the panel background, panel fill.
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.c("background"))
            st.padding = (1, 1, 1, 2)
        else:
            # indented plain text (assistant / mode-line / meta / inline tool).
            # use an invisible border so Textual never paints its default.
            st.background = "transparent"
            st.border_left = ("solid", theme.c("background"))
            st.padding = (0, 0, 0, 3)
        st.margin = (1, 0, 0, 0)

    def _tool_block(self) -> bool:
        """A completed tool run renders as a block iff it produced a result block."""
        if not isinstance(self._message, dict):
            return False
        run = self._message
        if run.get("status") != "completed":
            return False
        name = run.get("tool", "?")
        display = _tool_display(name)
        if display == "bash":
            return bool((run.get("output") or "").strip())
        if display == "edit":
            return bool((run.get("metadata") or {}).get("diff"))
        if display == "apply_patch":
            return bool((run.get("metadata") or {}).get("files"))
        if display == "todowrite":
            return bool((run.get("metadata") or {}).get("todos"))
        if display == "question":
            return (run.get("metadata") or {}).get("answers") is not None
        if display == "write":
            return bool((run.get("metadata") or {}).get("content"))
        if display == "generic":
            return bool((run.get("output") or "").strip())
        return False

    def _tool_collapse(self, run: dict[str, Any]) -> dict:
        """Mirror opencode's per-tool collapse limits (bash 10 lines, generic 3)."""
        output = (run.get("output") or "").strip()
        if not output:
            return {"output": output, "overflow": False}
        display = _tool_display(run.get("tool", "?"))
        if display == "bash":
            return collapse_tool_output(output, 10, 10 * 80)
        return collapse_tool_output(output, 3, 3 * 80)

    def _tool_overflow(self) -> bool:
        return isinstance(self._message, dict) and self._tool_collapse(self._message)["overflow"]

    # -- updates -----------------------------------------------------------
    def _refresh(self) -> None:
        self._apply_frame()
        self.update(self._build_content())

    def _start_spinner(self) -> None:
        if self._timer is None:
            self._timer = self.set_interval(SPINNER_INTERVAL, self._tick)

    def _stop_spinner(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        if self.role == "tool" and self._message.get("status") == "running":
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()
        elif self.role == "reasoning" and self.streaming:
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()

    def on_click(self, event: Any) -> None:
        if self.role == "reasoning":
            self.expanded = not self.expanded
        elif self.role == "tool":
            if self.content.get("tool") == "task":
                meta = self.content.get("metadata") or {}
                sid = meta.get("sessionId")
                if sid:
                    self.post_message(OpenTaskSession(sid))
            elif self._tool_overflow():
                self.expanded = not self.expanded

    def on_key(self, event: Any) -> None:
        if self.role == "reasoning" and self.has_focus and event.key in ("enter", "space"):
            event.stop()
            self.expanded = not self.expanded
        elif self.role == "tool" and self.has_focus and event.key in ("enter", "space"):
            if self.content.get("tool") == "task":
                meta = self.content.get("metadata") or {}
                sid = meta.get("sessionId")
                if sid:
                    event.stop()
                    self.post_message(OpenTaskSession(sid))

    def update_tool(self, tool_run: dict[str, Any]) -> None:
        self._message = tool_run
        if tool_run.get("status") == "running":
            self._start_spinner()
        else:
            self._stop_spinner()
            self.streaming = False
        self._refresh()

    def set_tool_metadata(self, key: str, value: Any) -> None:
        """Merge one metadata entry into the tool run in place (keeps input,
        output and the other metadata), then re-render. The engine supplies the
        task row in a running -> completed sequence; this lets the TUI enrich it
        with live data (duration, toolcall count, retry state) without losing
        what the engine already wrote."""
        if not isinstance(self._message, dict):
            return
        self._message.setdefault("metadata", {})[key] = value
        self._refresh()

    def pop_tool_metadata(self, key: str) -> None:
        if not isinstance(self._message, dict):
            return
        meta = self._message.get("metadata")
        if isinstance(meta, dict) and key in meta:
            del meta[key]
            self._refresh()

    def update_text(self, text: str) -> None:
        self._message = text
        self._refresh()

    def update_reasoning(self, text: str) -> None:
        self._message = text
        self._refresh()

    def append_stream(self, text: str) -> None:
        """Append one streamed chunk without rebuilding the whole text first."""
        if text:
            self._stream_parts.append(text)
            self.update_text("".join(self._stream_parts))

    def end_reasoning(self) -> None:
        self.streaming = False
        self._stop_spinner()
        if self._thought_started is not None:
            self._thought_seconds = time.monotonic() - self._thought_started
            self._thought_started = None
        # keep thoughts collapsed by default: show `+ Thought for 5.0s: <title>`
        # and let the user click / Enter / Ctrl+Shift+E to expand the full body
        self.expanded = False
        self._refresh()


class ChatView(VerticalScroll):
    """Scrollable list of message bubbles + streaming cursor."""

    messages: reactive = reactive([])
    streaming: reactive = reactive(False)
    streaming_text: reactive = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream_bubble: MessageBubble | None = None
        self._reasoning_bubble: MessageBubble | None = None
        self._compaction_bubble: MessageBubble | None = None
        # Queued user bubbles in submission order (mirrors opencode's FIFO
        # prompt queue). The engine clears them one-by-one as it folds each
        # prompt into the running turn.
        self._queued_bubbles: list[MessageBubble] = []
        # True while the user is reading the newest output (at the bottom).
        # When they scroll up to re-read history, we stop yanking the view back
        # down on every stream delta / tool update.
        self._follow_bottom = True
        # True when this chat belongs to a sub-agent session: the arrow keys
        # navigate the session tree instead of scrolling (so ↑ inside a
        # sub-agent always returns to the parent, even when the chat — not the
        # prompt — has focus).
        self._session_is_child = False
        # First-open welcome banner (ASCII opencode logo). Mounted only when
        # the chat has no messages/stream yet, removed once the conversation
        # starts.
        self._welcome_logo: Static | None = None

    # -- first-open banner -------------------------------------------------
    def show_logo(self) -> None:
        """Show the opencode logo when the chat is empty (first launch)."""
        if self._welcome_logo is not None:
            return
        logo = Static(opencode_logo_text(), classes="chat-welcome-logo")
        self.mount(logo)
        self._welcome_logo = logo
        try:
            self.scroll_home(animate=False)
        except Exception:
            pass

    def _dismiss_logo(self) -> None:
        """Remove the first-open logo once the conversation begins."""
        logo = self._welcome_logo
        self._welcome_logo = None
        if logo is not None:
            try:
                logo.remove()
            except Exception:
                pass

    def welcome_empty(self) -> bool:
        """True while the first-open banner is still visible (no messages)."""
        return self._welcome_logo is not None

    # -- key routing -------------------------------------------------------
    async def on_key(self, event: Any) -> None:
        if getattr(self, "_session_is_child", False) and event.key in ("up", "left", "right"):
            event.stop()
            event.prevent_default()
            self.post_message(SessionNavRequested(event.key))
        # Not intercepted: let the key bubble to the App's binding dispatch
        # (scroll bindings, etc.). There is no base `on_key` message handler
        # to chain to, so just return.

    # -- scrolling ---------------------------------------------------------
    def _auto_scroll(self) -> None:
        """Scroll to the newest message, unless the user scrolled up to read
        earlier conversation (opencode keeps your position while the model
        keeps streaming/tool-running below)."""
        if self._follow_bottom:
            self.scroll_end(animate=False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        # Update the follow flag on any user scroll: at the bottom (or no room
        # to scroll) we keep following; anywhere above it we stop until the
        # user scrolls back down.
        try:
            self._follow_bottom = new_value >= self.max_scroll_y - 1
        except Exception:
            self._follow_bottom = True

    def append_user(self, text: str, agent: str = "build", queued: bool = False) -> MessageBubble:
        self._dismiss_logo()
        bubble = MessageBubble("user", text, agent=agent)
        bubble.queued = queued
        if queued:
            self._queued_bubbles.append(bubble)
        self.mount(bubble)
        self._auto_scroll()
        return bubble

    def append_directive(self, text: str, title: str = "") -> MessageBubble:
        """The parent's instruction rendered at the top of a sub-agent's chat
        (official opencode shows the task directive as the first message)."""
        self._dismiss_logo()
        bubble = MessageBubble("user", text, directive=title or "Task")
        bubble.add_class("directive")
        self.mount(bubble)
        self._auto_scroll()
        return bubble

    def promote_next_queued(self) -> str | None:
        """Clear the ` QUEUED ` badge on the oldest queued message.

        Called when the engine folds that prompt into the running turn
        (opencode's Session Drain) — the message keeps its place in the chat,
        just loses its badge. Returns the promoted text.
        """
        if not self._queued_bubbles:
            return None
        bubble = self._queued_bubbles.pop(0)
        try:
            bubble.queued = False
        except Exception:
            pass
        return str(bubble.content)

    def queued_count(self) -> int:
        return len(self._queued_bubbles)

    def clear(self) -> None:
        """Drop every bubble and reset the streaming cursors. Used when the
        main session is deleted and the workspace resets to a fresh one."""
        self._stream_bubble = None
        self._reasoning_bubble = None
        self._follow_bottom = True
        self._queued_bubbles = []
        for child in list(self.children):
            try:
                child.remove()
            except Exception:
                pass
        try:
            self.scroll_home(animate=False)
        except Exception:
            pass
        # A removed-but-unflushed logo widget may still be in `children` (Textual
        # defers DOM removal), so drop the reference: the banner belongs to the
        # app's very first open, not to a workspace reset.
        self._welcome_logo = None

    def append_assistant(self, text: str) -> None:
        self._dismiss_logo()
        self.mount(MessageBubble("assistant", text))
        self._auto_scroll()

    def append_meta(self, text: str) -> None:
        """Persistent command/system output (e.g. /models, /help, /config)."""
        self._dismiss_logo()
        self.mount(MessageBubble("meta", text))
        self._auto_scroll()

    def append_compaction(self, summary: str) -> None:
        """Render a ` Compaction ` divider + markdown summary (opencode's
        compaction part).

        Ends any in-flight reasoning/stream bubble first so the divider lands
        between the finished text and whatever the model continues after the
        summarized history is replayed.
        """
        self.end_reasoning()
        self.remove_last_stream_bubble()
        self._dismiss_logo()
        self.mount(MessageBubble("compaction", summary))
        self._auto_scroll()

    def begin_compaction_stream(self) -> None:
        """Start the live `▸ Compacted summary` bubble before the summary
        model streams; deltas append via stream_compaction_delta."""
        self._dismiss_logo()
        self.end_reasoning()
        self.remove_last_stream_bubble()
        bubble = MessageBubble("compaction", "", streaming=True)
        self._compaction_bubble = bubble
        self.mount(bubble)
        self._auto_scroll()

    def stream_compaction_delta(self, text: str) -> None:
        """Append one live summary token to the streaming compaction bubble."""
        bubble = self._compaction_bubble
        if bubble is None:
            self.begin_compaction_stream()
            bubble = self._compaction_bubble
        bubble.append_stream(text)
        self._auto_scroll()

    def end_compaction_stream(self, summary: str) -> None:
        """Finalize the live compaction bubble into the ` Compaction ` divider.

        The live bubble was rendering plain text while the summary streamed;
        on `compacted` we re-render the FULL final summary as markdown inside
        the divider (same look as the non-live path)."""
        bubble = self._compaction_bubble
        self._compaction_bubble = None
        if bubble is None:
            # No live bubble (compaction failed before the first delta) — fall
            # back to the static divider so the outcome is still visible.
            if summary:
                self.append_compaction(summary)
            return
        if not summary and not bubble.content:
            bubble.remove()
            return
        bubble._message = summary or bubble.content
        bubble.streaming = False  # re-render as the finished ` Compaction ` divider
        bubble._refresh()

    def append_tool(self, tool_run: dict[str, Any]) -> None:
        self._dismiss_logo()
        bubble = MessageBubble("tool", tool_run)
        if tool_run.get("status") == "running":
            # start the spinner right away so a fresh tool_call row isn't stuck
            # on a static frame until tool_start arrives
            bubble._start_spinner()
        self.mount(bubble)
        self._auto_scroll()

    def append_reasoning(self, text: str, seconds: float | None = None) -> None:
        """Render a finished `+ Thought` bubble (used when replaying saved
        history; live thoughts stream through begin_thinking/stream_reasoning)."""
        self._dismiss_logo()
        bubble = MessageBubble("reasoning", text)
        bubble._thought_seconds = seconds
        self.mount(bubble)
        self._auto_scroll()

    def begin_stream(self) -> None:
        self._dismiss_logo()
        self._stream_bubble = MessageBubble("assistant", "")
        self._stream_bubble.streaming = True
        self.mount(self._stream_bubble)
        self._auto_scroll()

    def stream_delta(self, text: str) -> None:
        if self._stream_bubble is None:
            self.begin_stream()
        self._stream_bubble.append_stream(text)
        self._auto_scroll()

    def stream_reasoning_delta(self, text: str) -> None:
        self._dismiss_logo()
        if self._reasoning_bubble is None:
            bubble = MessageBubble("reasoning", "")
            bubble.streaming = True
            bubble._start_spinner()
            bubble._thought_started = time.monotonic()
            # chronological order: each new thought mounts below the previous
            # tool runs (not above a stale empty stream bubble)
            self.mount(bubble)
            self._reasoning_bubble = bubble
        self._reasoning_bubble.append_stream(text)
        self._auto_scroll()

    def begin_thinking(self) -> None:
        """Mount an eager `Thinking...` bubble the moment a turn starts, before
        the first token arrives, so the UI reacts instantly to Enter (mirrors
        opencode). Real reasoning deltas stream into this same bubble; if no
        reasoning ever arrives, end_reasoning drops the empty placeholder."""
        self._dismiss_logo()
        if self._reasoning_bubble is not None:
            return
        bubble = MessageBubble("reasoning", "")
        bubble.streaming = True
        bubble._start_spinner()
        bubble._thought_started = time.monotonic()
        self.mount(bubble)
        self._reasoning_bubble = bubble
        self._auto_scroll()

    def end_reasoning(self) -> None:
        if self._reasoning_bubble is not None:
            if not self._reasoning_bubble.content:
                # the eager placeholder produced no actual reasoning — remove
                # it instead of leaving a misleading empty `Thought` line
                bubble = self._reasoning_bubble
                self._reasoning_bubble = None
                bubble._stop_spinner()
                try:
                    bubble.remove()
                except Exception:
                    pass
            else:
                self._reasoning_bubble.end_reasoning()
                self._reasoning_bubble = None
        self._auto_scroll()

    def end_stream(self, text: str = "") -> None:
        """Finalize the streaming cursor. An assistant bubble that never
        received any text is DROPPED instead of left as a blank line — e.g. a
        thought + tool turn whose final "summary" streamed an empty chunk."""
        if self._stream_bubble is not None:
            if text:
                self._stream_bubble.update_text(text)
            if self._stream_bubble.content:
                self._stream_bubble.streaming = False
                self._stream_bubble._refresh()
            else:
                try:
                    self._stream_bubble.remove()
                except Exception:
                    pass
            self._stream_bubble = None
        self._auto_scroll()

    def remove_last_stream_bubble(self) -> None:
        """Remove the empty streaming bubble left behind when there's no reply.

        A bubble that already holds partial text is KEPT (only its streaming
        cursor is ended) so a mid-stream error doesn't discard model output.
        """
        target = self._stream_bubble
        if target is None:
            target = None
            # The only empty assistant bubbles are streaming-cursor leftovers,
            # which always sit in the trailing assistant run at the bottom of
            # the chat. Walk just that run instead of a full container query
            # (O(call) → O(trailing run) per remove).
            for child in reversed(self.children):
                try:
                    role = child.role
                except Exception:
                    break
                if role != "assistant":
                    break
                if not child.content and not child.streaming:
                    target = child
                    break
        if target is not None:
            if target.content:
                target.streaming = False
            else:
                try:
                    target.remove()
                except Exception:
                    pass
        self._stream_bubble = None
        self._auto_scroll()

    def find_tool(self, tool: str, call_id: str = "") -> MessageBubble | None:
        candidates = []
        for child in self.query(MessageBubble):
            if child.role == "tool" and child.content.get("tool") == tool:
                candidates.append(child)
        if not candidates:
            return None
        if not call_id:
            return candidates[0]
        for child in candidates:
            if child.content.get("call_id") == call_id:
                return child
        return None

    def find_task(self, session_id: str) -> MessageBubble | None:
        """The task row whose metadata.sessionId matches a child session id.
        Mirrors the reverse lookup opencode does when a sub-agent's events
        bubble up to the parent chat."""
        for child in self.query(MessageBubble):
            if child.role != "tool" or child.content.get("tool") != "task":
                continue
            meta = child.content.get("metadata")
            if isinstance(meta, dict) and meta.get("sessionId") == session_id:
                return child
        return None

    def tool_runs(self) -> list[dict[str, Any]]:
        """Every tool run currently shown in this chat (opencode's per-session
        tool-part store; the Subagent parent row reads the child's to show the
        live current tool / toolcall count)."""
        return [
            child.content
            for child in self.query(MessageBubble)
            if child.role == "tool" and isinstance(child.content, dict)
        ]

    def update_tool_bubble(self, tool_run: dict[str, Any]) -> None:
        bubble = self.find_tool(tool_run.get("tool", ""), tool_run.get("call_id", ""))
        if bubble:
            bubble.update_tool(tool_run)
            self._auto_scroll()
            return True
        return False

    def last_reasoning(self) -> MessageBubble | None:
        found = None
        for child in self.query(MessageBubble):
            if child.role == "reasoning":
                found = child
        return found

    def toggle_last_reasoning(self) -> None:
        bubble = self.last_reasoning()
        if bubble is not None:
            bubble.expanded = not bubble.expanded
            bubble.focus()

    def watch_messages(self, value: list) -> None:
        self.refresh()
