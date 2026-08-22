"""Status bar: opencode-style bottom footer.

Mirrors opencode's session footer: the working directory on the left, and on
the right a muted context-size hint (`12,345 (6%)`) plus the `/status` hint.
The "working…" indicator lives in the prompt's status line only.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

from .theme import get_theme


class StatusBar(Static):
    """Single-line footer mirroring opencode's bottom chrome."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.directory = ""
        self.streaming = False
        self.usage: dict[str, int] = {}
        self.interrupt_armed = False
        self.retry_message = ""
        self._refresh()

    def set_retry_message(self, message: str) -> None:
        """Show a transient 'connection dropped — retrying…' hint in the footer."""
        self.retry_message = message or ""
        self._refresh()

    def set_interrupt_armed(self, armed: bool) -> None:
        self.interrupt_armed = armed
        self._refresh()

    def set_directory(self, directory: str) -> None:
        self.directory = directory or ""
        self._refresh()

    def set_header(self, **kwargs: Any) -> None:
        """Compatibility shim; opencode's footer only shows the directory."""
        self._refresh()

    def set_streaming(self, streaming: bool) -> None:
        # The "working…" indicator lives in the prompt status line; the footer
        # deliberately shows nothing extra while a turn streams.
        self.streaming = streaming

    def set_usage(self, usage: dict[str, int]) -> None:
        self.usage = usage or {}
        self._refresh()

    def _context_hint(self) -> str:
        """Used context window, opencode-style: `12,345 (6%)`.

        Uses the cumulative session token count (summed across every stream in
        the turn) and the active model's context window. The percentage is the
        fraction of the window that was touched, so it is clamped to [0, 100] —
        providers can report input+output (or cached/reasoning) tokens that
        sum past the nominal context, which would otherwise show `111%`.
        """
        if not self.usage:
            return ""
        total = self.usage.get("total_tokens") or 0
        if not total:
            return ""
        ctx = self.usage.get("context_size") or 0
        text = f"  {total:,}"
        if ctx:
            pct = min(100, round(total / ctx * 100))
            text += f" ({pct}%)"
        return text

    def _refresh(self) -> None:
        theme = get_theme("opencode")
        left = Text(self.directory or " ", style=theme.c("text_muted"))

        right = Text()
        if self.interrupt_armed:
            right.append("esc again to interrupt", style=theme.c("accent"))
            right.append("  ", style=theme.c("text_muted"))
        if self.retry_message:
            right.append(self.retry_message, style=theme.c("warning"))
            right.append("  ", style=theme.c("text_muted"))
        right.append(self._context_hint(), style=theme.c("text_muted"))
        right.append("  /status", style=theme.c("text_muted"))

        # push the right group to the far edge
        width = self.size.width if self.size else 80
        left_text = left.plain
        right_text = right.plain
        pad = max(2, width - len(left_text) - len(right_text))
        t = Text(left_text, style=theme.c("text_muted"))
        t.append(" " * pad, style=theme.c("text_muted"))
        t.append(right)
        self.update(t)

    def on_resize(self) -> None:
        self._refresh()