"""Token-ish length guards for context trimming and tool output capping."""

from __future__ import annotations

import re

# Rough token estimate: 4 chars per token (English-ish heuristic). Good enough
# for budget accounting without a tokenizer dependency.
CHARS_PER_TOKEN = 4

# Regex used by opencode to split words/tokens loosely
_WORD_SPLIT = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count (chars/4), capped at min 1 token for non-empty."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def trim_messages(messages: list[dict], budget: int) -> list[dict]:
    """Trim a message list so the estimated token count fits `budget`.

    Drops oldest messages first; keeps the most recent user message intact.
    O(n): sizes are computed once and dropped by a running total instead of
    re-summing the whole list (and popping the front) on every drop.
    """
    if budget <= 0:
        return messages
    n = len(messages)
    if n == 0:
        return messages
    # score each message
    sizes = [estimate_tokens(_msg_text(m)) for m in messages]
    total = sum(sizes)
    if total <= budget:
        return messages
    # keep the LAST message (the newest user prompt) no matter what
    dropped = 0
    while total > budget and n - dropped > 1:
        total -= sizes[dropped]
        dropped += 1
    if dropped:
        return list(messages[dropped:])
    return messages


def _msg_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    # tool_calls / parts style
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part.get("text", "")))
                elif "name" in part and "arguments" in part:
                    parts.append(str(part.get("name", "")) + str(part.get("arguments", "")))
            else:
                parts.append(str(part))
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            fn = tc.get("function", {})
            parts.append(str(fn.get("name", "")) + str(fn.get("arguments", "")))
    return " ".join(parts)


def truncate_text(text: str, max_chars: int = 50_000) -> str:
    """Truncate a long string, keeping the head and adding a marker."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    return head + f"\n... output truncated ... ({len(text) - max_chars} chars dropped)"


def collapse_output(text: str, max_lines: int = 10, max_chars: int = 2000) -> str:
    """Collapse tool output to `max_lines` for display, like opencode's TUI."""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        shown = lines[:max_lines]
        return "\n".join(shown) + f"\n... (+{len(lines) - max_lines} more lines)"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... ({len(text) - max_chars} chars dropped)"
    return text
