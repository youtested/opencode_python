"""todowrite tool: in-session todo list helper (registry key `todo`)."""

from __future__ import annotations

import json

from .registry import Tool, schema_with

VALID_STATUS = {"pending", "in_progress", "completed", "cancelled"}
VALID_PRIORITY = {"high", "medium", "low"}


def _todowrite(todos: list[dict], state: dict) -> dict:
    normalized = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        status = t.get("status", "pending")
        priority = t.get("priority", "medium")
        if status not in VALID_STATUS:
            status = "pending"
        if priority not in VALID_PRIORITY:
            priority = "medium"
        normalized.append(
            {"content": t.get("content", ""), "status": status, "priority": priority}
        )
    state["todos"] = normalized
    remaining = sum(1 for t in normalized if t["status"] not in ("completed", "cancelled"))
    return {
        "output": json.dumps(normalized, indent=2),
        "metadata": {"todos": normalized, "remaining": remaining},
    }


def tool(state: dict | None = None) -> Tool:
    state = state or {}
    description = """Create and maintain a structured task list for the current coding session.

When to use:
- 3+ distinct steps or actions to take
- The work is non-trivial and benefits from planning
- The user provides multiple tasks (numbered or comma-separated) or explicitly asks for a todo list
- New instructions arrive - capture them as todos

When NOT to use:
- The work is a single, straightforward task (or <3 trivial steps)
- The request is purely informational or conversational
- Tracking adds no organizational value

States:
- pending - not started
- in_progress - actively working (exactly ONE at a time)
- completed - finished successfully
- cancelled - no longer needed

Rules:
- Update status in real time; don't batch completions
- Mark completed only after the required work is actually done, including any required verification. Never based on intent.
- Keep exactly one in_progress while work remains
- If blocked or partial, keep it in_progress and add a follow-up todo describing the blocker
- Preserve user-provided commands verbatim (flags, args, order)
- Items should be specific and actionable; break large work into smaller steps"""

    def run(input: dict) -> dict:
        return _todowrite(input.get("todos", []), state)

    return Tool(
        name="todowrite",
        description=description,
        parameters=schema_with(
            {
                "todos": {
                    "type": "array",
                    "description": "The updated todo list for the session",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Brief description of the task"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["content", "status", "priority"],
                    },
                }
            },
            ["todos"],
        ),
        run=run,
        permission="todowrite",
    )
