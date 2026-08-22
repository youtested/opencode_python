"""remember tool: persistent cross-session memory (registry key `remember`).

Stores short notes tagged with the project they were created in (the resolved
worktree of the current working directory, or "" for global). The system prompt
loader injects the current project's + global notes at startup, so a rule saved
once is followed in every later session without any extra work.

One tool, four actions via the `action` param:
- add    (default) - save a note
- list   - show notes for the current project + global notes
- delete - remove one note by id (see ids in `list`)
- clear  - wipe this project's notes

The store lives at <data dir>/memory.json and is capped so it never grows
without bound.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with

MAX_ENTRIES = 300
MAX_TEXT = 2000

_LOCK = threading.Lock()

_ACTIONS = ("add", "list", "delete", "clear")


def _memory_file() -> Path:
    return GPath.data / "memory.json"


def _load() -> dict[str, Any]:
    try:
        raw = _memory_file().read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return {"version": 1, "entries": []}
    if not isinstance(data, dict):
        data = {"version": 1}
    entries = data.get("entries")
    if not isinstance(entries, list):
        entries = []
    return {"version": data.get("version", 1), "entries": entries}


def _save(data: dict[str, Any]) -> None:
    path = _memory_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _project() -> str:
    from ..globals import resolve_worktree

    try:
        return str(resolve_worktree(Path.cwd()))
    except OSError:
        return ""


def _next_id(entries: list[dict]) -> int:
    return max((e.get("id", 0) for e in entries), default=0) + 1


def _fmt_date(ts: float) -> str:
    try:
        import datetime

        return datetime.date.fromtimestamp(ts).isoformat()
    except (OSError, ValueError, OverflowError):
        return "?"


def _add(text: str, project: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"output": "Nothing to remember: the text was empty.", "error": True}
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + "…"
    with _LOCK:
        data = _load()
        entries = data["entries"]
        entry = {
            "id": _next_id(entries),
            "text": text,
            "project": project,
            "created": time.time(),
        }
        entries.append(entry)
        if len(entries) > MAX_ENTRIES:
            entries[: len(entries) - MAX_ENTRIES] = []
        _save(data)
    where = "project memory" if project else "global memory"
    return {"output": f"Remembered ({where}, id {entry['id']}):\n{text}"}


def _list(project: str) -> dict:
    with _LOCK:
        data = _load()
    entries = data["entries"]
    proj = [e for e in entries if e.get("project") == project]
    glob = [e for e in entries if not e.get("project")]

    def fmt(group: list[dict]) -> str:
        lines = []
        for e in group:
            lines.append(
                f"{e.get('id', 0)}. ({_fmt_date(e.get('created', 0))}) "
                f"{str(e.get('text', '')).replace(chr(10), ' ')}"
            )
        return "\n".join(lines) if lines else "(none)"

    out: list[str] = []
    out.append(f"Project memory for {project or '(current folder)'}:")
    out.append(fmt(proj))
    if glob:
        out.append("Global memory:")
        out.append(fmt(glob))
    out.append(f"Total notes: {len(entries)}")
    return {"output": "\n".join(out)}


def _delete(spec: Any, project: str) -> dict:
    with _LOCK:
        data = _load()
        entries = data["entries"]
        target_id = None
        if isinstance(spec, bool):
            target_id = None
        elif isinstance(spec, (int, float)):
            target_id = int(spec)
        else:
            m = re.fullmatch(r"\s*#?(\d+)\s*", str(spec))
            if m:
                target_id = int(m.group(1))
        if target_id is None:
            return {
                "output": "Give the id of the note to delete (see `list` for ids).",
                "error": True,
            }
        kept = [e for e in entries if e.get("id") != target_id]
        if len(kept) == len(entries):
            return {"output": f"No note with id {target_id}.", "error": True}
        data["entries"] = kept
        _save(data)
    return {"output": f"Deleted note {target_id}."}


def _clear(project: str) -> dict:
    with _LOCK:
        data = _load()
        entries = data["entries"]
        if project:
            data["entries"] = [e for e in entries if e.get("project") != project]
            where = f"Cleared project memory for {project}."
        else:
            data["entries"] = []
            where = "Cleared global memory."
        _save(data)
    return {"output": f"{where} (was {len(entries)} notes.)"}


def tool() -> Tool:
    description = """Persistent memory: save, recall, and delete short notes that survive across sessions.

Use this to remember project rules, environment quirks, recurring fixes, or any
fact you don't want to repeat in a later ask. Notes are tagged with the current
project and are automatically loaded into the model's instructions at startup
for that project (plus any global notes), so a saved rule is followed every time.

Actions (via the `action` parameter):
- add (default): save a note. Requires `text`.
- list: show notes for the current project and any global notes, with their ids.
- delete: remove one note. Give the `id` shown by `list`.
- clear: remove all of the current project's notes.

When to use:
- The user says "remember ..." / "note that ..." / "keep this in mind ...".
- You discover a stable fact or convention mid-task that a later session should
  also follow (save it proactively, but keep notes short and reusable).

Examples:
- add: \"always ask before installing things on this machine\"
- list: \"what do you remember about this project?\"
- delete: \"forget note 2\""""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "add").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        project = _project()
        if action == "add":
            return _add(str(input.get("text") or ""), project)
        if action == "list":
            return _list(project)
        if action == "delete":
            return _delete(input.get("id"), project)
        return _clear(project)

    return Tool(
        name="remember",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "description": "What to do: add, list, delete, or clear.",
                    "enum": list(_ACTIONS),
                    "optional": True,
                },
                "text": {
                    "type": "string",
                    "description": "The note to remember (for action=add).",
                    "optional": True,
                },
                "id": {
                    "type": "integer",
                    "description": "Id of the note to delete (for action=delete, see list).",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
    )
