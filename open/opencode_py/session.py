"""Session persistence: JSON files under the data dir.

A session stores: id, title, created/completed times, directory, provider,
model, agent, and the OpenAI-style message history. Auto-save after each turn.

Sub-agent sessions (spawned via the `task` tool) are regular sessions with a
`parent_id` pointing at the session that spawned them.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .globals import Path as GPath

# Serializes all session-body and index writes. save_session can be called
# from the UI thread (exit / save-all) and from the in-flight autosave worker
# thread (TUI streaming) — the lock keeps those writers from interleaving and
# lets a `should_write` guard run atomically with the write it protects.
_WRITE_LOCK = threading.Lock()


def _as_number(value: Any, fallback: float = 0.0) -> float:
    """Coerce a persisted timestamp to a float, tolerant of garbage.

    A corrupt session file (crash mid-write, partial rename, hand-edit) can
    carry a string or object in `created`/`completed`; letting that leak into
    list sorting would crash the whole picker. Return the fallback instead.
    """
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return fallback


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_str(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    try:
        return str(value)
    except Exception:
        return fallback


class Session:
    def __init__(self, data: dict[str, Any] | Any, directory: str | None = None):
        # `data` comes from a JSON file the user could corrupt at any point;
        # never assume it's a dict — bad JSON yields None/list/str and would
        # crash every listener with AttributeError otherwise.
        data = _as_dict(data)
        self.id = _as_str(data.get("id"), uuid.uuid4().hex) or uuid.uuid4().hex
        self.title = _as_str(data.get("title"))
        self.created = _as_number(data.get("created"), time.time())
        self.completed = None if data.get("completed") is None else _as_number(data.get("completed"))
        self.directory = _as_str(data.get("directory")) or directory or ""
        self.provider = _as_str(data.get("provider"))
        self.model = _as_str(data.get("model"))
        self.agent = _as_str(data.get("agent"), "build") or "build"
        self.parent_id = data.get("parent_id")
        self.messages: list[dict[str, Any]] = _as_list(data.get("messages"))
        self.metadata: dict[str, Any] = _as_dict(data.get("metadata"))
        # Lightweight listing extras (populated by the session index, unused by
        # resume/export which always load the full file).
        self.has_messages = bool(data.get("has_messages")) if "has_messages" in data else bool(self.messages)
        self.first_user_text = _as_str(data.get("first_user_text"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "completed": self.completed,
            "directory": self.directory,
            "provider": self.provider,
            "model": self.model,
            "agent": self.agent,
            "parent_id": self.parent_id,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @property
    def path(self) -> Path:
        return session_path(self.id)


def session_path(session_id: str) -> Path:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe_id:
        safe_id = "invalid"
    return GPath.sessions_dir() / f"{safe_id}.json"


def _fsync_dir(directory: Path) -> None:
    """Durably persist a rename/delete in ``directory``.

    A rename followed by an fsync of the *file* only is not enough: the
    directory entry swap can still be lost on a sudden power loss unless the
    directory itself is flushed. Best effort — some filesystems can't open a
    directory for fsync, and that's fine (the write itself is the critical
    part).
    """
    dfd = -1
    try:
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        except OSError:
            pass
    except OSError:
        pass
    finally:
        if dfd >= 0:
            try:
                os.close(dfd)
            except OSError:
                pass


def _write_durable(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` durably. Returns True on success.

    A sudden phone reboot (process kill, power loss) loses everything still
    living in the OS page cache — ``Path.write_text`` was buffering the whole
    session body there, so an autosave right before a crash could leave the
    session file empty (0 bytes) or missing entirely. This helper makes the
    write crash-safe: write the new content to a temp file, ``fsync`` it so
    the bytes really reach flash, atomically rename it over ``path``, then
    ``fsync`` the directory so the rename itself is persisted too. After this
    returns True, the data survives any power cut. If the write/flush itself
    fails, the previous good body is left untouched (never replaced by a
    partial temp file).
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = None
    wrote = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        fh = os.fdopen(fd, "w", encoding="utf-8")
        fd = None  # ownership transferred to the file object
        with fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        wrote = True
    except OSError:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    if not wrote:
        return False
    try:
        os.replace(tmp, path)
    except OSError:
        # the temp file holds the complete text but the rename is impossible
        # (odd filesystem) — a direct write is a safe last resort here.
        try:
            path.write_text(text, encoding="utf-8")
        except OSError:
            return False
    _fsync_dir(directory)
    return True


def _read_optional(path: Path) -> str | None:
    """Read a file's text, or None if it's missing/empty/undecodable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return text if text else None


def _read_session_data(path: Path) -> dict[str, Any] | None:
    """Parse a session body with crash recovery.

    Tries the primary file first, then the previous-good ``.json.bak`` replica
    kept by save_session, then a leftover ``.json.tmp`` (a crash between
    writing the temp file and renaming it over the body). A 0-byte body — the
    signature of a non-durable write lost to a sudden reboot — transparently
    falls back to the last good copy instead of vanishing.
    """
    for candidate in (path, path.with_suffix(".json.bak"), path.with_suffix(".json.tmp")):
        text = _read_optional(candidate)
        if text is None:
            continue
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def save_session(
    session: Session,
    *,
    should_write: Callable[[], bool] | None = None,
) -> Path:
    GPath.sessions_dir().mkdir(parents=True, exist_ok=True)
    path = session_path(session.id)
    # Persist the message list EXACTLY as-is. Do NOT inject synthetic
    # "[interrupted — tool result missing]" tool messages here: the saved
    # transcript must be the real conversation. The request path already
    # repairs orphaned tool pairs for the provider (agent/loop.py), so a
    # resumed session that ends mid-tool-run stays provider-safe without
    # corrupting what the user sees on resume.
    text = json.dumps(session.to_dict(), indent=2)
    # Run the write-set (body + .bak + index + cache invalidation) atomically
    # with respect to other writers. Callers that must never overwrite a newer
    # save (the streaming autosave, which can race the exit save-all) pass a
    # `should_write` guard evaluated inside the lock, immediately before the
    # body hits disk.
    with _WRITE_LOCK:
        if should_write is not None and not should_write():
            return path
        # Keep the previous good body as a `.bak` replica so a crash that corrupts
        # the primary write (0-byte body) can still recover the last complete copy.
        # Only do real IO when the serialized body actually changed — the in-turn
        # autosave otherwise re-serializes the same body every tick.
        if text != _read_optional(path):
            _write_durable(path.with_suffix(".json.bak"), _read_optional(path) or text)
            if not _write_durable(path, text):
                # atomic write failed — never lose the new body (best effort)
                try:
                    path.write_text(text, encoding="utf-8")
                except OSError:
                    pass
        _index_insert(session.path, session)
        clear_session_cache()
    return path


def _index_insert(path: Path, session: Session) -> None:
    """Add or update one entry in the persistent index."""
    index = _read_index()
    files: dict[str, Any] = (index or {}).get("files") or {}
    files[path.name] = {
        "id": session.id,
        "title": session.title,
        "created": session.created,
        "completed": session.completed,
        "directory": session.directory,
        "provider": session.provider,
        "model": session.model,
        "agent": session.agent,
        "parent_id": session.parent_id,
        "has_messages": bool(session.messages),
        "first_user_text": _first_user_text(session.messages),
        "mtime_ns": _stat_mtime_ns(path),
        "size": _stat_size(path),
    }
    _write_index(files)


def load_session(session_id: str) -> Session | None:
    path = session_path(session_id)
    data = _read_session_data(path)
    if data is None:
        return None
    sess = Session(data)
    # A crash left the primary body corrupt (0-byte write) and we recovered
    # from the `.bak`/`.tmp` replica — immediately restore the primary so a
    # later crash on the session doesn't fall back to a staler copy.
    text = _read_optional(path)
    if text is None:
        try:
            _write_durable(path, json.dumps(sess.to_dict(), indent=2))
        except OSError:
            pass
    return sess


# Session listing cache: keyed by path -> (mtime_ns, size, session). Session
# files only change when a turn completes, so a picker that re-opens seconds
# later can be served from cache instead of re-reading + re-parsing every file
# body. The OpenCode session picker (Ctrl+R) rebuilds the list on each open,
# and with hundreds of sessions the repeated scans showed up as a multi-second
# hang.
_session_cache: dict[str, tuple[int, int, Session]] = {}
_session_cache_loaded = False
# Reuse the last scan for a short window without re-statting every file. All
# app-side mutations go through save_session/delete_session, which invalidate
# the cache directly, so a fresh picker open within the window is served from
# memory instantly (stat'ing 800+ files on this phone's filesystem is ~0.3s).
_SESSION_LIST_TTL = 2.0
_session_cache_time = 0.0


def _index_path() -> Path:
    return GPath.sessions_dir() / ".sessions-index.json"


def _is_session_file(name: str) -> bool:
    """True for session body files (any `*.json` except the listing index and
    an in-progress tmp write)."""
    return (
        name.endswith(".json")
        and name != ".sessions-index.json"
        and not name.endswith(".json.tmp")
    )


def _read_index() -> dict[str, Any] | None:
    try:
        data = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != 1:
        return None
    return data


def _write_index(files: dict[str, Any]) -> None:
    data = {"version": 1, "files": files}
    _write_durable(_index_path(), json.dumps(data))


def _index_sessions(sessions: list[Session]) -> list[Session]:
    """Persist a metadata-only index of all sessions for fast future lists."""
    files: dict[str, Any] = {}
    for s in sessions:
        files[s.path.name] = {
            "id": s.id,
            "title": s.title,
            "created": s.created,
            "completed": s.completed,
            "directory": s.directory,
            "provider": s.provider,
            "model": s.model,
            "agent": s.agent,
            "parent_id": s.parent_id,
            "has_messages": bool(s.messages),
            "first_user_text": _first_user_text(s.messages),
            "mtime_ns": _stat_mtime_ns(s.path),
            "size": _stat_size(s.path),
        }
    _write_index(files)
    GPath.sessions_dir().mkdir(parents=True, exist_ok=True)
    return sessions


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content or "")
        text = text.strip()
        if text:
            return text[:60]
    return ""


def _stat_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _suggested(first_user: str) -> str:
    text = first_user.strip()
    return text[:60] if text else ""


def _session_from_index(name: str, meta: dict[str, Any]) -> Session | None:
    """Rebuild a listing Session from index metadata (no body read)."""
    if meta.get("id") is None:
        return None
    return Session(
        {
            "id": meta.get("id"),
            "title": meta.get("title", ""),
            "created": meta.get("created", 0.0),
            "completed": meta.get("completed"),
            "directory": meta.get("directory", ""),
            "provider": meta.get("provider", ""),
            "model": meta.get("model", ""),
            "agent": meta.get("agent", "build"),
            "parent_id": meta.get("parent_id"),
            "metadata": meta.get("metadata", {}),
            "has_messages": bool(meta.get("has_messages", True)),
            "first_user_text": meta.get("first_user_text", ""),
        }
    )


def _cache_fresh() -> bool:
    """True when the in-memory cache is fresh enough to serve.

    Within the TTL window the cache is served directly (app mutations already
    invalidate it). After the window the on-disk files are re-statted to catch
    external changes; if any differ the cache is dropped.
    """
    global _session_cache_time
    if not _session_cache_loaded or not _session_cache:
        return False
    if (time.monotonic() - _session_cache_time) < _SESSION_LIST_TTL:
        return True
    dir_path = GPath.sessions_dir()
    try:
        entries = [
            e for e in os.scandir(dir_path) if _is_session_file(e.name)
        ]
    except OSError:
        return False
    if len(entries) != len(_session_cache):
        return False
    for e in entries:
        try:
            st = e.stat()
        except OSError:
            return False
        cached = _session_cache.get(e.name)
        if cached is None or (st.st_mtime_ns, st.st_size) != (cached[0], cached[1]):
            return False
    return True


def clear_session_cache() -> None:
    """Drop the listing cache (called after saves/deletes keep it coherent)."""
    global _session_cache_loaded, _session_cache_time
    _session_cache.clear()
    _session_cache_loaded = False
    _session_cache_time = 0.0


def _cache_sessions(sessions: list[Session]) -> list[Session]:
    global _session_cache_loaded, _session_cache_time
    _session_cache.clear()
    for s in sessions:
        try:
            st = s.path.stat()
        except OSError:
            continue
        _session_cache[s.path.name] = (st.st_mtime_ns, st.st_size, s)
    _session_cache_loaded = True
    _session_cache_time = time.monotonic()
    return sessions


def list_sessions() -> list[Session]:
    if _cache_fresh():
        return sorted((e[2] for e in _session_cache.values()), key=lambda s: s.created, reverse=True)
    index = _read_index()
    sessions: list[Session] = []
    if index is not None:
        for sess in _enrich_index(index):
            sessions.append(sess)
    if sessions:
        sessions.sort(key=lambda s: s.created, reverse=True)
        return _cache_sessions(sessions)

    # Full scan fallback (also rebuilds the index for next time). Raw bodies
    # may be corrupt (crash mid-write, partial rename, hand-edit) — recover
    # from the `.bak`/`.tmp` replica when possible, and otherwise skip anything
    # unparseable instead of letting one bad file break the picker.
    dir_path = GPath.sessions_dir()
    dir_path.mkdir(parents=True, exist_ok=True)
    for path in dir_path.glob("*.json"):
        if not _is_session_file(path.name):
            continue
        data = _read_session_data(path)
        if data is None:
            continue
        try:
            sessions.append(Session(data))
        except (TypeError, ValueError, AttributeError):
            continue
    sessions.sort(key=lambda s: s.created, reverse=True)
    _index_sessions(sessions)
    return _cache_sessions(sessions)


def _enrich_index(index: dict[str, Any]) -> list[Session]:
    """Build listing Sessions from the index, including the suggested title
    hint, only when every indexed file is still present and unchanged."""
    files: dict[str, Any] = index.get("files") or {}
    out: list[Session] = []
    dir_path = GPath.sessions_dir()
    try:
        entries = {
            e.name: e
            for e in os.scandir(dir_path)
            if _is_session_file(e.name)
        }
    except OSError:
        return []
    for name, meta in files.items():
        e = entries.get(name)
        if e is None:
            return []  # index stale
        try:
            st = e.stat()
        except OSError:
            return []
        if st.st_mtime_ns != meta.get("mtime_ns") or st.st_size != meta.get("size"):
            return []  # a body changed after indexing — full rescan needed
        sess = _session_from_index(name, meta)
        if sess is None:
            return []
        out.append(sess)
    if len(entries) != len(out):
        return []  # a file exists that the index doesn't know — full rescan
    return out


def delete_session(session_id: str) -> bool:
    path = session_path(session_id)
    if path.exists():
        path.unlink()
        for extra in (path.with_suffix(".json.bak"), path.with_suffix(".json.tmp")):
            try:
                if extra.exists():
                    extra.unlink()
            except OSError:
                pass
        _fsync_dir(path.parent)
        index = _read_index()
        if index is not None:
            files = index.get("files") or {}
            if path.name in files:
                del files[path.name]
                _write_index(files)
        clear_session_cache()
        return True
    return False


def group_sessions(sessions: list[Any]) -> list[tuple[str, list[Any]]]:
    """Group sessions into opencode-style sections: ``Today``, ``Yesterday``,
    then one label per older day (``Monday, August 17``), newest-first.

    Accepts ``Session`` objects or dicts (must carry ``created``). Returns an
    ordered list of ``(label, [sessions])``; each group's sessions are
    newest-first.
    """
    import datetime

    def _created(obj: Any) -> float:
        if isinstance(obj, dict):
            return _as_number(obj.get("created"))
        return _as_number(getattr(obj, "created", None))

    now = datetime.datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - datetime.timedelta(days=1)

    buckets: dict[str, list[Any]] = {}
    order: list[str] = []
    for s in sorted(sessions, key=_created, reverse=True):
        created = _created(s)
        if created >= today_start.timestamp():
            label = "Today"
        elif created >= yesterday_start.timestamp():
            label = "Yesterday"
        else:
            # Include the YEAR: the same month/day recurs every year, and the
            # weekday repeat cycle (5/6/11 years, e.g. Aug 17 2020 = Aug 17
            # 2026 = a Monday) lets sessions years apart collide onto one label
            # and silently merge into the SAME group. Day + year is unique.
            label = datetime.datetime.fromtimestamp(created).strftime("%A, %B %d, %Y")
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(s)
    return [(label, buckets[label]) for label in order]


def suggested_title(session: Any) -> str:
    """Title for display when a session has none saved: derive it from the
    first user message (opencode's behaviour). Accepts a Session or dict."""
    if isinstance(session, dict):
        title = session.get("title", "")
        first = session.get("first_user_text", "") if not session.get("messages") else ""
        messages = session.get("messages") or []
    else:
        title = session.title or ""
        first = getattr(session, "first_user_text", "") if not getattr(session, "messages", None) else ""
        messages = session.messages or []
    if title:
        return title
    if first:
        return first.strip()[:60]
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content or "")
        if text.strip():
            return text.strip()[:60]
    return ""


def session_to_markdown(session: Session) -> str:
    """Render a session transcript to Markdown, tool calls included, for
    sharing/review files (the ``/export`` command writes this)."""
    import datetime

    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(p.get("text", ""))
                if isinstance(p, dict)
                else (
                    f"{p.get('name', '')}({p.get('arguments', '')})"
                    if isinstance(p, dict) and "name" in p
                    else str(p)
                )
                for p in content
            )
        return str(content)

    title = session.title or "Session"
    lines: list[str] = [f"# {title}", ""]
    lines += [f"- **ID**: `{session.id}`"]
    if session.directory:
        lines.append(f"- **Directory**: `{session.directory}`")
    if session.provider:
        lines.append(f"- **Provider**: {session.provider}")
    if session.model:
        lines.append(f"- **Model**: {session.model}")
    if session.agent:
        lines.append(f"- **Agent**: {session.agent}")
    if session.created:
        try:
            lines.append(f"- **Created**: {datetime.datetime.fromtimestamp(session.created).isoformat()}")
        except (OSError, ValueError, OverflowError):
            pass
    lines += ["", "---", ""]

    for msg in session.messages:
        role = msg.get("role")
        content = _text(msg.get("content", ""))
        if role == "system":
            continue
        if role == "compaction" or msg.get("compaction"):
            lines += ["## Compaction", "", content, ""]
            continue
        if role == "user":
            lines += ["## User", "", content, ""]
            continue
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            lines += ["## Assistant", ""]
            if content:
                lines.append(content)
            for call in tool_calls:
                fn = call.get("function") or {}
                try:
                    args = json.dumps(json.loads(fn.get("arguments", "{}")), indent=2)
                except json.JSONDecodeError:
                    args = str(fn.get("arguments", "{}"))
                lines += ["", f"- 🛠️ **{fn.get('name', '?')}**", "```json", args, "```"]
            reasoning = msg.get("reasoning_content")
            if reasoning:
                lines += ["", "<details><summary>Thought</summary>", "", str(reasoning), "", "</details>"]
            lines.append("")
            continue
        if role == "tool":
            lines += [
                "",
                f"**Tool result** (`{msg.get('name', '?')}`, id `{msg.get('tool_call_id', '')}`):",
                "",
                "```",
                str(content or "(no output)"),
                "```",
                "",
            ]
            continue
        lines += ["", f"**{role}**:", "", str(content), ""]

    return "\n".join(lines).rstrip() + "\n"


def new_session(
    *,
    directory: str | None = None,
    provider: str = "",
    model: str = "",
    agent: str = "build",
    title: str = "",
    parent_id: str | None = None,
) -> Session:
    return Session(
        {
            "id": uuid.uuid4().hex,
            "title": title,
            "created": time.time(),
            "directory": directory,
            "provider": provider,
            "model": model,
            "agent": agent,
            "parent_id": parent_id,
            "messages": [],
        }
    )
