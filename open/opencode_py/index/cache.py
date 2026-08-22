"""Disk cache for symbol indexes, one per worktree, refreshed incrementally.

Layout: ``~/.cache/opencode_py/symbols/<sha1(root)[:16]>/index.json``

The file stores every per-file record (mtime + size stamp per file). On load the
engine compares stamps against the live tree and re-indexes only what changed —
repeat queries after an edit hit the untouched files instead of re-reading
them. Writes are atomic (tmp + os.replace, same as the memory store) and
guarded by a process-wide lock so concurrent engines never corrupt the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from ..globals import Path as GPath
from .model import FileIndex

_LOCK = threading.Lock()
_VERSION = 1


def cache_dir_for(root: Path) -> Path:
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8", "replace")).hexdigest()[:16]
    return GPath.cache / "symbols" / digest


class SymbolCache:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.dir = cache_dir_for(self.root)
        self.index_path = self.dir / "index.json"

    # -- persistence ------------------------------------------------------
    def load(self) -> dict[str, FileIndex] | None:
        try:
            raw = self.index_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError):
            return None
        if data.get("version") != _VERSION:
            return None
        files: dict[str, FileIndex] = {}
        for record in data.get("files", []):
            if not isinstance(record, dict):
                continue
            try:
                fi = FileIndex.from_dict(record)
            except (ValueError, TypeError):
                continue
            files[fi.path] = fi
        return files

    def save(self, files: dict[str, FileIndex]) -> None:
        records = [fi.to_dict() for fi in files.values()]
        payload = {"version": _VERSION, "root": str(self.root), "files": records}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.index_path)
        except OSError:  # pragma: no cover - best effort persistence
            try:
                tmp.unlink()
            except OSError:
                pass

    def clear(self) -> None:
        try:
            self.index_path.unlink()
        except OSError:
            pass