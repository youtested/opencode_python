"""Data model for the symbol index.

IndexEntry records are intentionally small and JSON-serializable so the whole
index for a repo can be loaded/saved in one shot and random-access lookups stay
in-memory dict walks instead of disk queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Symbol:
    """A *definition* of a name in the codebase.

    ``kind`` is a stable token (function/method/class/variable/constant/
    module/interface/struct/enum/trait/type/macro/enum_member/import/package/
    arg/unknown). ``container`` is the dotted scope the symbol lives in
    (e.g. ``"SessionStore"`` for a method inside a Python class), ``signature``
    is the human-readable declaration line (``def run()`` / ``fn main()``).
    """

    name: str
    kind: str
    file: str
    line: int
    end_line: int
    signature: str = ""
    container: str = ""
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "container": self.container,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Symbol":
        return cls(
            name=str(d.get("name", "")),
            kind=str(d.get("kind", "unknown")),
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            end_line=int(d.get("end_line", 0)),
            signature=str(d.get("signature", "")),
            container=str(d.get("container", "")),
            language=str(d.get("language", "")),
        )


@dataclass
class Ref:
    """A *usage* of a name (a call, a plain read, an attribute access…).

    ``role`` distinguishes why the name was recorded: ``"call"`` (invoked as a
    function/method), ``"use"`` (identifier read), ``"import"`` or
    ``"attribute"``. Container is the enclosing function/class at that site.
    """

    name: str
    file: str
    line: int
    role: str = "use"
    container: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "role": self.role,
            "container": self.container,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ref":
        return cls(
            name=str(d.get("name", "")),
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            role=str(d.get("role", "use")),
            container=str(d.get("container", "")),
        )


@dataclass
class ImportRecord:
    """One top-level import edge of a file.

    ``module`` is the dependency as written (dotted name, file stem, or quoted
    relative path); ``local`` marks whether it resolved to a file in the same
    root (True), the stdlib (False), or is unknown (None). ``aliases`` lists
    names bound to it (``import x as y`` -> aliases=["y"]).
    """

    module: str
    file: str
    line: int
    local: bool | None = None
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "file": self.file,
            "line": self.line,
            "local": self.local,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImportRecord":
        aliases = d.get("aliases") or []
        return cls(
            module=str(d.get("module", "")),
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            local=d.get("local"),
            aliases=[str(a) for a in aliases],
        )


@dataclass
class FileIndex:
    """Per-file extraction. Everything is relative to the index root."""

    path: str  # path relative to the index root
    mtime: float
    size: int
    language: str = ""
    symbols: list[Symbol] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mtime": self.mtime,
            "size": self.size,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "refs": [r.to_dict() for r in self.refs],
            "imports": [i.to_dict() for i in self.imports],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileIndex":
        return cls(
            path=str(d.get("path", "")),
            mtime=float(d.get("mtime", 0) or 0),
            size=int(d.get("size", 0) or 0),
            language=str(d.get("language", "")),
            symbols=[Symbol.from_dict(s) for s in d.get("symbols", [])],
            refs=[Ref.from_dict(r) for r in d.get("refs", [])],
            imports=[ImportRecord.from_dict(i) for i in d.get("imports", [])],
        )


# ---------------------------------------------------------------------------
# Language registry helpers used by both the heuristic indexer and the query
# engine (for picking the right backend for a given file path).
# ---------------------------------------------------------------------------

# canonical identifier patterns, shared by the heuristic indexers
NAME_RE = "([A-Za-z_][A-Za-z0-9_$]*)"
DOTTED_RE = r"([A-Za-z_][A-Za-z0-9_$-]*(?:\.[A-Za-z0-9_-]+)*)"

# extensions -> language id (an entry in LANGUAGES)
EXTENSION_LANGS: dict[str, str] = {
    # Python (handled by the ast indexer; listed so language lookups agree)
    ".py": "python",
    ".pyi": "python",
    # JavaScript / TypeScript family
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "typescript",
    # C / C++
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cu": "cpp",
    ".cuh": "cpp",
    # JVM family
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    # Go, Rust, C#, Swift, PHP
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".swift": "swift",
    ".php": "php",
    # Shell family
    ".sh": "bash",
    ".bash": "bash",
    ".bats": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    # Lua / Ruby / Perl / R / SQL
    ".lua": "lua",
    ".rb": "ruby",
    ".rake": "ruby",
    ".pl": "perl",
    ".pm": "perl",
    ".r": "r",
    ".sql": "sql",
    # Web/doc-ish
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    ".vue": "javascript",
    ".svelte": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
}


def language_for(path: str) -> str:
    """Language id for a path, lowercased, or "" if unknown."""
    key = path.lower()
    idx = key.rfind(".")
    if idx < 0:
        # dotfiles like `.bashrc` still index as shell
        base = key.rsplit("/", 1)[-1]
        if base in (".bashrc", ".zshrc", ".profile", ".bash_profile"):
            return "bash"
        return ""
    ext = key[idx:]
    return EXTENSION_LANGS.get(ext, "")