"""Config loading / deep-merge for opencode_py.

Mirrors opencode's config model (ConfigV1.Info) without pydantic:
  - Files: opencode.json / opencode.jsonc (JSONC = JSON with comments/trailing commas)
    discovered from the project dir (walking up to worktree) then the user-level
    config dir. Later sources override earlier (deep merge).
  - Env overrides: OPENCODE_CONFIG (file), OPENCODE_CONFIG_CONTENT (raw JSON),
    OPENCODE_PERMISSION (JSON merged into permission).
  - Config variable substitution: {env:VAR}, {file:path}.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .globals import Path as GPath

_VAR_RE = re.compile(r"\{env:([A-Za-z0-9_]+)\}|\{file:([^}]+)\}")


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc(text))


def _strip_jsonc(text: str) -> str:
    """JSONC -> JSON: strip // and /* */ comments (keeps strings intact).

    Handles nested block comments and treats string escapes (including \\uXXXX
    unicode escapes) as opaque so a comment marker can never appear from inside
    an escape sequence.
    """
    out = []
    i = 0
    in_str = False
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                nxt = text[i + 1]
                out.append(nxt)
                i += 2
                # skip the rest of a \uXXXX unicode escape atomically so its
                # hex digits can never be confused with comment markers
                if nxt == "u" and i + 4 <= n:
                    for _ in range(4):
                        out.append(text[i])
                        i += 1
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                depth = 1
                i += 2
                while depth > 0 and i < n:
                    if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                        depth -= 1
                        i += 2
                    elif text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                        depth += 1
                        i += 2
                    else:
                        i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (returns a new dict). Lists are replaced."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def substitute_vars(text: str, base_dir: Path) -> str:
    """Replace {env:VAR} and {file:path} in a config string."""
    def repl(m: re.Match) -> str:
        env = m.group(1)
        if env:
            return os.environ.get(env, "")
        fpath = m.group(2)
        p = Path(fpath).expanduser()
        if not p.is_absolute():
            p = base_dir / p
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""
    return _VAR_RE.sub(repl, text)


def _apply_vars(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        return substitute_vars(value, base_dir)
    if isinstance(value, list):
        return [_apply_vars(v, base_dir) for v in value]
    if isinstance(value, dict):
        return {k: _apply_vars(v, base_dir) for k, v in value.items()}
    return value


@dataclass
class Config:
    provider: str = "opencode"
    model: str = "deepseek-v4-flash-free"
    small_model: str = "north-mini-code-free"
    default_agent: str = "build"
    subagent_depth: int = 1
    username: str = field(default_factory=lambda: os.environ.get("USER", "user"))
    instructions: list[str] = field(default_factory=list)
    permission: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, Any] = field(default_factory=dict)
    rotation: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    theme: str = "opencode"
    diff_style: str = "split"  # "split" (auto side-by-side >120 cols) | "stacked" (unified always)
    diff_wrap_mode: str = "word"  # "word" | "none"
    suppress_backgrounds: bool = False
    tool_output_max_lines: int = 2000
    tool_output_max_bytes: int = 51200
    compaction_enabled: bool = True
    compaction_tail_turns: int = 2
    context_budget: int = 120000
    bash_default_timeout: int = 120
    model_read_timeout: float = 300.0
    auto_retry: bool = True
    auto_retry_count: int = 50
    auto_approve: bool = False
    rotation_lock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path | None = None) -> "Config":
        raw = data
        cfg = cls()
        cfg.raw = raw
        base = base_dir or Path.cwd()
        if "provider" in data:
            cfg.provider = data["provider"]
        if "model" in data:
            cfg.model = _parse_model(data["model"], cfg.provider)
        if "small_model" in data:
            cfg.small_model = _parse_model(data["small_model"], cfg.provider)
        if "default_agent" in data:
            cfg.default_agent = data["default_agent"]
        if "subagent_depth" in data:
            try:
                cfg.subagent_depth = int(data["subagent_depth"])
            except (ValueError, TypeError):
                pass
        if "username" in data:
            cfg.username = data["username"]
        if "instructions" in data:
            cfg.instructions = [str(x) for x in data["instructions"]]
        if "permission" in data:
            perm = _apply_vars(data["permission"], base)
            # opencode allows `permission: "ask"` as a global default action;
            # normalize it to `{"*": <action>}` so consumer code (permission
            # engine, /config validate) always sees a dict and a malformed
            # value can never crash the engine at startup.
            if isinstance(perm, str):
                perm = {"*": perm}
            cfg.permission = perm if isinstance(perm, dict) else {}
        if "agent" in data:
            cfg.agents = _apply_vars(data["agent"], base)
        if "agents" in data:
            # user wrote the plural field name: honor it (overrides singular)
            cfg.agents = _apply_vars(data["agents"], base)
        if "provider" in data and isinstance(data.get("provider"), dict):
            # 'provider' key is the default provider id; custom providers live under
            # 'providers' to avoid ambiguity with the top-level model selection.
            pass
        if "providers" in data:
            cfg.providers = _apply_vars(data["providers"], base)
        if "commands" in data:
            cfg.commands = data["commands"]
        if "rotation" in data:
            # Sanitize so a malformed `rotation` (string, list of non-dicts /
            # entries without a provider/model) can never crash build_rotation
            # or Rotation.stream at startup. Non-dict lanes are dropped; a
            # wholly-invalid value just falls back to the default lane.
            rot = data["rotation"]
            if isinstance(rot, list):
                cfg.rotation = [
                    {k: v for k, v in lane.items() if k in ("provider", "model")}
                    for lane in rot
                    if isinstance(lane, dict) and lane.get("provider") and lane.get("model")
                ]
            else:
                cfg.rotation = []
        if "system_prompt" in data:
            cfg.system_prompt = data["system_prompt"]
        if "theme" in data:
            cfg.theme = data["theme"]
        if "diff" in data:
            d = data["diff"]
            if isinstance(d, dict):
                cfg.diff_style = str(d.get("style", cfg.diff_style))
                cfg.diff_wrap_mode = str(d.get("wrap", cfg.diff_wrap_mode))
                cfg.suppress_backgrounds = bool(d.get("suppress_backgrounds", cfg.suppress_backgrounds))
        if "tool_output" in data:
            to = data["tool_output"]
            try:
                cfg.tool_output_max_lines = int(to.get("max_lines", 2000))
            except (ValueError, TypeError):
                pass
            try:
                cfg.tool_output_max_bytes = int(to.get("max_bytes", 51200))
            except (ValueError, TypeError):
                pass
        if "context_budget" in data:
            try:
                cfg.context_budget = int(data["context_budget"])
            except (ValueError, TypeError):
                pass
        if "bash_default_timeout" in data:
            try:
                cfg.bash_default_timeout = int(data["bash_default_timeout"])
            except (ValueError, TypeError):
                pass
        if "model_read_timeout" in data:
            try:
                cfg.model_read_timeout = float(data["model_read_timeout"])
            except (ValueError, TypeError):
                pass
        if "auto_retry" in data:
            cfg.auto_retry = bool(data["auto_retry"])
        if "auto_retry_count" in data:
            try:
                cfg.auto_retry_count = int(data["auto_retry_count"])
            except (ValueError, TypeError):
                pass
        if "rotation_lock" in data:
            cfg.rotation_lock = bool(data["rotation_lock"])
        if "compaction" in data:
            c = data["compaction"]
            cfg.compaction_enabled = bool(c.get("auto", True))
            try:
                cfg.compaction_tail_turns = int(c.get("tail_turns", 2))
            except (ValueError, TypeError):
                pass
        # OPENCODE_PERMISSION env overrides permission
        if os.environ.get("OPENCODE_PERMISSION"):
            try:
                override = json.loads(os.environ["OPENCODE_PERMISSION"])
                cfg.permission = deep_merge(cfg.permission, override)
            except json.JSONDecodeError:
                pass
        return cfg

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "small_model": self.small_model,
            "default_agent": self.default_agent,
            "subagent_depth": self.subagent_depth,
            "username": self.username,
            "instructions": self.instructions,
            "permission": self.permission,
            "agent": self.agents,
            "providers": self.providers,
            "commands": self.commands,
            "rotation": self.rotation,
            "system_prompt": self.system_prompt,
            "theme": self.theme,
            "diff": {
                "style": self.diff_style,
                "wrap": self.diff_wrap_mode,
                "suppress_backgrounds": self.suppress_backgrounds,
            },
            "tool_output": {
                "max_lines": self.tool_output_max_lines,
                "max_bytes": self.tool_output_max_bytes,
            },
            "context_budget": self.context_budget,
            "bash_default_timeout": self.bash_default_timeout,
            "model_read_timeout": self.model_read_timeout,
            "auto_retry": self.auto_retry,
            "auto_retry_count": self.auto_retry_count,
            "rotation_lock": self.rotation_lock,
            "compaction": {
                "auto": self.compaction_enabled,
                "tail_turns": self.compaction_tail_turns,
            },
        }


def _parse_model(model: str, default_provider: str) -> str:
    if "/" in model:
        return model
    return f"{default_provider}/{model}"


def _project_config_files(directory: Path) -> list[Path]:
    """Find opencode.json/opencode.jsonc walking up from directory to worktree root."""
    from .globals import resolve_worktree

    worktree = resolve_worktree(directory)
    files: list[Path] = []
    d = directory.resolve()
    while True:
        for name in ("opencode.jsonc", "opencode.json"):
            p = d / name
            if p.exists():
                files.append(p)
        if d == worktree:
            break
        if d.parent == d:
            break
        d = d.parent
    return files


def _user_config_file() -> Path | None:
    for name in ("opencode.jsonc", "opencode.json"):
        p = GPath.config / name
        if p.exists():
            return p
    return None


def load_config(directory: Path | None = None) -> Config:
    """Load config following opencode's merge order (later wins)."""
    directory = directory or Path.cwd()
    merged: dict[str, Any] = {}

    sources: list[tuple[Path, dict]] = []

    def add(path: Path) -> None:
        try:
            sources.append((path, _load_json(path)))
        except (OSError, json.JSONDecodeError):
            pass

    # user-level global config first (lowest priority), then project configs
    user = _user_config_file()
    if user:
        add(user)
    for p in _project_config_files(directory):
        add(p)

    # OPENCODE_CONFIG env points to an explicit file (overrides discovered)
    if os.environ.get("OPENCODE_CONFIG"):
        add(Path(os.environ["OPENCODE_CONFIG"]).expanduser())

    # OPENCODE_CONFIG_CONTENT inline JSON (highest)
    if os.environ.get("OPENCODE_CONFIG_CONTENT"):
        try:
            sources.append((Path.cwd(), json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])))
        except json.JSONDecodeError:
            pass

    for path, data in sources:
        data = _apply_vars(data, path.parent)
        merged = deep_merge(merged, data)

    return Config.from_dict(merged, directory)


def save_config(cfg: Config, path: Path | None = None) -> None:
    """Write the merged config (used by /config and /theme).

    Unknown top-level keys from the original file (`mcpServers`, `plugins`,
    `tools`, ...) are preserved alongside the normalized fields so a settings
    rewrite never silently drops them.
    """
    path = path or (GPath.config / "opencode.json")
    out = cfg.as_dict()
    for key, value in (cfg.raw or {}).items():
        if key not in out:
            out[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
