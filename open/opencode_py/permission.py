"""Permission engine: ask / allow / deny with pattern rules.

Mirrors opencode: rule = {permission, pattern, action}; LAST matching rule wins;
default action is "ask". Patterns: * -> .*, ? -> ., anchored, "/" normalization.
Headless default: auto-approve unless the rule denies (opencode --auto) OR a mode
is explicitly set to always-ask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# how many identical tool calls with identical input before it's a "doom loop"
DOOM_LOOP_THRESHOLD = 3
# cap on remembered "always allow" patterns (LRU-evicted, newest wins)
MAX_APPROVED_PATTERNS = 256


@dataclass
class Rule:
    permission: str
    pattern: str
    action: str  # allow | ask | deny


@dataclass
class PermissionEngine:
    rules: list[Rule] = field(default_factory=list)
    mode: str = "auto"  # auto | ask | deny
    # callback(user-facing description, always_patterns) -> "once"|"always"|"reject"
    ask_callback: Callable[[str, list[str]], str] | None = None
    _approved_patterns: list[str] = field(default_factory=list)
    _last_calls: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        mode: str = "auto",
        ask_callback: Callable[[str, list[str]], str] | None = None,
    ) -> "PermissionEngine":
        rules: list[Rule] = []
        for perm, value in config.items():
            if perm == "_comment":
                continue
            if isinstance(value, str):
                rules.append(Rule(permission=perm, pattern="*", action=value))
            elif isinstance(value, dict):
                for pattern, action in value.items():
                    if isinstance(action, str):
                        rules.append(Rule(permission=perm, pattern=pattern, action=action))
        engine = cls(rules=rules, mode=mode, ask_callback=ask_callback)
        engine.apply_defaults()
        return engine

    def apply_defaults(self) -> None:
        """opencode defaults: * allow, doom_loop ask, question deny."""
        defaults: dict[str, Any] = {
            "*": "allow",
            "doom_loop": "ask",
            "question": "deny",
            "plan_enter": "deny",
            "plan_exit": "deny",
            "read": {"*": "allow", "*.env": "ask", "*.env.*": "ask", "*.env.example": "allow"},
        }
        for perm, value in defaults.items():
            if self._has_permission(perm):
                continue
            if isinstance(value, str):
                self.rules.append(Rule(permission=perm, pattern="*", action=value))
            else:
                for pattern, action in value.items():
                    self.rules.append(Rule(permission=perm, pattern=pattern, action=action))

    def _has_permission(self, perm: str) -> bool:
        return any(r.permission == perm for r in self.rules)

    # -- pattern matching -------------------------------------------------
    @staticmethod
    def match(pattern: str, value: str) -> bool:
        """Wildcard match: * -> .*, ? -> ., anchored. '/' normalized."""
        pattern = pattern.replace("\\", "/")
        value = value.replace("\\", "/")
        # trailing ' *' matches "tool" or "tool <anything>"
        if pattern.endswith(" *"):
            base = pattern[:-2]
            base = re.escape(base).replace(r"\*", ".*").replace(r"\?", ".")
            regex = "^" + base + "( .*)?$"
        else:
            escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
            regex = "^" + escaped + "$"
        try:
            return re.match(regex, value) is not None
        except re.error:
            return False

    def evaluate(self, permission: str, input_value: str = "") -> str:
        """Return the action for a permission+input: allow | ask | deny."""
        # doom loop detection (bash tools repeat with identical input)
        if permission in ("bash", "write", "edit"):
            self._last_calls.append((permission, input_value))
            if len(self._last_calls) > DOOM_LOOP_THRESHOLD:
                self._last_calls.pop(0)
            recent = self._last_calls[-DOOM_LOOP_THRESHOLD:]
            if len(recent) == DOOM_LOOP_THRESHOLD and len({c for c in recent}) == 1:
                r = self._find_action("doom_loop", input_value)
                if r != "allow":
                    return r

        action = self._find_action(permission, input_value)
        if action == "ask":
            if self.mode == "deny":
                return "deny"
            if self.mode == "auto":
                return "allow"  # headless default: auto-approve (unless denied)
            return "ask"
        return action

    def _find_action(self, permission: str, input_value: str) -> str:
        # check explicit rules (last match wins), then "*" rule, then "ask"
        explicit = [r for r in self.rules if r.permission == permission]
        for r in reversed(explicit):
            if self.match(r.pattern, input_value):
                return r.action
        wildcard = [r for r in self.rules if r.permission == "*"]
        for r in reversed(wildcard):
            if self.match(r.pattern, input_value):
                return r.action
        return "ask"

    def ask(self, description: str, always_patterns: list[str] | None = None) -> bool:
        """Interactively ask the user; returns True if allowed."""
        always_patterns = always_patterns or []
        # if an always-allow pattern already approved, skip. Approved entries are
        # exact "permission input" strings; check literal equality FIRST so a
        # user-supplied command containing wildcard characters ("rm /*") can't
        # accidentally approve a broader pattern via the glob matcher below.
        for approved in self._approved_patterns:
            if any(p == approved for p in always_patterns):
                return True
            if any(self.match(approved, p) or self.match(p, approved) for p in always_patterns):
                return True
        if self.ask_callback is None:
            return True
        reply = self.ask_callback(description, always_patterns)
        if reply == "always":
            for p in always_patterns:
                # don't store megabyte-sized tool inputs as approval patterns
                if len(p) > 512:
                    continue
                if p in self._approved_patterns:
                    # refresh LRU position by re-appending (dedup first)
                    self._approved_patterns.remove(p)
                self._approved_patterns.append(p)
            # bounded memory: evict the oldest "always allow" approvals so a
            # long session's approval list never grows without limit.
            while len(self._approved_patterns) > MAX_APPROVED_PATTERNS:
                self._approved_patterns.pop(0)
            return True
        return reply == "once"

    def reset_doom_tracking(self) -> None:
        self._last_calls = []


def default_permissions(agent: str = "build") -> dict[str, Any]:
    """Mirror opencode's per-agent permission configs."""
    base: dict[str, Any] = {
        "*": "allow",
        "external_directory": {"*": "ask"},
        "read": {"*": "allow", "*.env": "ask", "*.env.*": "ask", "*.env.example": "allow"},
    }
    if agent == "build":
        base["question"] = "allow"
        base["plan_enter"] = "allow"
    elif agent == "plan":
        base["question"] = "allow"
        base["plan_exit"] = "allow"
        base["task"] = {"general": "deny"}
        # plan agent is read-only: deny every mutating tool (defense-in-depth;
        # the loop also filters these from the model's tool schemas)
        base["edit"] = {"*": "deny"}
        base["write"] = {"*": "deny"}
        base["bash"] = {"*": "deny"}
        base["apply_patch"] = {"*": "deny"}
    return base


def merge_permissions(user: dict[str, Any], agent: str = "build") -> dict[str, Any]:
    """Merge user config permission over the agent defaults."""
    defaults = default_permissions(agent)
    merged = dict(defaults)
    for perm, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(perm), dict):
            d = dict(merged[perm])
            d.update(value)
            merged[perm] = d
        else:
            merged[perm] = value
    return merged
