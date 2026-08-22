"""opencode theme colors + Textual theme registration.

Mirrors opencode's theme/assets/opencode.json dark palette so the Python TUI
picks up the exact same look: background #0a0a0a, backgroundPanel #141414,
backgroundElement #1e1e1e, primary #fab283, accent #9d7cd8, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

# Official opencode "opencode" dark theme.
OPENSE_DARK: dict[str, str] = {
    "background": "#0a0a0a",
    "background_panel": "#141414",
    "background_element": "#1e1e1e",
    "background_menu": "#141414",
    "border": "#484848",
    "border_active": "#606060",
    "border_subtle": "#3c3c3c",
    "primary": "#fab283",
    "secondary": "#5c9cf5",
    "accent": "#9d7cd8",
    "error": "#e06c75",
    "warning": "#f5a742",
    "success": "#7fd88f",
    "info": "#56b6c2",
    "text": "#eeeeee",
    "text_muted": "#808080",
    # diff (official diffAdded/diffRemoved/diffContext + backgrounds)
    "diff_added": "#4fd6be",
    "diff_removed": "#c53b53",
    "diff_context": "#828bb8",
    "diff_hunk_header": "#828bb8",
    "diff_highlight_added": "#b8db87",
    "diff_highlight_removed": "#e26a75",
    "diff_added_bg": "#20303b",
    "diff_removed_bg": "#37222c",
    "diff_context_bg": "#141414",
    "diff_line_number": "#8f8f8f",
    "diff_added_line_number_bg": "#1b2b34",
    "diff_removed_line_number_bg": "#2d1f26",
    # syntax (official syntax* scopes)
    "syntax_comment": "#808080",
    "syntax_keyword": "#9d7cd8",
    "syntax_function": "#fab283",
    "syntax_variable": "#e06c75",
    "syntax_string": "#7fd88f",
    "syntax_number": "#f5a742",
    "syntax_type": "#e5c07b",
    "syntax_operator": "#56b6c2",
    "syntax_punctuation": "#eeeeee",
# markdown (official markdown* colors)
    "markdown_heading": "#9d7cd8",
    "markdown_link": "#fab283",
    "markdown_link_text": "#5c9cf5",
    "markdown_code": "#7fd88f",
    "markdown_quote": "#e5c07b",
    "markdown_strong": "#f5a742",
    "markdown_hr": "#808080",
    "markdown_list_item": "#fab283",
    "markdown_code_block": "#eeeeee",
}

# Solarized stays as a lightweight alternative.
SOLARIZED: dict[str, str] = {
    "background": "#002b36",
    "background_panel": "#073642",
    "background_element": "#073642",
    "background_menu": "#073642",
    "border": "#586e75",
    "border_active": "#839496",
    "border_subtle": "#073642",
    "primary": "#cb4b16",
    "secondary": "#2aa198",
    "accent": "#6c71c4",
    "error": "#dc322f",
    "warning": "#b58900",
    "success": "#859900",
    "info": "#2aa198",
    "text": "#839496",
    "text_muted": "#586e75",
    "diff_added_bg": "#12332a",
    "diff_removed_bg": "#3d1f26",
    "diff_context_bg": "#073642",
}

THEMES: dict[str, dict[str, str]] = {
    "opencode": OPENSE_DARK,
    "solarized": SOLARIZED,
}

TEXTUAL_THEME_NAME = "opencode_py"


# Default agent colors, matched to opencode's local agent palette (ordered by
# the built-in agents: build -> secondary blue, plan -> accent purple, ...).
AGENT_COLORS: dict[str, str] = {
    "build": "#fab283",  # build's accent is the opencode primary orange
    "plan": "#5c9cf5",   # secondary blue
    "general": "#9d7cd8",
    "test": "#7fd88f",
}

_AGENT_PALETTE = [
    "#5c9cf5",  # secondary
    "#9d7cd8",  # accent
    "#7fd88f",  # success
    "#f5a742",  # warning
    "#fab283",  # primary
    "#e06c75",  # error
    "#56b6c2",  # info
]


@dataclass
class Theme:
    name: str
    colors: dict[str, str]

    def c(self, key: str) -> str:
        return self.colors.get(key, OPENSE_DARK.get(key, "#ffffff"))

    def agent_color(self, agent: str) -> str:
        """Color accent for an agent name, matching opencode's assignment."""
        if agent in AGENT_COLORS:
            return AGENT_COLORS[agent]
        return _AGENT_PALETTE[hash(agent) % len(_AGENT_PALETTE)]


def get_theme(name: str) -> Theme:
    return Theme(name=name, colors=THEMES.get(name, OPENSE_DARK))
