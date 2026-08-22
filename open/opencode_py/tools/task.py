"""Task tool: launch a sub-agent in its own session (mirrors opencode's task).

The tool's run() is resolved lazily from the Registry: the owning AgentLoop
installs itself as `registry.task_spawner` at construction time, so the tool
works headless (no spawner -> helpful error) and nests naturally (sub-agents
get their own registry with the same lazy hook).
"""

from __future__ import annotations

from .registry import Registry, Tool, schema_with


def tool(registry: Registry) -> Tool:
    def run(arguments: dict) -> dict:
        spawner = getattr(registry, "task_spawner", None)
        if spawner is None:
            return {
                "output": "task tool is unavailable here (no sub-agent runtime).",
                "error": True,
            }
        return spawner(arguments)

    return Tool(
        name="task",
        description=(
            "Launch a sub-agent to work on a task, streamed live in its own "
            "session (switch sessions to watch it work). Use for long-running, "
            "complex, or parallelizable work; give an isolated, self-contained "
            "prompt and a short description used as the session title."
        ),
        parameters=schema_with(
            {
                "prompt": {
                    "type": "string",
                    "description": "The instruction to the sub-agent.",
                },
                "description": {
                    "type": "string",
                    "description": "Short label shown in the UI (session title).",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "Agent type to use (build, plan, or general).",
                    "enum": ["build", "plan", "general"],
                },
            },
            required=["prompt"],
        ),
        run=run,
    )
