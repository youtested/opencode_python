"""Question tool: the model asks the user structured questions (opencode's).

Mirrors upstream `packages/opencode/src/tool/question.ts`. The run() resolves
the asker lazily from the Registry (like `task`): the owning AgentLoop installs
itself as `registry.question_asker` at construction time so the tool works
headless (no asker -> "user dismissed") and nests naturally.

The parameters/description mirror `question.txt` + the `question.v2` schema.
"""

from __future__ import annotations

from .registry import Registry, Tool, schema_with


def tool(registry: Registry) -> Tool:
    def run(arguments: dict) -> dict:
        asker = getattr(registry, "question_asker", None)
        if asker is None:
            return {
                "output": "question tool is unavailable here (no UI to ask the user).",
                "error": True,
                "denied": True,
            }
        questions = arguments.get("questions") or []
        if not isinstance(questions, list) or not questions:
            return {"output": "question: no questions provided.", "error": True}
        from ..question import parse_questions

        parsed = parse_questions(questions)
        if not parsed:
            return {"output": "question: questions must have text.", "error": True}
        try:
            answers = asker(parsed)
        except Exception as e:
            return {
                "title": f"Asked {len(parsed)} question{'s' if len(parsed) != 1 else ''}",
                "output": f"The user dismissed this question: {e}",
                "error": True,
                "denied": True,
            }
        formatted = [
            f'"{q.question}"="{"Unanswered" if not a else ", ".join(a or [])}"'
            for q, a in zip(parsed, answers)
        ]
        return {
            "title": f"Asked {len(parsed)} question{'s' if len(parsed) != 1 else ''}",
            "output": (
                "User has answered your questions: "
                + ", ".join(formatted)
                + ". You can now continue with the user's answers in mind."
            ),
            "metadata": {"answers": [list(a) for a in answers]},
        }

    return Tool(
        name="question",
        description=(
            "Use this tool when you need to ask the user questions during "
            "execution. This allows you to:\n"
            "1. Gather user preferences or requirements\n"
            "2. Clarify ambiguous instructions\n"
            "3. Get decisions on implementation choices as you work\n"
            "4. Offer choices to the user about what direction to take.\n\n"
            "Usage notes:\n"
            "- When `custom` is enabled (default), a \"Type your own answer\" "
            "option is added automatically; don't include \"Other\" or catch-all "
            "options\n"
            "- Answers are returned as arrays of labels; set `multiple: true` to "
            "allow selecting more than one\n"
            "- If you recommend a specific option, make that the first option in "
            "the list and add \"(Recommended)\" at the end of the label"
        ),
        parameters=schema_with(
            {
                "questions": {
                    "type": "array",
                    "description": "Questions to ask",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Complete question",
                            },
                            "header": {
                                "type": "string",
                                "description": "Very short label (max 30 chars)",
                            },
                            "options": {
                                "type": "array",
                                "description": "Available choices",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Display text (1-5 words, concise)",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Explanation of choice",
                                        },
                                    },
                                    "required": ["label"],
                                },
                            },
                            "multiple": {
                                "type": "boolean",
                                "description": "Allow selecting multiple choices",
                            },
                            "custom": {
                                "type": "boolean",
                                "description": "Allow typing a custom answer (default: true)",
                            },
                        },
                        "required": ["question", "options"],
                    },
                }
            },
            required=["questions"],
        ),
        run=run,
    )
