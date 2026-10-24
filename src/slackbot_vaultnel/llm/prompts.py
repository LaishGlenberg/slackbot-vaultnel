"""Prompts for background memory jobs (spec §5.2, §5.3).

These calls are cheap and must never leak into the reply path, so the prompts
are short, strict and ask for prose rather than JSON (except fact extraction,
which lands later).
"""

from __future__ import annotations

from collections.abc import Sequence

from slackbot_vaultnel.history.store import Turn

EXTRACTION_SYSTEM = (
    "You extract durable facts from Slack conversations. "
    "Return ONLY a JSON array. Each item must look like "
    '{"content": "one sentence", "scope": "user"|"channel"|"workspace"}. '
    "Use user for stable details or preferences about the speaker, channel for "
    "project decisions, constraints and team context, and workspace for "
    "org-wide rules. Skip opinions, pleasantries, restatements and anything "
    "that will be stale tomorrow. Return [] when nothing is worth keeping."
)

#: Fact extraction is a utility call; a large reply is itself a bad sign.
FACT_EXTRACTION_MAX_TOKENS = 512

SUMMARY_SYSTEM = (
    "You maintain a compact rolling summary of a Slack conversation. "
    "Keep decisions, constraints, names, numbers, and open questions. "
    "Drop pleasantries, acknowledgements, and restated boilerplate. "
    "Merge the previous summary with the new turns instead of listing them. "
    "Write plain prose only; do not mention that you are summarizing."
)


def build_summary_messages(
    previous_summary: str,
    turns: Sequence[Turn],
) -> list[dict[str, str]]:
    previous = previous_summary.strip() or "(none yet)"
    body = "\n".join(_format_turn(turn) for turn in turns)
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Previous summary:\n{previous}\n\n"
                f"Older turns to fold in:\n{body}\n\n"
                "Return the updated summary."
            ),
        },
    ]


def build_extraction_messages(turns: Sequence[Turn]) -> list[dict[str, str]]:
    body = "\n".join(_format_turn(turn) for turn in turns)
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation turns:\n{body}\n\nReturn the JSON array.",
        },
    ]


def _format_turn(turn: Turn) -> str:
    speaker = turn.display_name or turn.slack_user
    label = turn.role.capitalize() if not speaker else f"{speaker} ({turn.role})"
    return f"{label}: {turn.content}"
