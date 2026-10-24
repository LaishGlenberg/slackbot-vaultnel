"""Prompts for background memory jobs (spec §5.2, §5.3).

These calls are cheap and must never leak into the reply path, so the prompts
are short, strict and ask for prose rather than JSON (except fact extraction,
which lands later).
"""

from __future__ import annotations

from collections.abc import Sequence

from slackbot_vaultnel.history.store import Turn

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


def _format_turn(turn: Turn) -> str:
    speaker = turn.display_name or turn.slack_user
    label = turn.role.capitalize() if not speaker else f"{speaker} ({turn.role})"
    return f"{label}: {turn.content}"
