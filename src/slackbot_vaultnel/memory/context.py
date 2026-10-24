"""Prompt assembly: which memories survive, and in what order (spec §5.4).

This module is a pure-ish function of ``(request, store, budget)``: it reads
the thread, decides which tiers fit, and returns the message list. It never
calls the LLM, never summarizes, and never retrieves — facts and recalled
snippets arrive pre-selected on the request, which keeps retrieval (a later
slice) independent of assembly.

Two rules shape everything here:

* **Greedy in priority order.** Persona, then facts, then summary, then
  recall, then the short-term window. Lower-priority tiers get what is left.
* **The ceiling is hard.** If a tier cannot fit, it is trimmed or dropped, and
  the current message is truncated before the request is sent — a prompt that
  overflows the budget is a bug, not a warning.

Token counts come from ``Turn.tokens`` (computed at insert time), so assembly
does not re-tokenize history.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from slackbot_vaultnel.config import DEFAULT_SYSTEM_PROMPT, Settings
from slackbot_vaultnel.history.store import Store, Thread, Turn, utcnow
from slackbot_vaultnel.memory.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    count_tokens,
    tokens_for_messages,
    truncate_tokens,
)

FactScope = Literal["user", "channel", "workspace"]

#: Spec §5.4 tier ceilings. Only the overall budget is configurable; these keep
#: a single verbose tier from crowding out the others.
FACTS_TOKENS = 400
RECALL_TOKENS = 600
SNIPPET_TOKENS = 200
#: "min 4 messages" even when the short-term budget is smaller than their cost.
MIN_SHORT_TERM_MESSAGES = 4

FACT_SCOPES: tuple[FactScope, ...] = ("user", "channel", "workspace")
FACT_HEADERS: dict[FactScope, str] = {
    "user": "Known facts about this person:",
    "channel": "Known facts about this channel:",
    "workspace": "Known facts about this workspace:",
}
SUMMARY_HEADER = "Summary of this conversation so far:"
RECALL_HEADER = "Relevant messages from other conversations:"
SCOPE_RULES = (
    "Use the facts, summary and recalled messages above only when they are "
    "relevant. When they disagree with the current conversation, the current "
    "conversation wins."
)


@dataclass(frozen=True)
class PromptBudget:
    """Hard ceilings for one assembled prompt."""

    total: int = 6000
    short_term: int = 2000
    summary: int = 400
    facts: int = FACTS_TOKENS
    recall: int = RECALL_TOKENS
    min_short_term_messages: int = MIN_SHORT_TERM_MESSAGES
    thread_ttl_hours: float = 72.0

    @classmethod
    def from_settings(cls, settings: Settings) -> PromptBudget:
        return cls(
            total=settings.prompt_budget_tokens,
            short_term=settings.short_term_tokens,
            summary=settings.summary_max_tokens,
            facts=FACTS_TOKENS,
            recall=RECALL_TOKENS,
            min_short_term_messages=MIN_SHORT_TERM_MESSAGES,
            thread_ttl_hours=settings.thread_ttl_hours,
        )

    def __post_init__(self) -> None:
        if self.total <= 2 * MESSAGE_OVERHEAD_TOKENS:
            raise ValueError(
                f"prompt budget must exceed message overhead ({self.total} <= 2×"
                f"{MESSAGE_OVERHEAD_TOKENS})"
            )
        for name in ("short_term", "summary", "facts", "recall"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} budget cannot be negative")
        if self.min_short_term_messages < 0:
            raise ValueError("min_short_term_messages cannot be negative")


@dataclass(frozen=True)
class Fact:
    """A durable long-term memory. ``scope_key`` names the user/channel it belongs to."""

    content: str
    scope: FactScope = "channel"
    scope_key: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True)
class Snippet:
    """A recalled message from another thread, already ranked by the caller."""

    content: str
    display_name: str | None = None
    slack_user: str | None = None
    created_at: datetime | None = None
    channel_id: str | None = None
    thread_ts: str | None = None


@dataclass(frozen=True)
class PromptRequest:
    """Everything the assembly step needs besides the stored thread."""

    channel_id: str
    thread_ts: str
    text: str
    slack_user: str | None = None
    display_name: str | None = None
    persona: str = DEFAULT_SYSTEM_PROMPT
    include_speaker_names: bool = True
    now: datetime | None = None
    facts: Sequence[Fact] = field(default_factory=tuple)
    snippets: Sequence[Snippet] = field(default_factory=tuple)


async def build(
    request: PromptRequest,
    store: Store,
    budget: PromptBudget | None = None,
) -> list[dict[str, str]]:
    """Assemble the chat messages for one reply, never exceeding ``budget.total``."""

    budget = budget or PromptBudget()
    now = request.now or utcnow()
    thread = await store.get_or_create_thread(request.channel_id, request.thread_ts)
    fresh = _is_idle(thread, now, budget.thread_ttl_hours)

    facts_text = _facts_text(request.facts, budget.facts)
    summary_text = "" if fresh else _summary_text(thread.summary, budget.summary)
    recall_text = _recall_text(request.snippets, request, budget.recall)
    has_memory = bool(facts_text or summary_text or recall_text)

    system: list[str] = [_persona_text(request, has_memory=has_memory)]
    for text in (facts_text, summary_text, recall_text):
        if text:
            system.append(text)

    current = request.text
    current = _fit_mandatory(system, current, budget.total)
    current_msg = {"role": "user", "content": current}
    remaining = budget.total - tokens_for_messages([*_as_messages(system), current_msg])
    remaining = max(remaining, 0)

    history: list[Turn] = []
    if not fresh:
        window = await store.messages(
            thread.id,
            after_id=thread.summarized_through_msg_id,
            include_archived=False,
        )
        history = _select_window(
            window,
            slot=min(budget.short_term, remaining),
            minimum=budget.min_short_term_messages,
        )

    history_msgs = [{"role": turn.role, "content": turn.content} for turn in history]
    while (
        history_msgs
        and tokens_for_messages([*_as_messages(system), *history_msgs, current_msg]) > budget.total
    ):
        history_msgs.pop(0)

    return [*_as_messages(system), *history_msgs, current_msg]


# -- tiers ------------------------------------------------------------------


def _persona_text(request: PromptRequest, *, has_memory: bool) -> str:
    lines = [request.persona.strip() or DEFAULT_SYSTEM_PROMPT]
    if request.include_speaker_names:
        speaker = request.display_name or request.slack_user
        if speaker:
            lines.append(f"The current speaker is {speaker}.")
    if has_memory:
        lines.append(SCOPE_RULES)
    return "\n\n".join(lines)


def _facts_text(facts: Iterable[Fact], budget: int) -> str:
    if budget <= 0:
        return ""
    grouped: dict[str, list[str]] = {scope: [] for scope in FACT_SCOPES}
    for fact in facts:
        content = fact.content.strip()
        if content and fact.scope in grouped:
            grouped[fact.scope].append(f"- {content}")
    sections = [
        "\n".join([FACT_HEADERS[scope], *grouped[scope]]) for scope in FACT_SCOPES if grouped[scope]
    ]
    return truncate_tokens("\n\n".join(sections), budget) if sections else ""


def _summary_text(summary: str, budget: int) -> str:
    summary = summary.strip()
    if not summary or budget <= 0:
        return ""
    return truncate_tokens(f"{SUMMARY_HEADER}\n{summary}", budget)


def _recall_text(snippets: Iterable[Snippet], request: PromptRequest, budget: int) -> str:
    if budget <= 0:
        return ""
    lines: list[str] = []
    for snippet in snippets:
        if snippet.thread_ts and snippet.thread_ts == request.thread_ts:
            continue  # already in the short-term window (or deliberately hidden)
        author = _snippet_author(snippet, request.include_speaker_names)
        date = snippet.created_at.date().isoformat() if snippet.created_at else None
        label = f"{date} · {author}" if date and author else (date or author or "someone")
        body = truncate_tokens(snippet.content.strip(), SNIPPET_TOKENS)
        if body:
            lines.append(f"- [{label}] {body}")
    if not lines:
        return ""
    return truncate_tokens(f"{RECALL_HEADER}\n" + "\n".join(lines), budget)


def _snippet_author(snippet: Snippet, include_names: bool) -> str | None:
    if include_names and snippet.display_name:
        return snippet.display_name
    return snippet.slack_user or (snippet.display_name if include_names else None)


# -- short-term window ------------------------------------------------------


def _select_window(window: Sequence[Turn], *, slot: int, minimum: int) -> list[Turn]:
    """Newest-first selection, always at least ``minimum`` turns, oldest-first output."""

    selected: list[Turn] = []
    used = 0
    for turn in reversed(window):
        cost = turn.tokens + MESSAGE_OVERHEAD_TOKENS
        if len(selected) < minimum or used + cost <= slot:
            selected.append(turn)
            used += cost
        else:
            break
    selected.reverse()
    return selected


# -- budget fitting ---------------------------------------------------------


def _fit_mandatory(system: list[str], current: str, total: int) -> str:
    """Trim optional tiers, then the message itself, until persona+current fit.

    Mutates ``system`` in place (dropping lowest-priority tiers and, as a last
    resort, shortening the persona) and returns the possibly shortened current
    message.
    """

    def cost(text: str) -> int:
        return tokens_for_messages([*_as_messages(system), {"role": "user", "content": text}])

    while cost(current) > total and len(system) > 1:
        system.pop()  # recall, then summary, then facts — lowest priority first

    if cost(current) <= total:
        return current

    available = total - tokens_for_messages(_as_messages(system)) - MESSAGE_OVERHEAD_TOKENS
    # "" is a legitimate outcome here: no room for the question itself.
    current = truncate_tokens(current, max(available, 1))

    # Degenerate budgets: shrink the persona itself until the ceiling holds.
    while system and cost(current) > total:
        persona = system[0]
        shrink_by = max(total - cost(current), 8)
        shortened_persona = truncate_tokens(persona, max(count_tokens(persona) - shrink_by, 0))
        if shortened_persona:
            system[0] = shortened_persona
        else:
            system.pop(0)
    return current


def _as_messages(system: Sequence[str]) -> list[dict[str, str]]:
    return [{"role": "system", "content": text} for text in system]


# -- misc -------------------------------------------------------------------


def _is_idle(thread: Thread, now: datetime, ttl_hours: float) -> bool:
    updated = thread.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return now - updated > timedelta(hours=ttl_hours)
