"""Prompt assembly: tier selection and hard budget ceilings (spec §5.4).

Tests use ``InMemoryStore`` (proven equivalent to SQLite by the store contract
suite) and a fake clock, so nothing here is timing- or network-dependent.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fakes import InMemoryStore
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from slackbot_vaultnel.config import load_settings
from slackbot_vaultnel.history.store import Turn
from slackbot_vaultnel.memory.context import (
    Fact,
    PromptBudget,
    PromptRequest,
    Snippet,
    build,
)
from slackbot_vaultnel.memory.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    count_tokens,
    tokens_for_messages,
)

CHANNEL = "C0123456"
THREAD_TS = "1700000000.000100"
OTHER_THREAD = "1700000000.000200"
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def request(**overrides) -> PromptRequest:
    fields = {
        "channel_id": CHANNEL,
        "thread_ts": THREAD_TS,
        "text": "what is the plan?",
        "slack_user": "U01",
        "display_name": "Alice",
        "now": NOW,
    }
    fields.update(overrides)
    return PromptRequest(**fields)


async def seed(store, *contents: str, role: str = "user") -> list[Turn]:
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    turns = []
    for content in contents:
        turns.append(
            await store.append_message(
                thread.id,
                Turn(
                    role=role,
                    content=content,
                    tokens=count_tokens(content),
                    display_name="Alice",
                ),
            )
        )
    return turns


def total(messages) -> int:
    return tokens_for_messages(messages)


def content_of(text: str) -> str:
    return f"{text} " + "filler words here " * 20


# -- shape ------------------------------------------------------------------


async def test_minimal_prompt_is_persona_then_current_message():
    store = InMemoryStore(clock=Clock())

    messages = await build(request(), store)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[-1]["content"] == "what is the plan?"
    assert "Vaultnel" in messages[0]["content"]


async def test_speaker_name_is_resolved_into_the_system_prompt():
    store = InMemoryStore(clock=Clock())

    messages = await build(request(display_name="Alice"), store)

    assert "Alice" in messages[0]["content"]


async def test_speaker_falls_back_to_the_slack_user_id():
    store = InMemoryStore(clock=Clock())

    messages = await build(request(display_name=None, slack_user="U42"), store)

    assert "U42" in messages[0]["content"]


async def test_speaker_omitted_when_names_are_disabled_or_unknown():
    store = InMemoryStore(clock=Clock())

    disabled = await build(request(include_speaker_names=False), store)
    unknown = await build(request(display_name=None, slack_user=None), store)

    assert "Alice" not in disabled[0]["content"]
    assert "U42" not in unknown[0]["content"]


async def test_history_keeps_roles_and_ends_with_the_current_message():
    store = InMemoryStore(clock=Clock())
    await seed(store, "first", role="user")
    await seed(store, "reply", role="assistant")

    messages = await build(request(text="and then?"), store)

    assert [(m["role"], m["content"]) for m in messages[-3:]] == [
        ("user", "first"),
        ("assistant", "reply"),
        ("user", "and then?"),
    ]
    assert messages[-1]["content"] == "and then?"


# -- tiers ------------------------------------------------------------------


async def test_live_window_starts_after_the_summary_watermark():
    store = InMemoryStore(clock=Clock())
    turns = await seed(store, "old one", "old two", "recent one", "recent two")
    thread = await store.get_thread(CHANNEL, THREAD_TS)
    await store.set_summary(thread.id, "they discussed the old stuff", turns[1].id)

    messages = await build(request(), store)

    rendered = "\n".join(m["content"] for m in messages)
    assert "old one" not in rendered
    assert "old two" not in rendered
    assert "recent one" in rendered
    assert "recent two" in rendered


async def test_summary_tier_appears_when_present():
    store = InMemoryStore(clock=Clock())
    turns = await seed(store, "a", "b")
    thread = await store.get_thread(CHANNEL, THREAD_TS)
    await store.set_summary(thread.id, "they chose Fly.io with sqlite", turns[-1].id)

    messages = await build(request(), store)

    summary = [m for m in messages if "Fly.io with sqlite" in m["content"]]
    assert len(summary) == 1
    assert summary[0]["role"] == "system"


async def test_summary_is_truncated_to_its_tier_budget():
    store = InMemoryStore(clock=Clock())
    turns = await seed(store, "a")
    thread = await store.get_thread(CHANNEL, THREAD_TS)
    await store.set_summary(thread.id, "long summary " * 200, turns[-1].id)
    budget = PromptBudget(summary=25)

    messages = await build(request(), store, budget)

    summaries = [m for m in messages if "long summary" in m["content"]]
    assert len(summaries) == 1
    assert count_tokens(summaries[0]["content"]) <= 25
    assert summaries[0]["content"].endswith("…")


async def test_facts_are_grouped_user_scope_before_channel_scope():
    store = InMemoryStore(clock=Clock())
    facts = [
        Fact(content="ships on Fly.io", scope="channel"),
        Fact(content="prefers short answers", scope="user"),
        Fact(content="the team is called Vaultnel", scope="workspace"),
    ]

    messages = await build(request(facts=facts), store)

    text = "\n".join(m["content"] for m in messages)
    user_at = text.index("prefers short answers")
    channel_at = text.index("ships on Fly.io")
    workspace_at = text.index("the team is called Vaultnel")
    assert user_at < channel_at < workspace_at
    assert "Known facts" in text


async def test_facts_respect_their_tier_budget():
    store = InMemoryStore(clock=Clock())
    facts = [Fact(content=content_of(f"fact {i}")) for i in range(10)]
    budget = PromptBudget(facts=40)

    messages = await build(request(facts=facts), store, budget)

    facts_messages = [m for m in messages if "Known facts" in m["content"]]
    assert len(facts_messages) == 1
    assert count_tokens(facts_messages[0]["content"]) <= 40
    assert facts_messages[0]["content"].count("- ") < len(facts)


async def test_snippets_are_labelled_with_date_and_author():
    store = InMemoryStore(clock=Clock())
    snippets = [
        Snippet(
            content="we settled on WAL mode for sqlite",
            display_name="Bob",
            created_at=datetime(2026, 10, 20, 9, 30, tzinfo=UTC),
            thread_ts=OTHER_THREAD,
        )
    ]

    messages = await build(request(snippets=snippets), store)

    text = "\n".join(m["content"] for m in messages)
    assert "2026-10-20" in text
    assert "Bob" in text
    assert "WAL mode" in text


async def test_snippets_from_the_current_thread_are_dropped():
    store = InMemoryStore(clock=Clock())
    snippets = [
        Snippet(content="duplicate of current window", thread_ts=THREAD_TS),
        Snippet(content="genuinely from elsewhere", thread_ts=OTHER_THREAD),
    ]

    messages = await build(request(snippets=snippets), store)

    text = "\n".join(m["content"] for m in messages)
    assert "duplicate of current window" not in text
    assert "genuinely from elsewhere" in text


async def test_recall_budget_clips_each_snippet_and_the_total():
    store = InMemoryStore(clock=Clock())
    snippets = [
        Snippet(content=content_of(f"snippet {i}"), thread_ts=OTHER_THREAD) for i in range(6)
    ]
    budget = PromptBudget(recall=120)

    messages = await build(request(snippets=snippets), store, budget)

    recall = [m for m in messages if "Relevant" in m["content"]]
    assert len(recall) == 1
    assert count_tokens(recall[0]["content"]) <= 120
    assert recall[0]["content"].count("- ") < len(snippets)


async def test_tier_order_is_persona_facts_summary_recall_history_current():
    store = InMemoryStore(clock=Clock())
    turns = await seed(store, "earlier turn")
    thread = await store.get_thread(CHANNEL, THREAD_TS)
    await store.set_summary(thread.id, "summary words", turns[-1].id)
    await seed(store, "window turn")
    facts = [Fact(content="fact words", scope="user")]
    snippets = [Snippet(content="snippet words", thread_ts=OTHER_THREAD)]

    messages = await build(request(facts=facts, snippets=snippets), store)

    order = [
        next(i for i, m in enumerate(messages) if needle in m["content"])
        for needle in ("fact words", "summary words", "snippet words", "window turn")
    ]
    assert order == sorted(order)
    assert order[0] == 1  # persona is always first
    assert messages[-1]["content"] == "what is the plan?"


# -- short-term budget ------------------------------------------------------


async def test_short_term_keeps_the_newest_turns_that_fit():
    store = InMemoryStore(clock=Clock())
    for i in range(8):
        await seed(store, content_of(f"turn {i}"))
    budget = PromptBudget(short_term=80)

    messages = await build(request(), store, budget)

    kept = [m["content"] for m in messages if m["content"].startswith("turn ")]
    assert kept, "expected at least one turn to survive"
    assert kept[-1].startswith("turn 7")
    assert "turn 0" not in "\n".join(kept)


async def test_at_least_four_turns_survive_a_tiny_short_term_budget():
    store = InMemoryStore(clock=Clock())
    for i in range(6):
        await seed(store, f"turn {i}")
    budget = PromptBudget(short_term=1)

    messages = await build(request(), store, budget)

    kept = [m["content"] for m in messages if m["content"].startswith("turn ")]
    assert kept == ["turn 2", "turn 3", "turn 4", "turn 5"]


async def test_history_is_a_contiguous_suffix_of_the_live_window():
    store = InMemoryStore(clock=Clock())
    for i in range(10):
        await seed(store, f"turn {i}")
    budget = PromptBudget(short_term=40)

    messages = await build(request(), store, budget)

    kept = [m["content"] for m in messages if m["content"].startswith("turn ")]
    expected = [f"turn {i}" for i in range(10 - len(kept), 10)]
    assert kept == expected


# -- soft reset -------------------------------------------------------------


async def test_idle_thread_past_the_ttl_starts_with_a_fresh_context():
    clock = Clock()
    store = InMemoryStore(clock=clock)
    turns = await seed(store, "stale turn")
    thread = await store.get_thread(CHANNEL, THREAD_TS)
    await store.set_summary(thread.id, "stale summary", turns[-1].id)
    later = NOW + timedelta(hours=73)

    messages = await build(request(now=later), store)

    rendered = "\n".join(m["content"] for m in messages)
    assert "stale turn" not in rendered
    assert "stale summary" not in rendered
    assert messages[-1]["content"] == "what is the plan?"


async def test_idle_thread_still_gets_facts_and_recall():
    clock = Clock()
    store = InMemoryStore(clock=clock)
    await seed(store, "stale turn")
    later = NOW + timedelta(hours=73)

    messages = await build(
        request(
            now=later,
            facts=[Fact(content="durable fact", scope="user")],
            snippets=[Snippet(content="old snippet", thread_ts=OTHER_THREAD)],
        ),
        store,
    )

    text = "\n".join(m["content"] for m in messages)
    assert "durable fact" in text
    assert "old snippet" in text


# -- hard ceiling -----------------------------------------------------------


async def test_budget_maps_from_settings(minimal_env):
    settings = load_settings(env_file=None)

    budget = PromptBudget.from_settings(settings)

    assert budget.total == settings.prompt_budget_tokens
    assert budget.short_term == settings.short_term_tokens
    assert budget.summary == settings.summary_max_tokens
    assert budget.facts == 400
    assert budget.recall == 600
    assert budget.min_short_term_messages == 4
    assert budget.thread_ttl_hours == settings.thread_ttl_hours


async def test_giant_current_message_is_truncated_not_dropped():
    store = InMemoryStore(clock=Clock())
    budget = PromptBudget(total=200, short_term=50)

    messages = await build(request(text="question " * 2000), store, budget)

    assert messages[-1]["role"] == "user"
    assert count_tokens(messages[-1]["content"]) > 0
    assert total(messages) <= 200


def random_text(max_words: int = 30) -> st.SearchStrategy[str]:
    return st.lists(
        st.sampled_from(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
        min_size=1,
        max_size=max_words,
    ).map(" ".join)


@hyp_settings(max_examples=80, deadline=None)
@given(
    history=st.lists(random_text(40), max_size=12),
    summary=st.one_of(st.none(), random_text(80)),
    facts=st.lists(random_text(20), max_size=6),
    snippets=st.lists(random_text(60), max_size=5),
    question=random_text(80),
)
@pytest.mark.asyncio
async def test_prompt_never_exceeds_the_ceiling(history, summary, facts, snippets, question):
    budget = PromptBudget(
        total=180,
        short_term=60,
        summary=30,
        facts=30,
        recall=40,
    )
    store = InMemoryStore(clock=Clock())
    turns = []
    for i, content in enumerate(history):
        turns.extend(
            await seed(store, f"m{i} {content}", role="user" if i % 2 == 0 else "assistant")
        )
    if summary is not None and turns:
        thread = await store.get_thread(CHANNEL, THREAD_TS)
        await store.set_summary(thread.id, summary, turns[-1].id)

    messages = await build(
        request(
            text=question,
            facts=[Fact(content=f"fact {i} {text}") for i, text in enumerate(facts)],
            snippets=[Snippet(content=text, thread_ts=OTHER_THREAD) for text in snippets],
        ),
        store,
        budget,
    )

    assert total(messages) <= budget.total
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].strip()
    assert messages[0]["role"] == "system"


@hyp_settings(max_examples=80, deadline=None)
@given(
    history=st.lists(random_text(40), min_size=1, max_size=12),
    question=random_text(40),
)
@pytest.mark.asyncio
async def test_window_stays_a_suffix_under_any_budget(history, question):
    budget = PromptBudget(total=300, short_term=70)
    store = InMemoryStore(clock=Clock())
    contents = [f"m{i} {content}" for i, content in enumerate(history)]
    for content in contents:
        await seed(store, content)

    messages = await build(request(text=question), store, budget)

    kept = [m["content"] for m in messages if m["content"].startswith("m")]
    assert kept == contents[len(contents) - len(kept) :]
    assert total(messages) <= 300
    assert messages[-1]["content"] == question
    # overhead bookkeeping matches the estimator
    assert total(messages) == sum(
        count_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages
    )


def test_degenerate_budget_is_rejected():
    with pytest.raises(ValueError, match="overhead"):
        PromptBudget(total=8)

    with pytest.raises(ValueError, match="negative"):
        PromptBudget(short_term=-1)
