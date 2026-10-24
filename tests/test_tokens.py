import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from slackbot_vaultnel.memory.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    count_tokens,
    tokens_for_messages,
    truncate_tokens,
)

hyp_settings.register_profile("default", max_examples=200, deadline=None)
hyp_settings.load_profile("default")


def test_empty_text_counts_zero():
    assert count_tokens("") == 0


def test_short_text_is_in_a_sane_range():
    # cl100k_base: "hello world" -> 2 tokens. Allow slack for a tokenizer swap.
    assert 1 <= count_tokens("hello world") <= 4


def test_counting_is_stable_for_the_same_input():
    assert count_tokens("the quick brown fox") == count_tokens("the quick brown fox")


def test_non_ascii_text_is_counted():
    assert count_tokens("こんにちは世界") > 0
    assert count_tokens("naïve café — ✓") > 0


def test_whitespace_only_is_counted():
    assert count_tokens("      ") >= 1


@given(st.text(max_size=120), st.data())
def test_a_prefix_costs_at_most_its_own_bytes(text: str, data: st.DataObject):
    """BPE can never emit more tokens than there are bytes.

    Note the *strict* form ``count(prefix) <= count(text)`` is false for real
    tokenizers: the pre-tokenizer regex splits on the whole string, so 'gno'
    costs 2 tokens while 'gnore' costs 1. Prefix monotonicity is therefore not
    an invariant we can rely on, only this byte bound.
    """

    cut = data.draw(st.integers(min_value=0, max_value=len(text)))
    prefix = text[:cut]
    assert count_tokens(prefix) <= len(prefix.encode("utf-8"))


@given(st.text(max_size=300))
def test_count_never_exceeds_byte_length(text: str):
    """The hard upper bound: one token per byte is the worst case."""

    assert count_tokens(text) <= len(text.encode("utf-8"))


@given(st.text(min_size=1, max_size=300))
def test_any_non_empty_text_costs_at_least_one_token(text: str):
    assert count_tokens(text) >= 1


@given(st.text(max_size=120))
def test_counts_are_never_negative(text: str):
    assert count_tokens(text) >= 0


@given(st.text(max_size=200), st.integers(min_value=0, max_value=60))
def test_truncation_respects_the_budget(text: str, budget: int):
    assert count_tokens(truncate_tokens(text, budget)) <= budget


@given(st.text(max_size=200), st.integers(min_value=1, max_value=60))
def test_truncation_is_idempotent(text: str, budget: int):
    once = truncate_tokens(text, budget)
    assert truncate_tokens(once, budget) == once


@given(st.text(max_size=200), st.integers(min_value=1, max_value=60))
def test_truncation_only_shortens(text: str, budget: int):
    result = truncate_tokens(text, budget)
    assert len(result) <= len(text) + 1  # +1 for the ellipsis


def test_short_text_passes_through_untouched():
    text = "a short sentence"

    assert truncate_tokens(text, 100) == text


def test_truncated_text_is_marked_with_an_ellipsis():
    text = "word " * 500

    result = truncate_tokens(text, 20)

    assert result.endswith("…")
    assert len(result) < len(text)


def test_zero_budget_returns_empty_string():
    assert truncate_tokens("anything at all", 0) == ""


def test_negative_budget_is_a_programming_error():
    with pytest.raises(ValueError):
        truncate_tokens("anything", -1)


def test_messages_are_summed_with_per_message_overhead():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    expected = count_tokens("hello") + count_tokens("hi there") + 2 * MESSAGE_OVERHEAD_TOKENS

    assert tokens_for_messages(messages) == expected


def test_message_tokens_ignore_missing_content():
    assert tokens_for_messages([{"role": "assistant"}]) == MESSAGE_OVERHEAD_TOKENS


@given(st.lists(st.text(max_size=40), max_size=6))
def test_more_messages_never_cost_less(contents: list[str]):
    base = [{"role": "user", "content": c} for c in contents]
    assert tokens_for_messages(base) <= tokens_for_messages(
        [*base, {"role": "user", "content": "x"}]
    )
