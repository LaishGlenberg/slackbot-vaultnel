"""Model-based (stateful) tests for the Store protocol.

A shadow model — plain lists and counters — is updated by the same rules that
drive the store. Invariants then compare the two after every step, so any
order of append / compact / extract / archive has to keep the store honest.
The machine runs against both implementations to catch divergence between the
SQLite store and the in-memory fake the handler tests rely on.
"""

from __future__ import annotations

import asyncio
import itertools

import pytest
from fakes import InMemoryStore
from hypothesis import settings as hyp_settings
from hypothesis import stateful
from hypothesis import strategies as st

from slackbot_vaultnel.history.store import SqliteStore, StoredFact, Turn

KEEP_RECENT = 2
KEYS = [("C1", "1.0"), ("C1", "2.0"), ("C2", "1.0")]

hyp_settings.register_profile("stateful", max_examples=40, deadline=None, stateful_step_count=25)


class StoreMachine(stateful.RuleBasedStateMachine):
    def __init__(self, store: object) -> None:
        super().__init__()
        self._loop = asyncio.new_event_loop()
        self._store = store
        self._run(self._store.open())

        self._opened: set[tuple[str, str]] = set()
        self._contents: dict[tuple[str, str], list[str]] = {}
        self._ids: dict[tuple[str, str], list[int]] = {}
        self._folded: dict[tuple[str, str], int] = {}
        self._summary: dict[tuple[str, str], str] = {}
        self._extracted: dict[tuple[str, str], int] = {}
        self._archived: dict[tuple[str, str], bool] = {}
        self._hidden_until: dict[tuple[str, str], int] = {}
        self._facts: dict[tuple[str, str], list[str]] = {}

    # -- plumbing ----------------------------------------------------------

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def teardown(self) -> None:
        try:
            self._run(self._store.close())
        finally:
            self._loop.close()
        super().teardown()

    def _open(self, key: tuple[str, str]):
        thread = self._run(self._store.get_or_create_thread(*key))
        if key not in self._opened:
            self._opened.add(key)
            self._contents[key] = []
            self._ids[key] = []
            self._folded[key] = 0
            self._summary[key] = ""
            self._extracted[key] = 0
            self._archived[key] = False
            self._hidden_until[key] = 0
        elif self._archived[key]:
            # The store revived an archived thread: context starts empty from
            # the newest message, and previously archived rows stay hidden.
            self._archived[key] = False
            self._folded[key] = len(self._contents[key])
            self._summary[key] = ""
        return thread

    def _thread(self, key: tuple[str, str]):
        thread = self._run(self._store.get_thread(*key))
        assert thread is not None, f"thread {key} disappeared"
        return thread

    # -- rules -------------------------------------------------------------

    @stateful.rule(
        key=st.sampled_from(KEYS),
        text=st.text(min_size=1, max_size=20),
        tokens=st.integers(min_value=0, max_value=10),
    )
    def append_turn(self, key, text, tokens):
        thread = self._open(key)
        saved = self._run(self._store.append_message(thread.id, Turn("user", text, tokens)))
        self._contents[key].append(text)
        self._ids[key].append(saved.id)

    @stateful.rule(
        key=st.sampled_from(KEYS),
        text=st.sampled_from(["needle", "haystack", "other words"]),
    )
    def append_recallable(self, key, text):
        thread = self._open(key)
        saved = self._run(self._store.append_message(thread.id, Turn("user", text, 2)))
        self._contents[key].append(text)
        self._ids[key].append(saved.id)

    @stateful.rule(
        channel=st.sampled_from(["C1", "C2"]),
        thread_ts=st.sampled_from(["1.0", "2.0"]),
        fold=st.integers(min_value=1, max_value=3),
    )
    def compact(self, channel, thread_ts, fold):
        key = (channel, thread_ts)
        thread = self._open(key)
        start = self._folded[key]
        available = len(self._contents[key]) - start - KEEP_RECENT
        if available <= 0:
            return
        count = min(fold, available)
        folded = self._contents[key][start : start + count]
        through = self._ids[key][start + count - 1]
        summary = " | ".join(folded)
        self._run(self._store.set_summary(thread.id, summary, through))
        self._summary[key] = summary
        self._folded[key] = start + count

    @stateful.rule(
        key=st.sampled_from(KEYS),
        fold=st.integers(min_value=1, max_value=3),
    )
    def extract_facts(self, key, fold):
        thread = self._open(key)
        start = self._extracted[key]
        available = len(self._contents[key]) - start
        if available <= 0:
            return
        count = min(fold, available)
        self._run(self._store.mark_extracted(thread.id, self._ids[key][start + count - 1]))
        self._extracted[key] = start + count

    @stateful.rule(
        scope=st.sampled_from(["user", "channel", "workspace"]),
        scope_key=st.sampled_from(["U1", "C1", "*"]),
        content=st.text(min_size=1, max_size=20),
    )
    def add_fact(self, scope, scope_key, content):
        inserted = self._run(
            self._store.add_fact(StoredFact(scope=scope, scope_key=scope_key, content=content))
        )
        if inserted:
            self._facts.setdefault((scope, scope_key), []).append(content)

    @stateful.rule(key=st.sampled_from(KEYS))
    def archive(self, key):
        self._open(key)
        self._run(self._store.archive_thread(*key))
        self._archived[key] = True
        self._hidden_until[key] = len(self._contents[key])

    # -- invariants --------------------------------------------------------

    @stateful.invariant()
    def thread_metadata_matches_the_model(self):
        for key in self._opened:
            thread = self._thread(key)
            folded = self._folded[key]
            extracted = self._extracted[key]
            assert thread.summary == self._summary[key]
            assert thread.summarized_through_msg_id == (self._ids[key][folded - 1] if folded else 0)
            assert thread.last_extracted_msg_id == (
                self._ids[key][extracted - 1] if extracted else 0
            )
            assert thread.archived is self._archived[key]

    @stateful.invariant()
    def live_window_is_the_unfolded_suffix(self):
        for key in self._opened:
            thread = self._thread(key)
            window = self._run(
                self._store.messages(thread.id, after_id=thread.summarized_through_msg_id)
            )
            assert [m.content for m in window] == self._contents[key][self._folded[key] :]

    @stateful.invariant()
    def archived_rows_stay_hidden(self):
        for key in self._opened:
            thread = self._thread(key)
            visible = self._run(self._store.messages(thread.id, include_archived=False))
            assert [m.content for m in visible] == self._contents[key][self._hidden_until[key] :]

    @stateful.invariant()
    def ids_are_strictly_increasing_per_thread(self):
        for key in self._opened:
            assert self._ids[key] == sorted(set(self._ids[key]))

    @stateful.invariant()
    def message_count_matches_the_model(self):
        for key in self._opened:
            thread = self._thread(key)
            assert self._run(self._store.message_count(thread.id)) == len(self._contents[key])

    @stateful.invariant()
    def facts_match_the_model_without_duplicates(self):
        for (scope, scope_key), contents in self._facts.items():
            stored = self._run(self._store.facts(scope, scope_key))
            assert [fact.content for fact in stored] == contents
            assert len({fact.content_hash for fact in stored}) == len(stored)

    @stateful.invariant()
    def recall_only_returns_live_rows_outside_the_current_thread(self):
        for channel, thread_ts in self._opened:
            matches = self._run(
                self._store.search_messages(
                    channel_id=channel,
                    terms=["needle", "haystack", "other"],
                    exclude_thread_ts=thread_ts,
                    limit=5,
                )
            )
            for match in matches:
                hit = (match.channel_id, match.thread_ts)
                assert match.channel_id == channel
                assert hit != (channel, thread_ts)
                assert not self._archived[hit], f"archived thread {hit} leaked into recall"
                visible = self._contents[hit][self._hidden_until[hit] :]
                assert match.content in visible


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(request, tmp_path):
    counter = itertools.count()

    if request.param == "memory":
        return InMemoryStore
    return lambda: SqliteStore(tmp_path / f"stateful-{next(counter)}.db")


def _run_machine(factory):
    stateful.run_state_machine_as_test(
        lambda: StoreMachine(factory()),
        settings=hyp_settings.get_profile("stateful"),
    )


def test_store_survives_arbitrary_interleavings(store_factory):
    _run_machine(store_factory)
