"""Shared test fixtures.

Tests never read the real ``.env`` — every Settings construction passes
``_env_file=None`` and drivers use ``monkeypatch.setenv``. This fixture clears
every known variable so a stray shell export cannot change a test result.
"""

import pytest

from slackbot_vaultnel.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-1234567890-abcdefghij")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-1-A0123456789-abcdefghij")
    monkeypatch.setenv("LLM_API_KEY", "sk-0123456789abcdef")
