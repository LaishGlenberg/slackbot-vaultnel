import json

import pytest
from pydantic import ValidationError

from slackbot_vaultnel.config import (
    DEFAULT_SYSTEM_PROMPT,
    PROVIDERS,
    ConfigError,
    Provider,
    Settings,
    load_settings,
)


def build(**overrides) -> Settings:
    return load_settings(env_file=None, **overrides)


def test_minimal_env_loads_with_documented_defaults(minimal_env):
    settings = load_settings(env_file=None)

    assert settings.llm_provider is Provider.DEEPSEEK
    assert settings.resolved_model == "deepseek-chat"
    assert settings.resolved_base_url == "https://api.deepseek.com"
    assert settings.resolved_utility_model == "deepseek-chat"
    assert settings.prompt_budget_tokens == 6000
    assert settings.short_term_tokens == 2000
    assert settings.keep_recent_turns == 6
    assert settings.stream_interval == 1.0
    assert settings.thread_ttl_hours == 72
    assert settings.include_speaker_names is True


def test_missing_required_values_are_reported_by_env_name(minimal_env, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN")
    monkeypatch.delenv("LLM_API_KEY")

    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None)

    message = str(exc.value)
    assert "SLACK_BOT_TOKEN" in message
    assert "LLM_API_KEY" in message
    assert "slack_bot_token" not in message


def test_openrouter_provider_switches_base_url_and_model(minimal_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")

    settings = load_settings(env_file=None)

    assert settings.resolved_base_url == "https://openrouter.ai/api/v1"
    assert settings.resolved_model == PROVIDERS[Provider.OPENROUTER].model


def test_explicit_base_url_and_model_beat_provider_defaults(minimal_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.internal/v1")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4")

    settings = load_settings(env_file=None)

    assert settings.resolved_base_url == "https://proxy.internal/v1"
    assert settings.resolved_model == "anthropic/claude-sonnet-4"
    assert settings.resolved_utility_model == "anthropic/claude-sonnet-4"


def test_utility_model_is_independently_configurable(minimal_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("LLM_UTILITY_MODEL", "deepseek-chat")

    settings = load_settings(env_file=None)

    assert settings.resolved_model == "deepseek-reasoner"
    assert settings.resolved_utility_model == "deepseek-chat"


def test_unknown_provider_is_rejected(minimal_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None)

    assert "LLM_PROVIDER" in str(exc.value)


def test_system_prompt_defaults_to_builtin(minimal_env):
    assert build().resolved_system_prompt == DEFAULT_SYSTEM_PROMPT


def test_inline_system_prompt_overrides_builtin(minimal_env):
    assert build(system_prompt="Be terse.").resolved_system_prompt == "Be terse."


def test_system_prompt_file_beats_inline(minimal_env, tmp_path):
    prompt_file = tmp_path / "persona.md"
    prompt_file.write_text("You are a pirate.", encoding="utf-8")

    settings = build(system_prompt="ignored", system_prompt_file=prompt_file)

    assert settings.resolved_system_prompt == "You are a pirate."


def test_missing_system_prompt_file_names_the_variable(minimal_env, tmp_path):
    settings = build(system_prompt_file=tmp_path / "nope.md")

    with pytest.raises(ConfigError) as exc:
        _ = settings.resolved_system_prompt

    assert "SYSTEM_PROMPT_FILE" in str(exc.value)


def test_prompt_budget_must_exceed_short_term_window(minimal_env):
    with pytest.raises(ConfigError) as exc:
        build(prompt_budget_tokens=1500, short_term_tokens=2000)

    assert "PROMPT_BUDGET_TOKENS" in str(exc.value)


@pytest.mark.parametrize(
    "overrides, env_name",
    [
        ({"keep_recent_turns": 1}, "KEEP_RECENT_TURNS"),
        ({"stream_interval": 0.1}, "STREAM_INTERVAL"),
        ({"short_term_tokens": 0}, "SHORT_TERM_TOKENS"),
        ({"recall_top_k": -1}, "RECALL_TOP_K"),
        ({"compact_trigger_tokens": 0}, "COMPACT_TRIGGER_TOKENS"),
    ],
)
def test_out_of_range_values_are_rejected(minimal_env, overrides, env_name):
    with pytest.raises(ConfigError) as exc:
        build(**overrides)

    assert env_name in str(exc.value)


def test_redacted_summary_never_contains_full_secrets(minimal_env):
    summary = build().redacted_summary()
    dumped = json.dumps(summary)

    assert "xoxb-1234567890-abcdefghij" not in dumped
    assert "xapp-1-A0123456789-abcdefghij" not in dumped
    assert "sk-0123456789abcdef" not in dumped
    assert summary["slack_bot_token"].startswith("xoxb-1")
    assert summary["llm_provider"] == "deepseek"
    assert summary["llm_model"] == "deepseek-chat"


def test_repr_does_not_leak_secrets(minimal_env):
    text = repr(build())

    assert "sk-0123456789abcdef" not in text
    assert "xoxb-1234567890-abcdefghij" not in text


def test_env_file_is_read_when_no_variables_are_set(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SLACK_BOT_TOKEN=xoxb-file\n"
        "SLACK_APP_TOKEN=xapp-file\n"
        "LLM_API_KEY=sk-file\n"
        "LLM_PROVIDER=openrouter\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file)

    assert settings.slack_bot_token.get_secret_value() == "xoxb-file"
    assert settings.llm_provider is Provider.OPENROUTER


def test_ensure_db_dir_creates_parent(minimal_env, tmp_path):
    settings = build(db_path=tmp_path / "nested" / "dir" / "v.db")

    settings.ensure_db_dir()

    assert (tmp_path / "nested" / "dir").is_dir()


def test_every_field_has_a_uppercase_env_name(minimal_env):
    """Guards the .env.example table against drift."""

    assert {name.upper() for name in Settings.model_fields} >= {
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "LLM_API_KEY",
        "LLM_UTILITY_MODEL",
        "SHORT_TERM_TOKENS",
        "SUMMARY_MAX_TOKENS",
        "COMPACT_TRIGGER_TOKENS",
        "KEEP_RECENT_TURNS",
        "FACT_EXTRACTION_EVERY",
        "FACT_MAX_PER_SCOPE",
        "RECALL_TOP_K",
        "PROMPT_BUDGET_TOKENS",
        "THREAD_TTL_HOURS",
        "MEMORY_QUEUE_MAX",
        "DB_PATH",
        "STREAM_INTERVAL",
        "LOG_LEVEL",
    }


def test_validation_error_type_is_not_leaked_to_callers(minimal_env):
    """load_settings always raises ConfigError, never pydantic.ValidationError."""

    with pytest.raises(ConfigError) as exc:
        build(keep_recent_turns=0)

    assert not isinstance(exc.value, ValidationError)
