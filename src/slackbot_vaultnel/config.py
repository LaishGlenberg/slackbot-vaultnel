"""Typed configuration with fail-fast startup validation.

All settings come from environment variables (or a ``.env`` file). Validation
errors are always surfaced as :class:`ConfigError` whose message names the
*environment variables* involved, so an operator can fix the deployment without
reading Python tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = (
    "You are Vaultnel, a helpful assistant in a Slack workspace. "
    "Lead with the answer, then the supporting detail. Keep replies short and "
    "skimmable, use Slack markdown, and keep code snippets minimal but runnable. "
    "If you are unsure, say so instead of inventing details."
)


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class Provider(StrEnum):
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    model: str


PROVIDERS: dict[Provider, ProviderDefaults] = {
    Provider.DEEPSEEK: ProviderDefaults(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    ),
    Provider.OPENROUTER: ProviderDefaults(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Slack
    slack_bot_token: SecretStr
    slack_app_token: SecretStr
    slack_signing_secret: SecretStr = SecretStr("")

    # LLM
    llm_provider: Provider = Provider.DEEPSEEK
    llm_api_key: SecretStr
    llm_model: str | None = None
    llm_utility_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Persona
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    system_prompt_file: Path | None = None
    include_speaker_names: bool = True

    # Memory tiers
    short_term_tokens: int = Field(default=2000, ge=1)
    summary_max_tokens: int = Field(default=400, ge=1)
    compact_trigger_tokens: int = Field(default=2500, ge=1)
    keep_recent_turns: int = Field(default=6, ge=2)
    fact_extraction_every: int = Field(default=4, ge=1)
    fact_max_per_scope: int = Field(default=100, ge=1)
    recall_top_k: int = Field(default=3, ge=1)
    prompt_budget_tokens: int = Field(default=6000, ge=1)
    thread_ttl_hours: float = Field(default=72.0, gt=0)
    memory_queue_max: int = Field(default=100, ge=1)

    # Runtime
    db_path: Path = Path("data/vaultnel.db")
    max_concurrent_llm: int = Field(default=4, ge=1)
    stream_interval: float = Field(default=1.0, ge=0.3)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_budget_headroom(self) -> Settings:
        if self.prompt_budget_tokens <= self.short_term_tokens:
            raise ValueError(
                "PROMPT_BUDGET_TOKENS must be greater than SHORT_TERM_TOKENS "
                f"({self.prompt_budget_tokens} <= {self.short_term_tokens})"
            )
        if self.compact_trigger_tokens <= self.summary_max_tokens:
            raise ValueError(
                "COMPACT_TRIGGER_TOKENS must be greater than SUMMARY_MAX_TOKENS "
                f"({self.compact_trigger_tokens} <= {self.summary_max_tokens})"
            )
        return self

    @property
    def resolved_base_url(self) -> str:
        return self.llm_base_url or PROVIDERS[self.llm_provider].base_url

    @property
    def resolved_model(self) -> str:
        return self.llm_model or PROVIDERS[self.llm_provider].model

    @property
    def resolved_utility_model(self) -> str:
        return self.llm_utility_model or self.resolved_model

    @property
    def resolved_system_prompt(self) -> str:
        if self.system_prompt_file is None:
            return self.system_prompt
        try:
            return self.system_prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(
                f"SYSTEM_PROMPT_FILE could not be read: {self.system_prompt_file} ({exc.strerror})"
            ) from None

    def ensure_db_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def redacted_summary(self) -> dict[str, str | int]:
        """Safe to log: secrets reduced to an identifying prefix."""

        return {
            "llm_provider": str(self.llm_provider),
            "llm_model": self.resolved_model,
            "llm_base_url": self.resolved_base_url,
            "slack_bot_token": _redact(self.slack_bot_token),
            "slack_app_token": _redact(self.slack_app_token),
            "llm_api_key": _redact(self.llm_api_key),
            "db_path": str(self.db_path),
            "prompt_budget_tokens": self.prompt_budget_tokens,
        }


def _redact(secret: SecretStr) -> str:
    value = secret.get_secret_value()
    if not value:
        return "<unset>"
    return f"{value[:6]}…"


def _format_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors(include_url=False):
        location = error["loc"]
        name = str(location[0]).upper() if location else "settings"
        lines.append(f"  {name}: {error['msg']}")
    return "Invalid configuration:\n" + "\n".join(lines)


def load_settings(env_file: str | Path | None = ".env", **overrides: object) -> Settings:
    """Build settings, translating pydantic errors into operator-facing ones."""

    try:
        return Settings(_env_file=env_file, **overrides)
    except ValidationError as exc:
        raise ConfigError(_format_error(exc)) from None
