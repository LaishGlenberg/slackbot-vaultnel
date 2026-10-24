"""Typed configuration with fail-fast startup validation.

All settings come from environment variables (or a ``.env`` file). Validation
errors are always surfaced as :class:`ConfigError` whose message names the
*environment variables* involved, so an operator can fix the deployment without
reading Python tracebacks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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


class ThinkingLevel(StrEnum):
    """How much reasoning the model may do before answering.

    The provider spells "off" as ``none``; the other levels pass through as-is.
    """

    OFF = "off"
    LOW = "low"
    HIGH = "high"
    MAX = "max"


REASONING_EFFORT: dict[ThinkingLevel, str] = {
    ThinkingLevel.OFF: "none",
    ThinkingLevel.LOW: "low",
    ThinkingLevel.HIGH: "high",
    ThinkingLevel.MAX: "max",
}


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    model: str


PROVIDERS: dict[Provider, ProviderDefaults] = {
    Provider.DEEPSEEK: ProviderDefaults(
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    ),
    Provider.OPENROUTER: ProviderDefaults(
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
    ),
}


#: Server names become model-facing tool prefixes, so they must be a safe identifier.
MCP_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class MCPServerConfig(BaseModel):
    """One remote MCP server. Secrets are named, never inlined (spec §6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    token_env: str | None = None
    tools: tuple[str, ...] = ()

    @field_validator("url")
    @classmethod
    def _absolute_https(cls, value: str) -> str:
        parts = urlsplit(value.strip())
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("must be an absolute http(s) URL")
        if parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS:
            raise ValueError("must use https (http is allowed for localhost only)")
        return value.strip()

    @field_validator("token_env")
    @classmethod
    def _env_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _ENV_NAME_RE.match(value):
            raise ValueError("token_env must name an environment variable, not a token")
        return value

    @field_validator("tools")
    @classmethod
    def _clean_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name.strip() for name in value if name.strip())


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
    thinking_level: ThinkingLevel = ThinkingLevel.OFF

    # Persona
    system_prompt: str | None = None
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

    # Integrations (MCP): JSON object of {name: {url, token_env?, tools?}}
    mcp_servers: Annotated[dict[str, MCPServerConfig], NoDecode] = Field(default_factory=dict)
    mcp_rounds: int = Field(default=3, ge=1)
    mcp_calls_per_turn: int = Field(default=4, ge=1)
    mcp_result_tokens: int = Field(default=1200, ge=1)
    mcp_timeout_seconds: float = Field(default=30.0, gt=0)

    # Runtime
    db_path: Path = Path("data/vaultnel.db")
    max_concurrent_llm: int = Field(default=4, ge=1)
    stream_interval: float = Field(default=1.0, ge=0.3)
    log_level: str = "INFO"

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _parse_mcp_servers(cls, value: object) -> object:
        """`MCP_SERVERS` is JSON; a blank value means no integrations."""

        if isinstance(value, str):
            if not value.strip():
                return {}
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"must be a JSON object ({exc.msg} at position {exc.pos})"
                ) from None
        if isinstance(value, dict):
            for name in value:
                if not isinstance(name, str) or not MCP_SERVER_NAME_RE.match(name):
                    raise ValueError(
                        f"server name {name!r} must match {MCP_SERVER_NAME_RE.pattern} "
                        "(it becomes a tool name prefix)"
                    )
        return value

    @field_validator(
        "llm_model",
        "llm_utility_model",
        "llm_base_url",
        "system_prompt",
        "system_prompt_file",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """`KEY=` in a .env template must behave like the key being absent."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

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
    def reasoning_effort(self) -> str:
        return REASONING_EFFORT[self.thinking_level]

    @property
    def resolved_utility_model(self) -> str:
        return self.llm_utility_model or self.resolved_model

    @property
    def resolved_system_prompt(self) -> str:
        if self.system_prompt_file is not None:
            try:
                return self.system_prompt_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise ConfigError(
                    "SYSTEM_PROMPT_FILE could not be read: "
                    f"{self.system_prompt_file} ({exc.strerror})"
                ) from None
        return self.system_prompt or DEFAULT_SYSTEM_PROMPT

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
