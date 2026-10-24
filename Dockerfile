# syntax=docker/dockerfile:1

# ---- build: resolve dependencies, install the project, warm the tokenizer ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Bake the BPE table in so the first reply does not fetch it at runtime.
RUN /app/.venv/bin/python -c \
    "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---- runtime: small, non-root, one writable volume for SQLite ----
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/app/tiktoken-cache \
    DB_PATH=/data/vaultnel.db

RUN groupadd --system vaultnel \
    && useradd --system --gid vaultnel --uid 10001 --home-dir /app vaultnel \
    && mkdir -p /data /app/tiktoken-cache \
    && chown -R vaultnel:vaultnel /data /app

WORKDIR /app
COPY --from=builder --chown=vaultnel:vaultnel /app/.venv /app/.venv
COPY --from=builder --chown=vaultnel:vaultnel /app/tiktoken-cache /app/tiktoken-cache

USER vaultnel
VOLUME ["/data"]

CMD ["python", "-m", "slackbot_vaultnel"]
