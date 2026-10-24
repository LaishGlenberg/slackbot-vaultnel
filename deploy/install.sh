#!/usr/bin/env bash
# One-shot VM setup for Vaultnel: installs Docker, drops the updater + systemd
# timer in /opt/vaultnel, creates the .env template, then pulls the image and
# starts the bot. Safe to re-run: .env is never overwritten, the rest is.
#
#   curl -fsSLO https://raw.githubusercontent.com/LaishGlenberg/slackbot-vaultnel/main/deploy/install.sh
#   sudo bash install.sh
#   sudo nano /opt/vaultnel/.env     # paste the three required values
#   sudo bash install.sh             # pulls the image and starts the bot
set -euo pipefail

CONF_DIR="${CONF_DIR:-/opt/vaultnel}"
RAW_BASE="${RAW_BASE:-https://raw.githubusercontent.com/LaishGlenberg/slackbot-vaultnel/main/deploy}"
IMAGE="${VAULTNEL_IMAGE:-us-west2-docker.pkg.dev/vaultnel/gcloud-backend/vaultnel:latest}"
ENV_FILE="$CONF_DIR/.env"
REQUIRED=(SLACK_BOT_TOKEN SLACK_APP_TOKEN LLM_API_KEY)

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run this with sudo"

if ! command -v docker >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    say "installing docker + curl"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io curl
fi
systemctl enable --now docker

say "fetching deploy files into $CONF_DIR"
install -d -m 755 "$CONF_DIR"
for file in update.sh vaultnel-update.service vaultnel-update.timer; do
    curl -fsSL --retry 3 -o "$CONF_DIR/$file" "$RAW_BASE/$file" \
        || die "could not download $file from $RAW_BASE"
done
chmod 755 "$CONF_DIR/update.sh"

if [ ! -f "$ENV_FILE" ]; then
    say "creating $ENV_FILE"
    cat >"$ENV_FILE" <<EOF
# Vaultnel runtime config. Every option is documented in .env.example in the
# repo; only these three have no default.
SLACK_BOT_TOKEN=
SLACK_APP_TOKEN=
LLM_API_KEY=
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-flash
LLM_MAX_TOKENS=2048
LOG_LEVEL=INFO
VAULTNEL_IMAGE=$IMAGE
EOF
    chmod 600 "$ENV_FILE"
    say "fill in SLACK_BOT_TOKEN, SLACK_APP_TOKEN and LLM_API_KEY:"
    printf '    sudo nano %s\nthen re-run this script.\n' "$ENV_FILE"
    exit 0
fi

# CRLF or spaces around '=' make Docker's --env-file parser reject the file.
sed -i 's/\r$//' "$ENV_FILE"
chmod 600 "$ENV_FILE"

missing=()
for key in "${REQUIRED[@]}"; do
    value="$(grep -E "^${key}=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
    [ -n "$value" ] || missing+=("$key")
done
if [ "${#missing[@]}" -gt 0 ]; then
    die "empty in $ENV_FILE: ${missing[*]}"
fi

if ! grep -qE '^VAULTNEL_IMAGE=' "$ENV_FILE"; then
    say "adding VAULTNEL_IMAGE to $ENV_FILE"
    printf 'VAULTNEL_IMAGE=%s\n' "$IMAGE" >>"$ENV_FILE"
fi

say "installing the update timer"
install -m 644 "$CONF_DIR/vaultnel-update.service" /etc/systemd/system/
install -m 644 "$CONF_DIR/vaultnel-update.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now vaultnel-update.timer

say "pulling the image and starting the bot"
"$CONF_DIR/update.sh"

say "done. logs:"
printf '    sudo docker logs -f vaultnel\n'
