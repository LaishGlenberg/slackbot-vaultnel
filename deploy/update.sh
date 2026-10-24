#!/usr/bin/env bash
# Pull the latest Vaultnel image from Artifact Registry and restart it if it changed.
# Runs on the VM, as root, from the vaultnel-update timer (or by hand).
set -euo pipefail

CONF_DIR="${CONF_DIR:-/opt/vaultnel}"
ENV_FILE="$CONF_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "missing $ENV_FILE" >&2
    exit 1
fi

IMAGE="$(grep -E '^VAULTNEL_IMAGE=' "$ENV_FILE" | tail -1 | cut -d= -f2-)"
if [ -z "$IMAGE" ]; then
    echo "VAULTNEL_IMAGE is not set in $ENV_FILE" >&2
    exit 1
fi
REGISTRY="${IMAGE%%/*}"

# The VM's service account needs roles/artifactregistry.reader. Logging in every
# run keeps the short-lived metadata token fresh.
TOKEN="$(curl -sf -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')"
printf '%s' "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "$REGISTRY" >/dev/null

docker pull --quiet "$IMAGE" >/dev/null

current="$(docker inspect --format '{{.Image}}' vaultnel 2>/dev/null || true)"
target="$(docker inspect --format '{{.Id}}' "$IMAGE")"
if [ "$current" = "$target" ]; then
    exit 0
fi

echo "updating vaultnel to $target"
docker rm -f vaultnel >/dev/null 2>&1 || true
docker run -d \
    --name vaultnel \
    --restart unless-stopped \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --env-file "$ENV_FILE" \
    -e DB_PATH=/data/vaultnel.db \
    -v vaultnel-data:/data \
    "$IMAGE" >/dev/null

# Keep the disk honest without guessing at ages: the running image is tagged,
# so deleting every dangling image except the two newest leaves exactly three
# (current + two rollback candidates) however often we deploy.
docker image ls -q --filter dangling=true | tail -n +3 \
    | xargs -r docker image rm >/dev/null 2>&1 || true
