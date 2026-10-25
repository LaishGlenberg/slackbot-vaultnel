"""``deploy/update.sh`` behaviour, exercised against stub ``docker``/``curl``.

The script runs on the VM as root, but its decisions are plain shell, so the
tests drive it with a fake ``docker`` on ``PATH`` that records every call. That
keeps the "exactly one image" invariant honest without a daemon.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "update.sh"

IMAGE = "us-west2-docker.pkg.dev/vaultnel/gcloud-backend/vaultnel:latest"

DOCKER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_LOG"
cmd="$1"
shift || true
case "$cmd" in
    login) cat >/dev/null ;;
    inspect)
        case "$2" in
            *Image*) echo "sha256:running" ;;
            *Id*) echo "sha256:${STUB_TARGET:-running}" ;;
        esac
        ;;
esac
exit 0
"""

CURL_STUB = """#!/usr/bin/env bash
cat >/dev/null
printf '%s' '{"access_token":"fake-token"}'
"""


def _run(tmp_path: Path, target: str) -> list[str]:
    """Run update.sh with the container's image matching or not; return calls."""
    conf = tmp_path / "opt"
    conf.mkdir()
    (conf / ".env").write_text(f"VAULTNEL_IMAGE={IMAGE}\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("curl", CURL_STUB)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)

    log = tmp_path / "docker.log"
    log.touch()

    env = dict(os.environ)
    env.update(
        CONF_DIR=str(conf),
        PATH=f"{bin_dir}:{env['PATH']}",
        DOCKER_LOG=str(log),
        STUB_TARGET=target,
    )
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, capture_output=True)
    return log.read_text().splitlines()


def test_no_op_run_still_prunes_images(tmp_path: Path) -> None:
    """The timer fires every 5 minutes and almost always finds nothing to do.

    Exiting early must not skip cleanup: an untagged image left behind by an
    interrupted run would otherwise never be reclaimed.
    """
    calls = _run(tmp_path, target="running")

    assert not any(call.startswith(("run ", "rm ")) for call in calls)
    assert "image prune -a -f" in calls


def test_changed_image_restarts_and_prunes(tmp_path: Path) -> None:
    calls = _run(tmp_path, target="fresh")

    assert "rm -f vaultnel" in calls
    assert any(call.startswith("run -d ") for call in calls)
    assert calls.count("image prune -a -f") >= 1
