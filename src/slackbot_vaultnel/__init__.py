"""Vaultnel: a Slack AI assistant with tiered conversation memory."""

__all__ = ["main"]


def main() -> int:
    from slackbot_vaultnel.__main__ import main as entrypoint

    return entrypoint()
