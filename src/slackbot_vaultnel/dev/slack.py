"""An in-process Slack stand-in for local runs.

Only the methods ``Handlers`` and ``Renderer`` call are implemented. Messages
are stored by timestamp and replaced on ``chat_update``, so the terminal can
show exactly what Slack would display once a streamed answer is finished.
"""

from __future__ import annotations

from collections import OrderedDict

from slackbot_vaultnel.slack.render import PLACEHOLDER, strip_status


class FakeSlack:
    def __init__(self, *, user_name: str = "Local User") -> None:
        self._next_ts = 0
        self._messages: OrderedDict[str, str] = OrderedDict()
        self._ephemeral: list[str] = []
        self._reactions: list[tuple[str, str, str]] = []
        self._user_name = user_name

    async def chat_postMessage(
        self, *, channel: str, thread_ts: str | None = None, text: str
    ) -> dict[str, object]:
        ts = self._ts()
        self._messages[ts] = text
        return {"ok": True, "channel": channel, "ts": ts}

    async def chat_update(self, *, channel: str, ts: str, text: str) -> dict[str, object]:
        self._messages[ts] = text
        return {"ok": True, "channel": channel, "ts": ts}

    async def reactions_add(self, *, channel: str, timestamp: str, name: str) -> dict[str, object]:
        self._reactions.append(("add", name, timestamp))
        return {"ok": True}

    async def reactions_remove(
        self, *, channel: str, timestamp: str, name: str
    ) -> dict[str, object]:
        self._reactions.append(("remove", name, timestamp))
        return {"ok": True}

    async def users_info(self, *, user: str) -> dict[str, object]:
        return {
            "ok": True,
            "user": {
                "id": user,
                "name": "local",
                "real_name": self._user_name,
                "profile": {"display_name": self._user_name},
            },
        }

    async def chat_postEphemeral(self, *, channel: str, user: str, text: str) -> dict[str, object]:
        self._ephemeral.append(text)
        return {"ok": True}

    def take_messages(self) -> list[str]:
        """The rendered messages posted since the last call, in thread order."""

        messages: list[str] = []
        for text in self._messages.values():
            if text == PLACEHOLDER:
                continue
            cleaned = strip_status(text).strip()
            if cleaned:
                messages.append(cleaned)
        self._messages.clear()
        return messages

    def take_ephemeral(self) -> list[str]:
        messages = list(self._ephemeral)
        self._ephemeral.clear()
        return messages

    @property
    def reactions(self) -> list[tuple[str, str, str]]:
        return list(self._reactions)

    def _ts(self) -> str:
        self._next_ts += 1
        return f"dev-{self._next_ts}"
