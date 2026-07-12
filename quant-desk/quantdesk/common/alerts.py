"""Discord alerting via webhook.

One notifier, three severities. P0 alerts prefix @here. If the webhook is
unset the notifier is a silent no-op — alerting must never block the desk
(plan §9: unavailability of any non-mandatory component never prevents the
desk from getting safer).
"""
from __future__ import annotations

import os

import httpx

MAX_DISCORD_LEN = 1900  # under Discord's 2000-char content limit

SEVERITY_PREFIX = {
    "info": "",
    "p1": "⚠️ **P1** ",
    "p0": "🚨 **P0** @here ",
}


class DiscordNotifier:
    def __init__(self, webhook_url: str | None = None,
                 http_client: httpx.Client | None = None):
        self.webhook_url = (webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")).strip()
        self._client = http_client

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, message: str, severity: str = "info") -> bool:
        """Post a message. Returns True on success; never raises."""
        if not self.enabled:
            return False
        content = SEVERITY_PREFIX.get(severity, "") + message
        if len(content) > MAX_DISCORD_LEN:
            content = content[: MAX_DISCORD_LEN - 25] + "\n… (truncated)"
        try:
            if self._client is not None:
                resp = self._client.post(self.webhook_url, json={"content": content})
            else:
                with httpx.Client(timeout=10.0) as c:
                    resp = c.post(self.webhook_url, json={"content": content})
            return resp.status_code in (200, 204)
        except Exception:
            return False
