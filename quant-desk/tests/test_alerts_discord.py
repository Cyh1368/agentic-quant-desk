import httpx

from quantdesk.common.alerts import MAX_DISCORD_LEN, DiscordNotifier

WEBHOOK = "https://discord.com/api/webhooks/123/abc"


def _notifier(handler):
    return DiscordNotifier(WEBHOOK, http_client=httpx.Client(
        transport=httpx.MockTransport(handler)))


def test_send_posts_content_with_severity_prefix():
    seen = {}

    def handler(request):
        seen["json"] = request.read()
        return httpx.Response(204)

    n = _notifier(handler)
    assert n.send("unprotected position BTC", severity="p0")
    assert b"@here" in seen["json"] and b"unprotected position BTC" in seen["json"]


def test_disabled_without_webhook(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    n = DiscordNotifier("")
    assert not n.enabled
    assert n.send("anything") is False


def test_send_never_raises_and_truncates():
    def handler(request):
        raise httpx.ConnectError("down")

    n = _notifier(handler)
    assert n.send("x") is False  # network failure swallowed

    def ok(request):
        body = request.read().decode()
        assert len(body) < MAX_DISCORD_LEN + 200
        return httpx.Response(204)

    n2 = _notifier(ok)
    assert n2.send("y" * 5000)
