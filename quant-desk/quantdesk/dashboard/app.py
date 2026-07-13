"""Read-only web dashboard for the quant desk.

Serves the latest signals, sentiment (with tweet-level drill-down),
trade decisions, positions, and history straight from the sqlite
ledgers and snapshot files. Strictly read-only: every connection is
opened with mode=ro so the dashboard can never touch desk state.

Run: python -m quantdesk.dashboard  (port via QD_DASHBOARD_PORT, default 8420)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from quantdesk.features.sentiment import _population_mean, _population_weighted_mean

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("QD_DATA_DIR", REPO_ROOT / "data"))

app = Flask(__name__)


def _ro(db_name: str) -> sqlite3.Connection:
    path = DATA_DIR / db_name
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _payload(row: sqlite3.Row, key: str = "payload") -> dict:
    try:
        return json.loads(row[key] or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _sentiment_label(row: sqlite3.Row) -> str:
    # Label from the signed score (p_positive - p_negative) rather than argmax:
    # the scorer concentrates mass on neutral even for clearly directional
    # tweets, so argmax buries e.g. "buy more BTC" posts in the neutral bucket.
    score = row["sentiment"] or 0.0
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def _latest_snapshots() -> dict:
    """Newest market snapshot and newest sentiment snapshot from data/snapshots."""
    out = {"market": None, "sentiment": None}
    snap_dir = DATA_DIR / "snapshots"
    if not snap_dir.is_dir():
        return out
    files = sorted(snap_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        kind = "sentiment" if f.name.startswith("sentiment-") else "market"
        if out[kind] is None:
            try:
                out[kind] = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
        if out["market"] and out["sentiment"]:
            break
    return out


def _sentiment_rows(db: sqlite3.Connection, asset: str, hours: int) -> list[dict]:
    """Tweet rows shaped for the feature-pipeline aggregation helpers."""
    return [
        dict(r) | {"target_ambiguity": None}
        for r in db.execute(
            """select event_time, author_id, sentiment, author_followers
               from tweets
               where sentiment is not null and is_deleted = 0
                 and asset_mentions like ?
                 and datetime(event_time) >= datetime('now', ?)""",
            (f'%"{asset}"%', f"-{hours} hours"),
        )
    ]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/overview")
def overview():
    snaps = _latest_snapshots()
    market = snaps["market"] or {}
    sent = snaps["sentiment"] or {}

    with _ro("ledger.sqlite") as led:
        forecasts = [
            {
                "advisor_id": r["advisor_id"],
                "instrument_id": r["instrument_id"],
                "generated_at": r["generated_at"],
                **{
                    k: _payload(r).get(k)
                    for k in ("action", "raw_score", "horizon", "forecast_target",
                              "expires_at", "thesis")
                },
                "p_plus": _payload(r).get("calibrated_probability_positive"),
            }
            for r in led.execute(
                """select * from forecasts f
                   where generated_at = (select max(generated_at) from forecasts f2
                                         where f2.advisor_id = f.advisor_id
                                           and f2.instrument_id = f.instrument_id)
                   order by advisor_id, instrument_id"""
            )
        ]
        positions = [
            {
                "instrument_id": r["instrument_id"],
                "quantity": r["quantity"],
                "avg_price": r["avg_price"],
                "updated_at": r["updated_at"],
            }
            for r in led.execute("select * from positions_shadow order by instrument_id")
        ]
        halt = led.execute(
            "select * from halt_state order by changed_at desc limit 1"
        ).fetchone()
        intents = []
        for r in led.execute("select * from intents order by created_at desc limit 10"):
            p = _payload(r)
            intents.append({
                "created_at": r["created_at"],
                "instrument_id": r["instrument_id"],
                "queue_state": r["queue_state"],
                "side": p.get("side"),
                "effect": p.get("effect"),
                "quantity": p.get("quantity"),
                "order_type": p.get("order_type"),
                "limit_price": p.get("limit_price"),
                "stop_price": p.get("stop_price"),
            })
        verdicts = [
            {"evaluated_at": r["evaluated_at"], "verdict": r["verdict"],
             "reasons": _payload(r).get("reasons")}
            for r in led.execute(
                "select * from risk_verdicts order by evaluated_at desc limit 10")
        ]
        orders = []
        for r in led.execute("select * from orders order by created_at desc limit 10"):
            p = _payload(r)
            orders.append({
                "created_at": r["created_at"],
                "instrument_id": r["instrument_id"],
                "state": r["state"],
                "side": p.get("side"),
                "quantity": p.get("quantity"),
                "limit_price": p.get("limit_price"),
            })
        fills = [
            {"filled_at": r["filled_at"], "instrument_id": r["instrument_id"],
             "quantity": r["quantity"], "price": r["price"]}
            for r in led.execute("select * from fills order by filled_at desc limit 10")
        ]
        month_cost = led.execute(
            "select coalesce(sum(cost_usd), 0) from cost_log "
            "where occurred_at >= datetime('now', 'start of month')"
        ).fetchone()[0]

    with _ro("research.sqlite") as res:
        research = [
            {
                "advisor_id": r["advisor_id"],
                "instrument_id": r["instrument_id"],
                "generated_at": r["generated_at"],
                "action": ("flat" if _payload(r).get("abstain")
                           else _payload(r).get("direction")),
                "raw_score": _payload(r).get("confidence"),
                "p_plus": _payload(r).get("probability_positive"),
                "thesis": None,
            }
            for r in res.execute(
                """select * from research_forecasts f
                   where generated_at = (select max(generated_at) from research_forecasts f2
                                         where f2.advisor_id = f.advisor_id
                                           and f2.instrument_id = f.instrument_id)"""
            )
        ]

    sentiment_live = {}
    with _ro("sentiment.sqlite") as sdb:
        for asset in ("BTC", "ETH"):
            rows_24h = _sentiment_rows(sdb, asset, 24)
            cutoff_1h = datetime.now(timezone.utc) - timedelta(hours=1)
            rows_1h = [r for r in rows_24h
                       if datetime.fromisoformat(r["event_time"]) >= cutoff_1h]
            sentiment_live[asset] = {
                "mean_1h": _population_mean(rows_1h),
                "mean_24h": _population_mean(rows_24h),
                "weighted_1h": _population_weighted_mean(rows_1h),
                "weighted_24h": _population_weighted_mean(rows_24h),
                "n_24h": len(rows_24h),
            }

    return jsonify({
        "sentiment_live": sentiment_live,
        "market_features": market.get("features") or market.get("block", {}).get("features"),
        "market_generated_at": market.get("generated_at"),
        "sentiment_features": (sent.get("block") or {}).get("features") or sent.get("features"),
        "forecasts": forecasts,
        "research_forecasts": research,
        "positions": positions,
        "halt": dict(halt) if halt else None,
        "intents": intents,
        "risk_verdicts": verdicts,
        "orders": orders,
        "fills": fills,
        "llm_cost_month_usd": month_cost,
    })


@app.get("/api/tweets")
def tweets():
    asset = request.args.get("asset", "").upper()
    label = request.args.get("label", "")  # positive | negative | neutral | ""
    hours = min(int(request.args.get("hours", 168)), 24 * 90)
    limit = min(int(request.args.get("limit", 100)), 2000)
    offset = int(request.args.get("offset", 0))

    # datetime() normalizes the ISO 'T' separator so the comparison is temporal,
    # not lexicographic ('T' > ' ' would defeat the window filter otherwise).
    where = ["sentiment is not null", "is_deleted = 0",
             f"datetime(event_time) >= datetime('now', '-{hours} hours')"]
    params: list = []
    if asset in ("BTC", "ETH"):
        where.append("asset_mentions like ?")
        params.append(f'%"{asset}"%')

    with _ro("sentiment.sqlite") as db:
        rows = db.execute(
            f"""select tweet_id, event_time, author_handle, author_followers,
                       raw_text_restricted, sentiment, p_positive, p_negative,
                       p_neutral, scorer_confidence, asset_mentions,
                       like_count, retweet_count, view_count, is_retweet
                from tweets where {' and '.join(where)}
                order by event_time desc""",
            params,
        ).fetchall()

    out = []
    for r in rows:
        lab = _sentiment_label(r)
        if label and lab != label:
            continue
        out.append({
            "tweet_id": str(r["tweet_id"]),
            "url": f"https://x.com/{r['author_handle']}/status/{r['tweet_id']}",
            "event_time": r["event_time"],
            "author_handle": r["author_handle"],
            "author_followers": r["author_followers"],
            "text": r["raw_text_restricted"],
            "label": lab,
            "sentiment": r["sentiment"],
            "confidence": r["scorer_confidence"],
            "assets": json.loads(r["asset_mentions"] or "[]"),
            "likes": r["like_count"],
            "retweets": r["retweet_count"],
            "views": r["view_count"],
            "is_retweet": bool(r["is_retweet"]),
        })
    total = len(out)
    return jsonify({"total": total, "tweets": out[offset:offset + limit]})


@app.get("/api/history")
def history():
    days = min(int(request.args.get("days", 30)), 365)

    with _ro("ledger.sqlite") as led:
        forecasts = []
        for r in led.execute(
            "select * from forecasts where datetime(generated_at) >= datetime('now', ?) "
            "order by generated_at", (f"-{days} days",),
        ):
            p = _payload(r)
            forecasts.append({
                "generated_at": r["generated_at"],
                "advisor_id": r["advisor_id"],
                "instrument_id": r["instrument_id"],
                "action": p.get("action"),
                "raw_score": p.get("raw_score"),
                "p_plus": p.get("calibrated_probability_positive"),
            })
        fills = [
            {"filled_at": r["filled_at"], "instrument_id": r["instrument_id"],
             "quantity": r["quantity"], "price": r["price"],
             "side": _payload(r).get("side")}
            for r in led.execute(
                "select * from fills where datetime(filled_at) >= datetime('now', ?) "
                "order by filled_at desc", (f"-{days} days",))
        ]
        orders = []
        for r in led.execute(
            "select * from orders where datetime(created_at) >= datetime('now', ?) "
            "order by created_at desc", (f"-{days} days",),
        ):
            p = _payload(r)
            orders.append({
                "created_at": r["created_at"], "instrument_id": r["instrument_id"],
                "state": r["state"], "side": p.get("side"),
                "quantity": p.get("quantity"), "limit_price": p.get("limit_price"),
            })

    with _ro("research.sqlite") as res:
        for r in res.execute(
            "select * from research_forecasts where datetime(generated_at) >= datetime('now', ?) "
            "order by generated_at", (f"-{days} days",),
        ):
            p = _payload(r)
            forecasts.append({
                "generated_at": r["generated_at"],
                "advisor_id": r["advisor_id"] + " (research)",
                "instrument_id": r["instrument_id"],
                "action": "flat" if p.get("abstain") else p.get("direction"),
                "raw_score": p.get("confidence"),
                "p_plus": p.get("probability_positive"),
            })
    forecasts.sort(key=lambda f: f["generated_at"])

    hourly = []
    with _ro("sentiment.sqlite") as db:
        for asset in ("BTC", "ETH"):
            by_hour: dict[str, list[dict]] = {}
            for r in _sentiment_rows(db, asset, days * 24):
                hour = r["event_time"][:13].replace(" ", "T") + ":00"
                by_hour.setdefault(hour, []).append(r)
            for hour in sorted(by_hour):
                rows = by_hour[hour]
                hourly.append({
                    "hour": hour, "asset": asset, "n": len(rows),
                    "mean_sentiment": _population_mean(rows),
                    "weighted_sentiment": _population_weighted_mean(rows),
                })

    return jsonify({"forecasts": forecasts, "fills": fills, "orders": orders,
                    "sentiment_hourly": hourly})


def main() -> None:
    port = int(os.environ.get("QD_DASHBOARD_PORT", 8420))
    host = os.environ.get("QD_DASHBOARD_HOST", "127.0.0.1")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
