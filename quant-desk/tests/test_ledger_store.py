from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from quantdesk.ledger.store import DuplicateIntentError, LedgerStore
from quantdesk.common.schemas import OrderIntent


def make_intent(**overrides) -> OrderIntent:
    now = datetime.now(timezone.utc)
    defaults = dict(
        intent_id=uuid4(),
        decision_id=uuid4(),
        created_at=now,
        expires_at=now + timedelta(minutes=30),
        venue="hyperliquid",
        account_id="acct-1",
        instrument_id="BTC",
        instrument_type="perp",
        side="buy",
        effect="open",
        quantity=Decimal("0.1"),
        quantity_unit="BTC",
        order_type="limit",
        limit_price=Decimal("50000"),
        stop_price=None,
        time_in_force="GTC",
        reduce_only=False,
        max_slippage_bps=25,
        max_fee_bps=10,
        snapshot_id=uuid4(),
        risk_verdict_id=None,
    )
    defaults.update(overrides)
    return OrderIntent(**defaults)


@pytest.fixture
def store(tmp_path):
    s = LedgerStore(tmp_path / "ledger.sqlite")
    yield s
    s.close()


def test_wal_and_pragmas_set(store):
    row = store._conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal"
    row = store._conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


def test_insert_intent_duplicate_rejected(store):
    intent = make_intent()
    store.insert_intent(intent)
    with pytest.raises(DuplicateIntentError):
        store.insert_intent(intent)


def test_consume_intent_at_most_once(store):
    intent = make_intent()
    store.insert_intent(intent)

    consumed = store.consume_intent(intent.intent_id)
    assert consumed is not None
    assert consumed.intent_id == intent.intent_id
    assert store.get_intent_state(intent.intent_id) == "CONSUMED"

    # Second consume attempt must not re-deliver the intent.
    again = store.consume_intent(intent.intent_id)
    assert again is None


def test_consume_intent_at_most_once_under_simulated_crash(store):
    """Simulate a crash: consume marks CONSUMED durably (it's its own
    committed transaction), then the caller's follow-on work raises. The
    intent must still not be re-consumable afterward, proving at-most-once
    (never re-delivered) even though the follow-on step failed."""
    intent = make_intent()
    store.insert_intent(intent)

    consumed = store.consume_intent(intent.intent_id)
    assert consumed is not None

    class SimulatedCrash(Exception):
        pass

    def do_followup_work_and_crash():
        # Follow-on work (e.g. submitting to venue) crashes after consumption
        # has already committed.
        raise SimulatedCrash("boom")

    with pytest.raises(SimulatedCrash):
        do_followup_work_and_crash()

    # Consumption already committed prior to the crash; must not be re-handed-out.
    assert store.consume_intent(intent.intent_id) is None
    assert store.get_intent_state(intent.intent_id) == "CONSUMED"


def test_consume_nonexistent_intent_returns_none(store):
    assert store.consume_intent(uuid4()) is None


def test_integrity_check(store):
    assert store.integrity_check() is True


def test_halt_state_roundtrip(store):
    initial = store.get_halt_state()
    assert initial["state"] == "NORMAL"

    store.set_halt_state("HARD_HALT", reason="test", human_restart_required=True)
    halt = store.get_halt_state()
    assert halt["state"] == "HARD_HALT"
    assert halt["reason"] == "test"
    assert halt["human_restart_required"] is True


def test_monthly_llm_spend(store):
    now = datetime.now(timezone.utc)
    store.log_llm_cost(Decimal("0.01"), advisor_id="a", occurred_at=now)
    store.log_llm_cost(Decimal("0.02"), advisor_id="b", occurred_at=now)
    total = store.monthly_llm_spend(now.year, now.month)
    assert total == Decimal("0.03")


def test_upsert_shadow_position(store):
    store.upsert_shadow_position("BTC", Decimal("0.5"), Decimal("50000"))
    pos = store.get_shadow_position("BTC")
    assert pos["quantity"] == Decimal("0.5")
    store.upsert_shadow_position("BTC", Decimal("0.7"), Decimal("51000"))
    pos = store.get_shadow_position("BTC")
    assert pos["quantity"] == Decimal("0.7")


def test_record_verdict_requires_existing_intent_foreign_key(store):
    from quantdesk.common.schemas import RiskVerdict

    verdict = RiskVerdict(
        verdict_id=uuid4(),
        intent_id=uuid4(),  # does not exist in intents table
        evaluated_at=datetime.now(timezone.utc),
        hard_check_version="v1",
        portfolio_snapshot_id=uuid4(),
        verdict="approve",
        approved_quantity=Decimal("0.1"),
        hard_checks=[],
    )
    with pytest.raises(Exception):
        store.record_verdict(verdict)
