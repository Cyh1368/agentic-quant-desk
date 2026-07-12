from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantdesk.ledger.store import LedgerStore
from quantdesk.risk import halts

CONFIG = {
    "daily_loss_soft_halt_pct": 0.75,
    "daily_loss_hard_halt_pct": 1.25,
    "weekly_loss_hard_halt_pct": 2.5,
    "max_drawdown_kill_pct": 4,
    "human_restart_required_after_hard_halt": True,
}


@pytest.fixture
def store(tmp_path):
    s = LedgerStore(tmp_path / "ledger.sqlite")
    yield s
    s.close()


def test_initial_state_normal(store):
    assert halts.current_state(store)["state"] == halts.NORMAL


def test_allows_reduce_and_close_always_true(store):
    for state in (halts.NORMAL, halts.SOFT_HALT, halts.HARD_HALT, halts.KILL):
        store.set_halt_state(state)
        assert halts.allows(store, "reduce") is True
        assert halts.allows(store, "close") is True


def test_allows_open_only_when_normal(store):
    store.set_halt_state(halts.NORMAL)
    assert halts.allows(store, "open") is True
    store.set_halt_state(halts.SOFT_HALT)
    assert halts.allows(store, "open") is False
    store.set_halt_state(halts.HARD_HALT)
    assert halts.allows(store, "open") is False
    store.set_halt_state(halts.KILL)
    assert halts.allows(store, "open") is False


def test_daily_loss_soft_threshold_triggers_soft_halt(store):
    halts.apply_daily_pnl(store, Decimal("-0.80"), None, Decimal("0"), CONFIG)
    assert store.get_halt_state()["state"] == halts.SOFT_HALT


def test_daily_loss_hard_threshold_triggers_hard_halt(store):
    result = halts.apply_daily_pnl(store, Decimal("-1.30"), None, Decimal("0"), CONFIG)
    assert result["state"] == halts.HARD_HALT
    assert result["human_restart_required"] is True


def test_weekly_loss_triggers_hard_halt(store):
    result = halts.apply_daily_pnl(store, Decimal("0"), Decimal("-2.6"), Decimal("0"), CONFIG)
    assert result["state"] == halts.HARD_HALT


def test_drawdown_triggers_kill(store):
    result = halts.apply_daily_pnl(store, Decimal("0"), Decimal("0"), Decimal("-4.5"), CONFIG)
    assert result["state"] == halts.KILL


def test_kill_is_terminal_and_not_lifted_by_pnl(store):
    store.set_halt_state(halts.KILL, reason="test kill")
    result = halts.apply_daily_pnl(store, Decimal("0"), Decimal("0"), Decimal("0"), CONFIG)
    assert result["state"] == halts.KILL


def test_hard_halt_requires_human_restart_not_autoclear(store):
    halts.apply_daily_pnl(store, Decimal("-1.30"), None, Decimal("0"), CONFIG)
    assert store.get_halt_state()["state"] == halts.HARD_HALT
    # Good PnL alone should not clear a hard halt.
    result = halts.apply_daily_pnl(store, Decimal("0.5"), None, Decimal("0"), CONFIG)
    assert result["state"] == halts.HARD_HALT


def test_human_restart_clears_hard_halt(store):
    store.set_halt_state(halts.HARD_HALT, human_restart_required=True)
    result = halts.human_restart(store, operator="alice")
    assert result["state"] == halts.NORMAL


def test_human_restart_cannot_clear_kill(store):
    store.set_halt_state(halts.KILL)
    with pytest.raises(ValueError):
        halts.human_restart(store, operator="alice")


def test_manual_reset_requires_confirmation(store):
    store.set_halt_state(halts.KILL)
    with pytest.raises(ValueError):
        halts.manual_reset(store, operator="alice", confirmation="wrong")
    result = halts.manual_reset(store, operator="alice", confirmation="CONFIRM_KILL_RESET")
    assert result["state"] == halts.NORMAL


def test_soft_halt_autoclears_next_utc_day(store):
    store.set_halt_state(halts.SOFT_HALT, reason="test")
    # Manually backdate changed_at to yesterday to simulate day rollover.
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with store.transaction() as conn:
        conn.execute("UPDATE halt_state SET changed_at = ? WHERE id = 1", (yesterday,))
    result = halts.apply_daily_pnl(store, Decimal("0"), None, Decimal("0"), CONFIG)
    assert result["state"] == halts.NORMAL
