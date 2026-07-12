"""Halt state machine (plan §9).

States: NORMAL -> SOFT_HALT -> HARD_HALT -> KILL.

* SOFT_HALT: exposure increases prohibited; closes/reductions allowed;
  auto-clears at the next UTC day boundary.
* HARD_HALT: same restriction as soft, plus requires an authenticated
  human restart (``human_restart_required`` flag) before it can clear --
  it does NOT auto-clear.
* KILL: terminal. Everything HARD_HALT restricts, plus it never
  auto-clears and is not lifted by ``apply_daily_pnl``; only an explicit
  manual reset (outside this module, an operator action) can move out of
  KILL.

All transitions are persisted through a ``LedgerStore`` -- there is no
in-memory-only halt state in this system; the ledger is the single source
of truth so any process (watchdog, dashboard, restart of the main loop)
observes the same state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from quantdesk.ledger.store import LedgerStore

HaltState = Literal["NORMAL", "SOFT_HALT", "HARD_HALT", "KILL"]

NORMAL: HaltState = "NORMAL"
SOFT_HALT: HaltState = "SOFT_HALT"
HARD_HALT: HaltState = "HARD_HALT"
KILL: HaltState = "KILL"

_ORDER = {NORMAL: 0, SOFT_HALT: 1, HARD_HALT: 2, KILL: 3}

# Effects that reduce or eliminate exposure are always allowed, regardless
# of halt state (plan §9 authority rule: risk-reducing actions are always
# allowed when mechanically valid).
_REDUCING_EFFECTS = {"reduce", "close"}


def current_state(store: LedgerStore) -> dict:
    return store.get_halt_state()


def allows(store: LedgerStore, effect: str) -> bool:
    """Return True if an intent with the given ``effect`` may proceed given current halt state."""
    if effect in _REDUCING_EFFECTS:
        return True
    state = store.get_halt_state()["state"]
    return state == NORMAL


def _maybe_autoclear_soft(store: LedgerStore) -> dict:
    """SOFT_HALT auto-clears at the next UTC day boundary after it was set."""
    halt = store.get_halt_state()
    if halt["state"] != SOFT_HALT or halt["changed_at"] is None:
        return halt
    changed_at = datetime.fromisoformat(halt["changed_at"])
    today = datetime.now(timezone.utc).date()
    if changed_at.date() < today:
        store.set_halt_state(NORMAL, reason="soft_halt_autoclear_next_utc_day")
        return store.get_halt_state()
    return halt


def apply_daily_pnl(
    store: LedgerStore,
    daily_pnl_pct: Decimal,
    weekly_pnl_pct: Decimal | None,
    max_drawdown_pct: Decimal,
    config: dict,
) -> dict:
    """Evaluate PnL/drawdown thresholds and apply the resulting transition.

    ``daily_pnl_pct``, ``weekly_pnl_pct``, ``max_drawdown_pct`` are signed
    Decimal percentages (negative = loss / drawdown). ``config`` is the
    ``account`` section of desk config (thresholds are positive
    magnitudes: daily_loss_soft_halt_pct, daily_loss_hard_halt_pct,
    weekly_loss_hard_halt_pct, max_drawdown_kill_pct).

    A KILL state is never auto-lifted by this function. Transitions only
    ever move toward a *more* restrictive state here; recovery from
    SOFT_HALT happens via auto-clear, recovery from HARD_HALT/KILL
    requires an explicit human action outside this function.
    """
    halt = _maybe_autoclear_soft(store)
    if halt["state"] == KILL:
        return halt  # terminal

    daily_loss_soft = Decimal(str(config["daily_loss_soft_halt_pct"]))
    daily_loss_hard = Decimal(str(config["daily_loss_hard_halt_pct"]))
    weekly_loss_hard = Decimal(str(config["weekly_loss_hard_halt_pct"]))
    max_dd_kill = Decimal(str(config["max_drawdown_kill_pct"]))
    human_restart_required = bool(config.get("human_restart_required_after_hard_halt", True))

    target: HaltState = halt["state"]
    reason = None

    if max_drawdown_pct <= -max_dd_kill:
        target, reason = KILL, f"max_drawdown {max_drawdown_pct}% breached kill threshold {max_dd_kill}%"
    elif daily_pnl_pct <= -daily_loss_hard or (
        weekly_pnl_pct is not None and weekly_pnl_pct <= -weekly_loss_hard
    ):
        target, reason = HARD_HALT, f"loss breached hard-halt threshold (daily={daily_pnl_pct}%, weekly={weekly_pnl_pct}%)"
    elif daily_pnl_pct <= -daily_loss_soft:
        target, reason = SOFT_HALT, f"daily_pnl {daily_pnl_pct}% breached soft-halt threshold {daily_loss_soft}%"

    if target != halt["state"] and _ORDER[target] > _ORDER[halt["state"]]:
        store.set_halt_state(
            target,
            reason=reason,
            human_restart_required=(target in (HARD_HALT, KILL) and human_restart_required),
        )
        return store.get_halt_state()
    return halt


def human_restart(store: LedgerStore, operator: str) -> dict:
    """Clear a HARD_HALT via authenticated human action. Cannot clear KILL."""
    halt = store.get_halt_state()
    if halt["state"] == KILL:
        raise ValueError("KILL is terminal and cannot be cleared by human_restart; use manual_reset")
    if halt["state"] != HARD_HALT:
        return halt
    store.set_halt_state(NORMAL, reason=f"human_restart by {operator}")
    return store.get_halt_state()


def manual_reset(store: LedgerStore, operator: str, confirmation: str) -> dict:
    """Manually reset out of KILL. Requires an explicit confirmation string from an operator."""
    if confirmation != "CONFIRM_KILL_RESET":
        raise ValueError("manual_reset from KILL requires confirmation='CONFIRM_KILL_RESET'")
    store.set_halt_state(NORMAL, reason=f"manual_reset by {operator}")
    return store.get_halt_state()
