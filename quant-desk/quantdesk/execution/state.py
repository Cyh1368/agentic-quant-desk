"""Order state machine and deterministic client order IDs (plan §10).

Pure and deterministic: no wall-clock reads except via explicit params,
no randomness.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from uuid import UUID


class OrderState(str, Enum):
    CREATED = "CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PROTECTED = "PROTECTED"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class IllegalStateTransition(ValueError):
    pass


TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset(
        {OrderState.RISK_APPROVED, OrderState.REJECTED, OrderState.EXPIRED}
    ),
    OrderState.RISK_APPROVED: frozenset(
        {OrderState.SUBMITTING, OrderState.EXPIRED, OrderState.CANCELLED}
    ),
    OrderState.SUBMITTING: frozenset(
        {OrderState.ACKNOWLEDGED, OrderState.UNKNOWN, OrderState.REJECTED}
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.UNKNOWN,
            OrderState.PARTIALLY_FILLED,
        }
    ),
    OrderState.FILLED: frozenset({OrderState.PROTECTED, OrderState.UNKNOWN}),
    OrderState.PROTECTED: frozenset({OrderState.CLOSED, OrderState.UNKNOWN}),
    OrderState.CLOSED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
    ),
}


def validate_transition(current: OrderState, target: OrderState) -> None:
    """Raise IllegalStateTransition if `target` is not reachable from `current`."""
    if target not in TRANSITIONS[current]:
        raise IllegalStateTransition(
            f"illegal transition: {current.value} -> {target.value}"
        )


class OrderStateMachine:
    """Wraps a single order's current state with an in-memory audit trail."""

    def __init__(self, initial: OrderState = OrderState.CREATED) -> None:
        self._state = initial
        self.history: list[tuple[OrderState, OrderState, datetime | None]] = []

    @property
    def state(self) -> OrderState:
        return self._state

    def transition_to(self, target: OrderState, at: datetime | None = None) -> None:
        validate_transition(self._state, target)
        self.history.append((self._state, target, at))
        self._state = target


def client_order_id(intent_id: UUID | str) -> str:
    """Deterministic idempotency key derived purely from intent_id.

    Same intent_id (UUID or str form) always yields the identical string,
    so a retried submission of the same intent is venue-side idempotent.
    """
    digest = hashlib.sha256(str(intent_id).encode()).hexdigest()[:20]
    return f"qd-{digest}"
