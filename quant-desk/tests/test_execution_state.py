from __future__ import annotations

import uuid

import pytest

from quantdesk.execution.state import (
    IllegalStateTransition,
    OrderState,
    OrderStateMachine,
    client_order_id,
    validate_transition,
)


def test_legal_transitions_succeed():
    validate_transition(OrderState.CREATED, OrderState.RISK_APPROVED)
    validate_transition(OrderState.RISK_APPROVED, OrderState.SUBMITTING)
    validate_transition(OrderState.SUBMITTING, OrderState.ACKNOWLEDGED)
    validate_transition(OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED)
    validate_transition(OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED)
    validate_transition(OrderState.PARTIALLY_FILLED, OrderState.FILLED)
    validate_transition(OrderState.FILLED, OrderState.PROTECTED)
    validate_transition(OrderState.PROTECTED, OrderState.CLOSED)
    validate_transition(OrderState.UNKNOWN, OrderState.CANCELLED)


@pytest.mark.parametrize(
    "current,target",
    [
        (OrderState.CREATED, OrderState.FILLED),
        (OrderState.CLOSED, OrderState.CREATED),
        (OrderState.FILLED, OrderState.CREATED),
        (OrderState.REJECTED, OrderState.CREATED),
        (OrderState.EXPIRED, OrderState.RISK_APPROVED),
        (OrderState.CANCELLED, OrderState.ACKNOWLEDGED),
        (OrderState.ACKNOWLEDGED, OrderState.CREATED),
        (OrderState.SUBMITTING, OrderState.FILLED),
    ],
)
def test_illegal_transitions_raise(current, target):
    with pytest.raises(IllegalStateTransition):
        validate_transition(current, target)


def test_terminal_states_have_no_outgoing_transitions():
    from quantdesk.execution.state import TRANSITIONS

    for terminal in (
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.CANCELLED,
        OrderState.CLOSED,
    ):
        assert TRANSITIONS[terminal] == frozenset()


def test_order_state_machine_transitions_and_history():
    machine = OrderStateMachine()
    assert machine.state == OrderState.CREATED
    machine.transition_to(OrderState.RISK_APPROVED)
    assert machine.state == OrderState.RISK_APPROVED
    assert machine.history == [
        (OrderState.CREATED, OrderState.RISK_APPROVED, None)
    ]

    with pytest.raises(IllegalStateTransition):
        machine.transition_to(OrderState.FILLED)
    # failed transition must not change state
    assert machine.state == OrderState.RISK_APPROVED


def test_client_order_id_deterministic():
    intent_id = uuid.uuid4()
    a = client_order_id(intent_id)
    b = client_order_id(intent_id)
    c = client_order_id(str(intent_id))
    assert a == b == c
    assert a.startswith("qd-")
    assert len(a) == 23


def test_client_order_id_differs_across_intents():
    id1 = client_order_id(uuid.uuid4())
    id2 = client_order_id(uuid.uuid4())
    id3 = client_order_id(uuid.uuid4())
    assert len({id1, id2, id3}) == 3
