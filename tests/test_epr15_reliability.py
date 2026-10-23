"""Independent EPR15 reliability tests.

These tests deliberately model exchange truth separately from ``Engine.stop`` and
``Engine.entry``.  No network, credentials, or real order endpoints are used.
"""

from collections.abc import Callable

import pytest

from kasegu.core.engine import Engine
from kasegu.core.execution import PaperBroker
from kasegu.core.position import PosState
from kasegu.core.types import MarkPrice, Order, OrderStatus, OrderType, Purpose
from kasegu.market.data import MINUTE
from kasegu.strategies import create
from tests.test_epr15 import (
    RULES,
    S,
    T,
    epr_cfg,
    fill_entry,
    next_bar,
    open_stops,
    setup,
)


def open_entries(broker: PaperBroker, symbol: str = S) -> list[Order]:
    return [
        o
        for o in broker.orders.values()
        if o.symbol == symbol and o.purpose is Purpose.ENTRY and o.is_open
    ]


async def block_on_unknown_entry_cancel(eng: Engine, broker: PaperBroker, clock) -> Order:
    await next_bar(eng, clock)
    entry = eng.entry[S]

    async def unavailable(*_args):
        raise ConnectionError("scripted REST outage")

    broker.cancel = unavailable
    broker.query = unavailable
    await next_bar(eng, clock)
    assert S in eng.blocked
    assert open_entries(broker) == [broker.orders[entry.client_id]]
    return entry


async def test_stop_replacement_timeout_keeps_last_confirmed_stop_active():
    """A NEW local object is not proof that the replacement exists remotely."""
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    entry = broker.orders[eng.entry[S].client_id]

    broker._fill(entry, entry.qty * 0.4, entry.price, maker=True)
    await eng.drain()
    confirmed = open_stops(broker)[0]
    assert confirmed.qty == pytest.approx(eng.positions[S].qty)

    original_place, original_cancel = broker.place, broker.cancel
    trace: list[tuple[str, OrderType | None, str]] = []

    async def replacement_times_out(o: Order) -> Order:
        if o.type is OrderType.STOP_MARKET:
            raise TimeoutError("scripted timeout before exchange acceptance")
        trace.append(("place", o.type, o.client_id))
        return await original_place(o)

    async def record_cancel(symbol: str, client_id: str) -> Order | None:
        trace.append(("cancel", None, client_id))
        return await original_cancel(symbol, client_id)

    async def query_is_unknown(*_args):
        return None

    broker.place = replacement_times_out
    broker.cancel = record_cancel
    broker.query = query_is_unknown
    eng.exec.retries = 1

    # A late entry fill grows exposure and requires a larger stop.  The exchange
    # never accepts that stop, and querying its client id cannot resolve it.
    broker._fill(entry, entry.qty * 0.2, entry.price, maker=True)
    await eng.drain()

    assert S in eng.blocked
    if eng.positions[S].has_qty:
        # Keeping the position requires preserving the last confirmed, partially
        # covering stop and never claiming it is full coverage.
        assert [o.client_id for o in open_stops(broker)] == [confirmed.client_id]
        assert eng.stop[S].client_id == confirmed.client_id
        assert confirmed.qty < eng.positions[S].qty
    else:
        # Emergency flattening is also safe, but the confirmed stop must remain
        # until the reduce-only close is submitted and filled.
        market = next(i for i, item in enumerate(trace) if item[:2] == ("place", OrderType.MARKET))
        cancel = trace.index(("cancel", None, confirmed.client_id))
        assert market < cancel
        assert eng.trades[-1].reason == "stop_unknown"


async def test_reconcile_keeps_global_block_when_entry_cancel_is_unconfirmed():
    eng, broker, clock = setup(cfg=epr_cfg(symbols=[S, T]))
    entry = await block_on_unknown_entry_cancel(eng, broker, clock)

    await eng.sync()

    assert eng.exec.get(entry.client_id).is_open
    assert open_entries(broker) == [broker.orders[entry.client_id]]
    assert S in eng.blocked
    await next_bar(eng, clock, T)
    assert eng.entry[T] is None


async def test_reconcile_does_not_trust_an_incomplete_open_order_snapshot():
    eng, broker, clock = setup(cfg=epr_cfg(symbols=[S, T]))
    entry = await block_on_unknown_entry_cancel(eng, broker, clock)

    async def stale_open_orders(_symbol: str) -> list[Order]:
        return []

    async def flat(_symbol: str) -> float:
        return 0.0

    broker.open_orders = stale_open_orders
    broker.position_qty = flat
    await eng.sync()

    assert broker.orders[entry.client_id].is_open  # exchange truth hidden by stale REST data
    assert S in eng.blocked
    await next_bar(eng, clock, T)
    assert eng.entry[T] is None


async def test_reconcile_rest_failure_cannot_clear_an_existing_block():
    eng, broker, clock = setup()
    eng.blocked[S] = "prior_safety_block"

    async def stale_open_orders(_symbol: str) -> list[Order]:
        return []

    async def position_query_fails(_symbol: str) -> float:
        raise ConnectionError("scripted position endpoint outage")

    broker.open_orders = stale_open_orders
    broker.position_qty = position_query_fails
    try:
        await eng.sync()
    except ConnectionError:
        pass  # the baseline propagated; the fixed Engine contains and records it
    assert S in eng.blocked


async def test_partial_fill_cancel_race_converges_to_exposure_and_full_stop():
    """Cancellation ack is delivered before the fill it already accounts for."""
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    entry = broker.orders[eng.entry[S].client_id]
    original_cancel = broker.cancel
    injected = False

    async def fill_while_cancel_pending(symbol: str, client_id: str) -> Order | None:
        nonlocal injected
        if client_id != entry.client_id or injected:
            return await original_cancel(symbol, client_id)
        injected = True

        # Capture the late fill event, then deliberately deliver a cumulative
        # cancellation response first.  This is a reproducible out-of-order stream.
        listener = broker.listener
        held: list[tuple[Order, object]] = []
        broker.listener = lambda o, f: held.append((o, f))
        broker._fill(entry, entry.qty * 0.2, entry.price, maker=True)
        broker.listener = listener
        entry.status = OrderStatus.CANCELED
        cancellation = broker._copy(entry)
        broker.emit(cancellation)
        for order_event, fill_event in held:
            broker.emit(order_event, fill_event)
        return cancellation

    broker.cancel = fill_while_cancel_pending
    broker._fill(entry, entry.qty * 0.4, entry.price, maker=True)
    await eng.drain()

    expected = entry.qty * 0.6
    assert broker.pos[S] == pytest.approx(expected)
    assert eng.positions[S].qty == pytest.approx(expected)
    assert broker.orders[entry.client_id].status is OrderStatus.CANCELED
    assert [o.qty for o in open_stops(broker)] == [pytest.approx(expected)]
    assert eng.stop[S].client_id == open_stops(broker)[0].client_id


async def test_flat_sync_clears_stale_stop_before_same_size_reentry():
    eng, broker, clock = setup()
    await fill_entry(eng, clock)
    old_stop = eng.stop[S]
    prior_qty = eng.positions[S].qty

    # The position disappears outside this process.  Reconciliation cancels the
    # exchange stop and must also invalidate the local reference to it.
    broker.pos[S] = 0.0
    await eng.sync()
    assert eng.positions[S].state is PosState.FLAT
    assert not open_stops(broker)

    # A same-sized re-entry is the dangerous aliasing case: qty equality must not
    # make a canceled stop from the previous position look reusable.
    await fill_entry(eng, clock)
    fresh = open_stops(broker)
    assert eng.positions[S].qty == pytest.approx(prior_qty)
    assert len(fresh) == 1
    assert fresh[0].client_id != old_stop.client_id
    assert eng.stop[S].client_id == fresh[0].client_id


async def test_restart_cancels_pending_entry_before_entries_can_resume():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    pending_id = eng.entry[S].client_id
    assert open_entries(broker)

    restarted = Engine(
        eng.cfg, create(eng.cfg.strategy), broker, eng.market, RULES, eng.journal, clock=clock
    )
    restarted.restore()
    await restarted.sync()

    assert broker.orders[pending_id].status is OrderStatus.CANCELED
    assert not open_entries(broker)
    assert restarted.positions[S].state is PosState.FLAT


async def test_restart_with_protected_position_executes_overdue_time_exit():
    eng, broker, clock = setup()
    fill_price = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    stop_id = open_stops(broker)[0].client_id
    clock.t = opened + 20 * MINUTE

    restarted = Engine(
        eng.cfg, create(eng.cfg.strategy), broker, eng.market, RULES, eng.journal, clock=clock
    )
    restarted.restore()
    await restarted.sync()
    assert open_stops(broker)[0].client_id == stop_id

    await restarted.on_mark(MarkPrice(S, fill_price, 0, 0, clock.t))
    assert broker.pos[S] == pytest.approx(0)
    assert restarted.trades[-1].reason == "time_exit"
    assert restarted.trades[-1].hold_minutes == pytest.approx(20)
    assert not open_stops(broker)


async def test_restart_disagreement_places_resized_stop_before_canceling_old_stop():
    """Journal lags a late fill, so startup must preserve continuous coverage."""
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    entry = broker.orders[eng.entry[S].client_id]
    broker._fill(entry, entry.qty * 0.4, entry.price, maker=True)
    await eng.drain()
    old_stop = open_stops(broker)[0]
    journal_qty = eng.positions[S].qty

    # Crash boundary: exchange gets another fill but the process journals nothing.
    broker.listener = None
    broker._fill(entry, entry.qty * 0.2, entry.price, maker=True)
    exchange_qty = broker.pos[S]
    assert exchange_qty > journal_qty

    trace: list[tuple[str, str, float]] = []
    original_place: Callable = broker.place
    original_cancel: Callable = broker.cancel

    async def traced_place(o: Order) -> Order:
        if o.purpose is Purpose.STOP:
            trace.append(("place", o.client_id, o.qty))
        return await original_place(o)

    async def traced_cancel(symbol: str, client_id: str) -> Order | None:
        if client_id == old_stop.client_id:
            trace.append(("cancel", client_id, old_stop.qty))
        return await original_cancel(symbol, client_id)

    broker.place = traced_place
    broker.cancel = traced_cancel
    restarted = Engine(
        eng.cfg, create(eng.cfg.strategy), broker, eng.market, RULES, eng.journal, clock=clock
    )
    restarted.restore()
    await restarted.sync()

    assert restarted.positions[S].qty == pytest.approx(exchange_qty)
    assert [o.qty for o in open_stops(broker)] == [pytest.approx(exchange_qty)]
    actions = [item[0] for item in trace]
    assert actions.index("place") < actions.index("cancel")
