"""Fault injection: timeouts, failed queries, late fills, stale cancel answers, restarts."""

import pytest

from kasegu.core.engine import Engine
from kasegu.core.position import PosState
from kasegu.core.types import MarkPrice, Order, OrderStatus, OrderType, Purpose, Side
from kasegu.market.data import MINUTE
from kasegu.strategies import create
from tests.test_epr15 import RULES, S, T, epr_cfg, fill_entry, kinds, next_bar, open_stops, setup


class Timeout(Exception):
    pass


async def timeout(*a, **kw):
    raise Timeout("REST timeout")


async def not_found(*a, **kw):
    return None


def calls(broker):
    """Record place / cancel calls in order."""
    log = []
    place, cancel = broker.place, broker.cancel

    async def p(o):
        log.append(("place", o.type, o.client_id))
        return await place(o)

    async def c(sym, cid):
        log.append(("cancel", cid))
        return await cancel(sym, cid)

    broker.place, broker.cancel = p, c
    return log


def stale(broker, cid):
    """Cancel answers with the order's old open state, like a lagging read."""
    cancel = broker.cancel

    async def c(sym, x):
        if x == cid:
            return broker._copy(broker.orders[x]) if broker.orders[x].is_open else None
        return await cancel(sym, x)

    broker.cancel = c


def restart(eng):
    new = Engine(eng.cfg, create(eng.cfg.strategy), eng.broker, eng.market, RULES, eng.journal,
                 clock=eng.clock)  # fmt: skip
    new.restore()
    return new


# ---- P0-1: stops


async def test_stop_timeout_without_old_stop_closes_and_blocks():
    eng, broker, clock = setup()
    place = broker.place

    async def stop_timeout(o):
        if o.type is OrderType.STOP_MARKET:
            raise Timeout("REST timeout")
        return await place(o)

    broker.place, broker.query = stop_timeout, timeout
    await fill_entry(eng, clock)
    # unknown stop = maybe unprotected: closed with a reduce-only market order
    assert eng.blocked == {S: "stop_unknown"} and broker.pos[S] == 0
    assert eng.trades[-1].reason == "stop_unknown"
    # queries still fail: the reconcile keeps everything blocked
    await eng.sync()
    assert eng.blocked == {S: "sync_unconfirmed"}
    await next_bar(eng, clock)
    assert eng.entry[S] is None
    # the exchange answers: the stop never landed, state is clean again
    broker.query = not_found
    await eng.sync()
    assert not eng.blocked


async def test_resize_keeps_old_stop_until_new_is_confirmed():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    cancel = broker.cancel

    async def entry_cancel_lags(sym, cid):
        if cid == o.client_id:
            return broker._copy(broker.orders[cid])  # still NEW
        return await cancel(sym, cid)

    broker.cancel = entry_cancel_lags
    broker._fill(o, o.qty * 0.5, o.price, maker=True)
    await eng.drain()
    old = eng.stop[S]
    assert old.qty == pytest.approx(o.qty * 0.5) and eng.blocked == {S: "entry_cancel_unconfirmed"}

    # the rest fills late while the new stop times out
    log = calls(broker)
    place = broker.place

    async def stop_timeout(x):
        if x.type is OrderType.STOP_MARKET:
            raise Timeout("REST timeout")
        return await place(x)

    broker.place, broker.query = stop_timeout, timeout
    broker._fill(o, o.qty * 0.5, o.price, maker=True)
    await eng.drain()
    # the old stop was never cancelled before the position was closed
    exit_at = next(i for i, x in enumerate(log) if x[0] == "place" and x[1] is OrderType.MARKET)
    assert ("cancel", old.client_id) not in log[:exit_at]
    assert eng.trades[-1].reason == "stop_unknown" and broker.pos[S] == 0
    assert eng.blocked[S] in ("stop_unknown", "entry_cancel_unconfirmed")


async def test_resize_cancels_old_only_after_new_confirmed():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    broker._fill(o, o.qty * 0.5, o.price, maker=True)
    await eng.drain()
    old = eng.stop[S]
    log = calls(broker)
    # a fill that raced the entry cancel grows the position
    pos = eng.positions[S]
    pos.qty += 0.1
    broker.pos[S] += 0.1
    await eng._protect(S)
    new = next(i for i, x in enumerate(log) if x[0] == "place")
    assert log.index(("cancel", old.client_id)) > new
    assert [x.qty for x in open_stops(broker)] == [pytest.approx(pos.qty)]


async def test_old_stop_fills_while_being_replaced():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    broker._fill(o, o.qty * 0.5, o.price, maker=True)
    await eng.drain()
    old = broker.orders[eng.stop[S].client_id]
    cancel = broker.cancel

    async def triggers_first(sym, cid):
        if cid == old.client_id and old.is_open:
            broker._fill(old, old.qty, old.stop_price, maker=False)  # stop fires first
        return await cancel(sym, cid)

    broker.cancel = triggers_first
    pos = eng.positions[S]
    pos.qty += 0.1
    broker.pos[S] += 0.1
    await eng._protect(S)
    await eng.drain()
    # old stop closed its part, the new reduce-only stop covers the rest, never short
    assert broker.pos[S] == pytest.approx(0.1) and pos.qty == pytest.approx(0.1)
    await eng.on_mark(MarkPrice(S, eng.stop[S].stop_price - 1, 0, 0, clock.t + 5))
    assert broker.pos[S] == pytest.approx(0) and pos.state is PosState.FLAT
    assert len(eng.trades) == 1


# ---- P0-3: entry cancel after a partial fill


async def test_partial_fill_cancel_timeout_keeps_tracking():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    cancel, query = broker.cancel, broker.query
    broker.cancel = broker.query = timeout
    broker._fill(o, o.qty * 0.3, o.price, maker=True)
    await eng.drain()
    assert eng.entry[S] is not None and eng.blocked == {S: "entry_cancel_unconfirmed"}
    assert S in eng.exec.unknown
    # a late fill of the same order is still protected
    broker._fill(o, o.qty * 0.3, o.price, maker=True)
    await eng.drain()
    pos = eng.positions[S]
    assert pos.qty == pytest.approx(o.qty * 0.6)
    # new stop covers all; the old one could not be cancelled yet (reduce-only, harmless)
    assert eng.stop[S].qty == pytest.approx(pos.qty) and eng.stop[S].is_open
    assert len(open_stops(broker)) == 2
    # reconcile while the exchange is still unreachable: stays blocked
    await eng.sync()
    assert S in eng.blocked
    broker.cancel, broker.query = cancel, query
    await eng.sync()
    assert not eng.blocked and eng.entry[S] is None
    assert broker.orders[o.client_id].status is OrderStatus.CANCELED
    assert [x.qty for x in open_stops(broker)] == [pytest.approx(pos.qty)]


async def test_partial_fill_stale_cancel_answer_is_not_success():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    stale(broker, o.client_id)
    broker._fill(o, o.qty * 0.5, o.price, maker=True)
    await eng.drain()
    assert broker.orders[o.client_id].is_open
    assert eng.entry[S] is not None and eng.blocked == {S: "entry_cancel_unconfirmed"}


async def test_bar_cancel_stale_answer_blocks_new_entry():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = eng.entry[S]
    stale(broker, o.client_id)
    await next_bar(eng, clock)
    assert eng.entry[S] is o and eng.blocked == {S: "entry_cancel_unconfirmed"}
    assert len([x for x in broker.orders.values() if x.is_open]) == 1


async def test_entry_place_timeout_blocks_and_recheck_finds_it():
    eng, broker, clock = setup()
    place = broker.place
    landed = []

    async def lands_then_times_out(o):
        res = await place(o)  # reached the exchange, answer lost
        landed.append(res)
        raise Timeout("REST timeout")

    broker.place, query = lands_then_times_out, broker.query
    broker.query = timeout
    await next_bar(eng, clock)
    assert eng.blocked == {S: "order_unknown"}
    broker.place, broker.query = place, query
    await eng.sync()
    # found on the exchange and cancelled as a stray entry
    assert not eng.blocked and broker.orders[landed[0].client_id].status is OrderStatus.CANCELED


# ---- P0-2 / P0-4: sync


async def test_sync_failure_keeps_global_block():
    eng, broker, clock = setup(cfg=epr_cfg(symbols=[S, T]))
    open_orders = broker.open_orders

    async def fails_for_s(sym):
        if sym == S:
            raise Timeout("REST timeout")
        return await open_orders(sym)

    broker.open_orders = fails_for_s
    await eng.sync()
    assert eng.blocked == {S: "sync_unconfirmed"}
    await next_bar(eng, clock, T)
    assert eng.entry[T] is None
    broker.open_orders = open_orders
    await eng.sync()
    assert not eng.blocked


async def test_sync_entry_cancel_fails_stays_blocked():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    eng.blocked[S] = "order_unknown"
    stale(broker, eng.entry[S].client_id)
    await eng.sync()
    assert S in eng.blocked


async def test_sync_position_without_confirmed_stop_stays_blocked():
    eng, broker, clock = setup()
    await fill_entry(eng, clock)
    broker.orders[eng.stop[S].client_id].status = OrderStatus.CANCELED  # gone silently
    place = broker.place

    async def stop_timeout(o):
        if o.type is OrderType.STOP_MARKET:
            raise Timeout("REST timeout")
        return await place(o)

    broker.place, broker.query = stop_timeout, timeout
    await eng.sync()
    assert S in eng.blocked and "stop_unknown" in kinds(eng)


async def test_sync_flat_clears_stale_refs():
    eng, broker, clock = setup()
    await fill_entry(eng, clock)
    stale_stop = eng.stop[S]
    # the position closed on the exchange while we were away, stop gone too
    broker.pos[S] = 0
    broker.orders[stale_stop.client_id].status = OrderStatus.EXPIRED
    eng.close[S] = Order("vtxold", S, Side.SELL, OrderType.MARKET, 1, Purpose.EXIT)
    await eng.sync()
    assert eng.stop[S] is None and eng.entry[S] is None and eng.close[S] is None
    assert not eng.blocked
    # the next fill gets its own fresh stop
    await fill_entry(eng, clock)
    assert eng.stop[S].client_id != stale_stop.client_id
    assert [x.client_id for x in open_stops(broker)] == [eng.stop[S].client_id]


# ---- P1: crash leftovers and long downtime


async def test_restart_cancels_entry_left_by_a_crash():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    left = eng.entry[S].client_id  # the process dies here, the order stays on the exchange
    eng2 = restart(eng)
    await eng2.sync()
    assert broker.orders[left].status is OrderStatus.CANCELED and not eng2.blocked


async def test_restart_entry_filled_while_down_is_adopted_and_stopped():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    broker.listener = None  # dead process sees nothing
    broker._fill(o, o.qty * 0.5, o.price, maker=True)
    eng2 = restart(eng)
    await eng2.sync()
    pos = eng2.positions[S]
    assert pos.state is PosState.POSITION_OPEN and pos.qty == pytest.approx(o.qty * 0.5)
    assert broker.orders[o.client_id].status is OrderStatus.CANCELED
    assert [x.qty for x in open_stops(broker)] == [pytest.approx(pos.qty)]
    assert "position_adopted" in kinds(eng2) and not eng2.blocked


async def test_restart_after_15_minutes_exits_on_first_mark():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    clock.t = opened + 20 * MINUTE  # down for 20 minutes, exchange stop kept the position
    eng2 = restart(eng)
    await eng2.sync()
    assert eng2.positions[S].open_time == opened and not eng2.blocked
    await eng2.on_mark(MarkPrice(S, px, 0, 0, clock.t))
    assert eng2.trades[-1].reason == "time_exit" and broker.pos[S] == 0


async def test_stream_gap_over_15_minutes_exits_when_marks_return():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    eng.halted = True  # streams down, no marks
    clock.t = opened + 25 * MINUTE
    eng.halted = False
    await eng.on_mark(MarkPrice(S, px * 1.01, 0, 0, clock.t))
    assert eng.trades[-1].reason == "time_exit" and broker.pos[S] == 0


async def test_unconfirmed_cancel_on_close_blocks():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    stale(broker, eng.stop[S].client_id)
    await eng.on_mark(MarkPrice(S, px, 0, 0, eng.positions[S].open_time + 15 * MINUTE))
    assert eng.positions[S].state is PosState.FLAT
    assert eng.blocked == {S: "stop_cancel_unconfirmed"}


def test_check_sizes_lists_untradable_symbols():
    cfg = epr_cfg(symbols=[S, T], risk={"margin_budget": 1, "leverage": 1, "max_spread_pct": 1})
    eng, broker, clock = setup(cfg=cfg)
    # 0.99 USDT < 5 USDT min notional on both
    assert set(eng.check_sizes()) == {S, T} and "untradable" in kinds(eng)


# ---- R09: restart with stops that do not match the exchange position


def leftover_stop(broker, side, qty, px, filled=0.0):
    o = Order(f"vts{len(broker.orders):024d}", S, side, OrderType.STOP_MARKET, qty, Purpose.STOP,
              stop_price=px, reduce_only=True, filled_qty=filled)  # fmt: skip
    broker.orders[o.client_id] = o
    broker.live.setdefault(S, {})[o.client_id] = o
    return o


async def restart_with_mismatch(eng, broker, clock):
    """Exchange long 0.6 of the entry while the journal knows 0.4, plus odd stops."""
    await next_bar(eng, clock)
    entry = broker.orders[eng.entry[S].client_id]
    broker._fill(entry, entry.qty * 0.4, entry.price, maker=True)
    await eng.drain()
    old = broker.orders[eng.stop[S].client_id]
    broker.listener = None  # crash: the late fill and these stops are not journaled
    broker._fill(entry, entry.qty * 0.2, entry.price, maker=True)
    small = leftover_stop(broker, Side.SELL, entry.qty * 0.1, old.stop_price)
    wrong = leftover_stop(broker, Side.BUY, entry.qty * 0.6, old.stop_price * 1.1)
    return old, small, wrong


async def test_restart_multiple_and_wrong_side_stops_replaced_without_gap():
    eng, broker, clock = setup()
    old, small, wrong = await restart_with_mismatch(eng, broker, clock)
    log = calls(broker)
    eng2 = restart(eng)
    await eng2.sync()
    qty = broker.pos[S]
    stops = open_stops(broker)
    assert [x.qty for x in stops] == [pytest.approx(qty)] and stops[0].side is Side.SELL
    first_place = next(i for i, x in enumerate(log) if x[0] == "place")
    for o in (old, small):
        assert log.index(("cancel", o.client_id)) > first_place
    assert broker.orders[wrong.client_id].status is OrderStatus.CANCELED
    assert not eng2.blocked


async def test_restart_new_stop_timeout_keeps_old_stops_and_closes():
    eng, broker, clock = setup()
    old, small, wrong = await restart_with_mismatch(eng, broker, clock)
    log = calls(broker)
    place = broker.place

    async def stop_timeout(o):
        if o.type is OrderType.STOP_MARKET:
            raise Timeout("REST timeout")
        return await place(o)

    broker.place, broker.query = stop_timeout, timeout
    eng2 = restart(eng)
    await eng2.sync()
    # nothing confirmed covers the position: old stops are never cancelled first,
    # the position is closed reduce-only and entries stay blocked
    exit_at = next(i for i, x in enumerate(log) if x[0] == "place" and x[1] is OrderType.MARKET)
    assert ("cancel", old.client_id) not in log[:exit_at]
    assert ("cancel", small.client_id) not in log[:exit_at]
    assert broker.pos[S] == pytest.approx(0) and eng2.trades[-1].reason == "stop_unknown"
    assert S in eng2.blocked


async def test_restart_partly_triggered_stop_counts_remaining_only():
    eng, broker, clock = setup()
    await fill_entry(eng, clock)
    so = broker.orders[eng.stop[S].client_id]
    broker.listener = None
    # the stop child filled part of the order before the crash
    broker._fill(so, so.qty * 0.5, so.stop_price, maker=False)
    assert so.is_open and broker.pos[S] == pytest.approx(so.qty * 0.5)
    log = calls(broker)
    eng2 = restart(eng)
    await eng2.sync()
    # remaining 0.5 equals the position: kept, no replacement needed
    assert not [x for x in log if x[0] == "place"] and eng2.stop[S].client_id == so.client_id
    assert not eng2.blocked
