import pytest

from kasegu.config import AppConfig
from kasegu.core.engine import Engine
from kasegu.core.execution import PaperBroker
from kasegu.core.position import PosState
from kasegu.core.types import Fill, Kline, MarkPrice, OrderStatus, Purpose, Quote
from kasegu.exchange.rules import SymbolRules
from kasegu.market.data import MINUTE, MarketData
from kasegu.storage.journal import Journal
from kasegu.strategies import create

S = "BTCUSDT"
RULES = {S: SymbolRules(S, 0.01, 0.001, 0.001, 1000, 5)}


def bar(i, c):
    return Kline(S, i * MINUTE, (i + 1) * MINUTE - 1, c, c + 0.5, c - 0.5, c, 1)


class Clock:
    t = 0

    def __call__(self):
        return self.t


def setup(n=800, **cfg_kw):
    cfg = AppConfig(
        symbols=[S],
        fee={"slippage_pct": 0.0},
        risk={"margin_budget": 20, "leverage": 5, "max_spread_pct": 1.0, "max_notional": 1e6},
        **cfg_kw,
    )
    md = MarketData([S])
    md.load([bar(i, 100 + i * 0.01) for i in range(n)])
    clock = Clock()
    clock.t = n * MINUTE
    broker = PaperBroker(cfg.fee)
    j = Journal(":memory:", run_id="t")
    eng = Engine(cfg, create(cfg.strategy), broker, md, RULES, j, clock=clock)
    return eng, broker, clock


async def quote(eng, bid, ask, t):
    await eng.on_quote(Quote(S, bid, ask, 5, 5, t))


async def next_bar(eng, clock, c):
    i = eng.market.last_open(S) // MINUTE + 1
    clock.t = (i + 1) * MINUTE
    await quote(eng, c - 0.01, c, clock.t - 1)
    await eng.on_bar(bar(i, c))


async def test_full_cycle_trailing_exit():
    eng, broker, clock = setup()
    await next_bar(eng, clock, 108.0)
    pos, o = eng.positions[S], eng.entry[S]
    assert pos.state is PosState.PENDING_ENTRY and o is not None and o.price < 108.0

    # new bar replaces the unfilled entry
    first = o.client_id
    await next_bar(eng, clock, 108.0)
    assert eng.exec.get(first).status is OrderStatus.CANCELED
    o = eng.entry[S]
    assert o.client_id != first

    # quote trades through -> fill, stop placed
    px = o.price
    await quote(eng, px - 0.2, px - 0.1, clock.t + 1)
    assert pos.state is PosState.POSITION_OPEN and pos.entry_price == px
    assert eng.entry[S] is None and eng.stop[S].is_open
    assert eng.stop[S].stop_price == pytest.approx(px * 0.99, abs=0.01)

    # no new entry while in position
    await next_bar(eng, clock, 108.0)
    assert eng.entry[S] is None

    t = clock.t
    await quote(eng, px * 1.013, px * 1.0131, t)
    await eng.on_mark(MarkPrice(S, px * 1.013, 0, 0, t))
    assert pos.tier == 2
    await quote(eng, px * 1.009, px * 1.0091, t + 1)
    await eng.on_mark(MarkPrice(S, px * 1.009, 0, 0, t + 1))
    assert pos.state is PosState.FLAT
    tr = eng.trades[-1]
    assert tr.reason == "trail_t2" and tr.pnl > 0
    assert tr.meta["strategy"] == "atr_amplitude"
    assert not eng.stop[S] and broker.pos[S] == 0
    assert len(eng.journal.trades()) == 1


async def test_exchange_stop_and_max_hold():
    eng, broker, clock = setup()
    await next_bar(eng, clock, 108.0)
    px = eng.entry[S].price
    await quote(eng, px - 0.2, px - 0.1, clock.t + 1)
    stop = eng.stop[S].stop_price
    # mark gaps below stop before local check: exchange stop closes it
    await quote(eng, stop - 0.05, stop - 0.04, clock.t + 2)
    await eng.on_mark(MarkPrice(S, stop - 0.05, 0, 0, clock.t + 2))
    assert eng.positions[S].state is PosState.FLAT
    assert eng.trades[-1].reason in ("exchange_stop", "stop_loss")

    await next_bar(eng, clock, 108.0)
    px = eng.entry[S].price
    await quote(eng, px - 0.2, px - 0.1, clock.t + 1)
    opened = eng.positions[S].open_time
    await eng.on_mark(MarkPrice(S, px, 0, 0, opened + 60 * MINUTE))
    assert eng.trades[-1].reason == "max_hold"


async def test_partial_fill_cancels_rest():
    eng, broker, clock = setup()
    await next_bar(eng, clock, 108.0)
    o = broker.orders[eng.entry[S].client_id]
    broker._fill(o, o.qty / 2, o.price, maker=True)
    await eng.drain()
    pos = eng.positions[S]
    assert pos.state is PosState.POSITION_OPEN
    assert broker.orders[o.client_id].status is OrderStatus.CANCELED
    assert eng.stop[S].qty == pytest.approx(pos.qty)


async def test_funding_charged_once():
    eng, broker, clock = setup()
    await next_bar(eng, clock, 108.0)
    px = eng.entry[S].price
    await quote(eng, px - 0.2, px - 0.1, clock.t + 1)
    t, nxt = clock.t + 2, clock.t + 10
    await eng.on_mark(MarkPrice(S, px, 0.0001, nxt, t))
    await eng.on_mark(MarkPrice(S, px, 0.0001, nxt, nxt))
    await eng.on_mark(MarkPrice(S, px, 0.0001, nxt, nxt + 1))
    pos = eng.positions[S]
    assert pos.funding == pytest.approx(-pos.qty * px * 0.0001)


async def test_gap_backfill():
    eng, _, clock = setup()
    last = eng.market.last_open(S) // MINUTE

    async def backfill(sym, start):
        return [bar(i, 108.0) for i in range(start // MINUTE, last + 4)]

    eng.backfill = backfill
    clock.t = (last + 4) * MINUTE
    await quote(eng, 107.99, 108.0, clock.t - 1)
    await eng.on_bar(bar(last + 3, 108.0))
    assert eng.market.last_open(S) == (last + 3) * MINUTE
    assert eng.entry[S] is not None


async def test_halt_and_stale_block_entries():
    eng, _, clock = setup()
    eng.halted = True
    await next_bar(eng, clock, 108.0)
    assert eng.entry[S] is None
    eng.halted = False
    i = eng.market.last_open(S) // MINUTE + 1
    clock.t = (i + 1) * MINUTE + 60_000  # quote is a minute old
    await eng.on_bar(bar(i, 108.0))
    assert eng.entry[S] is None


async def test_sync_resets_unknown_local_position():
    eng, broker, clock = setup()
    pos = eng.positions[S]
    pos.apply(
        Fill(
            "x",
            S,
            *[__import__("kasegu.core.types").core.types.Side.BUY],
            1.0,
            100.0,
            0,
            True,
            0,
        ),
        Purpose.ENTRY,
    )
    await eng.sync()
    assert pos.state is PosState.FLAT


async def test_cancel_entries_on_shutdown():
    eng, broker, clock = setup()
    await next_bar(eng, clock, 108.0)
    o = eng.entry[S]
    assert o.is_open
    await eng.cancel_entries()
    assert eng.exec.get(o.client_id).status is OrderStatus.CANCELED
    assert await broker.open_orders(S) == []
