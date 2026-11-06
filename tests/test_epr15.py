import numpy as np
import pytest

from kasegu.config import AppConfig, ExitConfig, RiskConfig, StrategyConfig
from kasegu.core.engine import Engine
from kasegu.core.execution import PaperBroker
from kasegu.core.position import Position, PosState
from kasegu.core.risk import RiskManager
from kasegu.core.types import (
    Fill,
    Kline,
    MarkPrice,
    OrderStatus,
    OrderType,
    Purpose,
    Quote,
    Side,
    Signal,
    TimeInForce,
)
from kasegu.exchange.rules import SymbolRules
from kasegu.exits import create_exit
from kasegu.market.data import MINUTE, MarketData
from kasegu.storage.journal import Journal
from kasegu.strategies import create
from kasegu.strategies import indicators as ind
from tests.helpers import make_state

S, T = "BTCUSDT", "ETHUSDT"
RULES = {
    S: SymbolRules(S, 0.01, 0.001, 0.001, 1000, 5),
    T: SymbolRules(T, 0.01, 0.001, 0.001, 1000, 5),
}
EPR = {"id": "epr15", "k": 12}
EXIT = {"policy": "time_only", "stop_loss_pct": 3.0, "max_holding_minutes": 15}


def epr_cfg(**kw) -> AppConfig:
    base = dict(
        symbols=[S],
        strategy=EPR,
        exit=EXIT,
        fee={"slippage_pct": 0.0},
        risk={"margin_budget": 30, "leverage": 10, "max_spread_pct": 1.0},
    )
    return AppConfig(**{**base, **kw})


def bar(s, i, c, hl=0.5):
    return Kline(s, i * MINUTE, (i + 1) * MINUTE - 1, c, c + hl, c - hl, c, 1)


class Clock:
    t = 0

    def __call__(self):
        return self.t


def setup(n=800, cfg=None):
    # falling prices: close stays under the EMA240
    cfg = cfg or epr_cfg()
    md = MarketData(cfg.symbols)
    for s in cfg.symbols:
        md.load([bar(s, i, 200 - i * 0.05) for i in range(n)])
    clock = Clock()
    clock.t = n * MINUTE
    broker = PaperBroker(cfg.fee)
    eng = Engine(cfg, create(cfg.strategy), broker, md, RULES, Journal(":memory:"), clock=clock)
    return eng, broker, clock


async def quote(eng, bid, ask, t, s=S):
    await eng.on_quote(Quote(s, bid, ask, 5, 5, t))


async def next_bar(eng, clock, s=S, c=None):
    i = eng.market.last_open(s) // MINUTE + 1
    c = c if c is not None else eng.market[s].bars[-1].close - 0.05
    clock.t = (i + 1) * MINUTE
    await quote(eng, c - 0.01, c, clock.t - 1, s)
    await eng.on_bar(bar(s, i, c))


async def fill_entry(eng, clock, s=S):
    await next_bar(eng, clock, s)
    px = eng.entry[s].price
    await quote(eng, px - 0.02, px - 0.01, clock.t + 1, s)
    return px


def kinds(eng):
    return [r[0] for r in eng.journal.db.execute("SELECT kind FROM events").fetchall()]


def open_stops(broker, s=S):
    return [o for o in broker.orders.values() if o.symbol == s and o.is_open and o.stop_price]


# ---- signal


def test_long_only_below_ema():
    st = create(StrategyConfig(**EPR))
    down = make_state(np.linspace(200, 150, 800))
    up = make_state(np.linspace(150, 200, 800))
    sig = st.on_bar(down)
    assert sig is not None and sig.side is Side.BUY
    assert sig.metadata["ema"] > down.close[-1]
    # above the EMA: no order at all, never a short
    assert st.on_bar(up) is None


def test_distance_is_12_volatility_units_from_ask():
    st = create(StrategyConfig(**EPR))
    s = make_state(np.linspace(200, 150, 800), spread=0.02)
    sig = st.on_bar(s)
    c = s.close[-1]
    atr_pct = ind.atr(s.high, s.low, s.close, 60) / c * 100
    amp_pct = ind.amplitude_pct(s.high, s.low, s.close, 60)
    vol = (atr_pct + amp_pct) / 2
    assert sig.metadata["distance_pct"] == pytest.approx(12 * vol)
    assert sig.limit_price == pytest.approx(s.ask * (1 - 12 * vol / 100))
    assert sig.metadata["ref"] == s.ask


def test_warmup_720_bars():
    st = create(StrategyConfig(**EPR))
    assert st.warmup == 720
    assert st.on_bar(make_state(np.linspace(200, 150, 719))) is None


def test_frozen_parameters():
    with pytest.raises(ValueError, match="frozen"):
        create(StrategyConfig(id="epr15", k=3))
    with pytest.raises(ValueError, match="frozen"):
        create(StrategyConfig(id="epr15", k=12, ema_period=120))
    with pytest.raises(ValueError, match="frozen"):
        create(StrategyConfig(id="epr15", k=12, params={"sides": "both"}))
    st = create(StrategyConfig(**EPR))
    with pytest.raises(ValueError, match="exit"):
        st.check_exit(ExitConfig())  # default trailing
    with pytest.raises(ValueError, match="exit"):
        st.check_exit(ExitConfig(**{**EXIT, "max_holding_minutes": 30}))
    st.check_exit(ExitConfig(**EXIT))


def test_engine_refuses_wrong_exit():
    with pytest.raises(ValueError):
        setup(cfg=epr_cfg(exit={"policy": "trailing", "stop_loss_pct": 3.0}))


def test_limit_floored_to_tick_and_post_only():
    r = RiskManager(RiskConfig(max_spread_pct=1), RULES)
    q = Quote(S, 99.99, 100.0, 5, 5, 0)
    plan, why = r.check(Signal(S, Side.BUY, 88.0049, 0), q, 0, 0)
    assert why == "ok" and plan.price == 88.0
    # at or through the ask a post-only buy would take
    assert r.check(Signal(S, Side.BUY, 100.0, 0), q, 0, 0)[1] == "would_take"


# ---- exit policy


def test_time_only_never_trails():
    ex = create_exit(ExitConfig(**EXIT))
    p = Position(S, state=PosState.PENDING_ENTRY)
    p.apply(Fill("e", S, Side.BUY, 1, 100.0, 0, True, 0), Purpose.ENTRY)
    for i, px in enumerate([101, 103, 110, 104, 100.5]):
        assert ex.update(p, px, i * MINUTE) is None
    assert p.tier == -1 and p.stop_pct is None and p.peak_pct == pytest.approx(10)
    assert ex.update(p, 100.0, 15 * MINUTE - 1) is None
    assert ex.update(p, 100.0, 15 * MINUTE) == "time_exit"


def test_time_only_stop_loss():
    ex = create_exit(ExitConfig(**EXIT))
    p = Position(S, state=PosState.PENDING_ENTRY)
    p.apply(Fill("e", S, Side.BUY, 1, 100.0, 0, True, 0), Purpose.ENTRY)
    assert ex.update(p, 97.01, 1) is None
    assert ex.update(p, 97.0, 2) == "stop_loss"
    assert ex.stop_price(p) == pytest.approx(97.0)


def test_trailing_policy_is_default():
    assert create_exit(ExitConfig()).id == "trailing"


# ---- engine


async def test_entry_replaced_every_bar_as_post_only_buy():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = eng.entry[S]
    assert o.side is Side.BUY and o.tif is TimeInForce.GTX and o.type is OrderType.LIMIT
    assert o.price == RULES[S].price_down(o.price)
    first = o.client_id
    await next_bar(eng, clock)
    assert eng.exec.get(first).status is OrderStatus.CANCELED
    assert eng.entry[S].client_id != first
    assert len([x for x in broker.orders.values() if x.is_open]) == 1


async def test_no_entry_above_ema():
    eng, broker, clock = setup()
    # jump far above the EMA
    await next_bar(eng, clock, c=300.0)
    assert eng.entry[S] is None


async def test_fill_stop_then_time_exit_without_tiers():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    pos = eng.positions[S]
    stop = eng.stop[S]
    assert pos.state is PosState.POSITION_OPEN
    assert stop.reduce_only and stop.side is Side.SELL and stop.qty == pos.qty
    assert stop.stop_price == RULES[S].price_down(px * 0.97)
    opened = pos.open_time
    # big run up: trailing tiers would lock in profit, time_only must not
    for i, m in enumerate([1.01, 1.03, 1.05, 1.02]):
        await quote(eng, px * m - 0.01, px * m, opened + i * 1000)
        await eng.on_mark(MarkPrice(S, px * m, 0, 0, opened + i * 1000))
    assert pos.tier == -1 and pos.stop_pct is None
    assert "tier" not in kinds(eng)
    await eng.on_mark(MarkPrice(S, px * 1.02, 0, 0, opened + 15 * MINUTE - 1))
    assert pos.state is PosState.POSITION_OPEN
    await eng.on_mark(MarkPrice(S, px * 1.02, 0, 0, opened + 15 * MINUTE))
    tr = eng.trades[-1]
    assert tr.reason == "time_exit" and tr.meta["strategy"] == "epr15"
    assert broker.pos[S] == 0 and not open_stops(broker)


async def test_exchange_stop_first_then_no_second_close():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    stop = eng.stop[S].stop_price
    await quote(eng, stop - 0.1, stop - 0.09, opened + 1000)
    await eng.on_mark(MarkPrice(S, stop - 0.1, 0, 0, opened + 1000))
    assert len(eng.trades) == 1 and eng.trades[0].exit_price < px
    # later time exit has nothing to close
    await eng.on_mark(MarkPrice(S, px, 0, 0, opened + 15 * MINUTE))
    assert len(eng.trades) == 1 and broker.pos[S] == 0


async def test_time_exit_races_exchange_stop():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    pos = eng.positions[S]
    so = broker.orders[eng.stop[S].client_id]
    # the exchange stop fills but its event arrives late
    held, listener = [], broker.listener
    broker.listener = lambda o, f: held.append((o, f))
    broker._fill(so, so.qty, px * 0.97, maker=False)
    broker.listener = listener
    await eng.on_mark(MarkPrice(S, px, 0, 0, pos.open_time + 15 * MINUTE))
    # reduce-only exit is refused, no short is opened
    assert broker.pos[S] == pytest.approx(0) and "exit_rejected" in kinds(eng)
    assert pos.has_qty and not eng.trades
    eng.pending.extend(held)
    await eng.drain()
    assert len(eng.trades) == 1 and eng.trades[0].exit_price == pytest.approx(px * 0.97)
    assert pos.state is PosState.FLAT and broker.pos[S] == pytest.approx(0)
    # entries wait for a reconcile
    assert eng.blocked == {S: "exit_rejected"}
    await eng.sync()
    assert not eng.blocked


async def test_partial_fill_keeps_full_stop_cover():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    o = broker.orders[eng.entry[S].client_id]
    broker._fill(o, o.qty * 0.4, o.price, maker=True)
    await eng.drain()
    pos = eng.positions[S]
    assert broker.orders[o.client_id].status is OrderStatus.CANCELED
    assert [x.qty for x in open_stops(broker)] == [pytest.approx(pos.qty)]
    # a late fill raced the cancel: the stop grows, the old one is removed only after
    old = eng.stop[S].client_id
    broker._fill(o, o.qty * 0.2, o.price, maker=True)
    await eng.drain()
    assert pos.qty == pytest.approx(o.qty * 0.6)
    stops = open_stops(broker)
    assert len(stops) == 1 and stops[0].qty == pytest.approx(pos.qty)
    assert broker.orders[old].status is OrderStatus.CANCELED


async def test_fill_during_cancel_is_protected():
    eng, broker, clock = setup()
    await next_bar(eng, clock)
    cid = eng.entry[S].client_id
    cancel = broker.cancel

    async def fill_then_cancel(symbol, client_id):
        o = broker.orders[client_id]
        if client_id == cid and o.is_open:
            broker._fill(o, o.qty, o.price, maker=True)
        return await cancel(symbol, client_id)

    broker.cancel = fill_then_cancel
    await next_bar(eng, clock)
    pos = eng.positions[S]
    assert pos.state is PosState.POSITION_OPEN and eng.entry[S] is None
    assert [x.qty for x in open_stops(broker)] == [pytest.approx(pos.qty)]


async def test_unknown_cancel_keeps_order_and_blocks():
    eng, broker, clock = setup(cfg=epr_cfg(symbols=[S, T]))
    await next_bar(eng, clock)
    o = eng.entry[S]

    async def down(*a):
        raise ConnectionError("down")

    cancel, query = broker.cancel, broker.query
    broker.cancel = broker.query = down
    await next_bar(eng, clock)
    # no second entry next to an order that may still be live
    assert eng.entry[S] is o and len([x for x in broker.orders.values() if x.is_open]) == 1
    assert eng.blocked == {S: "entry_cancel_unconfirmed"}
    assert "entry_cancel_unconfirmed" in kinds(eng)
    await next_bar(eng, clock, T)
    assert eng.entry[T] is None  # blocked on every symbol
    broker.cancel, broker.query = cancel, query
    await eng.sync()
    assert not eng.blocked and not await broker.open_orders(S)
    await next_bar(eng, clock, T)
    assert eng.entry[T] is not None


async def test_lost_stop_is_replaced():
    eng, broker, clock = setup()
    await fill_entry(eng, clock)
    sid = eng.stop[S].client_id
    so = broker.orders[sid]
    so.status = OrderStatus.EXPIRED  # e.g. cancelled outside kasegu
    broker.emit(broker._copy(so))
    await eng.drain()
    assert "stop_lost" in kinds(eng)
    stops = open_stops(broker)
    assert len(stops) == 1 and stops[0].client_id != sid


async def test_restart_keeps_fill_time():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    eng2 = Engine(
        eng.cfg, create(eng.cfg.strategy), broker, eng.market, RULES, eng.journal, clock=clock
    )
    eng2.restore()
    await eng2.sync()
    pos = eng2.positions[S]
    assert pos.open_time == opened and len(open_stops(broker)) == 1
    await eng2.on_mark(MarkPrice(S, px, 0, 0, opened + 15 * MINUTE))
    assert eng2.trades[-1].reason == "time_exit" and broker.pos[S] == 0


async def test_stale_data_blocks_entry():
    eng, broker, clock = setup()
    i = eng.market.last_open(S) // MINUTE + 1
    c = eng.market[S].bars[-1].close - 0.05
    clock.t = (i + 1) * MINUTE + 30_000  # quote 30s old
    await quote(eng, c - 0.01, c, (i + 1) * MINUTE - 1)
    await eng.on_bar(bar(S, i, c))
    assert eng.entry[S] is None
    row = eng.journal.db.execute(
        "SELECT data FROM events WHERE kind = 'signal' ORDER BY id DESC"
    ).fetchone()
    assert "stale_data" in row[0]


async def test_per_symbol_budget_and_exposure():
    risk = {
        "margin_budget": 30,
        "leverage": 10,
        "max_spread_pct": 1,
        "max_symbol_notional": 300,
        "symbols": {T: {"margin_budget": 10, "max_notional": 120}},
    }
    eng, broker, clock = setup(cfg=epr_cfg(symbols=[S, T], risk=risk))
    await fill_entry(eng, clock, S)
    await next_bar(eng, clock, T)
    # no account wide position cap: T still trades with its own budget beside S's position
    assert eng.positions[S].has_qty
    o = eng.entry[T]
    assert o is not None and o.price * o.qty == pytest.approx(99, rel=0.02)  # 10 x 10 x 0.99
    # T's own cap below one trade: no order for T, S unaffected
    risk["symbols"][T]["max_notional"] = 80
    eng, broker, clock = setup(cfg=epr_cfg(symbols=[S, T], risk=risk))
    await next_bar(eng, clock, S)
    await next_bar(eng, clock, T)
    assert eng.entry[S] is not None and eng.entry[T] is None
    row = eng.journal.db.execute(
        "SELECT data FROM events WHERE kind = 'signal' AND symbol = ? ORDER BY id DESC", (T,)
    ).fetchone()
    assert "max_symbol_notional" in row[0]


def test_risk_symbols_must_be_traded():
    with pytest.raises(ValueError, match="not traded"):
        epr_cfg(risk={"symbols": {T: {"margin_budget": 5}}})


async def test_account_loss_stop_closes_everything_and_survives_restart():
    from kasegu.core.engine import reset_account_stop

    risk = {"margin_budget": 30, "leverage": 10, "max_spread_pct": 1, "max_account_loss": 5}
    cfg = epr_cfg(symbols=[S, T], risk=risk)
    eng, broker, clock = setup(cfg=cfg)
    px = await fill_entry(eng, clock, S)
    await next_bar(eng, clock, T)
    assert eng.positions[S].has_qty and eng.entry[T] is not None
    opened = eng.positions[S].open_time
    # 297 USDT long, -1%: about -3 USDT open loss, above the limit only with more
    await eng.on_mark(MarkPrice(S, px * 0.99, 0, 0, opened + MINUTE))
    assert not eng.stopped and eng.positions[S].has_qty
    await eng.on_mark(MarkPrice(S, px * 0.98, 0, 0, opened + 2 * MINUTE))
    assert eng.stopped.startswith("account loss")
    assert eng.entry[T] is None or not eng.entry[T].is_open  # resting entry pulled
    assert not eng.positions[S].has_qty and eng.trades[-1].reason == "account_stop"
    assert "account_stop" in kinds(eng)
    await next_bar(eng, clock, T)
    await next_bar(eng, clock, S)
    assert eng.entry[S] is None and eng.entry[T] is None  # nothing new opens
    assert eng.snapshot()["stopped"]

    # restart on the same journal: still stopped
    eng2 = Engine(cfg, create(cfg.strategy), PaperBroker(cfg.fee), eng.market, RULES, eng.journal,
                  clock=clock)  # fmt: skip
    eng2.restore()
    # the paper exit filled at the quote, not the mark, so the realized loss is smaller:
    # the saved stop, not the current pnl, keeps it stopped
    assert eng2.stopped.startswith("account loss")
    await next_bar(eng2, clock, S)
    assert eng2.entry[S] is None

    # reset: losses count from now, entries resume
    prev, _ = reset_account_stop(eng.journal)
    assert prev.startswith("account loss")
    eng3 = Engine(cfg, create(cfg.strategy), PaperBroker(cfg.fee), eng.market, RULES, eng.journal,
                  clock=clock)  # fmt: skip
    eng3.restore()
    assert not eng3.stopped and eng3.account_pnl() == pytest.approx(0)
    await next_bar(eng3, clock, S)
    assert eng3.entry[S] is not None


async def test_account_loss_off_by_default():
    eng, broker, clock = setup()
    px = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    await eng.on_mark(MarkPrice(S, px * 0.975, 0, 0, opened + MINUTE))
    assert not eng.stopped and eng.positions[S].has_qty


async def test_loss_limits_pause_epr15():
    cfg = epr_cfg(
        risk={
            "margin_budget": 30,
            "leverage": 10,
            "max_spread_pct": 1,
            "max_consecutive_losses": 1,
        }
    )
    eng, broker, clock = setup(cfg=cfg)
    px = await fill_entry(eng, clock)
    opened = eng.positions[S].open_time
    await eng.on_mark(MarkPrice(S, px * 0.99, 0, 0, opened + 15 * MINUTE))
    assert eng.trades[-1].net < 0 and eng.paused_until > clock.t
    await next_bar(eng, clock)
    assert eng.entry[S] is None


# ---- small account


def test_min_notional_and_lot_size():
    rules = {S: SymbolRules(S, 0.1, 0.01, 0.01, 1000, 100)}
    r = RiskManager(RiskConfig(margin_budget=5, leverage=10, max_spread_pct=1), rules)
    q = Quote(S, 99.9, 100.0, 5, 5, 0)
    # 5 x 10 x 0.99 = 49.5 USDT < 100 min notional: refused, never rounded up
    plan, why = r.check(Signal(S, Side.BUY, 88.0, 0), q, 0, 0)
    assert plan is None and why.startswith("notional")
    r = RiskManager(RiskConfig(margin_budget=15, leverage=10, max_spread_pct=1), rules)
    plan, why = r.check(Signal(S, Side.BUY, 88.07, 0), q, 0, 0)
    assert why == "ok" and plan.price == 88.0
    assert plan.qty == 1.68 and plan.notional >= 100  # 148.5 / 88 floored to the 0.01 step


def test_leverage_cannot_bypass_loss_budget():
    q = Quote(S, 99.9, 100.0, 5, 5, 0)
    sig = Signal(S, Side.BUY, 88.0, 0)

    def check(lev):
        cfg = RiskConfig(margin_budget=30, leverage=lev, max_spread_pct=1, max_loss_per_trade=9)
        r = RiskManager(cfg, RULES)
        r.stop_loss_pct = 3.0
        return r.check(sig, q, 0)[1]

    assert check(10) == "ok"  # 297 x 3% = 8.9 USDT
    with pytest.raises(ValueError):
        RiskConfig(leverage=20)  # above max_leverage
    cfg = RiskConfig(
        margin_budget=30,
        leverage=20,
        max_leverage=20,
        max_spread_pct=1,
        max_loss_per_trade=9,
    )
    r = RiskManager(cfg, RULES)
    r.stop_loss_pct = 3.0
    assert r.check(sig, q, 0)[1] == "max_loss_per_trade"
