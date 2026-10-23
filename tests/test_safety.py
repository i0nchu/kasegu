import json

import pytest

from kasegu.app import write_status
from kasegu.core.engine import DAY
from kasegu.core.position import PosState, Trade
from kasegu.core.types import MarkPrice, OrderType, Side
from kasegu.events import Event, EventBus
from kasegu.notify import telegram_text, webhook_sink
from tests.test_engine import S, next_bar, quote, setup


def events(eng):
    rows = eng.journal.db.execute("SELECT kind FROM events").fetchall()
    return [r[0] for r in rows]


async def test_adopts_unknown_position_with_stop():
    eng, broker, clock = setup()
    await eng.on_mark(MarkPrice(S, 108.0, 0, 0, clock.t))
    broker.pos[S] = -0.5  # short opened outside kasegu
    await eng.sync()
    pos = eng.positions[S]
    assert pos.state is PosState.POSITION_OPEN and pos.side is Side.SELL and pos.qty == 0.5
    assert pos.entry_price == 108.0 and S not in eng.blocked
    stops = [o for o in await broker.open_orders(S) if o.type is OrderType.STOP_MARKET]
    assert len(stops) == 1 and stops[0].side is Side.BUY and stops[0].qty == 0.5
    assert "position_adopted" in events(eng)


async def test_resized_position_alerts():
    eng, broker, clock = setup()
    pos = eng.positions[S]
    pos.side, pos.qty, pos.entry_price, pos.state = Side.BUY, 0.5, 100.0, PosState.POSITION_OPEN
    broker.pos[S] = 0.3
    await eng.sync()
    assert pos.qty == 0.3 and "position_resized" in events(eng)


async def test_stop_rejected_closes_position():
    eng, broker, clock = setup()
    place = broker.place

    async def reject_stops(o):
        if o.type is OrderType.STOP_MARKET:
            return broker._reject(broker._copy(o), "test")
        return await place(o)

    broker.place = reject_stops
    await next_bar(eng, clock, 108.0)
    px = eng.entry[S].price
    await quote(eng, px - 0.2, px - 0.1, clock.t + 1)  # entry fills, stop rejected
    assert "stop_rejected" in events(eng)
    assert eng.positions[S].state is PosState.FLAT
    assert eng.trades[-1].reason == "stop_rejected"


def loss(ts, net):
    return Trade(S, Side.BUY, 1, 100, 100, ts, ts, net, 0, 0, 0, 0, "stop_loss")


async def test_daily_loss_pauses_until_next_day():
    eng, broker, clock = setup()
    eng.cfg.risk.max_daily_loss = 1.0
    eng._closed(S, loss(clock.t, -0.6))
    assert eng.paused_until == 0
    eng._closed(S, loss(clock.t, -0.6))
    assert eng.paused_until == (clock.t // DAY + 1) * DAY and "loss_limit" in events(eng)
    await next_bar(eng, clock, 108.0)
    assert eng.entry[S] is None
    # a new UTC day starts from zero
    eng._count(eng.paused_until + 1, -0.1)
    assert eng.day_net == pytest.approx(-0.1)


async def test_loss_streak_pause():
    eng, broker, clock = setup()
    eng.cfg.risk.max_consecutive_losses = 2
    eng.cfg.risk.loss_pause_min = 30
    eng._closed(S, loss(clock.t, -0.1))
    eng._closed(S, loss(clock.t, 0.2))  # a win resets the streak
    eng._closed(S, loss(clock.t, -0.1))
    assert eng.paused_until == 0
    eng._closed(S, loss(clock.t, -0.1))
    assert eng.paused_until == clock.t + 30 * 60_000 and eng.streak == 0


async def test_restore_counts_todays_trades():
    eng, broker, clock = setup()
    eng.cfg.risk.max_daily_loss = 1.0
    eng.journal.trade(loss(clock.t, -1.5), ts=clock.t)
    eng.restore()
    assert eng.day_net == pytest.approx(-1.5) and eng.paused_until > clock.t


async def test_snapshot_is_json(tmp_path):
    eng, broker, clock = setup()
    await next_bar(eng, clock, 108.0)
    snap = eng.snapshot()
    assert snap["symbols"][S]["entry_order"] == eng.entry[S].price
    path = tmp_path / "status.json"
    write_status(path, snap)
    assert json.loads(path.read_text())["strategy"] == "atr_amplitude"


async def test_event_bus_levels_and_flush():
    got, crit = [], []

    async def all_(ev):
        got.append(ev.kind)

    async def bad(ev):
        raise RuntimeError("down")

    async def only_crit(ev):
        crit.append(ev.kind)

    bus = EventBus()
    bus.add(all_)
    bus.add(bad)  # a failing sink does not stop the others
    bus.add(only_crit, "critical")
    bus.start()
    bus.publish(Event("trade_closed", "info", "x"))
    bus.publish(Event("stop_rejected", "critical", "y", S))
    await bus.close()
    assert got == ["trade_closed", "stop_rejected"] and crit == ["stop_rejected"]


async def test_webhook_payload():
    sent = []

    class Http:
        async def post(self, url, **kw):
            sent.append((url, kw["json"]))

    await webhook_sink(Http(), "http://ui/events", "testnet")(Event("x", "info", "m", S, 1))
    url, body = sent[0]
    assert url == "http://ui/events"
    assert body == {
        "source": "kasegu",
        "tag": "testnet",
        "kind": "x",
        "level": "info",
        "message": "m",
        "symbol": S,
        "ts": 1,
        "data": {},
    }
    assert telegram_text(Event("stop_rejected", "critical", "m", S), "mainnet").startswith(
        "🚨 [mainnet] stop_rejected BTCUSDT"
    )


async def test_engine_publishes_alerts():
    eng, broker, clock = setup()
    got = []

    async def sink(ev):
        got.append(ev)

    eng.events = EventBus()
    eng.events.add(sink)
    eng.events.start()
    eng.alert("stop_rejected", S, "boom", "critical", reason="x")
    await eng.events.close()
    assert got[0].kind == "stop_rejected" and got[0].data == {"reason": "x"}


def test_health_check(tmp_path):
    import time

    from kasegu.health import main as health

    p = tmp_path / "status.json"
    assert health([str(p)]) == 1
    write_status(p, {"ts": int(time.time() * 1000)})
    assert health([str(p), "60"]) == 0
    write_status(p, {"ts": int(time.time() * 1000) - 120_000})
    assert health([str(p), "60"]) == 1
