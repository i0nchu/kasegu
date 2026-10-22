import pytest
from binance_common.errors import BadRequestError

from kasegu.config import ExchangeConfig, FeeConfig, Secrets
from kasegu.core.execution import new_id
from kasegu.core.position import PosState
from kasegu.core.types import Order, OrderStatus, OrderType, Purpose, Side, TimeInForce
from kasegu.exchange.trade import LiveBroker, num
from tests.test_engine import S, setup


class Resp:
    def __init__(self, d):
        self.d = d

    def data(self):
        return self.d


class Api:
    """Records calls; responses come from a dict of name -> value or callable."""

    def __init__(self, **resp):
        self.resp = resp
        self.calls = []

    def __getattr__(self, name):
        def fn(**kw):
            self.calls.append((name, kw))
            r = self.resp[name]
            if isinstance(r, Exception):
                raise r
            return Resp(r(**kw) if callable(r) else r)

        return fn


class Client:
    def __init__(self, **resp):
        self.rest_api = Api(**resp)


def make(**resp):
    b = LiveBroker(
        ExchangeConfig(), FeeConfig(maker=0.0002, taker=0.0005), Secrets(), Client(**resp)
    )
    seen = []
    b.listener = lambda o, f: seen.append((o, f))
    return b, b.client.rest_api, seen


def rest(o: Order, status="NEW", filled=0.0, avg=0.0):
    return {
        "clientOrderId": o.client_id,
        "symbol": o.symbol,
        "side": o.side.value,
        "type": o.type.value,
        "origQty": str(o.qty),
        "price": str(o.price or 0),
        "timeInForce": o.tif.value,
        "reduceOnly": o.reduce_only,
        "status": status,
        "executedQty": str(filled),
        "avgPrice": str(avg),
        "updateTime": 1,
    }


def entry(qty=0.3):
    return Order(
        new_id(Purpose.ENTRY),
        S,
        Side.BUY,
        OrderType.LIMIT,
        qty,
        Purpose.ENTRY,
        price=100.1,
        tif=TimeInForce.GTX,
    )


def trade_ev(o: Order, z, last, px, status, fee="0.006", cid=None, oid=1):
    return {
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "s": o.symbol,
            "c": cid or o.client_id,
            "S": o.side.value,
            "o": "LIMIT",
            "f": "GTX",
            "q": str(o.qty),
            "p": "100.1",
            "ap": str(px),
            "x": "TRADE",
            "X": status,
            "i": oid,
            "l": str(last),
            "z": str(z),
            "L": str(px),
            "N": "USDT",
            "n": fee,
            "T": 5,
            "m": True,
            "R": False,
        },
    }


def test_num_drops_float_noise():
    assert num(0.1 + 0.2) == 0.3


async def test_place_limit_sends_strings():
    o = entry(0.1 + 0.2)
    b, api, seen = make(new_order=lambda **kw: rest(o))
    res = await b.place(o)
    name, kw = api.calls[0]
    assert name == "new_order" and kw["quantity"] == 0.3 and kw["time_in_force"] == "GTX"
    assert kw["new_client_order_id"] == o.client_id and "reduce_only" not in kw
    assert res.status is OrderStatus.NEW and seen[0][1] is None


async def test_post_only_reject_and_unknown_raise():
    o = entry()
    b, _, seen = make(new_order=BadRequestError("maker only", -5022))
    res = await b.place(o)
    assert res.status is OrderStatus.REJECTED and res.reason == "post_only"
    b, _, _ = make(new_order=BadRequestError("timeout", -1007))
    with pytest.raises(BadRequestError):
        await b.place(o)


async def test_stream_fills_once_with_real_fee():
    o = entry()
    b, _, seen = make()
    b.on_user(trade_ev(o, 0.1, 0.1, 100.1, "PARTIALLY_FILLED", fee="0.002"))
    b.on_user(trade_ev(o, 0.1, 0.1, 100.1, "PARTIALLY_FILLED"))  # replayed event
    b.on_user(trade_ev(o, 0.3, 0.2, 100.1, "FILLED", fee="0.004"))
    fills = [f for _, f in seen if f]
    assert [f.qty for f in fills] == pytest.approx([0.1, 0.2])
    assert [f.fee for f in fills] == pytest.approx([0.002, 0.004]) and fills[0].maker


async def test_cancel_reports_fill_before_stream():
    o = entry()
    b, _, seen = make(cancel_order=lambda **kw: rest(o, "CANCELED", 0.1, 100.1))
    res = await b.cancel(S, o.client_id)
    f = seen[-1][1]
    assert res.status is OrderStatus.CANCELED and f.qty == pytest.approx(0.1) and f.maker
    assert f.fee == pytest.approx(0.1 * 100.1 * 0.0002)
    # the late stream event for the same qty adds nothing
    b.on_user(trade_ev(o, 0.1, 0.1, 100.1, "CANCELED"))
    assert sum(1 for _, x in seen if x) == 1


async def test_missing_order_query_returns_none():
    b, _, _ = make(query_order=BadRequestError("Order does not exist.", -2013))
    assert await b.query(S, new_id(Purpose.ENTRY)) is None


async def test_stop_is_algo_and_child_fill_maps_back():
    stop = Order(
        new_id(Purpose.STOP),
        S,
        Side.SELL,
        OrderType.STOP_MARKET,
        0.3,
        Purpose.STOP,
        stop_price=99.0,
        reduce_only=True,
    )
    algo = {
        "clientAlgoId": stop.client_id,
        "symbol": S,
        "side": "SELL",
        "quantity": "0.3",
        "triggerPrice": "99",
        "reduceOnly": True,
        "algoStatus": "NEW",
    }
    b, api, seen = make(new_algo_order=algo)
    res = await b.place(stop)
    name, kw = api.calls[0]
    assert name == "new_algo_order" and kw["algo_type"] == "CONDITIONAL"
    assert kw["client_algo_id"] == stop.client_id and kw["working_type"] == "MARK_PRICE"
    assert kw["reduce_only"] == "true" and res.is_open

    # child fill arrives before the ALGO_UPDATE that links it
    child = trade_ev(stop, 0.3, 0.3, 98.9, "FILLED", cid="autoclose-1", oid=77)
    child["o"]["m"] = False
    b.on_user(child)
    assert not any(f for _, f in seen)
    b.on_user(
        {
            "e": "ALGO_UPDATE",
            "E": 6,
            "o": {
                "caid": stop.client_id,
                "s": S,
                "S": "SELL",
                "q": "0.3",
                "X": "TRIGGERED",
                "ai": "77",
                "tp": "99",
                "R": True,
            },
        }
    )
    f = next(f for _, f in seen if f)
    assert f.client_id == stop.client_id and f.qty == pytest.approx(0.3) and not f.maker
    o = next(o for o, f in seen if f)
    assert o.purpose is Purpose.STOP and o.type is OrderType.STOP_MARKET


async def test_open_orders_skip_foreign():
    mine, other = entry(), entry()
    other.client_id = "web_123"
    b, _, _ = make(
        current_all_open_orders=[rest(mine), rest(other)],
        current_all_algo_open_orders=[],
    )
    assert [o.client_id for o in await b.open_orders(S)] == [mine.client_id]


async def test_sync_adopts_matching_stop():
    eng, broker, clock = setup()
    pos = eng.positions[S]
    # position and stop left over from a previous run
    pos.side, pos.qty, pos.entry_price, pos.state = Side.BUY, 0.5, 100.0, PosState.POSITION_OPEN
    broker.pos[S] = 0.5
    keep = Order(
        new_id(Purpose.STOP),
        S,
        Side.SELL,
        OrderType.STOP_MARKET,
        0.5,
        Purpose.STOP,
        stop_price=99.0,
        reduce_only=True,
    )
    extra = Order(
        new_id(Purpose.STOP),
        S,
        Side.SELL,
        OrderType.STOP_MARKET,
        0.2,
        Purpose.STOP,
        stop_price=99.0,
        reduce_only=True,
    )
    await broker.place(keep)
    await broker.place(extra)
    await eng.sync()
    open_ = await broker.open_orders(S)
    assert [o.client_id for o in open_] == [keep.client_id]
    assert eng.stop[S].client_id == keep.client_id


async def test_sync_cancels_stops_when_flat():
    eng, broker, clock = setup()
    stop = Order(
        new_id(Purpose.STOP),
        S,
        Side.SELL,
        OrderType.STOP_MARKET,
        0.5,
        Purpose.STOP,
        stop_price=99.0,
        reduce_only=True,
    )
    await broker.place(stop)
    await eng.sync()
    assert await broker.open_orders(S) == []


async def test_check_positions_blocks_mismatch():
    from kasegu.app import check_positions

    eng, _, _ = setup()

    class B:
        async def position_qty(self, s):
            return 0.4

    await check_positions(eng, B())
    assert S in eng.blocked


async def test_countdown_disarmed_before_stop():
    b, api, _ = make(auto_cancel_all_open_orders={})
    await b.arm(S, 120)
    await b.before_protect(S)
    assert [kw["countdown_time"] for _, kw in api.calls] == [120_000, 0]
    assert S not in b.armed
    await b.before_protect(S)  # not armed: no call
    assert len(api.calls) == 2

    b, api, _ = make(auto_cancel_all_open_orders=RuntimeError("timeout"))
    b.armed.add(S)
    await b.before_protect(S)
    assert S in b.armed  # retried by the live timer


async def test_engine_disarms_before_stop():
    eng, broker, clock = setup()
    calls = []

    async def hook(s):
        calls.append((s, eng.stop[s]))

    broker.before_protect = hook
    pos = eng.positions[S]
    pos.side, pos.qty, pos.entry_price, pos.state = Side.BUY, 0.5, 100.0, PosState.POSITION_OPEN
    broker.pos[S] = 0.5
    await eng.sync()
    assert calls == [(S, None)] and eng.stop[S].is_open


async def test_algo_cancel_trusts_exchange_ack():
    cid = new_id(Purpose.STOP)
    algo = {
        "clientAlgoId": cid,
        "symbol": S,
        "side": "SELL",
        "quantity": "0.3",
        "triggerPrice": "99",
        "reduceOnly": True,
        "algoStatus": "NEW",
    }
    b, _, seen = make(cancel_algo_order={"clientAlgoId": cid}, query_algo_order=algo)
    res = await b.cancel(S, cid)
    assert res.status is OrderStatus.CANCELED and seen[-1][0].status is OrderStatus.CANCELED


def test_user_stream_url():
    b, _, _ = make()
    url = b.user_url("abc")
    assert url == (
        "wss://fstream.binancefuture.com/private/ws?listenKey=abc"
        "&events=ORDER_TRADE_UPDATE/ALGO_UPDATE/ACCOUNT_UPDATE"
    )


def test_user_message_plain_or_wrapped():
    o = entry()
    b, _, seen = make()
    ev = trade_ev(o, 0.1, 0.1, 100.1, "PARTIALLY_FILLED")
    b._on_message({"stream": "x", "data": ev})
    b._on_message(trade_ev(o, 0.3, 0.2, 100.1, "FILLED"))
    b._on_message({"result": None, "id": 1})  # ack, ignored
    assert [f.qty for _, f in seen if f] == pytest.approx([0.1, 0.2])
