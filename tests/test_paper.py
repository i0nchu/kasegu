import pytest

from kasegu.config import FeeConfig
from kasegu.core.execution import Execution, PaperBroker, new_id, purpose_of
from kasegu.core.types import (
    MarkPrice,
    Order,
    OrderStatus,
    OrderType,
    Purpose,
    Quote,
    Side,
    TimeInForce,
)

FEES = FeeConfig(maker=0.0002, taker=0.0005, slippage_pct=0.0)


def q(bid, ask, t=0):
    return Quote("BTCUSDT", bid, ask, 1, 1, t)


def limit(side, px, qty=1.0):
    return Order(
        new_id(Purpose.ENTRY),
        "BTCUSDT",
        side,
        OrderType.LIMIT,
        qty,
        Purpose.ENTRY,
        price=px,
        tif=TimeInForce.GTX,
    )


def test_client_id():
    cid = new_id(Purpose.STOP)
    assert len(cid) <= 36 and purpose_of(cid) is Purpose.STOP
    assert purpose_of(new_id(Purpose.EXIT)) is Purpose.EXIT
    assert purpose_of(new_id(Purpose.ENTRY)) is Purpose.ENTRY
    assert purpose_of("other") is None


async def test_post_only_reject_and_fill():
    b, events = PaperBroker(FEES), []
    b.listener = lambda o, f: events.append((o, f))
    b.on_quote(q(99.9, 100.0))
    assert (await b.place(limit(Side.BUY, 100.0))).status is OrderStatus.REJECTED
    o = await b.place(limit(Side.BUY, 99.0))
    assert o.status is OrderStatus.NEW
    b.on_quote(q(98.9, 99.0, 1))  # touch only
    assert (await b.query("BTCUSDT", o.client_id)).status is OrderStatus.NEW
    b.on_quote(q(98.8, 98.9, 2))
    o2, f = events[-1]
    assert o2.status is OrderStatus.FILLED and f.maker and f.price == 99.0
    assert f.fee == pytest.approx(99.0 * 0.0002)
    assert await b.position_qty("BTCUSDT") == 1.0


async def test_reduce_only_and_stop():
    b = PaperBroker(FEES)
    b.on_quote(q(99.9, 100.0))
    ro = Order(
        new_id(Purpose.EXIT),
        "BTCUSDT",
        Side.SELL,
        OrderType.MARKET,
        1,
        Purpose.EXIT,
        reduce_only=True,
    )
    assert (await b.place(ro)).reason == "reduce_only"
    b.pos["BTCUSDT"] = 1.0
    st = Order(
        new_id(Purpose.STOP),
        "BTCUSDT",
        Side.SELL,
        OrderType.STOP_MARKET,
        5,
        Purpose.STOP,
        stop_price=99.0,
        reduce_only=True,
    )
    st = await b.place(st)
    assert st.qty == 1.0
    b.on_mark(MarkPrice("BTCUSDT", 99.5, 0, 0, 1))
    assert (await b.query("BTCUSDT", st.client_id)).is_open
    b.on_quote(q(98.9, 99.0, 2))
    b.on_mark(MarkPrice("BTCUSDT", 99.0, 0, 0, 3))
    done = await b.query("BTCUSDT", st.client_id)
    assert done.status is OrderStatus.FILLED and done.avg_price == 98.9
    assert b.pos["BTCUSDT"] == 0


class Flaky(PaperBroker):
    def __init__(self):
        super().__init__(FEES)
        self.calls = 0

    async def place(self, o):
        self.calls += 1
        res = await super().place(o)
        if self.calls == 1:
            raise ConnectionError("timeout after send")
        return res


async def test_submit_is_idempotent():
    b = Flaky()
    b.on_quote(q(99.9, 100.0))
    ex = Execution(b)
    res = await ex.submit(limit(Side.BUY, 99.0))
    assert res.status is OrderStatus.NEW
    assert b.calls == 1 and len(b.orders) == 1
