from kasegu.core.types import Kline, Quote
from kasegu.market.data import MINUTE, MarketData


def bar(i, c=100.0, closed=True):
    return Kline("BTCUSDT", i * MINUTE, (i + 1) * MINUTE - 1, c, c + 1, c - 1, c, 1, closed)


def test_add_dup_gap_open():
    md = MarketData(["BTCUSDT"])
    assert md.add(bar(0)) == "ok"
    assert md.add(bar(0)) == "dup"
    assert md.add(bar(1, closed=False)) == "open"
    assert md.add(bar(3)) == "gap"
    assert md.add(bar(1)) == "ok"
    assert md.last_open("BTCUSDT") == MINUTE


def test_state_and_stale():
    md = MarketData(["BTCUSDT"])
    assert md.state("BTCUSDT") is None
    md.load([bar(i, 100 + i) for i in range(5)])
    md.on_quote(Quote("BTCUSDT", 104, 105, 1, 1, 5 * MINUTE))
    st = md.state("BTCUSDT")
    assert len(st) == 5 and st.close[-1] == 104 and st.ask == 105
    assert not md.stale("BTCUSDT", 5 * MINUTE + 1000, 10_000)
    assert md.stale("BTCUSDT", 5 * MINUTE + 20_000, 10_000)
