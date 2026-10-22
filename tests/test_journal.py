from kasegu.core.types import Fill, Order, OrderType, Purpose, Side
from kasegu.storage.journal import Journal


def test_journal_roundtrip():
    j = Journal(":memory:", run_id="r1")
    j.event("signal", "BTCUSDT", price=1.0)
    o = Order("c1", "BTCUSDT", Side.BUY, OrderType.LIMIT, 1.0, Purpose.ENTRY, price=100.0)
    j.order(o, ts=1)
    j.fill(Fill("c1", "BTCUSDT", Side.BUY, 1.0, 100.0, 0.02, True, 2))
    j.save_state("BTCUSDT", {"peak": 1.5})
    assert j.load_state() == {"BTCUSDT": {"peak": 1.5}}
    assert j.db.execute("SELECT count(*) FROM fills").fetchone()[0] == 1
