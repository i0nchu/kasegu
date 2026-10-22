import pytest

from kasegu.config import ExitConfig
from kasegu.core.position import Position, PositionManager, PosState
from kasegu.core.types import Fill, Purpose, Side
from kasegu.exits.trailing import ExitManager

CFG = ExitConfig(
    stop_loss_pct=1.0,
    low_threshold=0.3,
    low_protect=0.05,
    first_threshold=0.6,
    trail_rate1=0.4,
    second_threshold=1.2,
    trail_rate2=0.25,
    max_holding_minutes=60,
)


def open_pos(side=Side.BUY, price=100.0, qty=1.0, t=0):
    p = Position("BTCUSDT", state=PosState.PENDING_ENTRY)
    p.apply(Fill("e", "BTCUSDT", side, qty, price, 0.02, True, t), Purpose.ENTRY)
    return p


def test_profit_pct_short():
    p = open_pos(Side.SELL)
    assert p.profit_pct(99.0) == pytest.approx(1.0)
    assert p.profit_pct(101.0) == pytest.approx(-1.0)


def test_stop_loss():
    ex, p = ExitManager(CFG), open_pos()
    assert ex.update(p, 99.5, 1) is None
    assert ex.update(p, 99.0, 2) == "stop_loss"
    assert ex.stop_price(p) == pytest.approx(99.0)
    assert ex.stop_price(open_pos(Side.SELL)) == pytest.approx(101.0)


def test_tiers_and_monotonic_stop():
    ex, p = ExitManager(CFG), open_pos()
    ex.update(p, 100.31, 1)
    assert (p.tier, p.stop_pct) == (0, pytest.approx(0.05))
    ex.update(p, 100.7, 2)
    assert p.tier == 1 and p.stop_pct == pytest.approx(0.7 * 0.6)
    ex.update(p, 101.3, 3)
    assert p.tier == 2 and p.stop_pct == pytest.approx(1.3 * 0.75)
    # trail stays when price retraces but is above stop
    assert ex.update(p, 101.0, 4) is None
    assert p.stop_pct == pytest.approx(1.3 * 0.75)
    assert ex.update(p, 100.9, 5) == "trail_t2"


def test_stop_never_lowers_on_tier_switch():
    # tier 1 at H=1.1 gives 0.66; tier 2 at H=1.2 gives 0.9 - check max() keeps higher
    cfg = CFG.model_copy(update={"trail_rate1": 0.1, "trail_rate2": 0.5})
    ex, p = ExitManager(cfg), open_pos()
    ex.update(p, 101.1, 1)
    s1 = p.stop_pct
    assert s1 == pytest.approx(1.1 * 0.9)
    ex.update(p, 101.2, 2)
    assert p.tier == 2
    assert p.stop_pct == pytest.approx(s1)


def test_tier0_exit():
    ex, p = ExitManager(CFG), open_pos()
    ex.update(p, 100.35, 1)
    assert ex.update(p, 100.04, 2) == "trail_t0"


def test_max_hold_from_first_fill():
    ex, p = ExitManager(CFG), open_pos(t=1_000)
    assert ex.update(p, 100.1, 1_000 + 59 * 60_000) is None
    assert ex.update(p, 100.1, 1_000 + 60 * 60_000) == "max_hold"


def test_partial_entry_and_close():
    p = Position("BTCUSDT", state=PosState.PENDING_ENTRY)
    p.apply(Fill("e", "BTCUSDT", Side.BUY, 0.4, 100.0, 0.01, True, 10), Purpose.ENTRY)
    assert p.state is PosState.POSITION_OPEN and p.open_time == 10
    p.apply(Fill("e", "BTCUSDT", Side.BUY, 0.6, 101.0, 0.01, True, 20), Purpose.ENTRY)
    assert p.entry_price == pytest.approx(100.6) and p.open_time == 10
    p.exit_reason = "max_hold"
    x = Fill("x", "BTCUSDT", Side.SELL, 0.5, 102.0, 0.03, False, 30)
    assert p.apply(x, Purpose.EXIT) is None
    t = p.apply(Fill("x", "BTCUSDT", Side.SELL, 0.5, 103.0, 0.03, False, 40), Purpose.EXIT)
    assert t.pnl == pytest.approx((102.5 - 100.6) * 1.0)
    assert t.fees == pytest.approx(0.08)
    assert t.reason == "max_hold" and t.hold_minutes == pytest.approx(30 / 60_000)
    assert p.state is PosState.FLAT and p.qty == 0


def test_state_roundtrip():
    p = open_pos(Side.SELL)
    p.peak_pct, p.stop_pct, p.tier = 1.5, 1.1, 2
    q = Position.from_dict(p.to_dict())
    assert q == p


def test_manager_counts():
    pm = PositionManager(["A", "B"])
    pm["A"].state = PosState.PENDING_ENTRY
    assert pm.open_count() == 1
