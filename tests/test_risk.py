import pytest

from kasegu.config import RiskConfig
from kasegu.core.risk import RiskManager
from kasegu.core.types import Quote, Side, Signal
from kasegu.exchange.rules import SymbolRules

RULES = {"BTCUSDT": SymbolRules("BTCUSDT", 0.1, 0.001, 0.001, 1000, 5)}
Q = Quote("BTCUSDT", 99.99, 100.0, 10, 10, 0)


def rm(**kw):
    return RiskManager(RiskConfig(margin_budget=20, leverage=5, cost_reserve_pct=0, **kw), RULES)


def sig(side=Side.BUY, px=99.03):
    return Signal("BTCUSDT", side, px, 0)


def test_plan_rounds_and_sizes():
    plan, why = rm().check(sig(), Q, 0, 0)
    assert why == "ok"
    assert plan.price == pytest.approx(99.0)
    assert plan.qty == pytest.approx(1.01)  # 100 / 99 floored to step


def test_short_rounds_up():
    plan, _ = rm().check(sig(Side.SELL, 101.01), Q, 0, 0)
    assert plan.price == pytest.approx(101.1)


@pytest.mark.parametrize(
    "kw,args,reason",
    [
        ({}, {"stale": True}, "stale_data"),
        ({}, {"unknown": True}, "unknown_order_state"),
    ],
)
def test_blocks(kw, args, reason):
    assert rm(**kw).check(sig(), Q, 0, 0, **args) == (None, reason)


def test_limits():
    assert rm(max_positions=1).check(sig(), Q, 1, 0)[1] == "max_positions"
    assert rm(max_notional=150).check(sig(), Q, 0, 100)[1] == "max_notional"
    wide = Quote("BTCUSDT", 99.0, 100.0, 10, 10, 0)
    assert rm().check(sig(), wide, 0, 0)[1].startswith("spread")
    assert rm().check(sig(px=100.5), Q, 0, 0)[1] == "would_take"
    assert rm(min_quote_qty=50).check(sig(), Q, 0, 0)[1] == "liquidity"
