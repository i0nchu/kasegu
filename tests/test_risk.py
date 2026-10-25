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
    assert rm(**kw).check(sig(), Q, 0, **args) == (None, reason)


def test_per_symbol_budget_and_cap():
    cfg = RiskConfig(
        margin_budget=20,
        leverage=5,
        cost_reserve_pct=0,
        max_symbol_notional=1000,
        symbols={"btcusdt": {"margin_budget": 10, "max_notional": 40}},
    )
    r = RiskManager(cfg, RULES)
    assert cfg.budget("BTCUSDT") == 10 and cfg.budget("ETHUSDT") == 20
    assert cfg.symbol_cap("BTCUSDT") == 40 and cfg.symbol_cap("ETHUSDT") == 1000
    # 10 x 5 = 50 USDT per trade on BTC, above its own 40 cap
    assert r.check(sig(), Q, 0)[1] == "max_symbol_notional"


def test_removed_keys_are_refused():
    with pytest.raises(ValueError, match="max_positions was removed"):
        RiskConfig(max_positions=3)
    with pytest.raises(ValueError, match="max_notional was removed"):
        RiskConfig(max_notional=500)
    with pytest.raises(ValueError):
        RiskConfig(max_postions=3)  # typos are refused too


def test_limits():
    # 1.01 x 99 = 100 USDT order: fits a 150 cap when flat, not on top of 100 already there
    assert rm(max_symbol_notional=150).check(sig(), Q, 0)[1] == "ok"
    assert rm(max_symbol_notional=150).check(sig(), Q, 100)[1] == "max_symbol_notional"
    wide = Quote("BTCUSDT", 99.0, 100.0, 10, 10, 0)
    assert rm().check(sig(), wide, 0, 0)[1].startswith("spread")
    assert rm().check(sig(px=100.5), Q, 0, 0)[1] == "would_take"
    assert rm(min_quote_qty=50).check(sig(), Q, 0, 0)[1] == "liquidity"
