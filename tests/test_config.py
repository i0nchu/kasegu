import pytest
from pydantic import ValidationError

from kasegu.config import AppConfig, load_config


def test_load_default_config():
    cfg = load_config("configs/config.example.yaml")
    assert cfg.mode == "paper"
    assert cfg.strategy.id == "atr_amplitude"
    assert cfg.strategy.k == 3.0
    assert cfg.exit.stop_loss_pct == 0.6
    assert cfg.exit.max_holding_minutes == 30
    assert cfg.exchange.testnet


def test_symbols_upper():
    cfg = AppConfig(symbols=["btcusdt"])
    assert cfg.symbols == ["BTCUSDT"]


def test_default_mode_not_live():
    assert AppConfig(symbols=["BTCUSDT"]).mode == "paper"


def test_leverage_limit():
    with pytest.raises(ValidationError):
        AppConfig(symbols=["BTCUSDT"], risk={"leverage": 20, "max_leverage": 10})


def test_tier_order():
    with pytest.raises(ValidationError):
        AppConfig(symbols=["BTCUSDT"], exit={"first_threshold": 2.0, "second_threshold": 1.0})


def test_load_epr15_configs():
    from kasegu.strategies import create

    for path in ("configs/epr15.example.yaml", "configs/epr15.research.yaml"):
        cfg = load_config(path)
        assert cfg.mode == "paper" and cfg.exchange.testnet
        assert cfg.strategy.id == "epr15" and cfg.exit.policy == "time_only"
        create(cfg.strategy).check_exit(cfg.exit)

    def notional(r):
        return r.margin_budget * r.leverage * (1 - r.cost_reserve_pct / 100)

    # research replication: same coins and size as the frozen run
    res = load_config("configs/epr15.research.yaml")
    assert len(res.symbols) == 40 and notional(res.risk) == pytest.approx(297)
    # small account example: small size inside every cap
    cfg = load_config("configs/epr15.example.yaml")
    r = cfg.risk
    assert notional(r) < 50 and r.leverage <= r.max_leverage <= 3
    assert notional(r) * cfg.exit.stop_loss_pct / 100 <= r.max_loss_per_trade
    # every symbol can hold one trade, the sum of the caps is the worst case
    assert notional(r) <= r.max_symbol_notional < 2 * notional(r)
    assert 0 < r.max_account_loss <= len(cfg.symbols) * r.max_symbol_notional
    assert r.max_daily_loss > 0 and r.max_consecutive_losses > 0
