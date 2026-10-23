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


def test_load_epr15_config():
    from kasegu.strategies import create

    cfg = load_config("configs/epr15.example.yaml")
    assert cfg.mode == "paper" and cfg.exchange.testnet
    assert cfg.strategy.id == "epr15" and cfg.exit.policy == "time_only"
    assert len(cfg.symbols) == 40
    create(cfg.strategy).check_exit(cfg.exit)
    # same notional as the research, inside the loss budget
    r = cfg.risk
    notional = r.margin_budget * r.leverage * (1 - r.cost_reserve_pct / 100)
    assert notional == pytest.approx(297)
    assert notional * cfg.exit.stop_loss_pct / 100 <= r.max_loss_per_trade
