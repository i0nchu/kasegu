import pytest
from pydantic import ValidationError

from kasegu.config import AppConfig, load_config


def test_load_default_config():
    cfg = load_config("configs/config.yaml")
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
