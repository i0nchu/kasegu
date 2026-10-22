from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["backtest", "paper", "live"]


class StrategyConfig(BaseModel):
    id: str = "atr_amplitude"
    k: float = Field(2.0, gt=0)
    ema_period: int = 240
    vol_period: int = 60
    reference: Literal["ask", "mid"] = "ask"
    params: dict[str, Any] = {}  # model specific parameters


class ExitConfig(BaseModel):
    # all values are unleveraged price return in %
    stop_loss_pct: float = Field(1.0, gt=0)
    low_threshold: float = 0.3
    low_protect: float = 0.05
    first_threshold: float = 0.6
    trail_rate1: float = Field(0.4, ge=0, lt=1)
    second_threshold: float = 1.2
    trail_rate2: float = Field(0.25, ge=0, lt=1)
    max_holding_minutes: int = Field(60, gt=0)

    @model_validator(mode="after")
    def _check_order(self):
        if not self.low_threshold <= self.first_threshold <= self.second_threshold:
            raise ValueError("tier thresholds must be ascending")
        return self


class RiskConfig(BaseModel):
    margin_budget: float = Field(20.0, gt=0)  # USDT per trade
    leverage: int = Field(5, ge=1)
    max_leverage: int = 10
    max_positions: int = Field(3, ge=1)
    max_notional: float = 500.0  # total exposure in USDT
    max_spread_pct: float = 0.05
    min_quote_qty: float = 0.0  # min top-of-book qty in base asset
    stale_after_s: float = 10.0
    cost_reserve_pct: float = 1.0  # % of notional kept for fees/slippage
    # circuit breakers on realized net pnl, 0 = off
    max_daily_loss: float = Field(0.0, ge=0)  # USDT per UTC day, pauses until next day
    max_consecutive_losses: int = Field(0, ge=0)
    loss_pause_min: int = Field(60, gt=0)  # pause after the loss streak

    @model_validator(mode="after")
    def _check_lev(self):
        if self.leverage > self.max_leverage:
            raise ValueError("leverage exceeds max_leverage")
        return self


class FeeConfig(BaseModel):
    maker: float = 0.0002
    taker: float = 0.0005
    slippage_pct: float = 0.02
    use_account_rate: bool = True


class ExchangeConfig(BaseModel):
    testnet: bool = True  # order endpoints
    data_testnet: bool = False  # market data endpoints
    rest_timeout_ms: int = 10_000
    history_bars: int = Field(1000, le=1500)
    heartbeat_s: float = 30.0


class BacktestConfig(BaseModel):
    start: date | None = None  # inclusive, UTC days
    end: date | None = None
    cache_dir: Path = Path("data/history")
    out_dir: Path = Path("data/backtests")


class LiveConfig(BaseModel):
    # exchange side countdown that cancels open orders if the bot stops refreshing it;
    # it also cancels stops, so it only runs while a symbol is flat
    # 0 = off; check with `python -m kasegu.exchange.check` before enabling
    auto_cancel_s: int = Field(0, ge=0)
    keepalive_min: float = Field(30.0, gt=0)  # listen key keepalive


class AlertConfig(BaseModel):
    min_level: Literal["info", "warning", "critical"] = "warning"  # for telegram
    telegram: bool = False  # needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    webhook_url: str = ""  # POST one JSON event per request, e.g. a WebUI
    webhook_min_level: Literal["info", "warning", "critical"] = "info"
    events_path: Path | None = Path("data/events.jsonl")  # all events as JSON lines


class PaperConfig(BaseModel):
    balance: float = 1000.0


class AppConfig(BaseModel):
    mode: Mode = "paper"
    symbols: list[str] = Field(min_length=1)
    timeframe: Literal["1m"] = "1m"
    strategy: StrategyConfig = StrategyConfig()
    exit: ExitConfig = ExitConfig()
    risk: RiskConfig = RiskConfig()
    fee: FeeConfig = FeeConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    paper: PaperConfig = PaperConfig()
    live: LiveConfig = LiveConfig()
    alerts: AlertConfig = AlertConfig()
    backtest: BacktestConfig = BacktestConfig()
    db_path: Path = Path("data/journal.db")
    status_path: Path | None = Path("data/status.json")  # snapshot for health checks / WebUI

    @model_validator(mode="after")
    def _upper(self):
        self.symbols = [s.upper() for s in self.symbols]
        return self


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_", env_file=".env", extra="ignore")

    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
