from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
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
    # trailing: fixed stop + 3 tier trailing + max hold; time_only: fixed stop + max hold
    policy: Literal["trailing", "time_only"] = "trailing"
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


class SymbolRisk(BaseModel):
    """Per symbol override of the risk defaults; unset fields fall back to `risk`."""

    model_config = ConfigDict(extra="forbid")

    margin_budget: float | None = Field(None, gt=0)  # USDT margin per trade
    max_notional: float | None = Field(None, ge=0)  # USDT exposure cap, 0 = off


# removed in favour of per symbol caps and the account loss stop
_REMOVED_RISK = {
    "max_positions": "use risk.max_symbol_notional / risk.symbols.<SYMBOL>.max_notional",
    "max_notional": "use risk.max_symbol_notional per symbol and risk.max_account_loss",
}


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    margin_budget: float = Field(20.0, gt=0)  # USDT margin per trade, default for every symbol
    leverage: int = Field(5, ge=1)
    max_leverage: int = 10
    # exposure cap per symbol in USDT notional: open position + resting entry, 0 = off
    max_symbol_notional: float = Field(0.0, ge=0)
    symbols: dict[str, SymbolRisk] = {}  # overrides of margin_budget / max_notional
    # account stop: realized net of every trade in the journal + open unrealized pnl at or
    # below -max_account_loss USDT cancels entries, closes positions and stops the strategy
    # until it is reset (python -m kasegu --reset-account-stop), 0 = off
    max_account_loss: float = Field(0.0, ge=0)
    max_loss_per_trade: float = Field(0.0, ge=0)  # USDT lost at the fixed stop, 0 = off
    max_spread_pct: float = 0.05
    min_quote_qty: float = 0.0  # min top-of-book qty in base asset
    stale_after_s: float = 10.0
    cost_reserve_pct: float = 1.0  # % of notional kept for fees/slippage
    # circuit breakers on realized net pnl, 0 = off
    max_daily_loss: float = Field(0.0, ge=0)  # USDT per UTC day, pauses until next day
    max_consecutive_losses: int = Field(0, ge=0)
    loss_pause_min: int = Field(60, gt=0)  # pause after the loss streak

    @model_validator(mode="before")
    @classmethod
    def _removed(cls, data):
        if isinstance(data, dict):
            old = [k for k in _REMOVED_RISK if k in data]
            if old:
                hint = "; ".join(f"risk.{k} was removed, {_REMOVED_RISK[k]}" for k in old)
                raise ValueError(hint)
        return data

    @model_validator(mode="after")
    def _check_lev(self):
        if self.leverage > self.max_leverage:
            raise ValueError("leverage exceeds max_leverage")
        self.symbols = {k.upper(): v for k, v in self.symbols.items()}
        return self

    def budget(self, symbol: str) -> float:
        o = self.symbols.get(symbol)
        return o.margin_budget if o and o.margin_budget is not None else self.margin_budget

    def symbol_cap(self, symbol: str) -> float:
        o = self.symbols.get(symbol)
        return o.max_notional if o and o.max_notional is not None else self.max_symbol_notional


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
        extra = set(self.risk.symbols) - set(self.symbols)
        if extra:
            raise ValueError(f"risk.symbols has symbols that are not traded: {sorted(extra)}")
        return self


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_", env_file=".env", extra="ignore")

    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
