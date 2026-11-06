from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from kasegu.config import ExchangeConfig


class GridEpisodeConfig(BaseModel):
    levels: int = Field(6, ge=1)
    mult: float = Field(1.5, gt=0)
    notional: float = Field(104.0, gt=0)  # USDT of all orders of one grid
    maker: float = 0.0002
    taker: float = 0.0005
    slip_in: float = 0.0002
    slip_out: float = 0.0005
    pen: float = 0.0005
    hold_hours: int = Field(72, gt=0)


class GridConfig(BaseModel):
    """Trend-aligned martingale grid, defaults = frozen candidate MGT-BOTH-k1.0-RV3-K3."""

    balance: float = Field(100.0, gt=0)  # USD account, margin per slot = balance / slots
    slots: int = Field(3, ge=1)  # coins at the same time
    top: int = 40  # universe: top by 30-day mean quote volume
    min_age_days: int = 60
    rv3_min: float = 0.0696  # 3-day realized daily vol needed to trade
    k: float = Field(1.0, gt=0)  # step = k * 7-day daily vol
    step_floor: float = 0.005
    step_cap: float = 0.10
    btc_sma: int = 100
    coin_sma: int = 50
    max_account_loss: float = Field(40.0, ge=0)  # USD, realized + open; stops new grids, 0 = off
    decision_delay_s: float = 20.0  # after 00:00 UTC, so the last 5m bar has closed
    out_dir: Path = Path("data/grid")
    grid: GridEpisodeConfig = GridEpisodeConfig()
    exchange: ExchangeConfig = ExchangeConfig()


def load_grid_config(path: str | Path) -> GridConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return GridConfig(**data)
