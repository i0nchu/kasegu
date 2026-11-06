from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from kasegu.config import ExchangeConfig


class BoldConfig(BaseModel):
    """Defaults: BTC 10x long, take profit at 2x equity, stop at 0.1x, 30 days max."""

    symbol: str = "BTCUSDT"
    side: int = Field(1)  # +1 long, -1 short
    stake: float = Field(100.0, gt=0)  # USD per try, every try starts from the stake again
    leverage: float = Field(10.0, gt=1)
    target: float = Field(2.0, gt=1)
    bust: float = Field(0.1, ge=0, lt=1)
    max_days: int = Field(30, gt=0)
    taker: float = 0.0005
    slip: float = 0.0005
    maintenance: float = 0.004  # maintenance margin rate, the stop must come before liquidation
    decision_delay_s: float = 20.0
    out_dir: Path = Path("data/bold")
    exchange: ExchangeConfig = ExchangeConfig()

    @model_validator(mode="after")
    def _stop_before_liquidation(self):
        if self.side not in (1, -1):
            raise ValueError("side must be 1 or -1")
        cost = 2 * self.leverage * (self.taker + self.slip)
        stop_move = (1 - self.bust - cost) / self.leverage
        liq_move = 1 / self.leverage - self.maintenance
        if stop_move >= liq_move:
            raise ValueError(
                f"stop at {stop_move:.2%} price move is beyond liquidation (~{liq_move:.2%}), "
                "lower the leverage or raise bust"
            )
        return self


def load_bold_config(path: str | Path) -> BoldConfig:
    return BoldConfig(**(yaml.safe_load(Path(path).read_text()) or {}))
