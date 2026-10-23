from kasegu.config import ExitConfig
from kasegu.core.types import Side
from kasegu.strategies.atr_amplitude import AtrAmplitude
from kasegu.strategies.base import MarketState, register

# frozen at git tag alpha-epr15-v1, research/alpha/docs/HANDOFF_codex.md
FROZEN = {"k": 12.0, "ema_period": 240, "vol_period": 60, "reference": "ask"}
FROZEN_EXIT = {"policy": "time_only", "stop_loss_pct": 3.0, "max_holding_minutes": 15}


@register
class Epr15(AtrAmplitude):
    """Extreme passive reversal: long only, post-only buy 12 volatility units below the ask
    while the 1m close is under EMA240. Exit with the time_only policy.

    The parameters are frozen; a config that differs is refused instead of silently
    running a different rule.
    """

    id = "epr15"

    def __init__(self, cfg):
        super().__init__(cfg)
        bad = {k: getattr(cfg, k) for k, v in FROZEN.items() if getattr(cfg, k) != v}
        if bad or cfg.params:
            raise ValueError(f"epr15 parameters are frozen at {FROZEN}, got {bad or cfg.params}")

    def side(self, last: float, trend: float) -> Side | None:
        return Side.BUY if last < trend else None

    def atr_base(self, s: MarketState, ref: float) -> float:
        # research: mean(TR, 60) / close
        return float(s.close[-1])

    def check_exit(self, cfg: ExitConfig):
        bad = {k: getattr(cfg, k) for k, v in FROZEN_EXIT.items() if getattr(cfg, k) != v}
        if bad:
            raise ValueError(f"epr15 needs exit {FROZEN_EXIT}, got {bad}")
