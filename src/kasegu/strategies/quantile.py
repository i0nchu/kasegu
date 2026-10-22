import numpy as np

from kasegu.core.types import Side
from kasegu.strategies.base import MarketState, register
from kasegu.strategies.distance import DistanceStrategy


def next_bar_excursion(s: MarketState, side: Side, n: int) -> np.ndarray:
    """Adverse move of the next bar vs this close, in %, for the last n closed pairs.

    Long: how far the next low dropped below close. Short: how far the next high rose above.
    Only uses bars that already closed, so no look-ahead.
    """
    c = s.close[-n - 1 : -1]
    if side is Side.BUY:
        x = (c - s.low[-n:]) / c * 100
    else:
        x = (s.high[-n:] - c) / c * 100
    return np.maximum(x, 0)


@register
class Quantile(DistanceStrategy):
    # params: q (quantile, 0-1), lookback (bars); distance = quantile * k
    id = "quantile"

    def __init__(self, cfg):
        super().__init__(cfg)
        if not 0 < self.q < 1 or self.lookback < 10:
            raise ValueError("quantile needs 0 < q < 1 and lookback >= 10")

    @property
    def q(self) -> float:
        return float(self.cfg.params.get("q", 0.95))

    @property
    def lookback(self) -> int:
        return int(self.cfg.params.get("lookback", 720))

    @property
    def warmup(self) -> int:
        return max(super().warmup, self.lookback + 1)

    def distance_pct(self, s: MarketState, side: Side, ref: float):
        x = next_bar_excursion(s, side, self.lookback)
        qv = float(np.quantile(x, self.q))
        return qv * self.cfg.k, {
            "quantile_pct": qv,
            "q": self.q,
            "lookback": self.lookback,
            "k": self.cfg.k,
        }
