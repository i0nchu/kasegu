from kasegu.core.types import Side
from kasegu.strategies import indicators as ind
from kasegu.strategies.base import MarketState, register
from kasegu.strategies.distance import DistanceStrategy


@register
class AtrOnly(DistanceStrategy):
    id = "atr_only"

    def distance_pct(self, s: MarketState, side: Side, ref: float):
        atr_pct = ind.atr(s.high, s.low, s.close, self.cfg.vol_period) / ref * 100
        return atr_pct * self.cfg.k, {"atr_pct": atr_pct, "k": self.cfg.k}
