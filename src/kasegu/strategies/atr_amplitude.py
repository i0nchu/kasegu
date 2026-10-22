from kasegu.core.types import Side
from kasegu.strategies import indicators as ind
from kasegu.strategies.base import MarketState, register
from kasegu.strategies.distance import DistanceStrategy


@register
class AtrAmplitude(DistanceStrategy):
    id = "atr_amplitude"

    def distance_pct(self, s: MarketState, side: Side, ref: float):
        n = self.cfg.vol_period
        atr_pct = ind.atr(s.high, s.low, s.close, n) / ref * 100
        amp_pct = ind.amplitude_pct(s.high, s.low, s.close, n)
        dist = (atr_pct + amp_pct) / 2 * self.cfg.k
        return dist, {"atr_pct": atr_pct, "amp_pct": amp_pct, "k": self.cfg.k}
