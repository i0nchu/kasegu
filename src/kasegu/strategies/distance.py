from abc import abstractmethod

from kasegu.core.types import Side, Signal
from kasegu.strategies import indicators as ind
from kasegu.strategies.base import MarketState, Strategy


class DistanceStrategy(Strategy):
    """EMA trend filter + limit price at a model specific distance.

    Subclasses only define the entry distance in %.
    """

    @property
    def warmup(self) -> int:
        # EMA needs a long warmup to remove the seed bias
        return max(self.cfg.ema_period * 3, self.cfg.vol_period + 1)

    def reference(self, s: MarketState) -> float:
        return s.ask if self.cfg.reference == "ask" else (s.bid + s.ask) / 2

    @abstractmethod
    def distance_pct(self, s: MarketState, side: Side, ref: float) -> tuple[float, dict] | None:
        """Return (distance %, metadata) or None."""

    def on_bar(self, s: MarketState) -> Signal | None:
        if len(s) < self.warmup or s.ask <= 0 or s.bid <= 0:
            return None
        trend = ind.ema(s.close, self.cfg.ema_period)
        last = s.close[-1]
        if last > trend:
            side = Side.BUY
        elif last < trend:
            side = Side.SELL
        else:
            return None
        if self.cfg.params.get("direction") == "against":
            # research: fade moves in the trend direction instead of against it
            side = side.opposite
        sides = self.cfg.params.get("sides", "both")
        if (sides == "long" and side is Side.SELL) or (sides == "short" and side is Side.BUY):
            return None

        ref = self.reference(s)
        res = self.distance_pct(s, side, ref)
        if res is None:
            return None
        dist, meta = res
        if dist <= 0:
            return None
        price = ref * (1 - dist / 100) if side is Side.BUY else ref * (1 + dist / 100)
        meta = {"strategy": self.id, "ema": trend, "ref": ref, "distance_pct": dist, **meta}
        return Signal(s.symbol, side, price, s.time, meta)
