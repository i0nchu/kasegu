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

    def side(self, last: float, trend: float) -> Side | None:
        """Trend following: buy dips above the EMA, sell rallies below it."""
        if last > trend:
            return Side.BUY
        if last < trend:
            return Side.SELL
        return None

    def reference(self, s: MarketState) -> float:
        return s.ask if self.cfg.reference == "ask" else (s.bid + s.ask) / 2

    @abstractmethod
    def distance_pct(self, s: MarketState, side: Side, ref: float) -> tuple[float, dict] | None:
        """Return (distance %, metadata) or None."""

    def on_bar(self, s: MarketState) -> Signal | None:
        if len(s) < self.warmup or s.ask <= 0 or s.bid <= 0:
            return None
        trend = ind.ema(s.close, self.cfg.ema_period)
        side = self.side(s.close[-1], trend)
        if side is None:
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
