from kasegu.config import ExitConfig
from kasegu.core.position import Position
from kasegu.core.types import Side


class ExitManager:
    """Fixed stop, three tier trailing stop and max holding time."""

    def __init__(self, cfg: ExitConfig):
        self.cfg = cfg

    def tier_stop(self, peak: float) -> tuple[int, float | None]:
        c = self.cfg
        if peak >= c.second_threshold:
            return 2, peak * (1 - c.trail_rate2)
        if peak >= c.first_threshold:
            return 1, peak * (1 - c.trail_rate1)
        if peak >= c.low_threshold:
            return 0, c.low_protect
        return -1, None

    def update(self, pos: Position, price: float, now: int) -> str | None:
        """Update peak / tier from a mark price. Returns exit reason or None."""
        if not pos.has_qty:
            return None
        p = pos.profit_pct(price)
        pos.peak_pct = max(pos.peak_pct, p)
        pos.trough_pct = min(pos.trough_pct, p)

        tier, stop = self.tier_stop(pos.peak_pct)
        pos.tier = max(pos.tier, tier)
        if stop is not None:
            # stop can only move up
            pos.stop_pct = stop if pos.stop_pct is None else max(pos.stop_pct, stop)

        if p <= -self.cfg.stop_loss_pct:
            return "stop_loss"
        if pos.stop_pct is not None and p < pos.stop_pct:
            return f"trail_t{pos.tier}"
        held = now - (pos.open_time or now)
        if held >= self.cfg.max_holding_minutes * 60_000:
            return "max_hold"
        return None

    def stop_price(self, pos: Position) -> float:
        """Exchange side protective stop price."""
        sl = self.cfg.stop_loss_pct / 100
        if pos.side is Side.BUY:
            return pos.entry_price * (1 - sl)
        return pos.entry_price * (1 + sl)
