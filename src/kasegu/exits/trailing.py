from kasegu.core.position import Position
from kasegu.exits.base import ExitPolicy, register


@register
class TrailingExit(ExitPolicy):
    """Fixed stop, three tier trailing stop and max holding time."""

    id = "trailing"

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
        if not pos.has_qty:
            return None
        p = self.track(pos, price)

        tier, stop = self.tier_stop(pos.peak_pct)
        pos.tier = max(pos.tier, tier)
        if stop is not None:
            # stop can only move up
            pos.stop_pct = stop if pos.stop_pct is None else max(pos.stop_pct, stop)

        if p <= -self.cfg.stop_loss_pct:
            return "stop_loss"
        if pos.stop_pct is not None and p < pos.stop_pct:
            return f"trail_t{pos.tier}"
        if self.timed_out(pos, now):
            return "max_hold"
        return None


# original name, kept for existing imports
ExitManager = TrailingExit
