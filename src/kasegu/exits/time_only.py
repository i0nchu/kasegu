from kasegu.core.position import Position
from kasegu.exits.base import ExitPolicy, register


@register
class TimeOnlyExit(ExitPolicy):
    """Fixed stop plus a time exit; no trailing tiers ever activate.

    Tier settings in the config are ignored.
    """

    id = "time_only"

    def update(self, pos: Position, price: float, now: int) -> str | None:
        if not pos.has_qty:
            return None
        p = self.track(pos, price)
        if p <= -self.cfg.stop_loss_pct:
            return "stop_loss"
        if self.timed_out(pos, now):
            return "time_exit"
        return None
