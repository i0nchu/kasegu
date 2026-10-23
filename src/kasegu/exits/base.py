from abc import ABC, abstractmethod
from typing import ClassVar

from kasegu.config import ExitConfig
from kasegu.core.position import Position
from kasegu.core.types import Side


class ExitPolicy(ABC):
    """Exit rules for an open position, independent of the entry strategy.

    Every policy keeps the fixed stop and the exchange side protective stop.
    """

    id: ClassVar[str]

    def __init__(self, cfg: ExitConfig):
        self.cfg = cfg

    def track(self, pos: Position, price: float) -> float:
        """Update MFE / MAE and return the current profit in %."""
        p = pos.profit_pct(price)
        pos.peak_pct = max(pos.peak_pct, p)
        pos.trough_pct = min(pos.trough_pct, p)
        return p

    def held_ms(self, pos: Position, now: int) -> int:
        return 0 if pos.open_time is None else now - pos.open_time

    def timed_out(self, pos: Position, now: int) -> bool:
        return self.held_ms(pos, now) >= self.cfg.max_holding_minutes * 60_000

    @abstractmethod
    def update(self, pos: Position, price: float, now: int) -> str | None:
        """Update from a mark price. Returns exit reason or None."""

    def stop_price(self, pos: Position) -> float:
        """Exchange side protective stop price."""
        sl = self.cfg.stop_loss_pct / 100
        if pos.side is Side.BUY:
            return pos.entry_price * (1 - sl)
        return pos.entry_price * (1 + sl)


_registry: dict[str, type[ExitPolicy]] = {}


def register(cls: type[ExitPolicy]) -> type[ExitPolicy]:
    _registry[cls.id] = cls
    return cls


def create_exit(cfg: ExitConfig) -> ExitPolicy:
    if cfg.policy not in _registry:
        raise ValueError(f"unknown exit policy: {cfg.policy} (have {sorted(_registry)})")
    return _registry[cfg.policy](cfg)
