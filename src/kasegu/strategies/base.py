from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from kasegu.config import StrategyConfig
from kasegu.core.types import Signal


@dataclass(frozen=True, slots=True)
class MarketState:
    # closed 1m bars only, oldest first
    symbol: str
    time: int  # close time of the last bar (ms)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    bid: float
    ask: float

    def __len__(self) -> int:
        return len(self.close)


class Strategy(ABC):
    id: ClassVar[str]

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars required before the first signal."""

    @abstractmethod
    def on_bar(self, s: MarketState) -> Signal | None:
        """Return a candidate entry or None."""


_registry: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    _registry[cls.id] = cls
    return cls


def create(cfg: StrategyConfig) -> Strategy:
    if cfg.id not in _registry:
        raise ValueError(f"unknown strategy: {cfg.id} (have {sorted(_registry)})")
    return _registry[cfg.id](cfg)


def available() -> list[str]:
    return sorted(_registry)
