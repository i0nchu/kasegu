"""Order-book reconstruction and conservative shadow maker-fill accounting.

Historical OHLC and aggTrades do not contain queue state.  This module therefore
returns UNKNOWN until a synchronized snapshot plus depth stream has been observed.
It never upgrades historical trade-through into a claimed maker fill.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class BookGap(RuntimeError):
    """Raised when a depth sequence gap makes local book state unknown."""


class FillStatus(StrEnum):
    WAITING = "waiting"
    PARTIAL = "partial"
    FILLED = "filled"
    EXPIRED = "expired"
    UNKNOWN = "unknown_book"


@dataclass(frozen=True)
class LatencyModel:
    base_ms: int
    jitter_ms: int = 0
    seed: int = 0

    def sample(self, order_id: str) -> int:
        """Stable symmetric jitter for repeatable latency sensitivity."""
        if self.jitter_ms <= 0:
            return max(self.base_ms, 0)
        raw = hashlib.sha256(f"{self.seed}:{order_id}".encode()).digest()[:8]
        unit = int.from_bytes(raw, "big") / 2**64
        jitter = round((unit * 2 - 1) * self.jitter_ms)
        return max(self.base_ms + jitter, 0)


@dataclass(frozen=True)
class DepthDelta:
    symbol: str
    event_time: int
    first_id: int
    final_id: int
    previous_id: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AggTrade:
    symbol: str
    trade_time: int
    price: float
    qty: float
    buyer_is_maker: bool

    @property
    def taker_side(self) -> str:
        return "sell" if self.buyer_is_maker else "buy"


@dataclass
class LocalBook:
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int | None = None
    event_time: int | None = None
    synchronized: bool = False

    def install_snapshot(
        self,
        last_update_id: int,
        bids: list[list[str]] | tuple[tuple[float, float], ...],
        asks: list[list[str]] | tuple[tuple[float, float], ...],
        event_time: int | None = None,
    ) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = int(last_update_id)
        self.event_time = event_time
        self.synchronized = False

    def apply(self, d: DepthDelta) -> bool:
        """Apply Binance diff-depth rules; return False for obsolete events."""
        if d.symbol != self.symbol or self.last_update_id is None:
            raise BookGap("snapshot missing or symbol mismatch")
        if d.final_id < self.last_update_id:
            return False
        if not self.synchronized:
            if not (d.first_id <= self.last_update_id <= d.final_id):
                raise BookGap("first delta does not bridge snapshot")
            self.synchronized = True
        elif d.previous_id != self.last_update_id:
            self.synchronized = False
            raise BookGap(
                f"depth gap for {self.symbol}: pu={d.previous_id}, expected={self.last_update_id}"
            )
        for levels, target in ((d.bids, self.bids), (d.asks, self.asks)):
            for price, qty in levels:
                if qty == 0:
                    target.pop(price, None)
                else:
                    target[price] = qty
        self.last_update_id = d.final_id
        self.event_time = d.event_time
        return True

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids and self.synchronized else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks and self.synchronized else None

    def depth(self, side: str, price: float) -> float | None:
        if not self.synchronized:
            return None
        return (self.bids if side == "buy" else self.asks).get(price, 0.0)


@dataclass
class ShadowOrder:
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    candidate_time: int
    cancel_time: int
    activation_time: int
    queue_ahead: float | None = None
    filled_qty: float = 0.0
    status: FillStatus = FillStatus.WAITING

    @classmethod
    def create(
        cls,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        candidate_time: int,
        cancel_time: int,
        latency_ms: int,
    ) -> ShadowOrder:
        return cls(
            order_id,
            symbol,
            side,
            price,
            qty,
            candidate_time,
            cancel_time,
            candidate_time + latency_ms,
        )

    def activate(self, book: LocalBook, queue_multiplier: float = 1.0) -> FillStatus:
        if not book.synchronized:
            self.status = FillStatus.UNKNOWN
            return self.status
        depth = book.depth(self.side, self.price)
        if depth is None:
            self.status = FillStatus.UNKNOWN
            return self.status
        self.queue_ahead = depth * queue_multiplier
        return self.status

    def on_trade(self, trade: AggTrade) -> float:
        """Conservatively count only opposite-aggressor volume at/through our price."""
        if self.status in (FillStatus.UNKNOWN, FillStatus.EXPIRED, FillStatus.FILLED):
            return 0.0
        if trade.symbol != self.symbol or trade.trade_time < self.activation_time:
            return 0.0
        if trade.trade_time >= self.cancel_time:
            self.expire(trade.trade_time)
            return 0.0
        eligible = (
            self.side == "buy" and trade.taker_side == "sell" and trade.price <= self.price
        ) or (self.side == "sell" and trade.taker_side == "buy" and trade.price >= self.price)
        if not eligible or self.queue_ahead is None:
            return 0.0
        volume = trade.qty
        consumed = min(self.queue_ahead, volume)
        self.queue_ahead -= consumed
        volume -= consumed
        fill = min(self.qty - self.filled_qty, volume)
        self.filled_qty += fill
        if self.filled_qty >= self.qty:
            self.status = FillStatus.FILLED
        elif self.filled_qty > 0:
            self.status = FillStatus.PARTIAL
        return fill

    def expire(self, now: int) -> FillStatus:
        if now >= self.cancel_time and self.status not in (FillStatus.FILLED, FillStatus.UNKNOWN):
            self.status = FillStatus.EXPIRED if self.filled_qty == 0 else FillStatus.PARTIAL
        return self.status
