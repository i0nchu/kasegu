"""Conservative, deterministic maker-fill model for recorded public data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class BookGap(RuntimeError):
    pass


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
        if self.jitter_ms <= 0:
            return max(self.base_ms, 0)
        raw = hashlib.sha256(f"{self.seed}:{order_id}".encode()).digest()[:8]
        unit = int.from_bytes(raw, "big") / 2**64
        return max(self.base_ms + round((unit * 2 - 1) * self.jitter_ms), 0)


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
    synchronized: bool = False

    def install_snapshot(self, update_id: int, bids: list, asks: list) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = int(update_id)
        self.synchronized = False

    def apply(self, delta: DepthDelta) -> bool:
        if self.last_update_id is None or delta.symbol != self.symbol:
            raise BookGap("snapshot missing or symbol mismatch")
        if delta.final_id < self.last_update_id:
            return False
        if not self.synchronized:
            if not delta.first_id <= self.last_update_id <= delta.final_id:
                raise BookGap("first delta does not bridge snapshot")
            self.synchronized = True
        elif delta.previous_id != self.last_update_id:
            self.synchronized = False
            raise BookGap(
                f"depth gap for {self.symbol}: "
                f"pu={delta.previous_id}, expected={self.last_update_id}"
            )
        for levels, target in ((delta.bids, self.bids), (delta.asks, self.asks)):
            for price, qty in levels:
                if qty == 0:
                    target.pop(price, None)
                else:
                    target[price] = qty
        self.last_update_id = delta.final_id
        return True

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
    created_ms: int
    cancel_ms: int
    activation_ms: int
    queue_ahead: float | None = None
    filled_qty: float = 0.0
    status: FillStatus = FillStatus.WAITING

    @classmethod
    def create(cls, candidate: dict, latency_ms: int) -> ShadowOrder:
        created = int(candidate["created_ms"])
        return cls(
            candidate["candidate_id"],
            candidate["symbol"],
            candidate["side"].lower(),
            float(candidate["price"]),
            float(candidate["qty"]),
            created,
            int(candidate["cancel_ms"]),
            created + latency_ms,
        )

    def activate(self, book: LocalBook) -> None:
        depth = book.depth(self.side, self.price)
        if depth is None:
            self.status = FillStatus.UNKNOWN
        else:
            self.queue_ahead = depth

    def on_trade(self, trade: AggTrade) -> None:
        if self.status in {FillStatus.UNKNOWN, FillStatus.EXPIRED, FillStatus.FILLED}:
            return
        if trade.symbol != self.symbol or trade.trade_time < self.activation_ms:
            return
        if trade.trade_time >= self.cancel_ms:
            self.expire()
            return
        eligible = (
            self.side == "buy" and trade.taker_side == "sell" and trade.price <= self.price
        ) or (self.side == "sell" and trade.taker_side == "buy" and trade.price >= self.price)
        if not eligible or self.queue_ahead is None:
            return
        volume = trade.qty
        consumed = min(self.queue_ahead, volume)
        self.queue_ahead -= consumed
        fill = min(self.qty - self.filled_qty, volume - consumed)
        self.filled_qty += fill
        if self.filled_qty >= self.qty:
            self.status = FillStatus.FILLED
        elif self.filled_qty > 0:
            self.status = FillStatus.PARTIAL

    def expire(self) -> None:
        if self.status not in {FillStatus.FILLED, FillStatus.UNKNOWN}:
            self.status = FillStatus.PARTIAL if self.filled_qty else FillStatus.EXPIRED
