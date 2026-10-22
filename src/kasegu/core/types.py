from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    GTX = "GTX"  # post-only


class OrderStatus(StrEnum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Purpose(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class Kline:
    symbol: str
    open_time: int  # ms
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    time: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid * 100


@dataclass(frozen=True, slots=True)
class MarkPrice:
    symbol: str
    price: float
    funding_rate: float
    next_funding_time: int
    time: int


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: Side
    limit_price: float
    signal_time: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Order:
    client_id: str
    symbol: str
    side: Side
    type: OrderType
    qty: float
    purpose: Purpose
    price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_price: float = 0.0
    reason: str = ""
    time: int = 0

    @property
    def remaining(self) -> float:
        return self.qty - self.filled_qty

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)


@dataclass(frozen=True, slots=True)
class Fill:
    client_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float  # in quote asset
    maker: bool
    time: int
