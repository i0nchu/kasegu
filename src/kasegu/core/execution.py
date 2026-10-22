import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable

from kasegu.config import FeeConfig
from kasegu.core.types import (
    Fill,
    Kline,
    MarkPrice,
    Order,
    OrderStatus,
    OrderType,
    Purpose,
    Quote,
    Side,
    TimeInForce,
)

log = logging.getLogger(__name__)

EPS = 1e-12
Listener = Callable[[Order, Fill | None], None]


_CODE = {Purpose.ENTRY: "n", Purpose.EXIT: "x", Purpose.STOP: "s"}
_PURPOSE = {v: k for k, v in _CODE.items()}


def new_id(purpose: Purpose) -> str:
    # <= 36 chars, unique across restarts
    return f"vt{_CODE[purpose]}{uuid.uuid4().hex[:24]}"


def purpose_of(client_id: str) -> Purpose | None:
    if not client_id.startswith("vt") or len(client_id) < 3:
        return None
    return _PURPOSE.get(client_id[2])


class Broker(ABC):
    """Order interface shared by paper, backtest and live."""

    def __init__(self):
        self.listener: Listener | None = None

    def emit(self, o: Order, f: Fill | None = None):
        if self.listener:
            self.listener(o, f)

    @abstractmethod
    async def place(self, o: Order) -> Order: ...

    @abstractmethod
    async def cancel(self, symbol: str, client_id: str) -> Order | None:
        """Cancel and return the final order state."""

    @abstractmethod
    async def query(self, symbol: str, client_id: str) -> Order | None: ...

    @abstractmethod
    async def open_orders(self, symbol: str) -> list[Order]: ...

    @abstractmethod
    async def position_qty(self, symbol: str) -> float:
        """Signed position size from the exchange."""

    # market hooks, used by simulated brokers
    def on_quote(self, q: Quote):  # noqa: B027
        pass

    def on_mark(self, m: MarkPrice):  # noqa: B027
        pass

    def on_bar(self, k: Kline):  # noqa: B027
        pass


class PaperBroker(Broker):
    """Simulated fills from live quotes.

    Limit orders fill at their price when the opposite side trades through.
    Market and triggered stop orders fill at the touch with slippage.
    """

    def __init__(self, fees: FeeConfig):
        super().__init__()
        self.fees = fees
        self.orders: dict[str, Order] = {}
        self.live: dict[str, dict[str, Order]] = {}  # symbol -> open orders
        self.pos: dict[str, float] = {}
        self.quotes: dict[str, Quote] = {}
        self.now = 0

    def _copy(self, o: Order) -> Order:
        return Order(**{k: getattr(o, k) for k in o.__slots__})

    async def place(self, o: Order) -> Order:
        o = self._copy(o)
        self.orders[o.client_id] = o
        self.live.setdefault(o.symbol, {})[o.client_id] = o
        q = self.quotes.get(o.symbol)
        pos = self.pos.get(o.symbol, 0.0)

        if o.reduce_only:
            if abs(pos) < EPS or (pos > 0) == (o.side is Side.BUY):
                return self._reject(o, "reduce_only")
            o.qty = min(o.qty, abs(pos))

        if o.type is OrderType.MARKET:
            if q is None:
                return self._reject(o, "no_quote")
            self._fill(o, o.remaining, self._taker_px(o.side, q), maker=False)
            return self._copy(o)

        if o.type is OrderType.LIMIT and o.tif is TimeInForce.GTX and q is not None:
            if (o.side is Side.BUY and o.price >= q.ask) or (
                o.side is Side.SELL and o.price <= q.bid
            ):
                return self._reject(o, "post_only")
        self.emit(self._copy(o))
        return self._copy(o)

    async def cancel(self, symbol: str, client_id: str) -> Order | None:
        o = self.orders.get(client_id)
        if o is None:
            return None
        if o.is_open:
            o.status = OrderStatus.CANCELED
            self.emit(self._copy(o))
        return self._copy(o)

    async def query(self, symbol: str, client_id: str) -> Order | None:
        o = self.orders.get(client_id)
        return self._copy(o) if o else None

    async def open_orders(self, symbol: str) -> list[Order]:
        return [self._copy(o) for o in self.orders.values() if o.symbol == symbol and o.is_open]

    async def position_qty(self, symbol: str) -> float:
        return self.pos.get(symbol, 0.0)

    def on_quote(self, q: Quote):
        self.quotes[q.symbol] = q
        self.now = max(self.now, q.time)
        for o in self._open(q.symbol, OrderType.LIMIT):
            if o.side is Side.BUY and q.ask < o.price:
                self._fill(o, o.remaining, o.price, maker=True)
            elif o.side is Side.SELL and q.bid > o.price:
                self._fill(o, o.remaining, o.price, maker=True)

    def on_mark(self, m: MarkPrice):
        self.now = max(self.now, m.time)
        for o in self._open(m.symbol, OrderType.STOP_MARKET):
            hit = m.price >= o.stop_price if o.side is Side.BUY else m.price <= o.stop_price
            if not hit:
                continue
            pos = self.pos.get(o.symbol, 0.0)
            if o.reduce_only and abs(pos) < EPS:
                o.status = OrderStatus.EXPIRED
                self.emit(self._copy(o))
                continue
            q = self.quotes.get(o.symbol)
            px = self._taker_px(o.side, q) if q else self._slip(o.side, m.price)
            qty = min(o.remaining, abs(pos)) if o.reduce_only else o.remaining
            self._fill(o, qty, px, maker=False)

    def _open(self, symbol: str, kind: OrderType) -> list[Order]:
        book = self.live.get(symbol, {})
        for cid in [c for c, o in book.items() if not o.is_open]:
            del book[cid]
        return [o for o in book.values() if o.type is kind]

    def _slip(self, side: Side, px: float) -> float:
        s = self.fees.slippage_pct / 100
        return px * (1 + s) if side is Side.BUY else px * (1 - s)

    def _taker_px(self, side: Side, q: Quote) -> float:
        return self._slip(side, q.ask if side is Side.BUY else q.bid)

    def _reject(self, o: Order, reason: str) -> Order:
        o.status, o.reason = OrderStatus.REJECTED, reason
        self.emit(self._copy(o))
        return self._copy(o)

    def _fill(self, o: Order, qty: float, px: float, maker: bool):
        if qty < EPS:
            return
        fee = qty * px * (self.fees.maker if maker else self.fees.taker)
        o.avg_price = (o.avg_price * o.filled_qty + px * qty) / (o.filled_qty + qty)
        o.filled_qty += qty
        o.status = OrderStatus.FILLED if o.remaining < EPS else OrderStatus.PARTIALLY_FILLED
        self.pos[o.symbol] = self.pos.get(o.symbol, 0.0) + qty * o.side.sign
        f = Fill(o.client_id, o.symbol, o.side, qty, px, fee, maker, self.now)
        self.emit(self._copy(o), f)


class Execution:
    """Local order book on top of a broker with idempotent submit."""

    def __init__(self, broker: Broker, retries: int = 3):
        self.broker = broker
        self.retries = retries
        self.orders: dict[str, Order] = {}
        self.unknown: set[str] = set()  # symbols with unknown order state

    async def submit(self, o: Order) -> Order:
        self.orders[o.client_id] = o
        for i in range(self.retries):
            try:
                res = await self.broker.place(o)
                self.unknown.discard(o.symbol)
                return self._merge(res)
            except Exception as e:
                log.warning("place %s failed: %s", o.client_id, e)
                # same client id: check whether the first attempt landed
                found = await self._safe_query(o)
                if found is not None:
                    return self._merge(found)
                await asyncio.sleep(0.2 * (i + 1))
        self.unknown.add(o.symbol)
        return o

    async def cancel(self, o: Order) -> Order:
        try:
            res = await self.broker.cancel(o.symbol, o.client_id)
        except Exception as e:
            log.warning("cancel %s failed: %s", o.client_id, e)
            res = None
        if res is None:
            res = await self._safe_query(o)
        if res is None:
            self.unknown.add(o.symbol)
            return o
        return self._merge(res)

    async def _safe_query(self, o: Order) -> Order | None:
        try:
            return await self.broker.query(o.symbol, o.client_id)
        except Exception as e:
            log.warning("query %s failed: %s", o.client_id, e)
            return None

    def _merge(self, o: Order) -> Order:
        cur = self.orders.get(o.client_id)
        # never move filled qty backwards
        if cur is not None and cur.filled_qty > o.filled_qty:
            return cur
        self.orders[o.client_id] = o
        return o

    def update(self, o: Order):
        self._merge(o)

    def get(self, client_id: str) -> Order | None:
        return self.orders.get(client_id)
