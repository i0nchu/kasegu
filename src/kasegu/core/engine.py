import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable

from kasegu.config import AppConfig
from kasegu.core.execution import Broker, Execution, new_id, purpose_of
from kasegu.core.position import Position, PositionManager, PosState, Trade
from kasegu.core.risk import RiskManager
from kasegu.core.types import (
    Fill,
    Kline,
    MarkPrice,
    Order,
    OrderStatus,
    OrderType,
    Purpose,
    Quote,
    TimeInForce,
)
from kasegu.exchange.rules import SymbolRules
from kasegu.exits.trailing import ExitManager
from kasegu.market.data import MINUTE, MarketData
from kasegu.storage.journal import Journal
from kasegu.strategies.base import Strategy

log = logging.getLogger(__name__)

Backfill = Callable[[str, int], Awaitable[list[Kline]]]


def wall_ms() -> int:
    return int(time.time() * 1000)


class Engine:
    """Event driven core shared by backtest, paper and live."""

    def __init__(
        self,
        cfg: AppConfig,
        strategy: Strategy,
        broker: Broker,
        market: MarketData,
        rules: dict[str, SymbolRules],
        journal: Journal,
        backfill: Backfill | None = None,
        clock: Callable[[], int] = wall_ms,
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.broker = broker
        self.market = market
        self.rules = rules
        self.journal = journal
        self.backfill = backfill
        self.clock = clock
        self.exec = Execution(broker)
        self.risk = RiskManager(cfg.risk, rules)
        self.exits = ExitManager(cfg.exit)
        self.positions = PositionManager(cfg.symbols)
        self.entry: dict[str, Order | None] = dict.fromkeys(cfg.symbols)
        self.stop: dict[str, Order | None] = dict.fromkeys(cfg.symbols)
        self.close: dict[str, Order | None] = dict.fromkeys(cfg.symbols)
        self.funding: dict[str, tuple[int, float]] = {}
        self.charged: dict[str, int] = {}
        self.trades: list[Trade] = []
        self.halted = False  # blocks new entries only
        self.blocked: set[str] = set()
        self.pending: deque[tuple[Order, Fill | None]] = deque()
        broker.listener = lambda o, f: self.pending.append((o, f))

    # ---- startup / recovery

    def restore(self):
        for s, d in self.journal.load_state().items():
            if s in self.positions.pos:
                self.positions.pos[s] = Position.from_dict(d)

    async def sync(self):
        """Reconcile local state with the exchange. Exchange state wins."""
        for s in self.cfg.symbols:
            stops = []
            for o in await self.broker.open_orders(s):
                p = purpose_of(o.client_id)
                if p is Purpose.STOP:
                    stops.append(o)
                elif p is not None:
                    await self.exec.cancel(o)
            await self.drain()
            pos, qty = self.positions[s], await self.broker.position_qty(s)
            if abs(qty) < 1e-12:
                if pos.state is not PosState.FLAT:
                    self.journal.event("sync_reset", s, local=pos.to_dict())
                pos.reset()
                self.blocked.discard(s)
                for o in stops:
                    await self.exec.cancel(o)
            elif not pos.has_qty or pos.side.sign * qty < 0:
                # position without local record: keep its stops, no new entries
                self.blocked.add(s)
                self.journal.event("sync_unknown_position", s, qty=qty)
                log.error("%s has exchange position %s without local state", s, qty)
            else:
                pos.qty = abs(qty)
                pos.state = PosState.POSITION_OPEN
                # adopt a matching stop from before the restart, drop the rest
                keep = next(
                    (
                        o
                        for o in stops
                        if abs(o.qty - pos.qty) < 1e-12 and o.side is pos.side.opposite
                    ),
                    None,
                )
                for o in stops:
                    if o is not keep:
                        await self.exec.cancel(o)
                if keep is not None:
                    self.exec.update(keep)
                self.stop[s] = keep
                await self._protect(s)
            await self.drain()
            self._save(s)

    # ---- market events

    async def on_quote(self, q: Quote):
        self.market.on_quote(q)
        self.broker.on_quote(q)
        await self.drain()

    async def on_bar(self, k: Kline):
        self.broker.on_bar(k)
        status = self.market.add(k)
        if status == "gap" and self.backfill:
            start = self.market.last_open(k.symbol) + MINUTE
            for b in await self.backfill(k.symbol, start):
                self.market.add(b)
            self.journal.event("backfill", k.symbol, ts=k.close_time, start=start)
            status = self.market.add(k)
            if status == "dup":
                status = "ok"
        await self.drain()
        if status == "ok":
            await self.entry_cycle(k.symbol)

    async def on_mark(self, m: MarkPrice):
        self.market.on_mark(m)
        self._funding(m)
        self.broker.on_mark(m)
        await self.drain()
        s, pos = m.symbol, self.positions[m.symbol]
        if not pos.has_qty:
            return
        reason = self.exits.update(pos, m.price, m.time)
        self._save(s, m.time)
        if reason and pos.state is PosState.POSITION_OPEN:
            await self._exit(s, reason, m.time)

    def _funding(self, m: MarkPrice):
        s, pos = m.symbol, self.positions[m.symbol]
        prev = self.funding.get(s)
        if prev and 0 < prev[0] <= m.time and pos.has_qty and self.charged.get(s) != prev[0]:
            # positive rate: longs pay shorts
            pay = -pos.side.sign * pos.qty * m.price * prev[1]
            pos.funding += pay
            self.charged[s] = prev[0]
            self.journal.event("funding", s, ts=m.time, amount=pay, rate=prev[1])
        self.funding[s] = (m.next_funding_time, m.funding_rate)

    # ---- entry

    async def entry_cycle(self, s: str):
        pos = self.positions[s]
        if pos.state not in (PosState.FLAT, PosState.PENDING_ENTRY):
            return
        old = self.entry[s]
        if old is not None and old.is_open:
            await self.exec.cancel(old)
            await self.drain()
        if pos.has_qty:
            return
        self.entry[s] = None
        pos.state = PosState.FLAT

        now = self.clock()
        st = self.market.state(s)
        if st is None or self.halted or s in self.blocked:
            log.info("%s skip: %s", s, "no data" if st is None else "halted")
            return
        sig = self.strategy.on_bar(st)
        if sig is None:
            log.info("%s close=%s no signal (%d bars)", s, st.close[-1], len(st))
            return
        stale = self.market.stale(s, now, self.cfg.risk.stale_after_s * 1000)
        plan, why = self.risk.check(
            sig,
            self.market[s].quote,
            self.positions.open_count(),
            self._exposure(),
            stale=stale,
            unknown=s in self.exec.unknown,
        )
        self.journal.event(
            "signal",
            s,
            ts=sig.signal_time,
            side=sig.side,
            price=sig.limit_price,
            ok=plan is not None,
            reason=why,
            **sig.metadata,
        )
        log.info(
            "%s signal %s @ %.8g dist=%.4f%% -> %s",
            s,
            sig.side,
            sig.limit_price,
            sig.metadata.get("distance_pct", 0),
            why,
        )
        if plan is None:
            return
        o = Order(
            new_id(Purpose.ENTRY),
            s,
            sig.side,
            OrderType.LIMIT,
            plan.qty,
            Purpose.ENTRY,
            price=plan.price,
            tif=TimeInForce.GTX,
            time=now,
        )
        pos.state = PosState.PENDING_ENTRY
        pos.meta = {"strategy": self.strategy.id, "k": self.cfg.strategy.k, **sig.metadata}
        self.entry[s] = o
        res = await self.exec.submit(o)
        self.journal.order(res)
        await self.drain()
        if res.status is OrderStatus.REJECTED and not pos.has_qty:
            pos.state = PosState.FLAT
            self.entry[s] = None

    def _exposure(self) -> float:
        total = 0.0
        for s, pos in self.positions.pos.items():
            mark = self.market[s].mark
            total += pos.notional(mark.price if mark else pos.entry_price)
            o = self.entry[s]
            if o is not None and o.is_open:
                total += o.remaining * o.price
        return total

    # ---- order events

    async def drain(self):
        while self.pending:
            o, f = self.pending.popleft()
            await self._on_order(o, f)

    async def _on_order(self, o: Order, f: Fill | None):
        self.exec.update(o)
        self.journal.order(o)
        purpose = purpose_of(o.client_id)
        if f is None or purpose is None:
            return
        self.journal.fill(f)
        s, pos = o.symbol, self.positions[o.symbol]

        log.info("%s %s fill %s %s @ %s", s, purpose, f.side, f.qty, f.price)
        if purpose is Purpose.ENTRY:
            pos.apply(f, Purpose.ENTRY)
            if o.is_open:
                # any fill stops further entry
                await self.exec.cancel(o)
            self.entry[s] = None
            await self._protect(s)
        else:
            t = pos.apply(f, purpose)
            if t is not None:
                self._closed(s, t)
                await self._cancel_protection(s)
        self._save(s, f.time)

    def _closed(self, s: str, t: Trade):
        if not t.reason:
            t.reason = "exchange_stop"
        self.trades.append(t)
        self.journal.trade(t, ts=t.close_time)
        log.info("%s closed %s net=%.4f reason=%s", s, t.side, t.net, t.reason)

    # ---- protection and exit

    async def _protect(self, s: str):
        """(Re)place the exchange side stop for the full position."""
        pos = self.positions[s]
        await self.broker.before_protect(s)
        old = self.stop[s]
        if old is not None and old.is_open:
            if abs(old.qty - pos.qty) < 1e-12:
                return
            await self.exec.cancel(old)
        stop_px = self.exits.stop_price(pos)
        r = self.rules[s]
        stop_px = r.price_down(stop_px) if pos.side.sign > 0 else r.price_up(stop_px)
        o = Order(
            new_id(Purpose.STOP),
            s,
            pos.side.opposite,
            OrderType.STOP_MARKET,
            pos.qty,
            Purpose.STOP,
            stop_price=stop_px,
            reduce_only=True,
            time=self.clock(),
        )
        self.stop[s] = o
        res = await self.exec.submit(o)
        self.journal.order(res)
        if res.status is OrderStatus.REJECTED:
            log.error("%s stop rejected: %s", s, res.reason)
            self.journal.event("stop_rejected", s, reason=res.reason)

    async def cancel_entries(self):
        """On shutdown: pull resting entries, keep positions and their stops."""
        for s in self.cfg.symbols:
            o = self.entry[s]
            if o is not None and o.is_open:
                await self.exec.cancel(o)
                self.journal.event("shutdown_cancel", s, client_id=o.client_id)
            await self.drain()  # a fill seen while cancelling still gets its stop

    async def flatten(self, reason: str):
        """Close every open position, e.g. at the end of a backtest."""
        for s, pos in self.positions.pos.items():
            if pos.has_qty and pos.state is PosState.POSITION_OPEN:
                await self._exit(s, reason, self.clock())

    async def _cancel_protection(self, s: str):
        for book in (self.stop, self.close):
            o = book[s]
            if o is not None and o.is_open:
                await self.exec.cancel(o)
            book[s] = None

    async def _exit(self, s: str, reason: str, ts: int):
        pos = self.positions[s]
        pos.state, pos.exit_reason = PosState.EXIT_PENDING, reason
        o = Order(
            new_id(Purpose.EXIT),
            s,
            pos.side.opposite,
            OrderType.MARKET,
            pos.qty,
            Purpose.EXIT,
            reduce_only=True,
            time=ts,
        )
        self.close[s] = o
        res = await self.exec.submit(o)
        self.journal.order(res)
        await self.drain()
        if res.status is OrderStatus.REJECTED and pos.has_qty:
            # retry on the next mark price
            pos.state = PosState.POSITION_OPEN
            self.journal.event("exit_rejected", s, ts=ts, reason=res.reason)

    def _save(self, s: str, ts: int | None = None):
        self.journal.save_state(s, self.positions[s].to_dict(), ts=ts)
