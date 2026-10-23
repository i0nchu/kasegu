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
    Side,
    TimeInForce,
)
from kasegu.events import Event, EventBus
from kasegu.exchange.rules import SymbolRules
from kasegu.exits import create_exit
from kasegu.market.data import MINUTE, MarketData
from kasegu.storage.journal import Journal
from kasegu.strategies.base import Strategy

log = logging.getLogger(__name__)

DAY = 86_400_000
_LOG_LEVEL = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.ERROR}

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
        events: EventBus | None = None,
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
        strategy.check_exit(cfg.exit)
        self.exits = create_exit(cfg.exit)
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
        self.events = events
        # loss circuit breakers
        self.day = -1  # UTC day number of day_net
        self.day_net = 0.0
        self.streak = 0  # consecutive losing trades
        self.paused_until = 0  # ms, no new entries before
        self.pause_reason = ""
        broker.listener = lambda o, f: self.pending.append((o, f))

    # ---- startup / recovery

    def restore(self):
        for s, d in self.journal.load_state().items():
            if s in self.positions.pos:
                self.positions.pos[s] = Position.from_dict(d)
        # today's closed trades count towards the loss limits after a restart
        now = self.clock()
        for t in self.journal.trades_since(now - now % DAY):
            self._count(t["close_time"], t["pnl"] - t["fees"] + t["funding"], alert=False)

    def alert(self, kind: str, s: str | None, msg: str, level: str = "info", **data):
        """Journal, log and publish an event for alerts / outside consumers."""
        self.journal.event(kind, s, level=level, message=msg, **data)
        log.log(_LOG_LEVEL[level], "%s%s", f"{s} " if s else "", msg)
        if self.events is not None:
            self.events.publish(Event(kind, level, msg, s, self.clock(), data))

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
                if pos.has_qty:
                    self.alert(
                        "position_gone",
                        s,
                        f"local {pos.side} {pos.qty} is flat on the exchange, state reset",
                        "warning",
                        local=pos.to_dict(),
                    )
                pos.reset()
                self.blocked.discard(s)
                for o in stops:
                    await self.exec.cancel(o)
            else:
                if not pos.has_qty or pos.side.sign * qty < 0:
                    await self._adopt(s, qty)
                elif abs(pos.qty - abs(qty)) > 1e-12:
                    self.alert(
                        "position_resized",
                        s,
                        f"local qty {pos.qty} -> exchange {abs(qty)}",
                        "warning",
                    )
                pos.qty = abs(qty)
                pos.state = PosState.POSITION_OPEN
                self.blocked.discard(s)
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
            self._log_position(k.symbol, k.close_time)

    async def on_mark(self, m: MarkPrice):
        self.market.on_mark(m)
        self._funding(m)
        self.broker.on_mark(m)
        await self.drain()
        s, pos = m.symbol, self.positions[m.symbol]
        if not pos.has_qty:
            return
        tier = pos.tier
        reason = self.exits.update(pos, m.price, m.time)
        if pos.tier != tier:
            log.info("%s tier %d -> %d, trailing stop %+.3f%%", s, tier, pos.tier, pos.stop_pct)
            self.journal.event("tier", s, ts=m.time, tier=pos.tier, stop_pct=pos.stop_pct)
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
        if now < self.paused_until:
            log.info("%s skip: paused (%s)", s, self.pause_reason)
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
            if not pos.has_qty:
                self.alert(
                    "entry_filled",
                    s,
                    f"entry {f.side} {f.qty} @ {f.price}",
                    side=f.side,
                    price=f.price,
                )
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
        self.alert(
            "trade_closed",
            s,
            f"closed {t.side} net {t.net:+.4f} USDT ({t.reason}, {t.hold_minutes:.1f}m)",
            net=t.net,
            reason=t.reason,
            entry=t.entry_price,
            exit=t.exit_price,
        )
        self._count(t.close_time, t.net)

    def _count(self, ts: int, net: float, alert: bool = True):
        """Update daily pnl and loss streak; pause new entries when a limit is hit."""
        r = self.cfg.risk
        if ts // DAY != self.day:
            self.day, self.day_net = ts // DAY, 0.0
        self.day_net += net
        self.streak = self.streak + 1 if net < 0 else 0
        until, why = 0, ""
        if r.max_daily_loss and self.day_net <= -r.max_daily_loss:
            until, why = (self.day + 1) * DAY, f"daily loss {self.day_net:.2f} USDT"
        elif r.max_consecutive_losses and self.streak >= r.max_consecutive_losses:
            until, why = ts + r.loss_pause_min * 60_000, f"{self.streak} losses in a row"
            self.streak = 0
        if until > self.paused_until:
            self.paused_until, self.pause_reason = until, why
            if alert:
                self.alert(
                    "loss_limit", None, f"new entries paused: {why}", "critical", until=until
                )

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
        log.info("%s exchange stop %s %s @ %s -> %s", s, o.side, o.qty, stop_px, res.status)
        if res.status is OrderStatus.REJECTED:
            # no exchange side protection: get out now
            self.alert("stop_rejected", s, f"stop rejected ({res.reason}), closing", "critical")
            if pos.state is PosState.POSITION_OPEN:
                await self._exit(s, "stop_rejected", self.clock())

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
        log.info("%s exit %s (%s)", s, pos.qty, reason)
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
            self.alert("exit_rejected", s, f"exit rejected ({res.reason}), retrying", "warning")

    async def _adopt(self, s: str, qty: float):
        """Exchange position without a matching local record: manage it from now on."""
        pos = self.positions[s]
        mark = self.market[s].mark
        quote = self.market[s].quote
        px = (
            await self.broker.entry_price(s)
            or (mark.price if mark else 0)
            or (quote.mid if quote else 0)
        )
        pos.reset()
        pos.side = Side.BUY if qty > 0 else Side.SELL
        pos.qty = pos.entry_qty = abs(qty)
        pos.entry_price = px
        pos.open_time = self.clock()
        pos.state = PosState.POSITION_OPEN
        pos.meta = {"adopted": True}
        self.alert(
            "position_adopted",
            s,
            f"took over exchange position {qty} @ {px}, stop and exits now apply",
            "warning",
            qty=qty,
            price=px,
        )

    def snapshot(self) -> dict:
        """Current state for status files and a WebUI."""
        now = self.clock()
        out = {}
        for s, pos in self.positions.pos.items():
            mark = self.market[s].mark
            e, st = self.entry[s], self.stop[s]
            out[s] = {
                "state": pos.state,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "mark": mark.price if mark else None,
                "pnl_pct": pos.profit_pct(mark.price) if mark and pos.has_qty else None,
                "peak_pct": pos.peak_pct,
                "tier": pos.tier,
                "trail_pct": pos.stop_pct,
                "held_min": (now - pos.open_time) / 60_000 if pos.open_time else None,
                "entry_order": e.price if e is not None and e.is_open else None,
                "stop_order": st.stop_price if st is not None and st.is_open else None,
                "blocked": s in self.blocked,
            }
        return {
            "ts": now,
            "strategy": self.strategy.id,
            "halted": self.halted,
            "paused_until": self.paused_until or None,
            "pause_reason": self.pause_reason if now < self.paused_until else "",
            "day_net": self.day_net,
            "loss_streak": self.streak,
            "trades": len(self.trades),
            "net": sum(t.net for t in self.trades),
            "symbols": out,
        }

    def _log_position(self, s: str, now: int):
        pos = self.positions[s]
        if not pos.has_qty:
            return
        mark = self.market[s].mark
        p = pos.profit_pct(mark.price) if mark else 0.0
        stop = f"{pos.stop_pct:+.3f}%" if pos.stop_pct is not None else "-"
        held = (now - (pos.open_time or now)) / 60_000
        log.info(
            "%s %s %s %s @ %.8g mark %s pnl %+.3f%% H %+.3f%% tier %d trail %s sl -%s%% held %.1fm",
            s,
            pos.state,
            pos.side,
            pos.qty,
            pos.entry_price,
            mark.price if mark else "-",
            p,
            pos.peak_pct,
            pos.tier,
            stop,
            self.cfg.exit.stop_loss_pct,
            held,
        )

    def _save(self, s: str, ts: int | None = None):
        self.journal.save_state(s, self.positions[s].to_dict(), ts=ts)
