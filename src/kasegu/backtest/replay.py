import bisect
import heapq
import logging
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from kasegu.backtest.data import Archive, days
from kasegu.backtest.metrics import summarize
from kasegu.config import AppConfig
from kasegu.core.engine import Engine
from kasegu.core.execution import PaperBroker, purpose_of
from kasegu.core.types import Kline, MarkPrice, OrderStatus, Purpose, Quote
from kasegu.exchange.rules import SymbolRules
from kasegu.market.data import MINUTE, MarketData
from kasegu.storage.journal import Journal
from kasegu.strategies import create

log = logging.getLogger(__name__)

FundingApi = Callable[[str, int, int], Awaitable[list[tuple[int, float]]]]


def ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    time: int
    price: float


class Funding:
    def __init__(self, rows: list[tuple[int, float]]):
        self.times = [t for t, _ in rows]
        self.rates = [r for _, r in rows]

    def at(self, t: int) -> tuple[int, float]:
        """Next funding time and its rate after t."""
        i = bisect.bisect_right(self.times, t)
        if i >= len(self.times):
            return 0, 0.0
        return self.times[i], self.rates[i]


def trade_bars(
    symbol: str,
    trades: Iterable[tuple[int, float, float]],
    start: int | None = None,
    seed: float | None = None,
) -> Iterator[Tick | Kline]:
    """Ticks from trades plus a closed 1m bar at each minute boundary.

    Empty minutes repeat the previous close so the bar series has no gaps.
    `start`/`seed` continue the series from a known bar time and close.
    """
    cur = None  # [open_time, o, h, l, c, v]
    if start is not None and seed is not None:
        cur = [start, seed, seed, seed, seed, 0.0]
    for t, px, qty in trades:
        if cur is not None and t < cur[0]:
            continue
        m = t - t % MINUTE
        while cur is not None and m > cur[0]:
            yield Kline(symbol, cur[0], cur[0] + MINUTE - 1, *cur[1:5], cur[5])
            c = cur[4]
            cur = [cur[0] + MINUTE, c, c, c, c, 0.0]
        if cur is None:
            cur = [m, px, px, px, px, 0.0]
        if cur[5] == 0:
            cur[1:5] = [px] * 4  # first trade of the minute sets the open
        cur[2], cur[3] = max(cur[2], px), min(cur[3], px)
        cur[4] = px
        cur[5] += qty
        yield Tick(symbol, t, px)
    if cur is not None:
        yield Kline(symbol, cur[0], cur[0] + MINUTE - 1, *cur[1:5], cur[5])


class Replay:
    """Tick replay of aggTrades through the same engine, risk and exits as paper/live.

    1m klines are only used to warm up indicators before the start date.
    """

    def __init__(
        self,
        cfg: AppConfig,
        rules: dict[str, SymbolRules],
        archive: Archive,
        start: date,
        end: date,
        funding_api: FundingApi | None = None,
    ):
        self.cfg, self.rules, self.archive = cfg, rules, archive
        self.start, self.end = start, end
        self.funding_api = funding_api
        self.now = 0
        self.last_px: dict[str, float] = {}
        self.last_mark: dict[str, int] = {}
        self.market = MarketData(cfg.symbols, maxlen=1500)
        self.broker = PaperBroker(cfg.fee)
        self.journal = Journal(":memory:", run_id="backtest")
        self.engine = Engine(
            cfg,
            create(cfg.strategy),
            self.broker,
            self.market,
            rules,
            self.journal,
            backfill=self.backfill,
            clock=lambda: self.now,
        )
        self.funding: dict[str, Funding] = {}

    async def load_funding(self):
        lo, hi = ms(self.start), ms(self.end + timedelta(1)) + 8 * 3_600_000
        for s in self.cfg.symbols:
            rows = self.archive.funding(s, self.start, self.end)
            last = rows[-1][0] if rows else lo - 1
            # monthly archives appear after the month ends, fill the rest from REST
            if last < hi - 8 * 3_600_000 and self.funding_api is not None:
                extra = await self.funding_api(s, max(last + 1, lo), hi)
                rows += [r for r in extra if r[0] > last]
                log.info("%s: %d funding rows from REST", s, len(extra))
            if not rows:
                log.warning("%s: no funding data, funding is not simulated", s)
            self.funding[s] = Funding(rows)

    def warmup(self):
        need = self.engine.strategy.warmup + 1
        n_days = need // 1440 + 1
        for s in self.cfg.symbols:
            for d in days(self.start - timedelta(n_days), self.start - timedelta(1)):
                self.market.load(self.archive.klines(s, d))

    async def backfill(self, symbol: str, start: int) -> list[Kline]:
        # safety net only, trade_bars already fills empty minutes
        d0 = datetime.fromtimestamp(start / 1000, UTC).date()
        out = []
        for d in days(d0, d0 + timedelta(1)):
            out += [k for k in self.archive.klines(symbol, d) if start <= k.open_time < self.now]
        return out

    async def tick(self, t: Tick, mark_every: int = 1000):
        self.now = max(self.now, t.time)
        half = self.rules[t.symbol].tick / 2
        if self.last_px.get(t.symbol) != t.price:
            self.last_px[t.symbol] = t.price
            await self.engine.on_quote(
                Quote(t.symbol, t.price - half, t.price + half, 1e9, 1e9, t.time)
            )
        if t.time - self.last_mark.get(t.symbol, 0) >= mark_every:
            self.last_mark[t.symbol] = t.time
            nxt, rate = self.funding[t.symbol].at(t.time)
            await self.engine.on_mark(MarkPrice(t.symbol, t.price, rate, nxt, t.time))

    async def bar(self, k: Kline):
        self.now = max(self.now, k.close_time)
        await self.tick(Tick(k.symbol, k.close_time, k.close), mark_every=0)
        await self.engine.on_bar(k)

    async def run_trades(self):
        for d in days(self.start, self.end):
            streams = []
            for s in self.cfg.symbols:
                last = self.market.last_open(s)
                seed = self.market[s].bars[-1].close if last is not None else None
                nxt = last + MINUTE if last is not None else ms(d)
                streams.append(trade_bars(s, self.archive.trades(s, d), nxt, seed))

            def key(ev):
                return ev.time if isinstance(ev, Tick) else ev.close_time

            for ev in heapq.merge(*streams, key=key):
                if isinstance(ev, Tick):
                    await self.tick(ev)
                else:
                    await self.bar(ev)

    async def run(self) -> dict:
        await self.load_funding()
        self.warmup()
        await self.run_trades()
        await self.engine.flatten("end_of_data")
        return self.report()

    def report(self) -> dict:
        orders = [
            o for o in self.engine.exec.orders.values() if purpose_of(o.client_id) is Purpose.ENTRY
        ]
        placed = [o for o in orders if o.status is not OrderStatus.REJECTED]
        filled = [o for o in placed if o.filled_qty > 0]
        out = summarize(self.engine.trades, len(placed), len(filled), self.cfg.fee.slippage_pct)
        out["open_positions"] = [s for s, p in self.engine.positions.pos.items() if p.has_qty]
        return out
