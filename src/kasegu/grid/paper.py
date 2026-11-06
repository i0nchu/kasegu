"""Paper runner of the trend-aligned grid: daily selection at 00:00 UTC, grids simulated on closed
5m klines with the research fill rules. Nothing is sent to the exchange.

Files in cfg.out_dir: state.json (restart), trades.csv (one row per finished grid),
decisions.jsonl (every daily selection with all candidates), daily.csv (equity at each decision),
status.json (current view)."""

import asyncio
import csv
import json
import logging
import time
from pathlib import Path

from kasegu.grid import signals
from kasegu.grid.config import GridConfig
from kasegu.grid.episode import BAR_MS, GridEpisode, GridParams

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
TRADE_FIELDS = [
    "symbol", "side", "start", "exit_time", "reason", "pnl", "fees", "funding", "levels",
    "step", "p0", "exit_price", "rv3", "sigma", "notional",
]  # fmt: skip


def _iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


class GridPaper:
    def __init__(self, cfg: GridConfig, market, now=None):
        self.cfg = cfg
        self.market = market
        self.now = now or (lambda: int(time.time() * 1000))
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.active: list[GridEpisode] = []
        self.realized = 0.0
        self.closed = 0
        self.day: int | None = None  # last decision day (ms)
        self.halted = False
        self.sem = asyncio.Semaphore(8)
        self._load()

    # persistence

    @property
    def state_path(self) -> Path:
        return self.out / "state.json"

    def _load(self):
        if not self.state_path.exists():
            return
        d = json.loads(self.state_path.read_text())
        self.active = [GridEpisode.from_dict(e) for e in d.get("active", [])]
        self.realized = d.get("realized", 0.0)
        self.closed = d.get("closed", 0)
        self.day = d.get("day")
        self.halted = d.get("halted", False)
        log.info("resumed: %d open grids, realized %.2f", len(self.active), self.realized)

    def _save(self):
        d = {
            "day": self.day,
            "realized": self.realized,
            "closed": self.closed,
            "halted": self.halted,
            "active": [e.to_dict() for e in self.active],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.state_path)

    def _append_csv(self, name: str, fields: list[str], row: dict):
        p = self.out / name
        new = not p.exists()
        with p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    def _jsonl(self, name: str, row: dict):
        with (self.out / name).open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # account

    def unrealized(self) -> float:
        return sum(e.upnl(e.last_close) for e in self.active if e.last_close is not None)

    def equity(self) -> float:
        return self.cfg.balance + self.realized + self.unrealized()

    def params(self) -> GridParams:
        g = self.cfg.grid
        return GridParams(
            levels=g.levels,
            mult=g.mult,
            notional=g.notional,
            margin=self.cfg.balance / self.cfg.slots,
            maker=g.maker,
            taker=g.taker,
            slip_in=g.slip_in,
            slip_out=g.slip_out,
            pen=g.pen,
            hold_bars=g.hold_hours * 12,
        )

    # grids

    async def _bars(self, e: GridEpisode):
        start = e.start if e.last_bar is None else e.last_bar + BAR_MS
        async with self.sem:
            ks = await self.market.klines(e.symbol, "5m", 1000, start)
            if not ks:
                return [], {}
            fund = await self.market.funding(e.symbol, ks[0].open_time, ks[-1].open_time + BAR_MS)
        per_bar: dict[int, float] = {}
        for t, rate in fund:
            b = t - t % BAR_MS
            per_bar[b] = per_bar.get(b, 0.0) + rate
        return ks, per_bar

    async def update(self):
        """Feed closed 5m bars to every open grid; finished grids go to trades.csv."""
        bars = (self._bars(e) for e in self.active)
        results = await asyncio.gather(*bars, return_exceptions=True)
        still = []
        for e, res in zip(self.active, results, strict=True):
            if isinstance(res, BaseException):
                log.warning("%s: bars failed: %s", e.symbol, res)
                still.append(e)
                continue
            ks, fund = res
            for k in ks:
                e.on_bar(k.open_time, k.open, k.high, k.low, k.close, fund.get(k.open_time, 0.0))
                if not e.open:
                    break
            if e.open:
                still.append(e)
            else:
                self._finish(e)
        self.active = still
        if self.cfg.max_account_loss and not self.halted:
            loss = self.realized + self.unrealized()
            if loss <= -self.cfg.max_account_loss:
                self.halted = True
                log.error("account loss %.2f reached the limit, no new grids", loss)
                self._jsonl("decisions.jsonl", {"time": _iso(self.now()), "halted": loss})
        self._save()

    def _finish(self, e: GridEpisode):
        self.realized += e.pnl
        self.closed += 1
        log.info(
            "%s %s grid closed: %s pnl %.3f levels %d", e.symbol,
            "long" if e.side == 1 else "short", e.reason, e.pnl, e.filled,
        )  # fmt: skip
        self._append_csv(
            "trades.csv",
            TRADE_FIELDS,
            {
                "symbol": e.symbol,
                "side": e.side,
                "start": _iso(e.start),
                "exit_time": _iso(e.exit_time),
                "reason": e.reason,
                "pnl": round(e.pnl, 6),
                "fees": round(e.fees, 6),
                "funding": round(e.funding, 6),
                "levels": e.filled,
                "step": round(e.step, 6),
                "p0": e.p0,
                "exit_price": e.exit_price,
                "rv3": e.info.get("rv3"),
                "sigma": e.info.get("sigma"),
                "notional": e.params.notional,
            },
        )

    # daily selection

    async def _klines(self, symbol: str, interval: str, limit: int):
        async with self.sem:
            return await self.market.klines(symbol, interval, limit)

    async def decide(self, day: int):
        """Open new grids for the UTC day starting at `day` (ms), using bars closed before it."""
        cfg = self.cfg
        rec: dict = {"day": _iso(day), "equity": round(self.equity(), 4), "open": len(self.active)}
        free = cfg.slots - len(self.active)
        if self.halted:
            rec["skip"] = "account loss limit"
        if free <= 0 or self.halted:
            self._jsonl("decisions.jsonl", rec)
            return
        perps = await self.market.perpetuals()
        need = max(cfg.coin_sma, 30) + 1
        syms = list(perps)
        daily = await asyncio.gather(
            *(self._klines(s, "1d", need) for s in syms), return_exceptions=True
        )
        dk = {
            s: [k for k in d if k.open_time < day]
            for s, d in zip(syms, daily, strict=True)
            if not isinstance(d, BaseException)
        }
        uni = signals.universe(
            {s: [k.volume for k in d] for s, d in dk.items()}, perps, day, cfg.top, cfg.min_age_days
        )
        btc_k = await self._klines("BTCUSDT", "1d", cfg.btc_sma + 1)
        btc = [k.close for k in btc_k if k.open_time < day]
        busy = {e.symbol for e in self.active}
        cands = [s for s in uni if s not in signals.EXCLUDE and s not in busy]
        hourly = await asyncio.gather(
            *(self._klines(s, "1h", 170) for s in cands), return_exceptions=True
        )
        rows = []
        for s, h in zip(cands, hourly, strict=True):
            if isinstance(h, BaseException):
                rows.append({"symbol": s, "skip": f"hourly failed: {h}"})
                continue
            closes = [k.close for k in h if k.open_time < day]
            rv3, sig = signals.rv_3d(closes), signals.sigma_7d(closes)
            coin = [k.close for k in dk.get(s, [])]
            side = signals.direction(btc, coin, cfg.btc_sma, cfg.coin_sma)
            rows.append({"symbol": s, "rv3": rv3, "sigma": sig, "side": side})
        ok = [
            r for r in rows
            if r.get("rv3") is not None and r["sigma"] is not None and r["side"] != 0
            and r["rv3"] >= cfg.rv3_min
        ]  # fmt: skip
        ok.sort(key=lambda r: -r["rv3"])
        chosen = ok[:free]
        for r in chosen:
            step = cfg.k * min(max(r["sigma"], cfg.step_floor), cfg.step_cap)
            e = GridEpisode(r["symbol"], r["side"], step, day, self.params(), info=r)
            self.active.append(e)
            log.info(
                "%s %s grid: step %.2f%% rv3 %.3f", r["symbol"],
                "long" if r["side"] == 1 else "short", step * 100, r["rv3"],
            )  # fmt: skip
        rec.update(
            universe=uni,
            btc_above_sma=signals.above_sma(btc, cfg.btc_sma),
            candidates=rows,
            chosen=[r["symbol"] for r in chosen],
        )
        self._jsonl("decisions.jsonl", rec)

    async def step(self):
        now = self.now()
        await self.update()
        day = now - now % DAY_MS
        if self.day != day and now - day >= self.cfg.decision_delay_s * 1000:
            self._append_csv(
                "daily.csv",
                ["day", "realized", "unrealized", "equity", "open"],
                {
                    "day": _iso(day),
                    "realized": round(self.realized, 4),
                    "unrealized": round(self.unrealized(), 4),
                    "equity": round(self.equity(), 4),
                    "open": len(self.active),
                },
            )
            await self.decide(day)
            self.day = day
            self._save()
        self.status()

    def status(self):
        d = {
            "ts": self.now(),  # for kasegu.health
            "time": _iso(self.now()),
            "equity": round(self.equity(), 4),
            "realized": round(self.realized, 4),
            "closed": self.closed,
            "halted": self.halted,
            "open": [
                {
                    "symbol": e.symbol,
                    "side": e.side,
                    "start": _iso(e.start),
                    "levels": e.filled,
                    "avg": e.avg() if e.qty else None,
                    "upnl": round(e.upnl(e.last_close), 4) if e.last_close else 0.0,
                }
                for e in self.active
            ],
        }
        tmp = self.out / "status.tmp"
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.out / "status.json")

    async def run(self):
        log.info("grid paper: %s", self.cfg.model_dump(mode="json", exclude={"exchange"}))
        while True:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("grid step failed, retrying next bar")
            now = self.now()
            # wake 15s after each 5m bar closes
            await asyncio.sleep((BAR_MS - now % BAR_MS) / 1000 + 15)
