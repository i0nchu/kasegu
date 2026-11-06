"""Paper runner: one try at a time. A try opens at 00:00 UTC and runs until target, bust or the time
limit; the next try opens at the next 00:00. Files in cfg.out_dir: state.json, tries.csv,
status.json."""

import asyncio
import csv
import json
import logging
import time
from pathlib import Path

from kasegu.bold.config import BoldConfig
from kasegu.bold.shot import BAR_MS, Shot, ShotParams

log = logging.getLogger(__name__)
DAY_MS = 86_400_000
FIELDS = ["start", "exit_time", "symbol", "side", "reason", "equity", "pnl_usd", "p0", "exit_price",
          "funding", "days"]  # fmt: skip


def _iso(ms):
    return "" if ms is None else time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


class BoldPaper:
    def __init__(self, cfg: BoldConfig, market, now=None):
        self.cfg, self.market = cfg, market
        self.now = now or (lambda: int(time.time() * 1000))
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.shot: Shot | None = None
        self.counts = {"target": 0, "bust": 0, "time": 0}
        self.pnl = 0.0
        self.last_end: int | None = None  # exit time of the last try
        p = self.out / "state.json"
        if p.exists():
            d = json.loads(p.read_text())
            self.shot = Shot.from_dict(d["shot"]) if d.get("shot") else None
            self.counts, self.pnl, self.last_end = d["counts"], d["pnl"], d.get("last_end")

    def _save(self):
        d = {"shot": self.shot.to_dict() if self.shot else None, "counts": self.counts,
             "pnl": self.pnl, "last_end": self.last_end}  # fmt: skip
        tmp = self.out / "state.tmp"
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.out / "state.json")

    def params(self) -> ShotParams:
        c = self.cfg
        return ShotParams(
            c.leverage, c.target, c.bust, c.taker, c.slip, c.max_days * 288,
            c.entry, c.tp_maker, c.maker, c.pen, c.entry_wait,
        )  # fmt: skip

    async def update(self):
        s = self.shot
        if s is None:
            return
        start = s.start if s.last_bar is None else s.last_bar + BAR_MS
        ks = await self.market.klines(s.symbol, "5m", 1000, start)
        if ks:
            fund = await self.market.funding(s.symbol, ks[0].open_time, ks[-1].open_time + BAR_MS)
            per_bar: dict[int, float] = {}
            for t, r in fund:
                per_bar[t - t % BAR_MS] = per_bar.get(t - t % BAR_MS, 0.0) + r
            for k in ks:
                s.on_bar(k.open_time, k.open, k.high, k.low, k.close, per_bar.get(k.open_time, 0.0))
                if not s.open:
                    break
        if not s.open:
            pnl = (s.equity - 1) * s.stake
            self.counts[s.reason] += 1
            self.pnl += pnl
            self.last_end = s.exit_time
            log.info("try closed: %s equity %.2fx pnl %.2f", s.reason, s.equity, pnl)
            p = self.out / "tries.csv"
            new = not p.exists()
            with p.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({
                    "start": _iso(s.start), "exit_time": _iso(s.exit_time), "symbol": s.symbol,
                    "side": s.side, "reason": s.reason, "equity": round(s.equity, 4),
                    "pnl_usd": round(pnl, 2), "p0": s.p0, "exit_price": s.exit_price,
                    "funding": round(s.funding, 5),
                    "days": round((s.exit_time - s.start) / DAY_MS, 2),
                })  # fmt: skip
            self.shot = None

    async def step(self):
        await self.update()
        now = self.now()
        day = now - now % DAY_MS
        free = self.last_end is None or self.last_end < day
        if self.shot is None and free and now - day >= self.cfg.decision_delay_s * 1000:
            c = self.cfg
            self.shot = Shot(c.symbol, c.side, day, self.params(), stake=c.stake)
            log.info("new try: %s %s %.0fx from %s", c.symbol, "long" if c.side == 1 else "short",
                     c.leverage, _iso(day))  # fmt: skip
            await self.update()
        self._save()
        self.status()

    def status(self):
        s = self.shot
        tries = sum(self.counts.values())
        d = {
            "ts": self.now(),
            "time": _iso(self.now()),
            "tries": tries,
            **self.counts,
            "hit_rate": round(self.counts["target"] / tries, 3) if tries else None,
            "pnl_usd": round(self.pnl, 2),
            "open": None if s is None else {
                "start": _iso(s.start), "p0": s.p0, "last": s.last_close,
                "equity": round(s.mark(s.last_close), 4) if s.p0 and s.last_close else None,
                "target_price": s.p0 * (1 + s.side * s.up_move()) if s.p0 else None,
                "stop_price": s.p0 * (1 - s.side * s.down_move()) if s.p0 else None,
            },
        }  # fmt: skip
        tmp = self.out / "status.tmp"
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.out / "status.json")

    async def run(self):
        log.info("bold paper: %s", self.cfg.model_dump(mode="json", exclude={"exchange"}))
        while True:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("bold step failed, retrying next bar")
            now = self.now()
            await asyncio.sleep((BAR_MS - now % BAR_MS) / 1000 + 15)
