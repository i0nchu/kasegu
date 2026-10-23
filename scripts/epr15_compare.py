"""Trade by trade comparison of the frozen EPR15 research run with the kasegu engine.

Research trades come from git tag alpha-epr15-v1 (1m bar simulator, research/sim.py).
The engine replays aggTrades tick by tick for every (symbol, day) that has research
trades and a cached aggTrades file, one symbol and one UTC day per run.

Each research trade is matched to an engine trade by symbol and fill minute, and every
difference gets one category: time (holding / bar semantics), fill (fill model),
exit (exit model), fees, slippage, funding, boundary (day edge of the replay).

Usage: uv run python scripts/epr15_compare.py [--limit N] [--jobs N] [--out DIR]
"""

import argparse
import asyncio
import io
import json
import logging
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from kasegu.app import load_rules
from kasegu.backtest.data import Archive
from kasegu.backtest.replay import Replay
from kasegu.config import load_config
from kasegu.market.data import MINUTE

TAG = "alpha-epr15-v1"
TRADES = "research/results/A3_candidate/EPR15/trades.csv"
CACHE = Path("data/history")
# research cost model, in %
MAKER, TAKER, SLIP, LEVEL_SLIP = 0.02, 0.05, 0.02, 0.03

log = logging.getLogger("epr15_compare")


def research_trades() -> pd.DataFrame:
    raw = subprocess.run(
        ["git", "show", f"{TAG}:{TRADES}"], capture_output=True, check=True, text=True
    ).stdout
    tr = pd.read_csv(io.StringIO(raw))
    tr["day"] = pd.to_datetime(tr.t, unit="ms", utc=True).dt.date
    return tr


def has_trades(symbol: str, d: date) -> bool:
    return (CACHE / f"daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip").exists()


def engine_cfg(symbol: str, d: date):
    cfg = load_config("configs/epr15.example.yaml")
    cfg.mode, cfg.symbols = "backtest", [symbol]
    cfg.backtest.start = cfg.backtest.end = d
    r = cfg.risk
    # one coin per run, no global caps, like the research
    r.max_positions, r.max_notional, r.max_loss_per_trade = 1, 1e9, 0
    r.max_daily_loss, r.max_consecutive_losses = 0, 0
    r.max_spread_pct = 1.0
    # replay quotes only move with trades; a quiet minute is not stale data here
    r.stale_after_s = 120
    cfg.fee.maker, cfg.fee.taker, cfg.fee.slippage_pct = MAKER / 100, TAKER / 100, SLIP
    return cfg


def run_day(job: tuple[str, str]) -> dict:
    symbol, day = job
    d = date.fromisoformat(day)
    cfg = engine_cfg(symbol, d)
    logging.disable(logging.INFO)

    async def go():
        rules = await load_rules(cfg)
        rp = Replay(cfg, rules, Archive(cfg.backtest.cache_dir), d, d)
        await rp.run()
        sig = []
        for ts, data in rp.journal.db.execute(
            "SELECT ts, data FROM events WHERE kind = 'signal' ORDER BY id"
        ):
            x = json.loads(data)
            sig.append({"ts": ts, "ok": x["ok"], "reason": x["reason"], "price": x["price"]})
        trades = []
        for t in rp.engine.trades:
            notional = t.entry_price * t.qty
            trades.append(
                {
                    "symbol": symbol,
                    "open_time": t.open_time,
                    "close_time": t.close_time,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "reason": t.reason,
                    "gross": t.pnl / notional * 100,
                    "fees": t.fees / notional * 100,
                    "funding": t.funding / notional * 100,
                    "net": t.net / notional * 100,
                    "qty": t.qty,
                }
            )
        return {"symbol": symbol, "day": day, "trades": trades, "signals": sig}

    try:
        return asyncio.run(go())
    except Exception as e:  # missing archive files etc.
        return {"symbol": symbol, "day": day, "error": repr(e), "trades": [], "signals": []}


def classify(r: pd.Series | None, e: dict | None, day_end: int) -> list[str]:
    """Why a research trade (r) and an engine trade (e) differ."""
    if r is None:
        return ["fill:engine_only"]
    if e is None:
        return ["fill:research_only"]
    out = []
    if abs(e["entry"] - r.entry) > 1e-9 * r.entry:
        out.append("fill:entry_price")
    if e["reason"] == "end_of_data" or r.exit_t + MINUTE > day_end:
        return out + ["boundary"]
    same_reason = (e["reason"] == "time_exit") == (r.reason == "max_hold")
    if not same_reason:
        # research stops on the 1m low (last trade), the engine on 1s mark samples
        out.append("exit:stop_trigger" if r.reason == "stop_loss" else f"exit:{e['reason']}")
    elif e["reason"] == "time_exit":
        # research: close of the 15th bar counted from the fill bar; engine: 15 min after fill
        out.append("time")
    else:
        out.append("exit:stop_path")
    r_fee = MAKER + TAKER * r.exit / r.entry
    if abs(e["fees"] - r_fee) > 1e-3:
        out.append("fees")
    if abs(e["funding"] - r.funding) > 1e-4:
        out.append("funding")
    return out


def compare(research: pd.DataFrame, runs: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for run in runs:
        s, day = run["symbol"], date.fromisoformat(run["day"])
        d0 = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        day_end = d0 + 86_400_000
        r_day = research[(research.symbol == s) & (research.day == day)]
        eng = {e["open_time"] - e["open_time"] % MINUTE: e for e in run["trades"]}
        sigs = {x["ts"]: x for x in run["signals"]}
        seen = set()
        for _, r in r_day.iterrows():
            e = eng.get(int(r.t))
            seen.add(int(r.t))
            cats = classify(r, e, day_end)
            sig = sigs.get(int(r.t) - 1)  # close of the bar that placed the order
            if e is None and r.t < d0 + MINUTE:
                cats.append("boundary")  # order placed before the replay day started
            elif e is None:
                busy = any(x["open_time"] < r.t <= x["close_time"] + MINUTE for x in run["trades"])
                if busy:
                    cats.append("time:in_position")  # engine still held the previous trade
                elif sig is not None and not sig["ok"]:
                    cats.append(f"fill:rejected:{sig['reason']}")
            rows.append(
                {
                    "symbol": s,
                    "day": run["day"],
                    "fill_min": int(r.t),
                    "matched": e is not None,
                    "categories": " ".join(cats),
                    "r_entry": r.entry,
                    "e_entry": e["entry"] if e else None,
                    "r_exit": r.exit,
                    "e_exit": e["exit"] if e else None,
                    "r_reason": r.reason,
                    "e_reason": e["reason"] if e else None,
                    "r_hold_min": (r.exit_t - r.t) / MINUTE + 1,
                    "e_hold_min": (e["close_time"] - e["open_time"]) / MINUTE if e else None,
                    "e_fill_offset_s": (e["open_time"] - r.t) / 1000 if e else None,
                    # research exits at the close of the 15th bar, t + 15 min
                    "e_exit_delay_s": (e["close_time"] - r.exit_t - MINUTE) / 1000 if e else None,
                    "r_net": r.net,
                    "e_net": e["net"] if e else None,
                    "d_net": e["net"] - r.net if e else None,
                    "r_funding": r.funding,
                    "e_funding": e["funding"] if e else None,
                    "e_signal": f"{sig['ok']}:{sig['reason']}" if sig else "none",
                }
            )
        for m, e in eng.items():
            if m in seen or m < d0:
                continue
            rows.append(
                {
                    "symbol": s,
                    "day": run["day"],
                    "fill_min": m,
                    "matched": False,
                    "categories": " ".join(classify(None, e, day_end)),
                    "e_entry": e["entry"],
                    "e_exit": e["exit"],
                    "e_reason": e["reason"],
                    "e_hold_min": (e["close_time"] - e["open_time"]) / MINUTE,
                    "e_net": e["net"],
                }
            )
    df = pd.DataFrame(rows)
    both = df[df.matched]
    ok = both[~both.categories.str.contains("boundary")]
    research_n = int(df.r_net.notna().sum())
    cats = Counter(c for x in df.categories for c in x.split())
    summary = {
        "symbol_days": len(runs),
        "errors": [r for r in runs if r.get("error")],
        "research_trades": research_n,
        "engine_trades": int(df.e_net.notna().sum()),
        "matched": int(len(both)),
        "match_rate": len(both) / research_n if research_n else None,
        "categories": dict(cats.most_common()),
        "entry_price_equal": int((abs(both.e_entry - both.r_entry) < 1e-9 * both.r_entry).sum()),
        "matched_ex_boundary": int(len(ok)),
        "avg_net_research": float(ok.r_net.mean()) if len(ok) else None,
        "avg_net_engine": float(ok.e_net.mean()) if len(ok) else None,
        "avg_d_net": float(ok.d_net.mean()) if len(ok) else None,
        "median_abs_d_net": float(ok.d_net.abs().median()) if len(ok) else None,
        "avg_hold_research": float(ok.r_hold_min.mean()) if len(ok) else None,
        "avg_hold_engine": float(ok.e_hold_min.mean()) if len(ok) else None,
        "exit_reasons_research": ok.r_reason.value_counts().to_dict() if len(ok) else {},
        "exit_reasons_engine": ok.e_reason.value_counts().to_dict() if len(ok) else {},
        "avg_net_research_all": float(df.r_net.mean()),
        "avg_net_engine_all": float(df.e_net.mean()),
    }
    primary = ok.categories.str.split().map(
        lambda c: next((x for x in c if x.startswith(("exit", "time"))), c[-1])
    )
    by = ok.groupby(primary)
    summary["d_net_by_category"] = {
        k: {"n": int(len(g)), "avg_d_net": float(g.d_net.mean())} for k, g in by
    }
    return df, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="first N symbol-days only")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", default="data/epr15_compare")
    p.add_argument("--reuse", action="store_true", help="compare the saved runs.json again")
    a = p.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s")

    research = research_trades()
    out = Path(a.out)
    if a.reuse:
        runs = json.loads((out / "runs.json").read_text())
        return report(research, runs, out)
    jobs = sorted(
        {
            (s, d.isoformat())
            for s, d in zip(research.symbol, research.day, strict=True)
            if has_trades(s, d)
        }
    )
    if a.limit:
        jobs = jobs[: a.limit]
    log.info("%d research trades, %d symbol-days with aggTrades", len(research), len(jobs))
    # rules once in this process so workers only read the cache
    cfg = engine_cfg(jobs[0][0], date.today())
    cfg.symbols = sorted({s for s, _ in jobs})
    asyncio.run(load_rules(cfg))

    out.mkdir(parents=True, exist_ok=True)
    runs = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, r in enumerate(ex.map(run_day, jobs), 1):
            runs.append(r)
            log.info("%d/%d %s %s: %d trades %s", i, len(jobs), r["symbol"], r["day"],
                     len(r["trades"]), r.get("error", ""))  # fmt: skip
    (out / "runs.json").write_text(json.dumps(runs))
    return report(research, runs, out)


def report(research: pd.DataFrame, runs: list[dict], out: Path) -> int:
    df, summary = compare(research, runs)
    df.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
