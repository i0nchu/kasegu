"""E4: replay sampled event days tick by tick in the real engine and compare with the 1m simulator.

Picks coin-days where the E3 lead group traded, downloads their aggTrades, runs the engine
backtest (same config: data/research/cfg_lead.yaml) one coin-day at a time, then matches
engine trades with simulator trades by symbol and entry minute.

Usage: uv run python research/e4_engine.py [n_days]
"""

import asyncio
import concurrent.futures as cf
import json
import logging
import sys
from datetime import date, timedelta

import pandas as pd
from runs import RUNS

from kasegu.app import run_backtest
from kasegu.backtest.data import Archive
from kasegu.config import load_config

EXP = "E4_engine"
SKIP = {"BTCUSDT", "ETHUSDT"}  # multi GB of ticks for a handful of trades
OUT = RUNS / EXP
log = logging.getLogger("e4")


def sample(n: int) -> pd.DataFrame:
    tr = pd.read_csv(RUNS / "E3_robust" / "L0_lead" / "trades.csv")
    tr["day"] = pd.to_datetime(tr.t, unit="ms").dt.date
    days = tr[~tr.symbol.isin(SKIP)][["symbol", "day"]].drop_duplicates()
    return days.sample(n=min(n, len(days)), random_state=7).sort_values(["day", "symbol"])


def prefetch(days: pd.DataFrame, cache: str):
    arc = Archive(cache)
    rel = []
    for s, d in days.itertuples(index=False):
        rel.append(f"daily/aggTrades/{s}/{s}-aggTrades-{d}.zip")
        p = d - timedelta(1)
        rel.append(f"daily/klines/{s}/1m/{s}-1m-{p}.zip")
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(arc.get, rel))


def main(n: int):
    logging.basicConfig(level=logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_config("data/research/cfg_lead.yaml")
    base.mode = "backtest"
    base.backtest.out_dir = OUT / "reports"
    days = sample(n)
    days.to_csv(OUT / "days.csv", index=False)
    prefetch(days, str(base.backtest.cache_dir))
    trades, cands = [], []
    for i, (s, d) in enumerate(days.itertuples(index=False)):
        cfg = base.model_copy(deep=True)
        cfg.symbols = [s]
        cfg.backtest.start = cfg.backtest.end = date.fromisoformat(str(d))
        try:
            rep = asyncio.run(run_backtest(cfg, "e4"))
        except Exception as e:  # missing archive day etc.
            print(f"{s} {d} failed: {e}", flush=True)
            continue
        for t in rep["trades"]:
            t["net"] = t["pnl"] - t["fees"] + t["funding"]
            trades.append({k: t[k] for k in t if k != "meta"} | {"day": str(d)})
        c = pd.read_csv(rep["candidates_file"])
        c["day"] = str(d)
        cands.append(c)
        print(f"[{i + 1}/{len(days)}] {s} {d}: {len(rep['trades'])} trades", flush=True)
    eng = pd.DataFrame(trades)
    eng.to_csv(OUT / "engine_trades.csv", index=False)
    pd.concat(cands).to_csv(OUT / "engine_candidates.csv.gz", index=False)
    compare(days)


def compare(days: pd.DataFrame | None = None):
    days = pd.read_csv(OUT / "days.csv") if days is None else days
    eng = pd.read_csv(OUT / "engine_trades.csv")
    sim = pd.read_csv(RUNS / "E3_robust" / "L0_lead" / "trades.csv")
    sim["day"] = pd.to_datetime(sim.t, unit="ms").dt.date.astype(str)
    keys = set(zip(days.symbol, days.day.astype(str), strict=True))
    sim = sim[[(s, d) in keys for s, d in zip(sim.symbol, sim.day, strict=True)]].copy()
    eng["m"] = eng.open_time // 60000 * 60000
    sim["m"] = sim.t
    m = sim.merge(eng, on=["symbol", "m"], how="outer", suffixes=("_sim", "_eng"), indicator=True)
    notional = 297
    both = m[m._merge == "both"].copy()
    both["entry_diff%"] = (both.entry_price / both.entry - 1) * 100
    both["exit_diff%"] = (both.exit_price / both.exit - 1) * 100
    both["net_eng%"] = both.net_eng / (both.qty * both.entry_price) * 100
    both["net_diff%"] = both["net_eng%"] - both.net_sim
    stops = both[both.reason_eng.isin(["exchange_stop", "stop_loss"])]
    out = {
        "coin_days": len(days),
        "sim_trades": int((m._merge != "right_only").sum()),
        "engine_trades": int((m._merge != "left_only").sum()),
        "matched": len(both),
        "sim_only": int((m._merge == "left_only").sum()),
        "engine_only": int((m._merge == "right_only").sum()),
        "sim_net_usdt_all": float(sim.net.sum() * notional / 100),
        "engine_net_usdt_all": float(eng.net.sum()),
        "matched_sim_avg%": float(both.net_sim.mean()),
        "matched_engine_avg%": float(both["net_eng%"].mean()),
        "engine_avg%_all": float((eng.net / (eng.qty * eng.entry_price) * 100).mean()),
        "same_exit_reason": float(
            (
                both.reason_sim.str[:5]
                == both.reason_eng.str.replace("exchange_stop", "stop_loss").str[:5]
            ).mean()
        ),
        "stop_exit_engine_avg%": float(stops["net_eng%"].mean()) if len(stops) else None,
        "engine_win_rate": float((eng.net > 0).mean()),
    }
    cand = pd.read_csv(OUT / "engine_candidates.csv.gz")
    # "rejected:spread 0.0507%" -> "rejected:spread"
    oc = cand.outcome.str.replace(r"^(rejected:\w+).*$", r"\1", regex=True)
    out["engine_candidate_outcomes"] = oc.value_counts().to_dict()
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    cols = [
        "symbol",
        "m",
        "_merge",
        "reason_sim",
        "reason_eng",
        "entry",
        "entry_price",
        "exit",
        "exit_price",
        "net_sim",
        "net_eng%",
        "net_diff%",
    ]
    both_all = m.copy()
    both_all["net_eng%"] = both_all.net_eng / (both_all.qty * both_all.entry_price) * 100
    both_all["net_diff%"] = both_all["net_eng%"] - both_all.net_sim
    both_all[cols].to_csv(OUT / "matched.csv", index=False)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1:] == ["compare"]:
        compare()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
