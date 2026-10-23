"""Experiment groups with full per-order logs.

Each group writes data/research/runs/<exp>/<group>/:
  config.json      settings snapshot of the group
  trades.csv       every fill: entry, exit, reason, per-trade pnl and signal features
  candidates/      every order the strategy wanted to place, per symbol (csv.gz), with
                   outcome filled / expired / rejected:<why> and the trade it became
  summary.json     totals, expectancy and candidate outcome counts
"""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from lab import IS_END, OUT, UNIVERSE, bars
from sim import Cfg, signals, simulate, tick_of

RUNS = OUT / "runs"


def expectancy(net: pd.Series) -> dict:
    """Win rate, payoff and the win rate needed to break even."""
    if net.empty:
        return {}
    w, lo = net[net > 0], net[net <= 0]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-lo.mean()) if len(lo) else 0.0
    p = len(w) / len(net)
    return {
        "trades": len(net),
        "win_rate": p,
        "avg_win_pct": aw,
        "avg_loss_pct": al,
        "payoff": aw / al if al else None,
        "breakeven_win_rate": al / (aw + al) if aw + al else None,
        "expectancy_pct": p * aw - (1 - p) * al,
    }


def _one(args):
    symbol, cfg, folder, control, seed, n_ctrl = args
    df, fund = bars(symbol)
    if df.empty:
        return pd.DataFrame(), {}
    cfg = replace(cfg, tick=tick_of(symbol))
    entries = None
    if control:
        # random bars where the strategy had a direction, same side, entered at the open
        sig = signals(df, cfg)
        ok = np.flatnonzero(sig.want.to_numpy()[:-1] != 0) + 1
        rng = np.random.default_rng(seed + hash(symbol) % 10_000)
        pick = np.sort(rng.choice(ok, size=min(n_ctrl.get(symbol, 0), len(ok)), replace=False))
        entries = np.c_[pick, sig.want.to_numpy()[pick - 1]]
        tr = simulate(df, cfg, fund, sig=sig, entries=entries)
        counts = {"random_entries": int(len(pick))}
    else:
        tr, cand = simulate(df, cfg, fund, log=True)
        (folder / "candidates").mkdir(parents=True, exist_ok=True)
        cand.insert(0, "symbol", symbol)
        cand.to_csv(folder / "candidates" / f"{symbol}.csv.gz", index=False)
        counts = cand.outcome.value_counts().to_dict()
    if not tr.empty:
        tr.insert(0, "symbol", symbol)
        tr.insert(1, "trade_id", range(len(tr)))
    return tr, counts


def group(
    exp: str,
    name: str,
    cfg: Cfg,
    symbols=UNIVERSE,
    control: bool = False,
    n_ctrl: dict | None = None,
    seed: int = 7,
    note: str = "",
) -> pd.DataFrame:
    folder = RUNS / exp / name
    folder.mkdir(parents=True, exist_ok=True)
    snap = {
        "group": name,
        "note": note,
        "control_random_entry": control,
        "symbols": symbols,
        "cfg": asdict(cfg),
    }
    (folder / "config.json").write_text(json.dumps(snap, indent=1))
    jobs = [(s, cfg, folder, control, seed, n_ctrl or {}) for s in symbols]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_one, jobs))
    tr = pd.concat([r[0] for r in res if not r[0].empty], ignore_index=True)
    tr["month"] = pd.to_datetime(tr.t, unit="ms").dt.strftime("%Y-%m")
    tr["oos"] = tr.t >= IS_END
    tr["net_usdt"] = tr.net * cfg.notional / 100
    tr.to_csv(folder / "trades.csv", index=False)
    counts: dict = {}
    for _, c in res:
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
    out = {
        "group": name,
        "net_usdt": float(tr.net_usdt.sum()),
        "avg_net_pct": float(tr.net.mean()),
        "avg_gross_pct": float(tr.gross.mean()),
        "cost_pct": float((tr.gross - tr.net).mean()),
        "is_avg_net_pct": float(tr[~tr.oos].net.mean()),
        "oos_avg_net_pct": float(tr[tr.oos].net.mean()),
        "pos_months": int((tr.groupby("month").net.sum() > 0).sum()),
        "months": int(tr.month.nunique()),
        "pos_coins": int((tr.groupby("symbol").net.sum() > 0).sum()),
        "coins": int(tr.symbol.nunique()),
        "exit_reasons": tr.reason.value_counts().to_dict(),
        "expectancy": expectancy(tr.net),
        "candidates": counts,
    }
    (folder / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    return tr


def load(exp: str, name: str) -> tuple[pd.DataFrame, dict]:
    f = RUNS / exp / name
    return pd.read_csv(f / "trades.csv"), json.loads((f / "summary.json").read_text())


def path(exp: str) -> Path:
    return RUNS / exp


def publish(exp: str):
    """Copy the small, reviewable part of an experiment into research/results/ for git.

    Full trades and candidate logs stay in data/research/runs/ (too large for git).
    """
    import shutil

    src, dst = RUNS / exp, Path("research/results") / exp
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.csv"):
        shutil.copy(f, dst / f.name)
    for g in sorted(p for p in src.iterdir() if p.is_dir()):
        d = dst / g.name
        d.mkdir(exist_ok=True)
        for name in ("config.json", "summary.json"):
            if (g / name).exists():
                shutil.copy(g / name, d / name)
        tr = pd.read_csv(g / "trades.csv")
        by = tr.groupby("symbol").agg(
            trades=("net", "size"), net_usdt=("net_usdt", "sum"), avg_net_pct=("net", "mean")
        )
        cands = sorted((g / "candidates").glob("*.csv.gz")) if (g / "candidates").exists() else []
        if cands:
            oc = pd.concat(
                pd.read_csv(c, usecols=["symbol", "outcome"])
                .value_counts()
                .rename("n")
                .reset_index()
                for c in cands
            ).pivot_table(index="symbol", columns="outcome", values="n", fill_value=0)
            by = by.join(oc)
        by.to_csv(d / "by_symbol.csv")
