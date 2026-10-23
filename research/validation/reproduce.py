"""Reproduce the frozen VDA candidate and adverse execution-only variants."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import sim  # noqa: E402
from download import month_range  # noqa: E402
from lab import UNIVERSE  # noqa: E402
from sim import Cfg  # noqa: E402

MONTHS = month_range("2024-10:2026-09")


def _one(args):
    symbol, cfg, snapshot, ticks = args
    sim.CACHE = snapshot
    d = sim.load_klines(symbol, MONTHS)
    f = sim.load_funding(symbol, MONTHS)
    tr = sim.simulate(d, replace(cfg, tick=ticks[symbol]), f)
    tr.insert(0, "symbol", symbol)
    return tr


def reproduce(snapshot: Path, frozen_trades: Path, out: Path, workers: int = 4) -> dict:
    ticks = json.loads((snapshot / "ticks.json").read_text())
    base = Cfg(
        k=12,
        trend="against",
        sides="long",
        sl=3.0,
        max_hold=30,
        extra={
            "months": MONTHS,
            "min_vol_ratio": 1.07,
            "max_ret60": -0.26,
            "min_atr_pct": 0.081,
        },
    )
    scenarios = {
        "frozen": base,
        "penetration_0.2pct": replace(base, penetrate=0.2),
        "stop_slip_0.3pct": replace(base, level_slip=0.3),
        "penetration_0.2_stop_slip_0.3": replace(base, penetrate=0.2, level_slip=0.3),
    }
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    frozen = pd.read_csv(frozen_trades)
    exact = {}
    for name, cfg in scenarios.items():
        jobs = [(s, cfg, snapshot, ticks) for s in UNIVERSE]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            tr = pd.concat(ex.map(_one, jobs), ignore_index=True)
        tr.insert(1, "trade_id", tr.groupby("symbol").cumcount())
        tr["net_usdt"] = tr.net * cfg.notional / 100
        tr.to_csv(out / f"reproduced_{name}.csv", index=False)
        rows.append(
            {
                "scenario": name,
                "trades": len(tr),
                "avg_net_pct": tr.net.mean(),
                "net_usdt_independent_accounts": tr.net_usdt.sum(),
                "positive_symbols": int((tr.groupby("symbol").net.sum() > 0).sum()),
                "positive_months": int(
                    (
                        tr.assign(month=pd.to_datetime(tr.t, unit="ms").dt.strftime("%Y-%m"))
                        .groupby("month")
                        .net.sum()
                        > 0
                    ).sum()
                ),
            }
        )
        if name == "frozen":
            a = frozen[["symbol", "t", "entry", "exit", "net"]]
            b = tr[["symbol", "t", "entry", "exit", "net"]]
            joined = a.merge(
                b,
                on=["symbol", "t"],
                how="outer",
                suffixes=("_published", "_new"),
                indicator=True,
            )
            both = joined[joined._merge == "both"]
            exact = {
                "published_trades": len(a),
                "reproduced_trades": len(b),
                "matched_keys": len(both),
                "published_only": int((joined._merge == "left_only").sum()),
                "reproduced_only": int((joined._merge == "right_only").sum()),
                "max_abs_net_diff_pct": float((both.net_published - both.net_new).abs().max()),
                "max_abs_entry_diff": float((both.entry_published - both.entry_new).abs().max()),
                "max_abs_exit_diff": float((both.exit_published - both.exit_new).abs().max()),
            }
    pd.DataFrame(rows).to_csv(out / "frozen_candidate_stress.csv", index=False)
    (out / "reproduction.json").write_text(json.dumps(exact, indent=2) + "\n")
    return {"exact": exact, "scenarios": rows}
