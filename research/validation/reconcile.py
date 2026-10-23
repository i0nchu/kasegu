"""Independent summaries of E4 one-minute simulator vs aggTrade engine replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def reconcile(matched_path: Path, engine_path: Path, out: Path) -> dict:
    m = pd.read_csv(matched_path)
    eng = pd.read_csv(engine_path)
    both = m[m._merge == "both"].copy()
    only_sim = int((m._merge == "left_only").sum())
    only_engine = int((m._merge == "right_only").sum())
    stop = both.reason_eng.isin(["exchange_stop", "stop_loss"])
    rows = []
    for label, x in (("all_matched", both), ("stops", both[stop]), ("non_stops", both[~stop])):
        rows.append(
            {
                "scope": label,
                "n": len(x),
                "sim_avg_pct": x.net_sim.mean(),
                "engine_avg_pct": x["net_eng%"].mean(),
                "mean_error_pct": x["net_diff%"].mean(),
                "median_error_pct": x["net_diff%"].median(),
                "p05_error_pct": x["net_diff%"].quantile(0.05),
                "p95_error_pct": x["net_diff%"].quantile(0.95),
                "mae_error_pct": x["net_diff%"].abs().mean(),
                "sign_disagreement": (np.sign(x.net_sim) != np.sign(x["net_eng%"])).mean(),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "execution_error.csv", index=False)
    per_trade = m[
        [
            "symbol",
            "m",
            "_merge",
            "reason_sim",
            "reason_eng",
            "net_sim",
            "net_eng%",
            "net_diff%",
        ]
    ].copy()
    per_trade.to_csv(out / "trade_reconciliation.csv", index=False)
    summary = {
        "simulator_level": "one-minute OHLC with assumed intrabar path",
        "engine_level": "aggTrades with synthetic last-trade +/- half-tick quotes",
        "real_order_book_observed": False,
        "matched": len(both),
        "sim_only": only_sim,
        "engine_only": only_engine,
        "match_rate_union": len(both) / len(m),
        "engine_total_net_usdt": float(eng.net.sum()),
        "matched_sim_avg_pct": float(both.net_sim.mean()),
        "matched_engine_avg_pct": float(both["net_eng%"].mean()),
        "engine_to_sim_expectancy_ratio": float(both["net_eng%"].mean() / both.net_sim.mean()),
        "matched_mean_error_pct": float(both["net_diff%"].mean()),
        "matched_abs_error_pct": float(both["net_diff%"].abs().mean()),
    }
    (out / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
