"""E6: what leverage does to the candidates, doubling odds versus ruin odds.

Input: the daily net return series of each candidate at its base sizing (E1 / E4 / E5). For an
account leverage multiple m the daily return is m * r (the same positions, m times larger), and
anything that takes equity to 0 or below is a liquidation (Binance closes the account).

Two views:
  history   the actual compounded path at m over the out-of-sample period (2024-01 to 2026-09)
  bootstrap 10,000 one-year paths drawn by a stationary block bootstrap of the daily returns
            (mean block 20 days, keeps volatility clusters and crash runs), from 2024-2026 for the
            point-in-time candidates, or 2020-2026 where marked.
            Reported: P(equity doubles within 90 / 180 / 365 days), P(ever loses 50% / 90%),
            median and 5th percentile of the final multiple.
The Kelly multiple mu / sigma^2 (daily) is the leverage that maximises long-run growth; twice
Kelly grows at zero on average and beyond that the account shrinks almost surely.

Daily returns understate intraday swings, so the ruin numbers are a lower bound.

Usage: PYTHONPATH=research/momentum uv run python research/momentum/e6_leverage.py
"""

import json

import numpy as np
import pandas as pd

import common as C

RNG = np.random.default_rng(7)
PATHS = 10_000
DAYS = 365
BLOCK = 20


def candidates() -> dict:
    """name -> (daily returns, bootstrap source). Point-in-time candidates are bootstrapped from
    2024-2026 only (the regime they were validated in); "full" also uses the 2020-2021 bull run."""
    out = {}
    e1 = pd.read_csv(C.RESULTS / "E1_trend/daily_returns.csv.gz", index_col=0, parse_dates=True)
    e5 = pd.read_csv(C.RESULTS / "E5_survivorship/daily_returns.csv.gz", index_col=0, parse_dates=True)
    e7 = pd.read_csv(C.RESULTS / "E7_breakout_pit/daily_returns.csv.gz", index_col=0, parse_dates=True)
    rot = [c for c in e5.columns if c.startswith("rot_f_L14_H7") and not c.endswith("_top20")]
    out["E1 tsmom_L 40 survivors (biased)"] = (e1["tsmom_L"], "oos")
    out["E5 tsmom_L point-in-time"] = (e5["tsmom_L"], "oos")
    out["E5 tsmom_L point-in-time, 2020-2026"] = (e5["tsmom_L"], "full")
    out["E5 rotation L14 point-in-time"] = (e5[rot].mean(axis=1), "oos")
    out["E7 dc4h_N40_M4_v2 long point-in-time"] = (e7["dc4h_N40_M4_v2_pit_L"], "oos")
    out["E7 squeeze L/S point-in-time"] = (e7["squeeze_pit_LS"], "oos")
    return {k: (v.dropna(), src) for k, (v, src) in out.items()}


def bootstrap_idx(n: int) -> np.ndarray:
    idx = np.empty((PATHS, DAYS), dtype=np.int64)
    for p in range(PATHS):
        i, d = RNG.integers(n), 0
        while d < DAYS:
            L = min(RNG.geometric(1 / BLOCK), DAYS - d)
            idx[p, d : d + L] = (i + np.arange(L)) % n
            d += L
            i = RNG.integers(n)
    return idx


def paths(r: np.ndarray, m: float, idx: np.ndarray) -> np.ndarray:
    x = np.maximum(1 + m * r[idx], 0.0)  # a day below -100% is a liquidation, equity stays 0
    return np.cumprod(x, axis=1)


def main():
    cands = candidates()
    out = C.RESULTS / "E6_leverage"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (s, src) in cands.items():
        oos = s[s.index >= pd.Timestamp("2024-01-01", tz=s.index.tz)].to_numpy()
        r = oos if src == "oos" else s.to_numpy()
        mu, var = r.mean(), r.var()
        kelly = mu / var
        idx = bootstrap_idx(len(r))
        ann_vol = r.std() * np.sqrt(365)
        print(f"{name}: daily mu {mu:.5f} vol(ann) {ann_vol:.3f} Kelly x{kelly:.1f}", flush=True)
        for m in sorted({1, 2, 3, 5, 10, 20, max(round(kelly, 1), 0.1), max(round(2 * kelly, 1), 0.1)}):
            if m <= 0:
                continue
            eq = paths(r, m, idx)
            peak = np.maximum.accumulate(eq, axis=1)
            hist = np.cumprod(np.maximum(1 + m * oos, 0))
            row = {
                "candidate": name,
                "mult": m,
                "ann_vol": round(ann_vol * m, 3),
                "kelly_mult": round(kelly, 1),
                "p_double_90d": round(float((eq[:, :90].max(1) >= 2).mean()), 3),
                "p_double_180d": round(float((eq[:, :180].max(1) >= 2).mean()), 3),
                "p_double_365d": round(float((eq.max(1) >= 2).mean()), 3),
                "p_lose50": round(float((eq.min(1) <= 0.5).mean()), 3),
                "p_lose90": round(float((eq.min(1) <= 0.1).mean()), 3),
                "p_dd50": round(float((np.where(peak > 0, eq / np.maximum(peak, 1e-12), 0).min(1) <= 0.5).mean()), 3),
                "median_1y": round(float(np.median(eq[:, -1])), 2),
                "p5_1y": round(float(np.percentile(eq[:, -1], 5)), 2),
                "oos_hist_multiple": round(float(hist[-1]), 2),
                "oos_hist_mdd": round(float((hist / np.maximum.accumulate(hist) - 1).min()), 3),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    json.dump({"paths": PATHS, "days": DAYS, "block": BLOCK, "seed": 7}, open(out / "config.json", "w"))
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
