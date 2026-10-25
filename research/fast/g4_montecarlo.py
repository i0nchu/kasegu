"""G4: Monte Carlo of the EPR15 account under the PR #11 risk model, by fill scenario and period.

Monte Carlo only re-shuffles what history and the fill calibration already contain; it adds no
new market evidence. What it adds is the spread of outcomes a year can have.

Inputs
  G2 ledger (deploy6 = the six coins of configs/epr15.example.yaml, 19.8 USDT per trade each)
  G3 calibration: fill probability by penetration bucket and the empirical slippage samples
Scenarios (fill model, slippage)
  base          every simulated fill fills (research rule), research slippage (0.02% exits)
  conservative  queue_med fill probabilities; exit / stop slippage drawn from the measured samples
  stress        queue_p90 fill probabilities (fills only on >= 0.2% flushes); exits from the worst
                25% of the exit samples, stops 0.46% worse than measured
Paths: 10,000 one-year paths, stationary block bootstrap of whole UTC days (mean block 10 days)
from the chosen period, so every day keeps all of its trades across coins together (the days
when several coins lose at once are drawn as one unit). Inside a path the engine rules apply in
entry-time order: max_daily_loss = 3 USDT pauses the rest of the UTC day, 3 losses in a row pause
120 minutes, and the account stop at -10 USDT (realized + the day's open loss, with every trade
of the day at its worst minute at once, a pessimistic bound) ends the path.
Periods: all (2022-01 to 2026-09), 2022, 2023, 2024, 2025-26.

Reported per scenario x period: P(account stop within a year), max drawdown quantiles (USDT and %
of the 100 USDT example balance), longest losing streak, P(at least one day hitting the daily
loss limit) and the mean number of such days, final net quantiles, trades per year.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/g4_montecarlo.py
"""

import json

import numba as nb
import numpy as np
import pandas as pd

import fast as F

OUT = F.RESULTS / "G4_montecarlo"
PATHS = 10_000
DAYS = 365
BLOCK = 10
LIMITS = dict(account=10.0, daily=3.0, streak=3, pause_ms=120 * 60_000, balance=100.0)
PERIODS = {
    "all": ("2022-01-01", "2026-10-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025-26": ("2025-01-01", "2026-10-01"),
}


@nb.njit(cache=True)
def run_paths(day0, day1, t_in, t_out, net, mae, pfill, is_stop, notional, xs, ss, days_idx,
              acct, dlim, slim, pause_ms, seed):  # fmt: skip
    np.random.seed(seed)
    P, D = days_idx.shape
    res = np.zeros((P, 8))
    for p in range(P):
        cum = 0.0
        peak = 0.0
        mdd = 0.0
        streak = 0
        worst_streak = 0
        lose_streak_run = 0
        breach_days = 0
        trades = 0
        stopped = -1
        for dd in range(D):
            d = days_idx[p, dd]
            day_net = 0.0
            open_low = 0.0
            paused_until = -1
            hit = False
            for k in range(day0[d], day1[d]):
                if t_in[k] < paused_until or hit:
                    continue
                if np.random.random() >= pfill[k]:
                    continue
                x = net[k]
                if is_stop[k]:
                    if len(ss) > 0:
                        x -= notional * (ss[np.random.randint(len(ss))] - 0.0005)
                elif len(xs) > 0:
                    x -= notional * (xs[np.random.randint(len(xs))] - 0.0002)
                trades += 1
                open_low += min(mae[k], 0.0) * notional
                if acct > 0 and cum + day_net + open_low <= -acct:
                    stopped = dd
                    # everything is closed at about the limit (worse if this trade alone is worse)
                    cum = min(cum + day_net + x, -acct)
                    break
                day_net += x
                lose_streak_run = lose_streak_run + 1 if x < 0 else 0
                worst_streak = max(worst_streak, lose_streak_run)
                streak = streak + 1 if x < 0 else 0
                if dlim > 0 and day_net <= -dlim:
                    hit = True
                elif slim > 0 and streak >= slim:
                    paused_until = t_out[k] + pause_ms
                    streak = 0
            if stopped >= 0:
                peak = max(peak, cum)
                mdd = min(mdd, cum - peak)
                break
            if hit:
                breach_days += 1
            cum += day_net
            peak = max(peak, cum)
            mdd = min(mdd, cum - peak)
        res[p, 0] = cum
        res[p, 1] = mdd
        res[p, 2] = worst_streak
        res[p, 3] = breach_days
        res[p, 4] = stopped
        res[p, 5] = trades
    return res


def bootstrap(n_days, rng):
    idx = np.empty((PATHS, DAYS), dtype=np.int64)
    for p in range(PATHS):
        i, d = rng.integers(n_days), 0
        while d < DAYS:
            L = min(rng.geometric(1 / BLOCK), DAYS - d)
            idx[p, d : d + L] = (i + np.arange(L)) % n_days
            d += L
            i = rng.integers(n_days)
    return idx


def scenario_inputs(L: pd.DataFrame, cal: dict, name: str):
    if name == "base":
        return np.ones(len(L)), np.zeros(0), np.zeros(0)
    model = "queue_med" if name == "conservative" else "queue_p90"
    edges = np.array(cal["pen_edges"])
    probs = np.array(cal["fill_prob"][model])
    b = np.clip(np.searchsorted(edges, L.pen.to_numpy(), side="right") - 1, 0, len(probs) - 1)
    pf = probs[b]
    xs, ss = np.array(cal["exit_slip"]), np.array(cal["stop_slip"])
    if name == "stress":
        # worst quarter of the exit samples; stops 0.46% worse, the engine mark-price replay gap
        xs, ss = xs[xs >= np.quantile(xs, 0.75)], ss + cal["stop_slip_stress_add"]
    return pf, xs, ss


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = json.loads((F.RESULTS / "G3_fills" / "calibration.json").read_text())
    L = pd.read_pickle(F.OUT / "G3_ledger_deploy6.pkl")  # G2 ledger + penetration from G3
    L = L.sort_values("t_in").reset_index(drop=True)
    notional = float(L.notional.iloc[0])
    rows = []
    rng = np.random.default_rng(5)
    for pname, (a, b) in PERIODS.items():
        lo, hi = pd.Timestamp(a).value // 10**6, pd.Timestamp(b).value // 10**6
        S = L[(L.t_in >= lo) & (L.t_in < hi)].reset_index(drop=True)
        day = (S.t_in // F.DAY).to_numpy()
        all_days = np.arange(lo // F.DAY, hi // F.DAY)
        day0 = np.searchsorted(day, all_days)
        day1 = np.searchsorted(day, all_days, side="right")
        idx = bootstrap(len(all_days), rng)
        for sc in ("base", "conservative", "stress"):
            pf, xs, ss = scenario_inputs(S, cal, sc)
            r = run_paths(day0, day1, S.t_in.to_numpy(), S.t_out.to_numpy(), S.net.to_numpy(),
                          S.mae.to_numpy(), pf, (S.why == 1).to_numpy(), notional, xs, ss, idx,
                          LIMITS["account"], LIMITS["daily"], LIMITS["streak"], LIMITS["pause_ms"], 17)  # fmt: skip
            fin, mdd, stk, br, st, n = r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5]
            rows.append({
                "period": pname, "scenario": sc, "hist_days": len(all_days), "hist_trades": len(S),
                "p_account_stop": (st >= 0).mean(),
                "median_days_to_stop": float(np.median(st[st >= 0])) if (st >= 0).any() else None,
                "mdd_med_usdt": np.median(mdd), "mdd_p95_usdt": np.quantile(mdd, 0.05),
                "mdd_p95_pct_of_100": np.quantile(mdd, 0.05) / LIMITS["balance"] * 100,
                "streak_med": np.median(stk), "streak_p95": np.quantile(stk, 0.95),
                "p_daily_limit_day": (br > 0).mean(), "daily_limit_days_mean": br.mean(),
                "net_p5": np.quantile(fin, 0.05), "net_med": np.median(fin), "net_p95": np.quantile(fin, 0.95),
                "p_loss": (fin < 0).mean(), "trades_med": np.median(n),
            })  # fmt: skip
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, float_format="%.4g")
    json.dump({"paths": PATHS, "days": DAYS, "block": BLOCK, "limits": LIMITS, "seed": 5}, open(OUT / "config.json", "w"))


if __name__ == "__main__":
    main()
