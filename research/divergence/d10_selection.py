"""D10: multiple-selection audit of DIV-S4 on 2021-01 to 2026-09 (holdout 2020 not touched).

1. Inventory: every D-step summary and whether its OOS columns were shown (they were in all).
2. D3 under the null: how many IS t > 2 out of 720 configurations chance alone gives, and where
   the chosen 4h MACD short ranks on IS alone.
3. Permutation test: shuffle the feature vectors of the 4h pivot events within each calendar
   month, or within each day (outcomes and timing stay, any link between feature and outcome
   breaks; the day version only lets the feature pick coins on the same day), then
   take the best tercile excess over all 16 D6 features x 2 sides x 2 tails plus the MACD+CVD
   top quintile. The observed MACD+CVD excess is compared with that maximum.
4. Deflated Sharpe ratio of the DIV-S4 account (W = 10%, max 3) for 100 / 720 / 1600 trials, with
   the trial Sharpe variance taken from the D3 grid.
Usage: PYTHONPATH=research/divergence python research/divergence/d10_selection.py
"""

import json
from math import e as E_
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

import common as C
import d6_features as D6
import d7_account as D7

OUT = C.RESULTS / "D10_selection"
RNG = np.random.default_rng(10)
NPERM = 1000


def inventory():
    rows = []
    for step, f in (("D1", "D1_events/summary.csv"), ("D2", "D2_fm/summary.csv"), ("D3", "D3_strategy/summary.csv"),
                    ("D4", "D4_controls/summary.csv"), ("D5", "D5_macd_robust/summary.csv"),
                    ("D6", "D6_features/buckets.csv"), ("D7", "D7_account/summary.csv")):  # fmt: skip
        S = pd.read_csv(C.RESULTS / f)
        oos = any("oos" in c for c in S.columns) or ("period" in S.columns and (S.period == "oos").any())
        rows.append({"step": step, "rows": len(S), "oos_shown": oos})
    return pd.DataFrame(rows)


def d3_null():
    S = pd.read_csv(C.RESULTS / "D3_strategy/summary.csv")
    S = S[S.is_t.notna()]
    key = (S.tf == "4h") & (S.ind == "macd") & (S.side == -1) & (S.trend == "any") & S.zone.isna() & (S.stop == "atr")
    key &= (S.k == 1.5) & (S.tpR == 2.0)
    rank = int((S.is_t > S[key].is_t.iloc[0]).sum() + 1)
    return {"configs": len(S), "is_t_gt2": int((S.is_t > 2).sum()), "expected_by_chance": 0.0228 * len(S),
            "both_t_gt2": int(((S.is_t > 2) & (S.oos_t > 2)).sum()), "chosen_is_rank": rank,
            "chosen_is_t": float(S[key].is_t.iloc[0]), "chosen_oos_t": float(S[key].oos_t.iloc[0]),
            "sd_of_is_t": float(S.is_t.std())}  # fmt: skip


def perm_test(D, by):
    D = D[(D.tx > 0) & (D.t < C.T_END)].reset_index(drop=True)
    key = (D.t // C.DAY).to_numpy() if by == "day" else pd.to_datetime(D.t, unit="ms").dt.to_period("M").astype(str).to_numpy()
    side = D.side.to_numpy()
    net = D.net.to_numpy()
    F = D[D6.FEATS].to_numpy()
    sh = side == -1
    # IS percentiles for the MACD+CVD score, as in D7 (short side)
    isg = D[sh & (D.t < C.T_OOS)]
    qm, qc = np.sort(isg.macd.dropna()), np.sort(isg.cvd.dropna())
    jm, jc = D6.FEATS.index("macd"), D6.FEATS.index("cvd")
    groups = [g for g in pd.Series(np.arange(len(D))).groupby(key).indices.values() if len(g) > 1]

    def stats(F):
        out = []
        for s in (-1, 1):
            m = side == s
            base = net[m].mean()
            for k in range(F.shape[1]):
                x = F[m, k]
                ok = ~np.isnan(x)
                lo, hi = np.quantile(x[ok], [1 / 3, 2 / 3])
                out.append(net[m][ok & (x >= hi)].mean() - base)
                out.append(net[m][ok & (x <= lo)].mean() - base)
        s2 = (np.searchsorted(qm, F[sh, jm]) / len(qm) + np.searchsorted(qc, F[sh, jc]) / len(qc)) / 2
        combo = net[sh][s2 >= np.quantile(s2, 0.8)].mean() - net[sh].mean()
        return np.array(out), combo

    fam, combo = stats(F)
    mx, cs, ms = [], [], []
    km = 2 * D6.FEATS.index("macd")  # short side, MACD top tercile
    for _ in range(NPERM):
        P = F.copy()
        for g in groups:
            P[g] = F[RNG.permutation(g)]
        a, c = stats(P)
        mx.append(max(a.max(), c))
        cs.append(c)
        ms.append(a[km])
    mx, cs, ms = np.array(mx), np.array(cs), np.array(ms)
    names = [f"{'short' if s < 0 else 'long'}:{f}:{tail}" for s in (-1, 1) for f in D6.FEATS for tail in ("top", "bottom")]
    best = names[int(np.argmax(fam))]
    return {"combo_excess_bp": 1e4 * combo, "best_single": best, "best_single_bp": 1e4 * fam.max(),
            "null_max_median_bp": 1e4 * np.median(mx), "null_max_p95_bp": 1e4 * np.quantile(mx, 0.95),
            "p_familywise_combo": float((mx >= combo).mean()), "p_single_combo": float((cs >= combo).mean()),
            "macd_top_excess_bp": 1e4 * fam[km], "p_single_macd_top": float((ms >= fam[km]).mean()),
            "shuffle_within": by, "perms": NPERM}  # fmt: skip


def dsr(r, trials, var_t):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado) on daily returns."""
    r = r[r != 0] if (r != 0).sum() > 100 else r
    sr = r.mean() / r.std()
    n = len(r)
    sk = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    ku = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    g = 0.5772
    var_sr = var_t / n  # trial t-values -> Sharpe on this many days
    sr0 = sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / trials) + g * norm.ppf(1 - 1 / (trials * E_)))
    z = (sr - sr0) * sqrt(n - 1) / sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2)
    return {"trials": trials, "sr_daily": sr, "sr0_daily": sr0, "dsr": float(norm.cdf(z)), "n_days": n}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    inv.to_csv(OUT / "inventory.csv", index=False)
    res = {"d3_null": d3_null()}
    D = pd.read_pickle("data/research/divergence_d6.pkl")
    res["perm"] = [perm_test(D, by) for by in ("month", "day")]
    K = pd.read_csv(C.RESULTS / "D7_account/ledger_macd_cvd_N3.csv.gz")
    r = C.daily(K, 0.10)
    # spread of trial results: variance of the D3 IS t-values (about 1 under no edge)
    var_t = float(pd.read_csv(C.RESULTS / "D3_strategy/summary.csv").is_t.var())
    res["dsr"] = [dsr(r.to_numpy(), n, var_t) for n in (100, 720, 1600)]
    mu, t, nd = C.day_t(K.t.to_numpy(), K.net.to_numpy())
    res["div_s4_day_t_full"] = t
    res["bonferroni_t_needed"] = {n: float(norm.ppf(1 - 0.05 / (2 * n))) for n in (100, 720, 1600)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(inv.to_string())
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
