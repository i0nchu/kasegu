"""E11 report: A-D alpha attribution (development vs pre-2021 hold-out), fill realism, market states.

Units, kept apart on purpose:
  R        trade result in multiples of the planned risk (entry to stop + costs); +1R = made what was risked
  net      trade result as a price return incl. costs and funding (what chart titles show, in %)
  account  equity return when each trade risks 1% of equity (account return = 1% x R summed over trades)
"""

import numpy as np
import pandas as pd

import common as C
import e11_regime as G

OUT = C.RESULTS / "E11_attrib"
POOL = ["1h", "4h", "12h", "1d"]


def stat(g):
    mu, tt, nd = C.day_t(g.t.to_numpy(), g.R.to_numpy())
    return {"n": len(g), "win": (g.R > 0).mean() if len(g) else np.nan, "R": mu, "t": tt,
            "net_bp": 1e4 * g.net.mean() if len(g) else np.nan}  # fmt: skip


def versions(D, A, thr):
    base = D[(D.thr == thr)]
    v = {"B": base[base.divfail == 0], "C": base[(base.divfail == 0) & base.macd_ctx],
         "D": base[(base.divfail == 0) & base.macd_ctx & base.r3]}  # fmt: skip
    if thr > 0:
        v["D+"] = base[(base.divfail == 1) & base.macd_ctx & base.r3]
        v["A"] = A
    return v


def attribution(D, A):
    rows = []
    for thr in (0.0005, 0.0):
        for name, g in versions(D, A, thr).items():
            for per in ("dev", "hold_pre"):
                gp = g[g.period == per]
                for tf in ["15m", *POOL, "1h-1d"]:
                    x = gp[gp.tf.isin(POOL)] if tf == "1h-1d" else gp[gp.tf == tf]
                    rows.append({"thr": thr, "version": name, "period": per, "tf": tf} | stat(x))
    return pd.DataFrame(rows)


def main():
    D = pd.read_pickle("data/research/divergence_e11.pkl")
    A = pd.read_pickle("data/research/divergence_e11_random.pkl")
    S = attribution(D, A)
    S.to_csv(OUT / "attribution.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(S[S.thr == 0.0005].pivot_table(index=["tf", "version"], columns="period", values=["n", "win", "R", "t"])
          .round(3).to_string())  # fmt: skip
    print(S[(S.thr == 0.0) & (S.tf == "1h-1d")].round(3).to_string())
    N = pd.read_csv(OUT / "fill_counts.csv")
    F = N[N.divfail == 0].groupby(["thr", "tf"])[["setups", "filled", "touched"]].sum()
    F["fill_rate"] = F.filled / F.setups
    F["touch_not_filled"] = F.touched / F.setups
    F.to_csv(OUT / "fill_rates.csv", float_format="%.4f")
    print(F.round(3).to_string())

    # market states: on C (MACD context, more trades) and D, pooled 1h-1d, main fill rule, no MACD exit
    M, R30 = G.market_states()
    rows = []
    for name in ("C", "D"):
        T = versions(D, A, 0.0005)[name]
        T = G.attach(T[T.tf.isin(POOL)].copy(), M, R30)
        dev, hold = T[T.period == "dev"], T[T.period == "hold_pre"]
        cuts = {f: dev[f].quantile([1 / 3, 2 / 3]).to_list() for f in G.FEATS}
        for per, x in (("dev", dev), ("hold_pre", hold)):
            t = G.table(x, cuts)
            t["set"], t["period"] = name, per
            rows.append(t)
    R = pd.concat(rows)
    R.to_csv(OUT / "market_states.csv", index=False, float_format="%.4f")
    print(R.pivot_table(index=["set", "feature", "tercile"], columns="period", values=["n", "R", "t"],
                        observed=True).round(3).to_string())  # fmt: skip


if __name__ == "__main__":
    main()
