"""E10 pooled view and timing: one configuration (stop, reward/risk, TP1 all at the body) over the timeframes
1h / 4h / 12h / 1d and both sides, rule3 only. Is the pooled R per trade positive in both periods, and do
trades cluster by year / month / weekday / hour?"""

import sys

import numpy as np
import pandas as pd

import common as C
import e10_report as R

TFS = ["1h", "4h", "12h", "1d"]


def main(stop_k=0.5, rr=1.5):
    D = pd.read_pickle("data/research/divergence_e10_all.pkl")
    T = D[D.rule3 & (D.exit == "body") & (D.stop_k == stop_k) & (D.rr == rr) & D.tf.isin(TFS)].copy()
    rows = []
    for name, g in [("pooled", T)] + [(f"{tf} {'long' if s > 0 else 'short'}", x) for (tf, s), x in T.groupby(["tf", "side"])]:
        for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END), ("all", C.T_IS, C.T_END)):
            x = g[(g.t >= t0) & (g.t < t1)]
            rows.append({"group": name, "per": tag} | R.gap(x))
    S = pd.DataFrame(rows)
    out = R.OUT
    S.to_csv(out / f"pooled_s{stop_k}_rr{rr}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    tm = R.timing(T)
    for k, v in tm.items():
        v.to_csv(out / f"timing_{k}_s{stop_k}_rr{rr}.csv", float_format="%.4f")
        print(k)
        print(v.round(3).to_string())
    # trades per month over time and longest gaps
    m = pd.to_datetime(T.t_in, unit="ms").dt.to_period("M").value_counts().sort_index()
    m = m.reindex(pd.period_range("2021-01", "2026-09", freq="M"), fill_value=0)
    print("trades/month: mean", round(m.mean(), 2), "median", m.median(), "zero months", int((m == 0).sum()),
          "max", int(m.max()), m.idxmax())  # fmt: skip
    m.to_csv(out / f"per_month_s{stop_k}_rr{rr}.csv")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(float(v) for v in a)) if a else main()
