"""D1 summary: per timeframe / side / indicator, divergence vs the same price pattern without it."""

import numpy as np
import pandas as pd

import common as C

D = pd.read_pickle("data/research/divergence_d1.pkl")
COST2 = 2 * C.COST
rows = []
for (tf, side), g in D.groupby(["tf", "side"]):
    ext = (g.side * g.dp) < 0  # new lower low / higher high
    for ind in ("macd", "hist", "rsi", "obv", "cvd", "vol"):
        groups = {
            "regular": ext & (g[ind] > 0),
            "confirmed": ext & (g[ind] <= 0),
            "hidden": ~ext & (g[ind] < 0),
        }
        for kind, m in groups.items():
            for H in (1, 3, 6, 12, 24):
                row = {"tf": tf, "side": side, "ind": ind, "kind": kind, "H": H}
                for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
                    mm = (m & pm).to_numpy()
                    r = g[f"r{H}"].to_numpy()[mm] - COST2
                    mu, t, nd = C.day_t(g.t.to_numpy()[mm], r)
                    row[f"{tag}_n"] = int(mm.sum())
                    row[f"{tag}_bp"] = 1e4 * mu
                    row[f"{tag}_t"] = t
                rows.append(row)
S = pd.DataFrame(rows)
S.to_csv(C.RESULTS / "D1_events/summary.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
# excess of regular divergence over the confirmed control, same H
k = ["tf", "side", "ind", "H"]
R = S[S.kind == "regular"].set_index(k)
Q = S[S.kind == "confirmed"].set_index(k)
X = R[["is_n", "is_bp", "is_t", "oos_n", "oos_bp", "oos_t"]].copy()
X["is_ex"] = R.is_bp - Q.is_bp
X["oos_ex"] = R.oos_bp - Q.oos_bp
print(X.round(1).to_string())
print(S[S.kind == "hidden"].sort_values("is_t", ascending=False).head(20).round(1).to_string())
