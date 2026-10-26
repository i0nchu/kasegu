"""E2b: feature effects after removing geometry.

excess = fail - (1 - recov): how much more often the setup fails than a driftless random walk
starting at the same spot between L1 and H1 would. Per feature quintile (IS cuts), mean excess in
percentage points, IS and OOS, with a day-clustered t.
"""

import numpy as np
import pandas as pd

import common as C
from e2_features import FEATS


def main():
    D = pd.read_pickle("data/research/divergence_e1.pkl").replace([np.inf, -np.inf], np.nan)
    D = D[D.label.isin([1, 2])].copy()
    D["excess"] = (D.label == 2) - (1 - D.recov.clip(0, 1))
    rows = []
    for (tf, side), g in D.groupby(["tf", "side"]):
        isg = g[g.t < C.T_OOS]
        for f in ["all"] + FEATS:
            if f == "all":
                b = np.zeros(len(g), int)
                ks = [0]
            else:
                b = np.digitize(g[f], isg[f].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
                b[g[f].isna().to_numpy()] = -1
                ks = range(5)
            for k in ks:
                row = {"tf": tf, "side": side, "feat": f, "q": k}
                for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
                    m = (b == k) & pm.to_numpy()
                    mu, t, _ = C.day_t(g.t.to_numpy()[m], g.excess.to_numpy()[m].astype(float))
                    row |= {f"{tag}_pp": 100 * mu, f"{tag}_t": t}
                rows.append(row)
    S = pd.DataFrame(rows)
    (C.RESULTS / "E2_features").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "E2_features/excess_quintiles.csv", index=False, float_format="%.3f")
    P = S[S.q.isin([0, 4])].pivot_table(index=["tf", "side", "feat"], columns="q", values=["is_pp", "oos_pp"])
    P["is_spread"] = P[("is_pp", 4)] - P[("is_pp", 0)]
    P["oos_spread"] = P[("oos_pp", 4)] - P[("oos_pp", 0)]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(S[S.feat == "all"].round(2).to_string())
    print(P.round(1).to_string())


if __name__ == "__main__":
    main()
