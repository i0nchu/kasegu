"""E2: which features at the trigger separate double tops (fail) from breakouts?

For each timeframe and side, every feature is cut into quintiles on IS quantiles; reported per
quintile: fail share among resolved setups (fail / (fail + break)) and the mean net of the base
trade (reversal entry, stop at H1, take profit at L1), IS and OOS.
"""

import numpy as np
import pandas as pd

import common as C

FEATS = ["retr", "impulse_atr", "dist_atr", "recov", "macd_gap", "macd_slope", "vol_leg", "vol_now",
         "flow1", "flow2", "flow_chg", "fund24", "fund_chg", "atr_chg", "rv_ratio", "bars_leg1",
         "bars_since", "trend200", "rsi_t", "rsi_h1"]  # fmt: skip


def main():
    D = pd.read_pickle("data/research/divergence_e1.pkl")
    D = D[D.label.isin([1, 2])].replace([np.inf, -np.inf], np.nan)
    D["fail"] = (D.label == 2).astype(float)
    rows = []
    for (tf, side), g in D.groupby(["tf", "side"]):
        isg = g[g.t < C.T_OOS]
        for f in FEATS:
            qs = isg[f].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            b = np.digitize(g[f], qs)
            b[g[f].isna().to_numpy()] = -1
            for k in range(5):
                row = {"tf": tf, "side": side, "feat": f, "q": k}
                for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
                    m = (b == k) & pm.to_numpy()
                    row[f"{tag}_n"] = m.sum()
                    row[f"{tag}_fail"] = g.fail.to_numpy()[m].mean()
                    row[f"{tag}_bp"] = 1e4 * np.nanmean(g["net_b0.0"].to_numpy()[m])
                rows.append(row)
    S = pd.DataFrame(rows)
    (C.RESULTS / "E2_features").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "E2_features/quintiles.csv", index=False, float_format="%.4f")
    # spread: fail share top minus bottom quintile, both periods
    sp = S.pivot_table(index=["tf", "side", "feat"], columns="q", values=["is_fail", "oos_fail"])
    out = pd.DataFrame({
        "is_q0": sp[("is_fail", 0)], "is_q4": sp[("is_fail", 4)],
        "oos_q0": sp[("oos_fail", 0)], "oos_q4": sp[("oos_fail", 4)],
    })
    out["is_spread"] = out.is_q4 - out.is_q0
    out["oos_spread"] = out.oos_q4 - out.oos_q0
    out["consistent"] = np.sign(out.is_spread) == np.sign(out.oos_spread)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(out.sort_values(["tf", "side", "is_spread"]).round(3).to_string())


if __name__ == "__main__":
    main()
