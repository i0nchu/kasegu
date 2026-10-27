"""D10 holdout: one-time check of the frozen DIV-S4 rule on 2020-05-15 to 2020-12-31.

Pre-registered in docs/DIVS4_LOG.md (D10) before this script was run. Nothing here is tuned:
percentile references and thresholds come from 2021-2023 (divergence_d9.pkl), account N = 3,
W = 10%, costs 0.2% round trip. Events before 2020-05-15 are dropped (indicator warm-up: data
starts 2020-05-01).
Usage: PYTHONPATH=research/divergence python research/divergence/d10_holdout.py
"""

import json

import numpy as np
import pandas as pd

import common as C
import d7_account as D7

OUT = C.RESULTS / "D10_holdout"
T0 = int(pd.Timestamp("2020-05-15", tz="UTC").value // 10**6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle("data/research/divergence_d9.pkl")
    H = pd.read_pickle("data/research/divergence_d9_holdout.pkl")
    H = H[(H.t >= T0) & (H.t < C.T_IS)]
    isg = D[D.t < C.T_OOS]
    th = json.loads((C.RESULTS / "D7_account/thresholds.json").read_text())

    def prank(f):
        q = np.sort(isg[f].dropna().to_numpy())
        return np.searchsorted(q, H[f].to_numpy()) / len(q)

    H = H.assign(side=-1, net=H.net_base, s2=(prank("macd") + prank("cvd")) / 2)
    R = {"ctl": H, "macd": H[H.macd >= th["macd_threshold"]], "macd_cvd": H[H.s2 >= th["macd_cvd_threshold"]]}
    rows = []
    for rule, T in R.items():
        K = D7.account(T, 3)
        for v in ("base", "realistic", "xs"):
            k = K.assign(net=K[f"net_{v}"])
            mu, t, nd = C.day_t(k.t.to_numpy(), k.net.to_numpy())
            r = C.daily(k, 0.10, T0, C.T_IS)
            s = C.stats(r)
            rows.append({"rule": rule, "variant": v, "n": len(k), "coins": k.sym.nunique(), "bp": 1e4 * mu, "day_t": t,
                         "win": (k.net > 0).mean(), "period_ret_w10": (1 + r).prod() - 1, "maxdd_w10": s["maxdd"],
                         "basket_bp": 1e4 * k.basket.mean()})  # fmt: skip
        K.to_csv(OUT / f"ledger_{rule}.csv", index=False)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    g = S.set_index(["rule", "variant"]).bp
    verdict = {"1_div_s4_xs_gt_0": bool(g["macd_cvd", "xs"] > 0),
               "2_div_s4_gt_control_raw": bool(g["macd_cvd", "base"] > g["ctl", "base"]),
               "3_div_s4_raw_gt_0": bool(g["macd_cvd", "base"] > 0)}  # fmt: skip
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    pd.set_option("display.width", 220)
    print(S.round(3).to_string())
    print(verdict)


if __name__ == "__main__":
    main()
