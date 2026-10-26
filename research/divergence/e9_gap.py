"""E9 distance to break even per setting (R and price bp via the median stop distance) and account level
(max 3 positions, one per coin, 1% equity risk per trade)."""

import pandas as pd

import common as C
import e7_account as E7

RISK = 0.01


def main():
    D = pd.read_pickle("data/research/divergence_e9.pkl")
    rows = []
    for key, g in D.groupby(["tf", "side", "x", "exit"]):
        row = dict(zip(["tf", "side", "x", "exit"], key))
        A = E7.account(g, 3).rename(columns={"t_out": "tx"})
        A["net"] = A.R
        for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END)):
            x = g[(g.t >= t0) & (g.t < t1)]
            win, loss = x.R[x.R > 0].mean(), x.R[x.R <= 0].mean()
            mu = x.R.mean()
            k = A[(A.t_in >= t0) & (A.t_in < t1)]
            s = C.stats(C.daily(k, RISK, t0, t1))
            row |= {f"{tag}_n": len(x), f"{tag}_win": (x.R > 0).mean(), f"{tag}_R": mu,
                    f"{tag}_gap_pp": 100 * max(-mu, 0) / (win - loss),
                    f"{tag}_gap_bp": 1e4 * max(-mu, 0) * x.risk.median(), f"{tag}_risk_bp": 1e4 * x.risk.median(),
                    f"{tag}_cagr": s["cagr"], f"{tag}_dd": s["maxdd"], f"{tag}_tpm": len(k) / ((t1 - t0) / C.DAY / 30.4)}  # fmt: skip
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(C.RESULTS / "E9_refined" / "gap_account.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
