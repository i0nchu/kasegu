"""D8: final checks on DIV-S4 (4h bearish MACD + CVD divergence short, max 3 positions).

- Position size: Monte Carlo one-year paths at W = 5%, 10%, 15%, 20%, 33% of equity per trade.
- Concentration: result without the 10 best days and without the 5 best coins.
- Trade-order Monte Carlo: shuffle OOS trade order to get the drawdown distribution.
- Overlap with EPR15 (research/fast G2 point-in-time ledger): daily PnL correlation.
"""

import sys

import numpy as np
import pandas as pd

import common as C
import d7_account as D7

OUT = C.RESULTS / "D8_final"
RNG = np.random.default_rng(11)


def main(epr_path=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("macd", "macd_cvd"):
        K = pd.read_csv(C.RESULTS / f"D7_account/ledger_{name}_N3.csv.gz")
        for W in (0.05, 0.10, 0.15, 0.20, 1 / 3):
            for tag, t0 in (("oos", C.T_OOS), ("all", C.T_IS)):
                r = C.daily(K[K.t >= t0], W, t0, C.T_END)
                s = C.stats(r)
                fin, dd = D7.mc(r)
                rows.append({"rule": name, "W": round(W, 3), "sample": tag, "cagr": s["cagr"],
                             "sharpe": s["sharpe"], "hist_maxdd": s["maxdd"],
                             "mc_median": np.median(fin), "mc_p05": np.quantile(fin, 0.05),
                             "mc_p95": np.quantile(fin, 0.95), "p_loss": (fin < 1).mean(),
                             "p_dd20": (dd < -0.2).mean(), "p_dd30": (dd < -0.3).mean(),
                             "mc_median_dd": np.median(dd)})  # fmt: skip
        # concentration
        day = K.groupby(K.tx // C.DAY).net.sum().sort_values(ascending=False)
        coin = K.groupby("sym").net.sum().sort_values(ascending=False)
        k1 = K[~(K.tx // C.DAY).isin(day.index[:10])]
        k2 = K[~K.sym.isin(coin.index[:5])]
        conc = {"rule": name, "bp_all": 1e4 * K.net.mean(), "bp_no_top10_days": 1e4 * k1.net.mean(),
                "bp_no_top5_coins": 1e4 * k2.net.mean(), "top10_days_share": day.head(10).sum() / K.net.sum(),
                "coins_traded": len(coin), "coins_positive": (coin > 0).mean()}  # fmt: skip
        # trade-order shuffle on OOS at W = 10%
        x = K[K.t >= C.T_OOS].net.to_numpy() * 0.10
        dds = []
        for _ in range(5000):
            eq = np.cumprod(1 + RNG.permutation(x))
            dds.append((eq / np.maximum.accumulate(eq) - 1).min())
        conc |= {"shuffle_dd_median_W10": np.median(dds), "shuffle_dd_p95_W10": np.quantile(dds, 0.05)}
        pd.Series(conc).to_json(OUT / f"concentration_{name}.json")
        print(conc)
        if epr_path:
            E = pd.read_csv(epr_path)
            e = E.groupby(E.t_out // C.DAY * C.DAY).net.sum()
            d = C.daily(K, 0.1)
            x = pd.concat([d, e], axis=1).fillna(0.0)
            x = x[(x.index >= C.T_IS) & (x.index < C.T_END)]
            print(name, "daily corr with EPR15", round(x.corr().iloc[0, 1], 3))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sizing_montecarlo.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
