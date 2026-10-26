"""E8 account level: max N open positions (one per coin), each trade risks RISK of equity at its stop
(position size = RISK / stop distance), so results are in R multiples like a position tool."""

import sys

import pandas as pd

import common as C
import e7_account as E7
import e8_wbottom as E8

RISK = 0.01


def prep(K):
    D = pd.read_pickle(f"data/research/divergence_e8_k{K}.pkl")
    D = D[(D.t_in >= 0) & (D.divfail == 1)].copy()
    D["risk"] = (D.px_in - D.stop).abs() / D.px_in + C.COST + 0.0002
    D["R"] = D.net / D.risk
    D["rule"] = E8.rule(D)
    return D


def run(D, N=3):
    rows = []
    for key, g in D.groupby(["tf", "side", "x", "stop_k", "exit"]):
        for st, gg in (("all", g), ("rule", g[g.rule])):
            A = E7.account(gg, N).rename(columns={"t_out": "tx"})
            A["net"] = A.R  # daily() books w * net; w = RISK
            row = dict(zip(["tf", "side", "x", "stop", "exit"], key)) | {"set": st}
            for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END)):
                k = A[(A.t_in >= t0) & (A.t_in < t1)]
                s = C.stats(C.daily(k, RISK, t0, t1))
                mu, tt, _ = C.day_t(k.t.to_numpy(), k.R.to_numpy())
                row |= {f"{tag}_n": len(k), f"{tag}_R": mu, f"{tag}_t": tt, f"{tag}_win": (k.R > 0).mean(),
                        f"{tag}_cagr": s["cagr"], f"{tag}_dd": s["maxdd"]}  # fmt: skip
            rows.append(row)
    return pd.DataFrame(rows)


def main(K):
    S = run(prep(K))
    S.to_csv(C.RESULTS / "E8_wbottom" / f"account_k{K}.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
