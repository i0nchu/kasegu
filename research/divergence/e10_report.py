"""E10 report: per timeframe / side / stop / reward-risk / exit, with the user's round-3 rules (rule3) vs without:
win rate, R per trade, distance to break even, account level (max 3 positions, 1% risk). Also win rate vs
reward-risk against the break-even line, and timing regularities (year, month, weekday, hour)."""

import numpy as np
import pandas as pd

import common as C
import e7_account as E7

RISK = 0.01
OUT = C.RESULTS / "E10_rules"


def gap(x):
    win, loss = x.R[x.R > 0].mean(), x.R[x.R <= 0].mean()
    mu = x.R.mean()
    _, tt, _ = C.day_t(x.t.to_numpy(), x.R.to_numpy())
    return {"n": len(x), "win": (x.R > 0).mean(), "R": mu, "t": tt, "gap_pp": 100 * max(-mu, 0) / (win - loss),
            "gap_bp": 1e4 * max(-mu, 0) * x.risk.median(), "risk_bp": 1e4 * x.risk.median()}  # fmt: skip


def account(g, t0, t1):
    k = g[(g.t_in >= t0) & (g.t_in < t1)]
    s = C.stats(C.daily(k.assign(net=k.R), RISK, t0, t1))
    return {"cagr": s["cagr"], "dd": s["maxdd"], "tpm": len(k) / ((t1 - t0) / C.DAY / 30.4)}


def summary(D):
    rows = []
    for key, g in D.groupby(["tf", "side", "stop_k", "rr", "exit"]):
        for st, gg in (("all", g), ("rule3", g[g.rule3])):
            row = dict(zip(["tf", "side", "stop_k", "rr", "exit"], key)) | {"set": st}
            A = E7.account(gg, 3).rename(columns={"t_out": "tx"}) if st == "rule3" else None
            for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END)):
                row |= {f"{tag}_{k}": v for k, v in gap(gg[(gg.t >= t0) & (gg.t < t1)]).items()}
                if A is not None:
                    row |= {f"{tag}_{k}": v for k, v in account(A, t0, t1).items()}
            rows.append(row)
    return pd.DataFrame(rows)


def rr_curve(D):
    """Realised win rate per theoretical reward/risk vs the break-even win rate with costs."""
    rows = []
    for key, g in D[D.rule3 & (D.exit == "body")].groupby(["tf", "rr"]):
        cost_r = ((C.COST + 0.0002) / g.risk).median()  # costs in R
        be = (1 + cost_r) / (1 + key[1])  # p * rr - (1 - p) - cost = 0
        rows.append({"tf": key[0], "rr": key[1], "n": len(g), "win": (g.R > 0).mean(), "breakeven_win": be,
                     "edge_pp": 100 * ((g.R > 0).mean() - be), "R": g.R.mean()})  # fmt: skip
    return pd.DataFrame(rows)


def timing(T):
    """T: one configuration's trades. Counts and mean R by year, month, weekday and UTC hour (entry time)."""
    ti = pd.to_datetime(T.t_in, unit="ms")
    out = {}
    for name, key in (("year", ti.dt.year), ("month", ti.dt.month), ("weekday", ti.dt.dayofweek),
                      ("hour", ti.dt.hour)):  # fmt: skip
        g = T.R.groupby(key.to_numpy())
        out[name] = pd.DataFrame({"n": g.size(), "share": g.size() / len(T), "R": g.mean(),
                                  "t": g.mean() / g.std() * np.sqrt(g.size())})  # fmt: skip
    return out


def main():
    D = pd.read_pickle("data/research/divergence_e10_all.pkl")
    OUT.mkdir(parents=True, exist_ok=True)
    S = summary(D)
    S.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    rc = rr_curve(D)
    rc.to_csv(OUT / "rr_curve.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 600)
    cols = ["tf", "side", "stop_k", "rr", "exit", "is_n", "is_win", "is_R", "is_gap_pp", "is_gap_bp", "is_cagr",
            "oos_n", "oos_win", "oos_R", "oos_gap_pp", "oos_gap_bp", "oos_cagr", "oos_tpm"]  # fmt: skip
    r3 = S[S.set == "rule3"]
    print(r3[cols].round(3).to_string())
    print(rc.round(3).to_string())
    # rule3 vs all, pooled over reward/risk and stops, full TP1 exit
    cmp_ = S[S.exit == "body"].groupby(["tf", "side", "set"])[["is_win", "is_R", "oos_win", "oos_R"]].mean()
    cmp_.to_csv(OUT / "rule3_vs_all.csv", float_format="%.4f")
    print(cmp_.round(3).to_string())


if __name__ == "__main__":
    main()
