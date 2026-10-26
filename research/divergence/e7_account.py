"""E7: the E6 day-weighted t-stats are often positive while the per-trade mean is negative: losses
cluster on days when many coins break out together. An account that holds at most N positions
earns something closer to the day average, so the main E6 variants are rerun through a position
cap (one per coin, signals skipped when full, ordered by fill time)."""

import numpy as np
import pandas as pd

import common as C


def account(T, N):
    T = T.sort_values("t_in")
    open_, keep = [], []
    for r in T.itertuples():
        open_ = [o for o in open_ if o[0] > r.t_in]
        if len(open_) >= N or any(o[1] == r.sym for o in open_):
            continue
        open_.append((r.t_out, r.sym))
        keep.append(r.Index)
    return T.loc[keep]


def main():
    D = pd.read_pickle("data/research/divergence_e6.pkl")
    D = D[D.net.notna() & (D.buf == 1.0)]
    rows = []
    for key, g in D.groupby(["tf", "side", "entry", "exit"]):
        for N in (1, 3, 5):
            K = account(g, N).rename(columns={"t_out": "tx"})
            row = dict(zip(["tf", "side", "entry", "exit"], key)) | {"N": N}
            for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END)):
                k = K[(K.t_in >= t0) & (K.t_in < t1)]
                r = C.daily(k, 1 / N, t0, t1)
                s = C.stats(r)
                row |= {f"{tag}_n": len(k), f"{tag}_bp": 1e4 * k.net.mean(), f"{tag}_win": (k.net > 0).mean(),
                        f"{tag}_cagr": s["cagr"], f"{tag}_sharpe": s["sharpe"]}  # fmt: skip
            rows.append(row)
    S = pd.DataFrame(rows)
    (C.RESULTS / "E7_account").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "E7_account/summary.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
