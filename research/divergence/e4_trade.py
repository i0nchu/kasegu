"""E4: the manual trade itself. Short (or long, mirror) at the trigger, stop above H1 (+0 / 0.5 /
1 ATR), take profit at L1, max 60 bars. Unfiltered, and filtered by the three features the user
reads by eye, with directions fixed from E2 (volume of the second leg below the first, volatility
contracting since H1, funding in the last 24h low), cut at IS medians.
"""

import numpy as np
import pandas as pd

import common as C


def main():
    D = pd.read_pickle("data/research/divergence_e1.pkl").replace([np.inf, -np.inf], np.nan)
    rows = []
    for (tf, side), g in D.groupby(["tf", "side"]):
        isg = g[g.t < C.T_OOS]
        med = isg[["vol_leg", "atr_chg", "fund24"]].median()
        f_vol = g.vol_leg < med.vol_leg
        f_atr = g.atr_chg < med.atr_chg
        f_fund = g.fund24 < med.fund24
        filters = {"none": pd.Series(True, index=g.index), "vol": f_vol, "atr": f_atr, "fund": f_fund,
                   "vol+atr": f_vol & f_atr, "all3": f_vol & f_atr & f_fund}  # fmt: skip
        for fn, fm in filters.items():
            for buf in ("0.0", "0.5", "1.0"):
                row = {"tf": tf, "side": side, "filter": fn, "stop_buf_atr": buf}
                for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
                    m = (fm & pm).to_numpy()
                    x = g[f"net_b{buf}"].to_numpy()[m]
                    mu, t, _ = C.day_t(g.t.to_numpy()[m], x)
                    row |= {f"{tag}_n": int(m.sum()), f"{tag}_bp": 1e4 * mu, f"{tag}_t": t,
                            f"{tag}_win": np.nanmean(x > 0)}  # fmt: skip
                rows.append(row)
    S = pd.DataFrame(rows)
    (C.RESULTS / "E4_trade").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "E4_trade/summary.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
