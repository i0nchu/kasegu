"""B2: order flow momentum. Does persistent one-sided taker pressure keep pushing the price?

Signal at the close of bar i: flow imbalance over the last W minutes
  imbW = (taker buy volume - taker sell volume) / volume,  W in 15, 60, 240
Time series test, per coin, decision every 15 minutes:
  deciles of imbW cut on year 1, forward return over H in 15, 60, 240 minutes from the close
  trade: long the top decile, short the bottom decile, taker in and out (0.12% round trip)
Controls:
  price    the same deciles built from the price return over W (plain price momentum)
  resid    flow left after removing what the price move explains (per coin regression of imbW
           on retW, fitted on year 1): is there information in flow beyond price?
Cross-sectional test: every H minutes long the 8 coins with the highest imbW and short the 8
lowest (the B3 portfolio machinery), dollar neutral.

Usage: uv run python research/alpha/b2_flow_momentum.py
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from b3_relative_value import COSTS, book
from common import fmt, folder, market, publish, save_json
from features import rsum
from stats import boot_mean

from data import GRID, MIN, UNIVERSE, Y2, matrix

EXP = "B2_flow_momentum"
WS = (15, 60, 240)
HS = (15, 60, 240)
STEP = 15
COST = 0.12
WARM = 2 * 1440


def _one(sym: str) -> pd.DataFrame:
    from data import panel

    d = panel(sym)
    c, v, tb, live = d["c"], d["v"], d["tb"], d["live"]
    n = len(c)
    idx = np.arange(WARM, n - max(HS) - 1, STEP)
    lc = np.cumsum(live)
    act = (lc[idx] - lc[idx - 60]) >= 55
    idx = idx[act]
    out = {"symbol": sym, "t": GRID[idx] + MIN}
    for W in WS:
        vs = rsum(v, W)
        imb = np.where(vs > 0, rsum(2 * tb - v, W) / np.where(vs > 0, vs, 1), 0.0)
        out[f"imb{W}"] = imb[idx]
        out[f"ret{W}"] = (c[idx] / c[idx - W] - 1) * 100
    for H in HS:
        out[f"f{H}"] = (c[idx + H] / c[idx] - 1) * 100
    return pd.DataFrame(out)


def ts_test(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y1 = ~d.y2
    for W in WS:
        # flow orthogonal to the price move, fitted per coin on year 1 only
        res = np.full(len(d), np.nan)
        for _s, g in d.groupby("symbol"):
            a = g[y1.loc[g.index]]
            X = np.c_[np.ones(len(a)), a[f"ret{W}"]]
            ok = np.isfinite(X).all(1) & np.isfinite(a[f"imb{W}"])
            b, *_ = np.linalg.lstsq(X[ok], a[f"imb{W}"].to_numpy()[ok], rcond=None)
            res[d.index.get_indexer(g.index)] = g[f"imb{W}"] - b[0] - b[1] * g[f"ret{W}"]
        d[f"res{W}"] = res
        for sig in (f"imb{W}", f"ret{W}", f"res{W}"):
            # deciles per coin from year 1
            q = np.full(len(d), -1)
            for _s, g in d.groupby("symbol"):
                cuts = np.nanpercentile(g[y1.loc[g.index]][sig], np.arange(10, 100, 10))
                q[d.index.get_indexer(g.index)] = np.digitize(g[sig], cuts)
            d["q"] = q
            for H in HS:
                sub = d.iloc[:: max(1, H // STEP)]  # non overlapping holds
                top, bot = sub[sub.q == 9], sub[sub.q == 0]
                trade = pd.concat([top.assign(pnl=top[f"f{H}"]), bot.assign(pnl=-bot[f"f{H}"])])
                m, lo, hi = boot_mean(trade.pnl - COST, trade.day.to_numpy(), b=500)
                rows.append(
                    {
                        "W": W,
                        "signal": sig[:3],
                        "H": H,
                        "top_fwd": top[f"f{H}"].mean(),
                        "bottom_fwd": bot[f"f{H}"].mean(),
                        "spread": top[f"f{H}"].mean() - bot[f"f{H}"].mean(),
                        "trades": len(trade),
                        "gross": trade.pnl.mean(),
                        "net": m,
                        "net_lo": lo,
                        "net_hi": hi,
                        "net_y1": (trade[~trade.y2].pnl - COST).mean(),
                        "net_y2": (trade[trade.y2].pnl - COST).mean(),
                        "ic": np.corrcoef(sub[sig].rank().fillna(0), sub[f"f{H}"].rank().fillna(0))[
                            0, 1
                        ],
                    }
                )
                print(fmt(pd.DataFrame(rows[-1:])), flush=True)
    return pd.DataFrame(rows)


def xs_test() -> pd.DataFrame:
    P = np.log(matrix("c", UNIVERSE))
    v = matrix("v", UNIVERSE)
    tb = matrix("tb", UNIVERSE)
    live = matrix("live", UNIVERSE, bool)
    lc = np.cumsum(live, axis=0)
    rows = []
    for W in WS:
        cv = np.cumsum(v, axis=0)
        cs = np.cumsum(2 * tb - v, axis=0)
        for H in HS:
            times = np.arange(WARM, len(GRID) - H, H)
            w_prev = np.zeros(len(UNIVERSE))
            rec = []
            for i in times:
                vs = cv[i] - cv[i - W]
                imb = np.where(vs > 0, (cs[i] - cs[i - W]) / np.where(vs > 0, vs, 1), np.nan)
                ok = ((lc[i] - lc[i - 60]) >= 55) & ((lc[i + H] - lc[i]) >= H - 5)
                w = -book(imb, ok)  # book() longs the lowest: flip to long the highest flow
                fr = np.nan_to_num(P[i + H] - P[i])
                rec.append((GRID[i] + MIN, float(w @ fr) * 100, float(np.abs(w - w_prev).sum())))
                w_prev = w
            r = pd.DataFrame(rec, columns=["t", "gross", "turnover"])
            r["day"] = r.t // 86_400_000
            row = {"W": W, "H": H, "rebalances": len(r), "turnover": r.turnover.mean()}
            for k, c in COSTS.items():
                m, lo, hi = boot_mean(
                    (r.gross - r.turnover * c).to_numpy(), r.day.to_numpy(), b=500
                )
                row.update(
                    {f"net_{k}_bp": m * 100, f"net_{k}_lo_bp": lo * 100, f"net_{k}_hi_bp": hi * 100}
                )
            row["gross_bp"] = r.gross.mean() * 100
            row["gross_y1_bp"] = r[r.t < GRID[Y2]].gross.mean() * 100
            row["gross_y2_bp"] = r[r.t >= GRID[Y2]].gross.mean() * 100
            rows.append(row)
            print(fmt(pd.DataFrame(rows[-1:])), flush=True)
    return pd.DataFrame(rows)


def main():
    out = folder(EXP)
    save_json(
        out / "config.json",
        {"exp": EXP, "windows": WS, "holds": HS, "step_min": STEP, "cost_ts_round_trip": COST,
         "xs_costs_per_side": COSTS, "symbols": UNIVERSE},
    )  # fmt: skip
    market()
    with ProcessPoolExecutor(5) as ex:
        d = pd.concat(list(ex.map(_one, UNIVERSE)), ignore_index=True)
    d["day"] = d.t // 86_400_000
    d["y2"] = d.t >= GRID[Y2]
    ts = ts_test(d)
    ts.to_csv(out / "timeseries.csv", index=False)
    xs = xs_test()
    xs.to_csv(out / "cross_section.csv", index=False)
    print(fmt(ts), fmt(xs), sep="\n\n")
    publish(EXP, ["config.json", "timeseries.csv", "cross_section.csv"])


if __name__ == "__main__":
    main()
