"""E4: the E3 survivors as portfolios, with exact mark-to-market and a parameter neighbourhood.

Trade-level t-stats in E3 treat 40 coins as independent, but breakouts cluster on the same days
(a market-wide move triggers many coins at once). Here every trade gets 1/40 of equity, so gross
exposure is at most 1x, and returns are marked on 4h closes: entry bar from the fill price,
exit bar to the fill price, P&L in coins held (not rebalanced), costs at both ends, funding
while held. The result is a 4h return
series whose Sharpe and drawdown can be compared with E1 / E2 and scaled by leverage in E6.

Neighbourhood: Donchian N in {10, 20, 30, 40}, chandelier M in {2, 3, 4}, volume filter
multiple in {1.5, 2, 3}; squeeze long-only and long/short. A real effect should be positive
across most of the grid, not only at one point.

Usage: PYTHONPATH=research/momentum uv run python research/momentum/e4_portfolio.py
"""

import numpy as np
import pandas as pd

import common as C
import e3_intraday as E3

SLOT = 1 / len(C.UNIVERSE)


def trades_to_returns(t: pd.DataFrame, B: dict, cost=C.COST) -> pd.Series:
    c = pd.DataFrame(B["c"]).ffill(limit=3).to_numpy()
    f = np.nan_to_num(B["fund"])
    r = np.zeros(len(c))
    for tr in t.itertuples():
        j, e, k, s = int(tr.j), int(tr.i_in), int(tr.i_out), tr.side
        # fixed number of coins from entry to exit: each bar earns its price change / entry price
        px = np.concatenate([[tr.px_in], np.nan_to_num(c[e:k, j], nan=tr.px_in), [tr.px_out]])
        px = pd.Series(px).replace(0, np.nan).ffill().to_numpy()
        r[e : k + 1] += SLOT * s * np.diff(px) / tr.px_in
        r[e : k + 1] -= SLOT * s * f[e : k + 1, j]
        r[e] -= SLOT * cost
        r[k] -= SLOT * cost
    return pd.Series(r, index=pd.to_datetime(B["t"], unit="ms", utc=True))


def dc_vol(B, N, M, mult):
    c, h, l = B["c"], B["h"], B["l"]
    hi = pd.DataFrame(h).rolling(N).max().shift(1).to_numpy()
    lo = pd.DataFrame(l).rolling(N).min().shift(1).to_numpy()
    hi2 = pd.DataFrame(h).rolling(max(N // 2, 2)).max().shift(1).to_numpy()
    lo2 = pd.DataFrame(l).rolling(max(N // 2, 2)).min().shift(1).to_numpy()
    sig = np.where(c > hi, 1, 0) + np.where(c < lo, -1, 0)
    if mult:
        qv = pd.DataFrame(B["qv"])
        sig = np.where((qv > mult * qv.rolling(20).mean()).to_numpy(), sig, 0)
    E3.EXIT_STATE["x"] = np.where(c < lo2, -1, 0) + np.where(c > hi2, 1, 0)
    rows = []
    for j, sym in enumerate(C.UNIVERSE):
        rows += E3.sim_4h(B, sig[:, j], lambda i, pos, best, a: best - pos * M * a, sym, j, M)
    return pd.DataFrame(rows)


def main():
    B = C.resample(C.panel(), 4)
    B["atr"] = E3.atr(B)
    out = C.RESULTS / "E4_portfolio"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for N in (10, 20, 30, 40):
        for M in (2, 3, 4):
            for mult in (0, 1.5, 2, 3):
                name = f"dc4h_N{N}_M{M}_v{mult}"
                t = dc_vol(B, N, M, mult)
                r = trades_to_returns(t, B)
                row = C.flat(name, C.split_metrics(r))
                row["trades"] = len(t)
                rows.append(row)
                curves[name] = r
                print(name, row["is_sharpe"], row["oos_sharpe"], row["all_cagr"], row["all_mdd"], flush=True)
    for both in (False, True):
        name = f"squeeze_{'LS' if both else 'L'}"
        t = E3.squeeze(B, both)
        r = trades_to_returns(t, B)
        row = C.flat(name, C.split_metrics(r))
        row["trades"] = len(t)
        rows.append(row)
        curves[name] = r
        print(name, row["is_sharpe"], row["oos_sharpe"], row["all_cagr"], row["all_mdd"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    daily = pd.DataFrame(curves).groupby(lambda x: x.floor("D")).apply(lambda g: (1 + g).prod() - 1)
    daily.to_csv(out / "daily_returns.csv.gz")
    cols = ["name", "trades", "is_cagr", "is_sharpe", "is_mdd", "oos_cagr", "oos_sharpe", "oos_mdd"]
    print(df[cols].to_string(index=False))
    grid = df[df.name.str.startswith("dc4h")]
    print("share of grid with Sharpe > 0: IS", (grid.is_sharpe > 0).mean(), "OOS", (grid.oos_sharpe > 0).mean())


if __name__ == "__main__":
    main()
