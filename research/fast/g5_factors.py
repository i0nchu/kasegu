"""G5: which coins EPR15 works on. Point-in-time coin factors against per-trade results.

Trades: G1, point-in-time top-40 universe (delisted coins included), base fills, 2022-01 to
2026-09, every symbol on its own (the PR #11 risk model has no account position cap).
Factors are computed from the daily panel with data up to the day before the trade:
  qv30      log 30-day mean quote volume            rank40   its rank inside the universe
  vol30     30-day std of daily returns             range30  30-day mean (high - low) / close
  wick30    30-day mean lower wick (min(o, c) - l) / c
  ret30     30-day return                           dd90     close / 90-day high - 1
  beta60    60-day beta to BTC                      corr60   60-day correlation to BTC
  fund30    30-day mean funding rate                age      days since the first daily bar
  prior90   the coin's own EPR15 net per trade over the previous 90 days (persistence)
Each factor: quintiles of trades by factor value, per-trade net and a day-clustered t, in the
design half (2022-01 to 2024-06) and the check half (2024-07 to 2026-09) separately. A rule is
chosen on the design half only and then reported on the check half.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/g5_factors.py
"""

import numpy as np
import pandas as pd

import e5_survivorship as E5
import fast as F

SPLIT = int(pd.Timestamp("2024-07-01", tz="UTC").value // 10**6)
FACTORS = ["qv30", "rank40", "vol30", "range30", "wick30", "ret30", "dd90", "beta60", "corr60",
           "fund30", "age", "prior90"]  # fmt: skip


def daily_factors() -> pd.DataFrame:
    D = E5.daily_panel()
    names = [str(s) for s in D["names"]]
    idx = pd.to_datetime(D["t"], unit="ms")

    def frame(k):
        return pd.DataFrame(D[k], index=idx, columns=names)

    o, h, l, c, qv, fu = (frame(k) for k in ("o", "h", "l", "c", "qv", "fund"))
    r = c.pct_change(fill_method=None)
    btc = r["BTCUSDT"]
    qv30 = qv.rolling(30, min_periods=30).mean()
    out = {
        "qv30": np.log10(qv30),
        "rank40": qv30.rank(axis=1, ascending=False),
        "vol30": r.rolling(30, min_periods=20).std(),
        "range30": ((h - l) / c).rolling(30, min_periods=20).mean(),
        "wick30": ((np.minimum(o, c) - l) / c).rolling(30, min_periods=20).mean(),
        "ret30": c / c.shift(30) - 1,
        "dd90": c / c.rolling(90, min_periods=30).max() - 1,
        "beta60": r.rolling(60, min_periods=40).cov(btc).div(btc.rolling(60, min_periods=40).var(), axis=0),
        "corr60": r.rolling(60, min_periods=40).corr(btc),
        "fund30": fu.rolling(30, min_periods=20).mean(),
        "age": c.notna().cumsum(),
    }  # fmt: skip
    # known at the close of day d, used for trades on day d+1
    long = pd.concat({k: v.shift(1).stack() for k, v in out.items()}, axis=1)
    long.index.names = ["day", "sym"]
    return long


def day_t(d: pd.DataFrame) -> tuple[float, float, int]:
    g = d.groupby(d.t // F.DAY).net.sum()
    return 100 * d.net.mean(), g.mean() / g.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan, len(d)


def main():
    out = F.RESULTS / "G5_factors"
    out.mkdir(parents=True, exist_ok=True)
    T = pd.read_pickle(F.OUT / "G1_trades.pkl")
    T = T[(T.uni == "pit") & (T.scen == "base") & (T.t >= F.T_IS + 365 * F.DAY)].copy()  # 2022-01 on
    T["day"] = pd.to_datetime(T.t // F.DAY * F.DAY, unit="ms")
    X = daily_factors().reset_index()
    T = T.merge(X, on=["day", "sym"], how="left")
    # own past EPR15 result: mean net of the coin's trades that closed in the previous 90 days
    T = T.sort_values("t")
    prior = []
    for _, g in T.groupby("sym"):
        tt, nn = g.t.to_numpy(), g.net.to_numpy()
        cs = np.concatenate([[0], np.cumsum(nn)])
        lo = np.searchsorted(tt, tt - 90 * F.DAY)
        hi = np.arange(len(tt))  # trades strictly before this one
        cnt = hi - lo
        prior.append(pd.Series(np.where(cnt >= 5, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan), index=g.index))
    T["prior90"] = pd.concat(prior)
    T.to_pickle(F.OUT / "G5_trades.pkl")

    rows = []
    for f in FACTORS:
        d = T.dropna(subset=[f])
        # quintile edges from the design half only, so the check half is not used to cut
        edges = np.unique(np.nanquantile(d[d.t < SPLIT][f], [0, 0.2, 0.4, 0.6, 0.8, 1]))
        edges[0], edges[-1] = -np.inf, np.inf
        d = d.assign(q=pd.cut(d[f], edges, labels=False))
        for q, g in d.groupby("q"):
            a = day_t(g[g.t < SPLIT])
            b = day_t(g[g.t >= SPLIT])
            rows.append({"factor": f, "q": int(q) + 1, "lo": edges[int(q)], "hi": edges[int(q) + 1],
                         "design_net_pct": a[0], "design_t": a[1], "design_n": a[2],
                         "check_net_pct": b[0], "check_t": b[1], "check_n": b[2]})  # fmt: skip
    Q = pd.DataFrame(rows)
    Q.to_csv(out / "quintiles.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(Q.to_string(index=False))


if __name__ == "__main__":
    main()
