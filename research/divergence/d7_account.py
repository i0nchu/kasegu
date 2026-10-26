"""D7: account-level backtest and Monte Carlo of the 4h bearish-divergence short (DIV-S4).

Events and per-trade outcomes come from D6 (every 4h higher-high pivot pair, entry at the next 5m
open, stop 1.5 ATR, take profit 3 ATR, exit after 4 days, fees + slippage + funding).
Rules compared:
  ctl      short every higher-high pivot (baseline: is it just "short alts after a high"?)
  macd     MACD line at the second high is lower than at the first by at least the IS 80th pct
  macd_cvd mean percentile of MACD and CVD divergence >= IS 80th pct (CVD = taker delta between
           the highs over the volume traded between them)
Account: each position uses a fraction W of equity at entry (no leverage at W * N <= 1), at most N
open positions, one per coin; signals arriving when full are skipped. Daily returns are booked on
the exit day. Monte Carlo: 10,000 one-year paths of 20-day blocks of daily returns.
"""

import numpy as np
import pandas as pd

import common as C

RNG = np.random.default_rng(7)


def rules(D):
    g = D[(D.side == -1) & (D.tx > 0)].copy()
    isg = g[g.t < C.T_OOS]

    def prank(f):
        q = np.sort(isg[f].dropna().to_numpy())
        return np.searchsorted(q, g[f].to_numpy()) / len(q)

    g["p_macd"] = prank("macd")
    g["p_cvd"] = prank("cvd")
    g["s2"] = (g.p_macd + g.p_cvd) / 2
    th2 = g[g.t < C.T_OOS].s2.quantile(0.8)
    th_macd = isg.macd.quantile(0.8)
    out = {"ctl": g, "macd": g[g.macd >= th_macd], "macd_cvd": g[g.s2 >= th2]}
    meta = {"macd_threshold": float(th_macd), "macd_cvd_threshold": float(th2)}
    return out, meta


def account(T, N, extra=0.0):
    """Pick trades in signal order under the position cap. extra: added cost per round trip."""
    T = T.sort_values("t")
    open_ = []  # (tx, sym)
    keep = []
    for r in T.itertuples():
        entry = r.t + 4 * 3_600_000  # entry right after the 4h confirmation bar closes
        open_ = [o for o in open_ if o[0] > entry]
        if len(open_) >= N or any(o[1] == r.sym for o in open_):
            continue
        open_.append((r.tx, r.sym))
        keep.append(r.Index)
    K = T.loc[keep].copy()
    K["net"] = K.net - extra
    return K


def perf(K, W, t0, t1):
    r = C.daily(K[(K.t >= t0) & (K.t < t1)], W, t0, t1)
    s = C.stats(r)
    k = K[(K.t >= t0) & (K.t < t1)]
    s |= {"trades": len(k), "per_month": len(k) / (len(r) / 30.4), "bp": 1e4 * k.net.mean(),
          "win": (k.net > 0).mean()}  # fmt: skip
    return s, r


def btc_daily():
    X = C.load("BTCUSDT")
    s = pd.Series(X["c"], index=X["t"] // C.DAY * C.DAY).groupby(level=0).last()
    return s.pct_change()


def mc(r, years=1.0, block=20, n=10_000):
    r = r.to_numpy()
    L = int(365 * years)
    nb = L // block + 1
    st = RNG.integers(0, len(r) - block, (n, nb))
    paths = np.stack([r[s:s + block] for s in st.ravel()]).reshape(n, -1)[:, :L]  # fmt: skip
    eq = np.cumprod(1 + paths, 1)
    dd = (eq / np.maximum.accumulate(eq, 1) - 1).min(1)
    return eq[:, -1], dd


def main():
    D = pd.read_pickle("data/research/divergence_d6.pkl")
    R, meta = rules(D)
    out = C.RESULTS / "D7_account"
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(meta).to_json(out / "thresholds.json")
    bd = btc_daily()
    rows, mrows = [], []
    for name, T in R.items():
        for N in (1, 3, 5):
            W = 1 / N if N > 1 else 0.5
            for slip in (0.0, 0.001, 0.002):
                K = account(T, N, slip)
                for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END), ("all", C.T_IS, C.T_END)):
                    s, r = perf(K, W, t0, t1)
                    x = pd.concat([r, bd.reindex(r.index)], axis=1).dropna()
                    b = np.polyfit(x.iloc[:, 1], x.iloc[:, 0], 1)
                    res = x.iloc[:, 0] - b[0] * x.iloc[:, 1]
                    s["beta_btc"] = b[0]
                    s["alpha_t"] = res.mean() / res.std() * np.sqrt(len(res))
                    rows.append({"rule": name, "N": N, "W": W, "extra_cost": slip, "period": tag} | s)
                if slip == 0.0:
                    K["yr"] = pd.to_datetime(K.t, unit="ms").dt.year
                    yr = K.groupby("yr").apply(lambda k: (1 + W * k.net).prod() - 1, include_groups=False)
                    rows[-1] |= {f"y{y}": v for y, v in yr.items()}
                    K.to_csv(out / f"ledger_{name}_N{N}.csv.gz", index=False)
                    for tag, t0 in (("oos", C.T_OOS), ("all", C.T_IS)):
                        _, r = perf(K, W, t0, C.T_END)
                        for lev in (1, 2, 3):
                            fin, dd = mc(r * lev)
                            mrows.append({"rule": name, "N": N, "sample": tag, "lev": lev,
                                          "median": np.median(fin), "p05": np.quantile(fin, 0.05),
                                          "p95": np.quantile(fin, 0.95), "p_loss": (fin < 1).mean(),
                                          "p_dd30": (dd < -0.3).mean(), "p_dd50": (dd < -0.5).mean(),
                                          "median_dd": np.median(dd)})  # fmt: skip
    S = pd.DataFrame(rows)
    M = pd.DataFrame(mrows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    M.to_csv(out / "montecarlo.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(meta)
    print(S[S.extra_cost == 0].round(3).to_string())
    print(S[(S.period == "oos")][["rule", "N", "extra_cost", "cagr", "sharpe", "maxdd", "bp"]].round(3).to_string())
    print(M.round(3).to_string())


if __name__ == "__main__":
    main()
