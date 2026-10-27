"""D9: technical audit of DIV-S4 (4h bearish MACD + CVD divergence short).

1. Rebuild every 4h higher-high pivot pair from 2020-05 (the 2020 rows are the holdout and are
   written to a separate file that this script never summarizes; see d10_holdout.py).
2. Reproduce the D7 ledger (rule, IS percentile threshold, account cap) and compare with the
   committed one.
3. Look-ahead checks: pivot confirmation lag, features recomputed on data truncated at the signal
   bar, universe flag timing, trades ending on missing data (delisting).
4. Execution variants: take profit needs a trade-through, entry delayed 5m / 1h, extra stop
   slippage, all combined.
5. Market-neutral excess: each trade against an equal-weight short of the eligible universe over
   the same window, to separate "this coin falls" from "alts fall".
Usage: PYTHONPATH=research/divergence python research/divergence/d9_audit.py
"""

import json
import multiprocessing as mp

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3
import d7_account as D7

N5 = 48
HOLD = 24
T_HOLD = int(pd.Timestamp("2020-05-01", tz="UTC").value // 10**6)
OUT = C.RESULTS / "D9_audit"
VARIANTS = {  # delay (5m bars), tp trade-through, extra stop slippage
    "base": (0, 0.0, 0.0),
    "tp_through": (0, 0.0005, 0.0),
    "delay_5m": (1, 0.0, 0.0),
    "delay_1h": (12, 0.0, 0.0),
    "stop_slip": (0, 0.0, 0.001),
    "realistic": (1, 0.0005, 0.0005),
}


@nb.njit(cache=True)
def outcome(o, h, l, c, fund, j0, stop_k, tp_k, hold, cost, thr, sx):  # noqa: E741
    """Short trades, each simulated alone (same rules as D6 outcome). stop_k / tp_k: distances as
    fractions of the entry. thr: the low must trade through the take profit by thr to fill.
    sx: extra slippage on stop fills. Returns net, exit bar, reason (0 time, 1 stop, 2 tp, 3 data end)."""
    n = len(j0)
    res = np.full(n, np.nan)
    xi = np.full(n, -1, np.int64)
    why = np.full(n, -1, np.int64)
    px0 = np.full(n, np.nan)
    for e in range(n):
        j = j0[e]
        if j >= len(o) or o[j] != o[j]:
            continue
        px = o[j]
        sp = px * (1 + stop_k[e])
        tq = px * (1 - tp_k[e])
        f = 0.0
        ex = np.nan
        w = 0
        m = j
        while m < len(o):
            if o[m] != o[m]:
                ex = c[m - 1]
                m -= 1
                w = 3
                break
            f += fund[m]
            op = o[m] if m > j else px
            if h[m] >= sp:
                ex = max(sp, op) * (1 + sx)
                w = 1
                break
            if l[m] <= tq * (1 - thr):
                ex = min(tq, op)
                w = 2
                break
            if m - j + 1 >= hold:
                ex = c[m]
                w = 0
                break
            if m == len(o) - 1:
                ex = c[m]
                w = 3
                break
            m += 1
        res[e] = -(ex / px - 1) - 2 * cost + f
        xi[e] = m
        why[e] = w
        px0[e] = px
    return res, xi, why, px0


def features(B, I, p, q):
    vol = pd.Series(B["qv"]).cumsum().to_numpy()
    return {"macd": -(I["macd"][p] - I["macd"][q]), "cvd": -(I["cvd"][p] - I["cvd"][q]) / (vol[p] - vol[q])}


def run_sym(sym):
    X = C.load(sym)
    B = C.resample(X, N5)
    if len(B["t"]) < 300:
        return None
    I = C.indicators(B)
    e = C.events(B, I)[-1]
    a = D3.atr(B)
    i, p, q, dp = e["i"], e["p"], e["q"], e["dp"]
    ok = (dp > 0) & B["elig"][i] & (B["t"][i] >= T_HOLD) & (i + 1 < len(B["t"]))
    i, p, q = i[ok], p[ok], q[ok]
    if not len(i):
        return None
    t_end = B["t"][i] + (N5 - 1) * C.M5
    j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
    sk = 1.5 * a[i] / B["c"][i]
    d = {"sym": sym, "t": B["t"][i], "i": i, "p": p, "q": q, "lag": i - p, "atr": sk / 1.5,
         "hi_p": B["h"][p], "hi_q": B["h"][q], "t_p": B["t"][p], "t_q": B["t"][q]}  # fmt: skip
    d |= features(B, I, p, q)
    for name, (dl, thr, sx) in VARIANTS.items():
        r, xi, why, px = outcome(X["o"], X["h"], X["l"], X["c"], X["fund"], j + dl, sk, 2 * sk, HOLD * N5 - dl,
                                 C.COST, thr, sx)  # fmt: skip
        d[f"net_{name}"] = r
        if name == "base":
            d["tx"] = np.where(xi >= 0, X["t"][np.clip(xi, 0, len(X["t"]) - 1)], -1)
            d["why"] = why
            d["px"] = px
            d["t_in"] = X["t"][np.clip(j, 0, len(X["t"]) - 1)]
    D = pd.DataFrame(d)
    # truncation test: recompute features using only data up to the end of the signal bar
    rng = np.random.default_rng(abs(hash(sym)) % 2**32)
    pick = rng.choice(len(i), min(5, len(i)), replace=False)
    bad = 0
    for k in pick:
        cut = int(np.searchsorted(X["t"], t_end[k]) + 1)
        Xt = {kk: v[:cut] for kk, v in X.items()}
        Bt = C.resample(Xt, N5)
        It = C.indicators(Bt)
        ft = features(Bt, It, np.array([p[k]]), np.array([q[k]]))
        at = D3.atr(Bt)[i[k]] / Bt["c"][i[k]]
        et = C.events(Bt, It)[-1]
        seen = i[k] in set(et["i"].tolist())
        same = (np.isclose(ft["macd"][0], D.macd.iloc[k], rtol=1e-9, atol=1e-15)
                and np.isclose(ft["cvd"][0], D.cvd.iloc[k], rtol=1e-9) and np.isclose(at, D.atr.iloc[k], rtol=1e-9)
                and seen and len(Bt["t"]) == i[k] + 1)  # fmt: skip
        bad += not same
    D.attrs["trunc"] = (len(pick), bad)
    return D


def basket():
    """Log index of an equal-weight basket of the eligible universe, rebalanced every 4h
    (bar start time -> cumulative log return through that bar's close)."""
    rets = []
    for s in C.coins():
        B = C.resample(C.load(s), N5)
        r = np.diff(B["c"], prepend=np.nan) / np.r_[np.nan, B["c"][:-1]]
        rets.append(pd.Series(np.where(B["elig"], r, np.nan), index=B["t"]))
    R = pd.concat(rets, axis=1).sort_index().clip(-0.5, 1.0)
    return np.log1p(R.mean(axis=1).fillna(0.0)).cumsum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    parts = [x for x in parts if x is not None]
    trunc = np.sum([x.attrs["trunc"] for x in parts], 0)
    D = pd.concat(parts, ignore_index=True)
    D = D[np.isfinite(D.net_base) & (D.tx > 0)]
    idx = basket()
    t4 = idx.index.to_numpy()
    a = np.searchsorted(t4, D.t.to_numpy())  # index value at the close of the signal bar ~ entry
    b = np.clip(np.searchsorted(t4, D.tx.to_numpy(), side="right") - 1, 0, len(t4) - 1)
    D["basket"] = np.exp(idx.to_numpy()[b] - idx.to_numpy()[np.clip(a, 0, len(t4) - 1)]) - 1
    D["net_xs"] = D.net_base + D.basket  # short coin, long the equal-weight basket
    D[D.t < C.T_IS].to_pickle("data/research/divergence_d9_holdout.pkl")
    D = D[D.t >= C.T_IS].reset_index(drop=True)
    D.to_pickle("data/research/divergence_d9.pkl")

    # rule and thresholds exactly as D7
    g = D.assign(side=-1, net=D.net_base)
    R, meta = D7.rules(g)
    old = json.loads((C.RESULTS / "D7_account/thresholds.json").read_text())
    chk = {"threshold_now": meta, "threshold_D7": old, "trunc_tested": int(trunc[0]), "trunc_mismatch": int(trunc[1]),
           "lag_not_3": int((D.lag != 3).sum()), "entry_before_signal_close": int((D.t_in < D.t + 4 * 3_600_000).sum()),
           "events": len(D)}  # fmt: skip
    K = D7.account(R["macd_cvd"], 3)
    K0 = pd.read_csv(C.RESULTS / "D7_account/ledger_macd_cvd_N3.csv.gz")
    m = K.merge(K0[["sym", "t", "net"]], on=["sym", "t"], how="outer", suffixes=("", "_d7"), indicator=True)
    chk |= {"ledger_now": len(K), "ledger_d7": len(K0), "ledger_both": int((m._merge == "both").sum()),
            "max_net_diff": float((m.net - m.net_d7).abs().max()), "bp_now": 1e4 * K.net.mean(),
            "bp_d7": 1e4 * K0.net.mean(), "data_end_exits": int((K.why == 3).sum()),
            "data_end_bp_sum": 1e4 * K[K.why == 3].net.sum() / len(K)}  # fmt: skip
    (OUT / "checks.json").write_text(json.dumps(chk, indent=1, default=float))
    print(json.dumps(chk, indent=1, default=float))

    rows = []
    for rule in ("ctl", "macd", "macd_cvd"):
        T = R[rule]
        for v in list(VARIANTS) + ["xs"]:
            T2 = T.assign(net=T[f"net_{v}"])
            K = D7.account(T2, 3)
            for tag, t0, t1 in (("is", C.T_IS, C.T_OOS), ("oos", C.T_OOS, C.T_END)):
                s, _ = D7.perf(K, 0.10, t0, t1)
                k = K[(K.t >= t0) & (K.t < t1)]
                mu, t, _ = C.day_t(k.t.to_numpy(), k.net.to_numpy())
                rows.append({"rule": rule, "variant": v, "period": tag, "n": len(k), "bp": 1e4 * k.net.mean(),
                             "day_t": t, "win": (k.net > 0).mean(), "cagr_w10": s["cagr"], "maxdd_w10": s["maxdd"]})  # fmt: skip
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "variants.csv", index=False, float_format="%.4f")
    K = D7.account(R["macd_cvd"], 3)
    why = K.groupby([np.where(K.t < C.T_OOS, "is", "oos"), "why"]).net.agg(["size", "mean"])
    why.to_csv(OUT / "exit_reasons.csv", float_format="%.5f")
    yrs = []
    for rule in ("ctl", "macd_cvd"):
        K = D7.account(R[rule], 3)
        K["yr"] = pd.to_datetime(K.t, unit="ms").dt.year
        y = K.groupby("yr").agg(n=("net", "size"), raw_bp=("net", "mean"), xs_bp=("net_xs", "mean"),
                                basket_bp=("basket", "mean"))  # fmt: skip
        y[["raw_bp", "xs_bp", "basket_bp"]] *= 1e4
        day = K.groupby(K.tx // C.DAY).net.sum().sort_values(ascending=False)
        k1 = K[~(K.tx // C.DAY).isin(day.index[:10])]
        y.loc["no_top10_days"] = [len(k1), 1e4 * k1.net.mean(), 1e4 * k1.net_xs.mean(), 1e4 * k1.basket.mean()]
        yrs.append(y.assign(rule=rule))
    Y = pd.concat(yrs)
    Y.to_csv(OUT / "yearly.csv", float_format="%.2f")
    print(Y.round(1))
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    print(why)


if __name__ == "__main__":
    main()
