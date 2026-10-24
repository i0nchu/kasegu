"""F1: session strategies on 5m bars, point-in-time universe, delisted coins included.

  orb  Opening range breakout. Session opens at S (00:00 UTC Asia, 08:00 London, 13:30 New York)
       and lasts 24h. The range is the high / low of the first R minutes. The first 5m close above
       the range high goes long at the next open (below the low: short), one trade per session.
       Stop = k_stop x range from entry, optional take profit = k_tp x range, flat at session end.
  im   Intraday momentum. At hour H of the UTC day, if the return since 00:00 exceeds thr in
       absolute value, trade in its direction until 24:00 (Gao et al. 2018 style, which crypto
       papers report on BTC). Optional stop at 1 x the day's range so far.

Every trade pays 2 x (0.05% taker + 0.05% slippage) plus funding. 1x notional per trade.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/f1_session.py
"""

import concurrent.futures as cf
import itertools

import numpy as np
import pandas as pd

import fast as F

SESS = {"asia": 0, "london": 8 * 60, "ny": 13 * 60 + 30}
ORB = [
    dict(fam="orb", sess=s, R=R, ks=ks, kt=kt, dirs=d)
    for s, R, ks, kt, d in itertools.product(SESS, (15, 30, 60, 120), (0.5, 1.0), (0, 2.0), ("long", "both"))
]
IM = [
    dict(fam="im", H=H, thr=thr, stop=st, dirs=d)
    for H, thr, st, d in itertools.product((6, 12, 18, 22), (0.0, 0.02, 0.05), (0, 1), ("long", "both"))
]
CONFIGS = ORB + IM


def sess_index(t, start_min):
    rel = t - start_min * 60_000
    return rel // F.DAY, (rel % F.DAY) // F.M5  # session id, bar number inside the session


def orb(X, cfg):
    sid, pos = sess_index(X["t"], SESS[cfg["sess"]])
    nr = cfg["R"] // 5
    df = pd.DataFrame({"sid": sid, "h": np.where(pos < nr, X["h"], np.nan), "l": np.where(pos < nr, X["l"], np.nan)})
    g = df.groupby("sid")
    rh = g.h.transform("max").to_numpy()
    rl = g.l.transform("min").to_numpy()
    full = df.assign(k=(pos < nr) & np.isfinite(X["h"])).groupby("sid").k.transform("sum").to_numpy() == nr
    c = X["c"]
    after = (pos >= nr) & full
    up = after & (c > rh)
    dn = after & (c < rl) if cfg["dirs"] == "both" else np.zeros_like(up)
    hit = up | dn
    first = hit & (pd.Series(hit).groupby(sid).cumsum().to_numpy() == 1)
    sig = np.where(first, np.where(up, 1, -1), 0)
    rng = (rh - rl) / c
    flat = pos == 287  # last 5m bar of the session
    return sig, cfg["ks"] * rng, cfg["kt"] * rng, 288, flat


def im(X, cfg):
    sid, pos = sess_index(X["t"], 0)
    c = X["c"]
    o0 = pd.Series(X["o"]).groupby(sid).transform("first").to_numpy()
    at = pos == cfg["H"] * 12 - 1  # close of the bar ending at hour H
    r = c / o0 - 1
    sig = np.where(at & (r > cfg["thr"]), 1, 0)
    if cfg["dirs"] == "both":
        sig = np.where(at & (r < -cfg["thr"]), -1, sig)
    hi = pd.Series(X["h"]).groupby(sid).cummax().to_numpy()
    lo = pd.Series(X["l"]).groupby(sid).cummin().to_numpy()
    stop = (hi - lo) / c if cfg["stop"] else np.nan
    return sig, stop, np.nan, 288, pos == 287


FAM = {"orb": orb, "im": im}


def run_coin(sym):
    X = F.load(sym)
    if X is None:
        return []
    out = []
    for k, cfg in enumerate(CONFIGS):
        sig, st, tp, hold, flat = FAM[cfg["fam"]](X, cfg)
        d = F.trades(X, sig, st, tp, hold, flat)
        d["cfg"] = k
        d["sym"] = sym
        out.append(d[["cfg", "sym", "t", "tx", "side", "gross", "net", "why"]])
    return out


def main():
    syms = F.coins()
    parts = []
    with cf.ProcessPoolExecutor(12) as ex:
        for r in ex.map(run_coin, syms, chunksize=4):
            parts += r
    T = pd.concat(parts, ignore_index=True)
    out = F.RESULTS / "F1_session"
    out.mkdir(parents=True, exist_ok=True)
    T.to_pickle(F.OUT / "F1_trades.pkl")
    rows = []
    for k, cfg in enumerate(CONFIGS):
        d = T[T.cfg == k]
        rows.append(F.summarize(d, cfg["fam"], **{a: b for a, b in cfg.items() if a != "fam"}))
        for side in (1, -1):
            if cfg.get("dirs") == "both":
                rows.append(F.summarize(d[d.side == side], cfg["fam"] + ("_L" if side > 0 else "_S"),
                                        **{a: b for a, b in cfg.items() if a != "fam"}))  # fmt: skip
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(S.sort_values("is_bp", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
