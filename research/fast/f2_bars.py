"""F2: bar-pattern momentum on 5m / 15m / 1h bars, point-in-time universe, delisted coins included.

  shock  A bar whose return is > z x its trailing 1-day std and whose volume is > v x its trailing
         1-day mean. mom: trade in the bar's direction, rev: against it. Stop 2 x ATR(24 bars),
         time exit after H bars.
  dc     Donchian breakout: close above the highest high of the previous N bars (1 day or 3 days of
         bars) -> long, below the lowest low -> short. Optional volume filter (bar volume > 2x the
         1-day mean) and trend filter (close on the same side of the 3-day mean). Stop k x ATR, take
         profit 2k x ATR or none, time exit after 1 day.
  flow   Taker-buy imbalance: taker-buy share of the bar's volume above 0.5 + d (or below 0.5 - d)
         together with a volume > 2x mean bar -> trade with the flow, hold H bars, 2 ATR stop.

Same execution and costs as F1 (0.20% round trip + funding, 1x notional per trade).

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/f2_bars.py
"""

import concurrent.futures as cf
import itertools

import numpy as np
import pandas as pd

import fast as F

TF = {"5m": 1, "15m": 3, "1h": 12}
BPD = {k: 288 // v for k, v in TF.items()}  # bars per day

SHOCK = [
    dict(fam="shock", tf=tf, z=z, v=v, mode=m, H=H)
    for tf, z, v, m, H in itertools.product(TF, (3, 5), (3,), ("mom", "rev"), (1, 4, 12))
]
DC = [
    dict(fam="dc", tf=tf, days=nd, vol=vf, trend=tr, k=k, tp=tp)
    for tf, nd, vf, tr, k, tp in itertools.product(TF, (1, 3), (0, 1), (0, 1), (1.5, 3.0), (0, 1))
]
FLOW = [dict(fam="flow", tf=tf, d=d, H=H) for tf, d, H in itertools.product(TF, (0.15, 0.25), (1, 4, 12))]
CONFIGS = SHOCK + DC + FLOW


def atr(B, n):
    pc = np.concatenate([[np.nan], B["c"][:-1]])
    tr = np.fmax(B["h"] - B["l"], np.fmax(np.abs(B["h"] - pc), np.abs(B["l"] - pc)))
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def shock(B, cfg, n):
    c = B["c"]
    r = c / B["o"] - 1
    sd = pd.Series(r).shift(1).rolling(n, min_periods=n // 2).std().to_numpy()
    vm = pd.Series(B["qv"]).shift(1).rolling(n, min_periods=n // 2).mean().to_numpy()
    ev = (np.abs(r) > cfg["z"] * sd) & (B["qv"] > cfg["v"] * vm)
    s = np.sign(r) * (1 if cfg["mode"] == "mom" else -1)
    sig = np.where(ev, s, 0)
    return sig, 2 * B["atr"] / c, np.nan, cfg["H"]


def dc(B, cfg, n):
    c = B["c"]
    N = n * cfg["days"]
    hh = pd.Series(B["h"]).shift(1).rolling(N, min_periods=N).max().to_numpy()
    ll = pd.Series(B["l"]).shift(1).rolling(N, min_periods=N).min().to_numpy()
    up, dn = c > hh, c < ll
    if cfg["vol"]:
        vm = pd.Series(B["qv"]).shift(1).rolling(n, min_periods=n // 2).mean().to_numpy()
        up, dn = up & (B["qv"] > 2 * vm), dn & (B["qv"] > 2 * vm)
    if cfg["trend"]:
        ma = pd.Series(c).rolling(3 * n, min_periods=3 * n).mean().to_numpy()
        up, dn = up & (c > ma), dn & (c < ma)
    # only the first bar of a breakout run, not every bar that stays above the old high
    up = up & ~np.concatenate([[False], up[:-1]])
    dn = dn & ~np.concatenate([[False], dn[:-1]])
    sig = np.where(up, 1, np.where(dn, -1, 0))
    a = B["atr"] / c
    return sig, cfg["k"] * a, 2 * cfg["k"] * a if cfg["tp"] else np.nan, n


def flow(B, cfg, n):
    share = B["tb"] / B["qv"]
    vm = pd.Series(B["qv"]).shift(1).rolling(n, min_periods=n // 2).mean().to_numpy()
    big = B["qv"] > 2 * vm
    sig = np.where(big & (share > 0.5 + cfg["d"]), 1, np.where(big & (share < 0.5 - cfg["d"]), -1, 0))
    return sig, 2 * B["atr"] / B["c"], np.nan, cfg["H"]


FAM = {"shock": shock, "dc": dc, "flow": flow}


def run_coin(sym):
    X = F.load(sym)
    if X is None:
        return []
    bars = {}
    for tf, m in TF.items():
        B = X if m == 1 else F.resample(X, m)
        B["atr"] = atr(B, 24)
        bars[tf] = B
    out = []
    for k, cfg in enumerate(CONFIGS):
        B = bars[cfg["tf"]]
        with np.errstate(invalid="ignore", divide="ignore"):
            sig, st, tp, hold = FAM[cfg["fam"]](B, cfg, BPD[cfg["tf"]])
        d = F.trades(B, np.nan_to_num(sig).astype(int), st, tp, hold)
        d["cfg"] = k
        d["sym"] = sym
        out.append(d[["cfg", "sym", "t", "tx", "side", "gross", "net", "why"]])
    return out


def main():
    parts = []
    with cf.ProcessPoolExecutor(12) as ex:
        for r in ex.map(run_coin, F.coins(), chunksize=4):
            parts += r
    T = pd.concat(parts, ignore_index=True)
    T.to_pickle(F.OUT / "F2_trades.pkl")
    out = F.RESULTS / "F2_bars"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, cfg in enumerate(CONFIGS):
        d = T[T.cfg == k]
        kw = {a: b for a, b in cfg.items() if a != "fam"}
        rows.append(F.summarize(d, cfg["fam"], **kw))
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(S.sort_values("is_bp", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
