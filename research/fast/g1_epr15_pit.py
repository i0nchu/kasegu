"""G1: EPR15 on the point-in-time universe (delisted coins included) and on unused 2022-2024 data.

EPR15 rules from research/alpha/docs/HANDOFF_codex.md (frozen, tag alpha-epr15-v1), rewritten here
on 1m klines without reusing research/sim.py:
  trend   close of the decision bar < EMA(240, adjust=False); EMA needs >= 720 bars
  vol%    (mean(TR, 60) / close * 100 + mean((H - L) / C * 100, 60)) / 2
  limit   close * (1 - 12 * vol% / 100), Post-Only buy, alive for the next 1m bar only
  fill    next bar's low < limit * (1 - pen) (pen = 0 base; the research used low + tick/2 < limit)
  exit    3% stop (fills at stop - 0.03% - slippage, or the open when it gaps; checked from the
          fill bar on, the fill bar's low counts) or the close of the 15th bar counting the fill
          bar as 1, minus slippage. One position per coin; the next order can be placed at the
          close of the exit bar.
  costs   maker 0.02% in, taker 0.05% out, slippage 0.02% (+ stop_slip extra on stops),
          funding ignored (15-minute holds)

Universes:
  surv40  the research's 40 coins (today's survivors), every bar they have data
  pit     a coin may place orders on day d only if it was in the top-40-by-30-day-volume set at
          the close of day d-1 (listed >= 60 days); delisted coins included
Periods: R = 2024-10 to 2026-09 (the research sample), U = 2022-01 to 2024-09 (never used).

Inference: daily mean of trades, t-stat over days; also excluding the worst / best market days.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/g1_epr15_pit.py
"""

import concurrent.futures as cf
import json

import numba as nb
import numpy as np
import pandas as pd

import fast as F

M1 = 60_000
T_U = int(pd.Timestamp("2022-01-01", tz="UTC").value // 10**6)
T_R = int(pd.Timestamp("2024-10-01", tz="UTC").value // 10**6)
SURV40 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "SUIUSDT",
    "1000PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "TIAUSDT",
    "INJUSDT", "SEIUSDT", "ENAUSDT", "1000BONKUSDT", "FETUSDT", "ORDIUSDT", "WLDUSDT",
    "1000FLOKIUSDT", "AAVEUSDT", "ATOMUSDT", "CRVUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT",
    "JUPUSDT", "ONDOUSDT", "PENDLEUSDT", "UNIUSDT", "XLMUSDT",
]  # fmt: skip
SCEN = {  # name: (penetration, extra stop slippage)
    "base": (0.0, 0.0),
    "pen0.2": (0.002, 0.0),
    "stop+0.3": (0.0, 0.003),
    "both": (0.002, 0.003),
}


def load1m(sym: str) -> dict | None:
    p = F.OUT / "1m" / f"{sym}.npz"
    if not p.exists():
        files = sorted((F.HIST / "klines" / sym / "1m").glob("*.zip"))
        if not files:
            return None
        k = pd.concat([F._read_zip(f) for f in files]).iloc[:, [0, 1, 2, 3, 4]].astype(float)
        k.columns = ["t", "o", "h", "l", "c"]
        k["t"] = k.t.astype(np.int64)
        k.loc[k.t > 10**14, "t"] //= 1000
        k = k.drop_duplicates("t").sort_values("t")
        grid = np.arange(k.t.iloc[0], k.t.iloc[-1] + M1, M1, dtype=np.int64)
        k = k.set_index("t").reindex(grid)
        if F._EL is None:
            F._EL = F._elig_days()
        dt, el = F._EL
        di = np.searchsorted(dt, grid, side="right") - 1
        e = el[sym][np.clip(di, 0, len(dt) - 1)] & (di >= 0) if sym in el else np.zeros(len(grid), bool)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, t=grid, e=e, **{c: k[c].to_numpy() for c in ("o", "h", "l", "c")})
    z = np.load(p)
    return {k: z[k] for k in z.files}


def indicators(X):
    o, h, l, c = X["o"], X["h"], X["l"], X["c"]
    # EMA over a series with gaps: carry the EMA across missing bars (pandas ewm ignores NaN rows)
    ema = pd.Series(c).ewm(span=240, adjust=False, ignore_na=True).mean().to_numpy()
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(60, min_periods=60).mean().to_numpy()
    amp = pd.Series((h - l) / c * 100).rolling(60, min_periods=60).mean().to_numpy()
    vol = (atr / c * 100 + amp) / 2
    warm = pd.Series(np.isfinite(c)).cumsum().to_numpy() >= 720
    return ema, vol, warm


@nb.njit(cache=True)
def sim(o, h, l, c, ema, vol, ok, pen, sslip):
    n = len(c)
    out = np.empty((n // 16 + 1, 4))
    k = 0
    i = 0
    maker, taker, slip = 0.0002, 0.0005, 0.0002
    while i < n - 15:
        if not ok[i] or not (c[i] < ema[i]) or not (vol[i] > 0) or np.isnan(l[i + 1]):
            i += 1
            continue
        lim = c[i] * (1 - 12 * vol[i] / 100)
        j = i + 1
        if not (l[j] < lim * (1 - pen)):
            i += 1
            continue
        stop = lim * 0.97
        ex = np.nan
        why = 0
        m = j
        while m < j + 15:
            if np.isnan(o[m]):
                ex = c[m - 1] * (1 - slip)
                m -= 1
                break
            if l[m] <= stop:
                px = stop if m == j else min(stop, o[m])
                ex = px * (1 - 0.0003 - slip - sslip)
                why = 1
                break
            if m == j + 14:
                ex = c[m] * (1 - slip)
                break
            m += 1
        net = ex / lim - 1 - maker - taker * ex / lim
        out[k, 0] = j
        out[k, 1] = m
        out[k, 2] = net
        out[k, 3] = why
        k += 1
        i = m  # the next order is placed at the close of the exit bar
    return out[:k]


def run_coin(sym):
    X = load1m(sym)
    if X is None:
        return []
    ema, vol, warm = indicators(X)
    rows = []
    for uni in ("surv40", "pit"):
        if uni == "surv40" and sym not in SURV40:
            continue
        ok = warm & np.isfinite(X["c"]) & (X["e"] if uni == "pit" else True)
        for name, (pen, ss) in SCEN.items():
            r = sim(X["o"], X["h"], X["l"], X["c"], ema, vol, ok, pen, ss)
            d = pd.DataFrame(r, columns=["i", "x", "net", "stop"])
            d["t"] = X["t"][d.i.astype(int)]
            d["uni"], d["scen"], d["sym"] = uni, name, sym
            rows.append(d[["uni", "scen", "sym", "t", "net", "stop"]])
    return rows


def stats(d: pd.DataFrame) -> dict:
    g = d.groupby(d.t // F.DAY).net.mean()
    tt = g.mean() / g.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan
    mon = d.groupby(pd.to_datetime(d.t, unit="ms").dt.to_period("M")).net.sum()
    trim = g.drop(g.nsmallest(max(1, len(g) // 50)).index).drop(g.nlargest(max(1, len(g) // 50)).index)
    return {
        "trades": len(d), "coins": d.sym.nunique(), "days": len(g),
        "avg_net_pct": 100 * d.net.mean(), "win": (d.net > 0).mean(), "stop_rate": d.stop.mean(),
        "day_t": tt, "trim2pct_day_mean_pct": 100 * trim.mean(),
        "pos_months": f"{(mon > 0).sum()}/{len(mon)}",
    }  # fmt: skip


def main():
    syms = sorted(set(F.coins()) | set(SURV40))
    parts = []
    with cf.ProcessPoolExecutor(10) as ex:
        for r in ex.map(run_coin, syms, chunksize=2):
            parts += r
    T = pd.concat(parts, ignore_index=True)
    T.to_pickle(F.OUT / "G1_trades.pkl")
    out = F.RESULTS / "G1_epr15_pit"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (uni, scen), d in T.groupby(["uni", "scen"]):
        for per, m in (("R 2024-10..2026-09", d.t >= T_R), ("U 2022-01..2024-09", (d.t >= T_U) & (d.t < T_R))):
            rows.append({"uni": uni, "scen": scen, "period": per, **stats(d[m])})
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    # the PIT coins that are not survivors: what the research universe left out
    d = T[(T.uni == "pit") & (T.scen == "base") & ~T.sym.isin(SURV40)]
    by = d.groupby("sym").net.agg(["count", "mean", "sum"]).sort_values("sum")
    by.to_csv(out / "non_survivor_coins.csv", float_format="%.4g")
    json.dump({"scenarios": SCEN}, open(out / "config.json", "w"))
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    print(by.head(15))


if __name__ == "__main__":
    main()
