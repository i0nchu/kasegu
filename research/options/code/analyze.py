"""OTM short-dated option lottery study on Deribit history (BTC/ETH)."""
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
R = Path(__file__).resolve().parents[1] / "results"
R.mkdir(exist_ok=True)
MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
TICK = 0.0001
BINS = [0.0, 0.02, 0.04, 0.06, 0.09, 0.13]
LAB = ["0-2%", "2-4%", "4-6%", "6-9%", "9-13%"]


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, v, call):
    if T <= 0:
        return max(0.0, S - K) if call else max(0.0, K - S)
    sd = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    return S * ncdf(d1) - K * ncdf(d2) if call else K * ncdf(-d2) - S * ncdf(-d1)


def load(cur):
    t = pd.read_csv(D / f"opt_trades_{cur}_0809.csv.gz")
    p = t.instrument.str.split("-", expand=True)
    ex = p[1].apply(lambda s: pd.Timestamp(2000 + int(s[-2:]), MON[s[-5:-2]], int(s[:-5]), 8, tz="UTC"))
    t["expiry"] = ex
    t["K"] = p[2].astype(float)
    t["call"] = p[3] == "C"
    t["ts"] = pd.to_datetime(t.first_ts, unit="ms", utc=True)
    t["dte"] = (t.expiry - t.ts).dt.total_seconds() / 86400
    S = t.first_index
    t["otm"] = np.where(t.call, t.K / S - 1, 1 - t.K / S)
    dl = pd.read_csv(D / f"delivery_{cur}.csv.gz")
    dl["date"] = pd.to_datetime(dl.date).dt.date
    t["ST"] = t.expiry.dt.date.map(dict(zip(dl.date, dl.price)))
    t["dte_d"] = t.dte.round().astype(int)
    t = t[(t.dte_d.isin([1, 2, 3])) & (t.otm >= 0) & (t.otm < BINS[-1]) & t.ST.notna()].copy()
    t["bucket"] = pd.cut(t.otm, BINS, labels=LAB, right=False)
    return t


def costs(t, ticks=1.0):
    """USD cost per 1 underlying unit incl. ask-side ticks and taker fee; net settle value."""
    px = t.vwap + ticks * TICK
    fee = np.minimum(0.0003, 0.125 * px)
    cost = (px + fee) * t.first_index
    pay = np.where(t.call, np.maximum(0, t.ST - t.K), np.maximum(0, t.K - t.ST))
    sfee = np.minimum(0.00015 * t.ST, 0.125 * pay)
    return cost, pay - sfee


def pick(t):
    """One instrument per day/type/dte/bucket: the most traded."""
    return t.sort_values("amount").groupby(["day", "call", "dte_d", "bucket"], observed=True).tail(1)


def single_stats(t, ticks=1.0):
    t = t.copy()
    t["cost"], t["val"] = costs(t, ticks)
    t["mult"] = t.val / t.cost
    t["prem_pct"] = t.vwap * 100
    g = t.groupby(["call", "dte_d", "bucket"], observed=True)
    out = g.agg(n=("mult", "size"), prem_btc_pct=("prem_pct", "median"), ev=("mult", "mean"),
                p_win=("mult", lambda m: (m > 1).mean()), p2=("mult", lambda m: (m >= 2).mean()),
                p5=("mult", lambda m: (m >= 5).mean()), p10=("mult", lambda m: (m >= 10).mean()),
                p20=("mult", lambda m: (m >= 20).mean()), iv=("mean_iv", "median")).reset_index()
    return t, out


def roll(t, call, dte, bucket, targets=(2, 5, 10), frac=1.0, horizon=90, start=100.0, floor=1.0):
    """Bet frac of equity each round, next round at expiry. Returns hit rates from every start day."""
    s = t[(t.call == call) & (t.dte_d == dte) & (t.bucket == bucket)]
    by = {d: r for d, r in zip(s.day, s.mult)}
    days = sorted(by)
    res = []
    for d0 in days:
        d0t = pd.Timestamp(d0)
        eq, d, hit = start, d0t, {}
        while (d - d0t).days <= horizon and eq >= floor:
            m = by.get(d.date().isoformat())
            if m is None:
                d += pd.Timedelta(days=1)
                continue
            eq = eq * (1 - frac) + eq * frac * m
            d += pd.Timedelta(days=dte)
            for k in targets:
                if k not in hit and eq >= start * k:
                    hit[k] = (d - d0t).days
            if len(hit) == len(targets):
                break
        res.append({f"hit{k}": k in hit for k in targets} | {f"days{k}": hit.get(k) for k in targets} | {"d0": d0, "end": eq})
    return pd.DataFrame(res)


if __name__ == "__main__":
    pass
