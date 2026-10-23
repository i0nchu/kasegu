"""A2: would a real resting order have been filled, and what happens from the true fill time?

For lead orders (A1 rule L_dn_k12) on coin-days with cached aggTrades:
  touch    the 1m bar rule of A1: the bar low traded below the limit (minus half a tick)
  through  a trade printed strictly below the limit: the whole price level was consumed, so
           a resting order at the limit was filled whatever its place in the queue
  touch_only  touched but never traded through: a queued order may or may not be filled
Markouts start at the first trade at or below the limit (touch) or the first trade below it
(through), using the last trade price at fill time + H minutes.

Usage: uv run python research/alpha/a2_fill_realism.py
"""

import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from common import fmt, folder, publish, save_json
from stats import boot_mean

from data import MIN

EXP = "A2_fill_realism"
SRC = Path("data/research/runs/A1_passive_reversal/fills.csv.gz")
TRADES = Path("data/history/daily/aggTrades")
COST = 0.09
HS = (1, 5, 15, 30)


def read_trades(symbol: str, date: str) -> pd.DataFrame | None:
    p = TRADES / symbol / f"{symbol}-aggTrades-{date}.zip"
    if not p.exists():
        return None
    with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
        d = pd.read_csv(f, header=None, usecols=[1, 2, 5, 6])
    if not str(d.iloc[0, 0]).replace(".", "").isdigit():
        d = d.iloc[1:]
    d.columns = ["p", "q", "t", "m"]
    d = d.astype({"p": float, "q": float, "t": np.int64})
    d["m"] = d.m.astype(str).str.lower() == "true"
    return d.sort_values("t", kind="stable").reset_index(drop=True)


def _day(args) -> list[dict]:
    (symbol, date), g = args
    tr = read_trades(symbol, date)
    if tr is None or tr.empty:
        return []
    # the next day for markouts that cross midnight
    nxt = str((pd.Timestamp(date) + pd.Timedelta(days=1)).date())
    tr2 = read_trades(symbol, nxt)
    if tr2 is not None:
        tr = pd.concat([tr, tr2], ignore_index=True)
    t, p, q = tr.t.to_numpy(), tr.p.to_numpy(), tr.q.to_numpy()
    rows = []
    for e in g.itertuples():
        a, b = np.searchsorted(t, [e.t, e.t + MIN])
        if b <= a:
            continue
        pw, qw, tw = p[a:b], q[a:b], t[a:b]
        L = e.limit
        at = np.flatnonzero(pw <= L)
        thr = np.flatnonzero(pw < L)
        r = {
            "symbol": symbol,
            "t": e.t,
            "limit": L,
            "event": e.event,
            "bar_m30": e.m30,
            "touch": len(at) > 0,
            "through": len(thr) > 0,
            "vol_at_or_below": float(qw[at].sum()),
            "vol_at_limit": float(qw[pw == L].sum()),
            "secs_to_touch": (tw[at[0]] - e.t) / 1000 if len(at) else np.nan,
        }
        for nm, idx in (("touch", at), ("through", thr)):
            if not len(idx):
                continue
            tf = tw[idx[0]]
            for H in HS:
                k = np.searchsorted(t, tf + H * MIN, side="right") - 1
                r[f"{nm}_m{H}"] = (p[k] - L) / L * 100
        rows.append(r)
    return rows


def main():
    out = folder(EXP)
    fx = pd.read_csv(SRC)
    lead = fx[fx.rule == "L_dn_k12"].copy()
    lead["date"] = pd.to_datetime(lead.t, unit="ms").dt.strftime("%Y-%m-%d")
    have = {(p.parent.name, p.stem.split("-aggTrades-")[1]) for p in TRADES.glob("*/*.zip")}
    lead = lead[[(s, d) in have for s, d in zip(lead.symbol, lead.date, strict=True)]]
    with ProcessPoolExecutor(6) as ex:
        parts = list(ex.map(_day, list(lead.groupby(["symbol", "date"]))))
    d = pd.DataFrame([r for p in parts for r in p])
    d["day"] = d.t // 86_400_000
    d.to_csv(out / "fills_tick.csv", index=False)
    rows = []
    for nm, s, col in (
        ("bar rule (A1), m30 from bar data", d, "bar_m30"),
        ("touch, from first trade at limit", d[d.touch], "touch_m30"),
        ("through, from first trade below limit", d[d.through], "through_m30"),
        ("touch only (queue risk), from touch", d[d.touch & ~d.through], "touch_m30"),
    ):
        for ev, x in (("all", s), ("ex_crash", s[s.event != "crash"])):
            m, lo, hi = boot_mean(x[col] - COST, x.day.to_numpy())
            rows.append(
                {
                    "fill_model": nm,
                    "days": ev,
                    "n": len(x),
                    "net30": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "median": float((x[col] - COST).median()),
                    "win": float((x[col] > COST).mean()),
                }
            )
    t = pd.DataFrame(rows)
    t.to_csv(out / "fill_models.csv", index=False)
    summ = {
        "orders_checked": len(d),
        "bar_fills_with_no_trade_at_limit": int((~d.touch).sum()),
        "through_rate": float(d.through.mean()),
        "touch_only_rate": float((d.touch & ~d.through).mean()),
        "median_secs_to_touch": float(d.secs_to_touch.median()),
        "profile_through": {f"m{H}": float(d[f"through_m{H}"].mean() - COST) for H in HS},
        "profile_touch_only": {
            f"m{H}": float(d[d.touch & ~d.through][f"touch_m{H}"].mean() - COST) for H in HS
        },
    }
    save_json(out / "summary.json", summ)
    print(fmt(t), summ, sep="\n")
    publish(EXP, ["fill_models.csv", "summary.json", "fills_tick.csv"])


if __name__ == "__main__":
    main()
