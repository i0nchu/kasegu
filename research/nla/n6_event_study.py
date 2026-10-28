"""NLA-02 event study: forward returns after listing, raw and in excess of BTC and an alt basket.

Anchors are 1h, 6h and 24h after the first trade; horizons 1h, 6h, 24h, 7d and 30d after the
anchor (30d only where the 31-day window allows). Prices are 5-minute VWAPs ending at the anchor and
at the exit, never the first bar. Only dev events are read (split.csv); the holdout is skipped.

Alt basket: equal weight of the 50 USDT spot pairs with the largest trailing 30-day quote volume
among pairs listed at least 180 days, rebalanced daily, hourly returns.

Usage: PYTHONPATH=research/nla uv run python research/nla/n6_event_study.py [dev|val]
Needs: n3_dataset.py, n4_benchmarks.py, n5_split.py outputs
Output: data/research/nla/event_returns_<split>.csv, research/nla/results/N2_event_study/*.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/research/nla")
RES = Path("research/nla/results/N2_event_study")
H, MIN = 3_600_000, 60_000
ANCHORS = {"1h": 1, "6h": 6, "24h": 24}
HORIZONS = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}
WIN = 31 * 24  # hours of bars per event


def alt_basket() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / "spot_1h.npz")
    t, c, q = z["t"], z["close"].astype(np.float64), z["qvol"].astype(np.float64)
    seen = np.isfinite(c)
    first = np.where(seen.any(0), seen.argmax(0), len(t))
    age = np.arange(len(t))[:, None] - first[None, :]
    q30 = pd.DataFrame(q).rolling(720, min_periods=1).sum().to_numpy()
    r = np.log(c[1:] / c[:-1])
    ret = np.zeros(len(t))
    for d0 in range(24, len(t) - 1, 24):  # daily rebalance from the previous day's ranks
        ok = (age[d0] >= 180 * 24) & seen[d0]
        v = np.where(ok, q30[d0], -1)
        top = np.argsort(-v)[:50]
        top = top[v[top] > 0]
        blk = r[d0 : d0 + 24][:, top]
        with np.errstate(invalid="ignore"):
            ret[d0 + 1 : d0 + 25] = np.nanmean(np.where(np.isfinite(blk), blk, np.nan), axis=1)
    return t, np.cumsum(np.nan_to_num(ret))


def btc() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(OUT / "bench_btc_1m.npz")
    return z["t"], np.log(z["c"])


def at(t: np.ndarray, x: np.ndarray, when: float) -> float:
    i = np.searchsorted(t, when, side="right") - 1
    return x[i] if i >= 0 else np.nan


def vwap5(k: np.ndarray, when: float) -> float:
    m = (k[:, 0] >= when - 5 * MIN) & (k[:, 0] < when)
    v, qv = k[m, 5].sum(), k[m, 6].sum()
    return np.log(qv / v) if v > 0 else np.nan


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    assert which in ("dev", "val"), "the holdout is not read before NLA-06"
    sp = pd.read_csv(RES.parent / "N1_dataset" / "split.csv")
    ev = pd.read_csv(OUT / "events.csv").merge(sp[["event_id", "split"]], on="event_id")
    ev = ev[ev["split"] == which]
    bt, bx = btc()
    at_, ax = alt_basket()
    rows = []
    for e in ev.itertuples():
        k = np.load(OUT / "bars" / f"{e.event_id}.npz")["k"]
        t0 = k[0, 0]
        for an, ah in ANCHORS.items():
            ta = t0 + ah * H
            pa = vwap5(k, ta)
            for hn, hh in HORIZONS.items():
                te = ta + hh * H
                if te > t0 + WIN * H or te > k[-1, 0] + MIN:
                    r = np.nan
                else:
                    r = vwap5(k, te) - pa
                rows.append(
                    dict(
                        event_id=e.event_id,
                        anchor=an,
                        horizon=hn,
                        ret=r,
                        btc=at(bt, bx, te) - at(bt, bx, ta),
                        alt=at(at_, ax, te - H)
                        - at(at_, ax, ta - H),  # hourly index, closed hours only
                    )
                )
    r = pd.DataFrame(rows).merge(
        ev[["event_id", "listing_category", "market_type", "year", "delisted", "asset_id"]],
        on="event_id",
    )
    r["ex_btc"] = r["ret"] - r["btc"]
    r["ex_alt"] = r["ret"] - r["alt"]
    r.to_csv(OUT / f"event_returns_{which}.csv", index=False)
    summarize(r, which)


def stats(x: pd.Series) -> pd.Series:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(
        dict(
            n=n,
            mean=x.mean(),
            median=x.median(),
            pos=(x > 0).mean(),
            std=x.std(),
            t=x.mean() / x.std() * np.sqrt(n) if n > 2 and x.std() > 0 else np.nan,
            p10=x.quantile(0.1),
            p90=x.quantile(0.9),
        )
    )


def summarize(r: pd.DataFrame, which: str):
    RES.mkdir(parents=True, exist_ok=True)
    out = []
    for col in ("ret", "ex_btc", "ex_alt"):
        for keys, g in r.groupby(["anchor", "horizon"]):
            out.append(
                dict(
                    zip(["anchor", "horizon"], keys, strict=True),
                    measure=col,
                    group="all",
                    **stats(g[col]),
                )
            )
        for keys, g in r.groupby(["anchor", "horizon", "listing_category"]):
            out.append(
                dict(
                    zip(["anchor", "horizon", "group"], keys, strict=True),
                    measure=col,
                    **stats(g[col]),
                )
            )
        for keys, g in r.groupby(["anchor", "horizon", "year"]):
            out.append(
                dict(
                    anchor=keys[0],
                    horizon=keys[1],
                    group=f"y{keys[2]}",
                    measure=col,
                    **stats(g[col]),
                )
            )
    s = pd.DataFrame(out)
    s.to_csv(RES / f"summary_{which}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    order = {k: i for i, k in enumerate(HORIZONS)}
    v = s[(s["group"] == "all")].copy()
    v["o"] = v["horizon"].map(order)
    print(
        v.sort_values(["measure", "anchor", "o"]).drop(columns="o").round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
