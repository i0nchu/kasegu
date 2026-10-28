"""NLA-03 single-factor tests on dev only, as pre-registered in LOG N3.

Usage: PYTHONPATH=research/nla uv run python research/nla/n7_factors.py
Needs: events.csv, bars/, split.csv, bench_btc_1m.npz, spot_1h.npz
Output: data/research/nla/factors_dev.csv, research/nla/results/N3_factors/*.csv
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from n6_event_study import MIN, H, alt_basket, at, vwap5

OUT = Path("data/research/nla")
RES = Path("research/nla/results/N3_factors")
ANCHORS = {"1h": 1, "6h": 6, "24h": 24}
HORIZONS = {"24h": 24, "7d": 168, "30d": 720}
TAIL = {"24h": 0.30, "7d": 0.50, "30d": 1.00}
FACTORS = ["early_ret", "clv", "vwap_dev", "taker_imb", "vol_decay"]
BIG = ["perp_after_spot", "perp_first", "spot_first", "spot_perp_same_day"]
PRIMARY = ["24h", "7d"]
N_PRIMARY = len(FACTORS) * len(ANCHORS) * len(PRIMARY) * 2
BOOT = 2000


def vwap(k: np.ndarray, a: float, b: float) -> float:
    m = (k[:, 0] >= a) & (k[:, 0] < b)
    v = k[m, 5].sum()
    return k[m, 6].sum() / v if v > 0 else np.nan


def factors(k: np.ndarray, t0: float, ta: float) -> dict:
    w = k[(k[:, 0] >= t0) & (k[:, 0] < ta)]
    pa = vwap5(k, ta)
    ref = vwap(k, t0, t0 + 15 * MIN)
    late = w[w[:, 0] >= t0 + 15 * MIN]
    hi, lo = (late[:, 2].max(), late[:, 3].min()) if len(late) else (np.nan, np.nan)
    qv = w[:, 6]
    half = (t0 + ta) / 2
    v1, v2 = qv[w[:, 0] < half].mean(), qv[w[:, 0] >= half].mean()
    p = np.exp(pa)
    return dict(
        early_ret=pa - np.log(ref) if ref > 0 else np.nan,
        clv=(p - lo) / (hi - lo) if hi > lo else np.nan,
        vwap_dev=pa - np.log(vwap(k, t0, ta)),
        taker_imb=(2 * w[:, 8].sum() - qv.sum()) / qv.sum() if qv.sum() > 0 else np.nan,
        vol_decay=np.log(v2 / v1) if v1 > 0 and v2 > 0 else np.nan,
    )


def build() -> pd.DataFrame:
    sp = pd.read_csv(RES.parent / "N1_dataset" / "split.csv")
    ev = pd.read_csv(OUT / "events.csv").merge(sp[["event_id", "split"]], on="event_id")
    ev = ev[ev["split"] == "dev"]
    dup = (ev["listing_category"] == "spot_perp_same_day") & (ev["market_type"] == "spot")
    print("dev events", len(ev), "- same-day spot duplicates dropped", int(dup.sum()))
    ev = ev[~dup]
    at_, ax = alt_basket()
    rows = []
    for e in ev.itertuples():
        k = np.load(OUT / "bars" / f"{e.event_id}.npz")["k"]
        t0, tend = k[0, 0], k[-1, 0] + MIN
        for an, ah in ANCHORS.items():
            ta = t0 + ah * H
            row = dict(
                event_id=e.event_id,
                anchor=an,
                shortable=bool(e.perp_listing_time <= ta)
                if e.perp_listing_time == e.perp_listing_time
                else False,
            )
            row |= factors(k, t0, ta) if ta <= tend else dict.fromkeys(FACTORS, np.nan)
            pa = vwap5(k, ta)
            for hn, hh in HORIZONS.items():
                te = ta + hh * H
                ok = te <= min(tend, t0 + 31 * 24 * H)
                raw = np.exp(vwap5(k, te) - pa) - 1 if ok else np.nan
                alt = np.exp(at(at_, ax, te - H) - at(at_, ax, ta - H)) - 1
                row[f"raw_{hn}"] = raw
                row[f"ex_{hn}"] = raw - alt
            rows.append(row)
    f = pd.DataFrame(rows).merge(
        ev[
            [
                "event_id",
                "listing_category",
                "market_type",
                "year",
                "first_trade_time",
                "data_quality_flags",
            ]
        ],
        on="event_id",
    )
    f["month"] = pd.to_datetime(f["first_trade_time"], unit="ms").dt.strftime("%Y-%m")
    f.to_csv(OUT / "factors_dev.csv", index=False)
    return f


def group_pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile within category x year; years with < 10 events pooled per category."""
    n = df.groupby(["listing_category", "year"])[col].transform("count")
    g = df["listing_category"] + "_" + np.where(n >= 10, df["year"].astype(str), "small")
    return df.groupby(g)[col].rank(pct=True)


def pval(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else np.nan


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan


def auc(score: np.ndarray, tail: np.ndarray) -> float:
    pos, neg = score[tail], score[~tail]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot(df: pd.DataFrame, fn, rng) -> tuple[float, float]:
    """Point estimate and month-cluster bootstrap standard error."""
    est = fn(df)
    months = df["month"].unique()
    idx = {m: np.flatnonzero(df["month"].to_numpy() == m) for m in months}
    bs = []
    for _ in range(BOOT):
        pick = rng.choice(months, len(months))
        bs.append(fn(df.iloc[np.concatenate([idx[m] for m in pick])]))
    return est, np.nanstd(bs)


def test(f: pd.DataFrame, fac: str, an: str, hn: str, rng) -> dict:
    d = f[(f["anchor"] == an)].dropna(subset=[fac, f"ex_{hn}"]).copy()
    d["fx"] = group_pct(d, fac)
    d["fy"] = group_pct(d, f"ex_{hn}")
    d["tail"] = d[f"ex_{hn}"] >= TAIL[hn]
    d["q"] = np.ceil(d["fx"] * 5).clip(1, 5).astype(int)
    out = dict(
        factor=fac,
        anchor=an,
        horizon=hn,
        n=len(d),
        n_tail=int(d["tail"].sum()),
        n_missing=int(((f["anchor"] == an) & f[fac].isna()).sum()),
    )
    out["ic_raw"] = d[fac].rank().corr(d[f"ex_{hn}"].rank())
    ic, se = boot(d, lambda x: corr(x["fx"].to_numpy(), x["fy"].to_numpy()), rng)
    out |= dict(ic_ctrl=ic, ic_se=se, ic_p=pval(ic / se))
    a, se = boot(d, lambda x: auc(x["fx"].to_numpy(), x["tail"].to_numpy()), rng)
    out |= dict(tail_auc=a, auc_se=se, auc_p=pval((a - 0.5) / se))
    for q in range(1, 6):
        g = d[d["q"] == q]
        out |= {
            f"q{q}_mean": g[f"ex_{hn}"].mean(),
            f"q{q}_median": g[f"ex_{hn}"].median(),
            f"q{q}_tail": g["tail"].mean(),
            f"q{q}_raw_mean": g[f"raw_{hn}"].mean(),
        }
    out["spread"] = out["q5_mean"] - out["q1_mean"]
    w = d[f"ex_{hn}"].clip(d[f"ex_{hn}"].quantile(0.01), d[f"ex_{hn}"].quantile(0.99))
    out["spread_w1"] = w[d["q"] == 5].mean() - w[d["q"] == 1].mean()
    lo, hi = out["q1_tail"], out["q5_tail"]
    out["tail_ratio"] = max(lo, hi) / min(lo, hi) if min(lo, hi) > 0 else np.inf
    yrs = {
        y: corr(g["fx"].to_numpy(), g["fy"].to_numpy())
        for y, g in d.groupby("year")
        if len(g) >= 15
    }
    out["years"] = len(yrs)
    out["year_same"] = np.mean([np.sign(v) == np.sign(ic) for v in yrs.values()]) if yrs else np.nan
    out["year_ic"] = ";".join(f"{y}:{v:+.2f}" for y, v in yrs.items())
    cats = {
        c: corr(g["fx"].to_numpy(), g["fy"].to_numpy())
        for c, g in d.groupby("listing_category")
        if c in BIG
    }
    out["cat_same"] = int(sum(np.sign(v) == np.sign(ic) for v in cats.values()))
    out |= {f"ic_{c}": v for c, v in cats.items()}
    s = d[d["shortable"]]
    out["n_short"] = len(s)
    out["ic_short"] = corr(s["fx"].to_numpy(), s["fy"].to_numpy())
    return out


def main():
    cache = OUT / "factors_dev.csv"
    f = pd.read_csv(cache) if cache.exists() else build()
    rng = np.random.default_rng(7)
    res = pd.DataFrame(
        [test(f, fac, an, hn, rng) for fac in FACTORS for an in ANCHORS for hn in HORIZONS]
    )
    alpha = 0.05 / N_PRIMARY
    prim = res["horizon"].isin(PRIMARY)
    res["pass_ic"] = prim & (res["ic_p"] < alpha)
    res["pass_auc"] = prim & (res["auc_p"] < alpha)
    RES.mkdir(parents=True, exist_ok=True)
    res.to_csv(RES / "factor_tests.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = [
        "factor",
        "anchor",
        "horizon",
        "n",
        "n_tail",
        "ic_raw",
        "ic_ctrl",
        "ic_p",
        "tail_auc",
        "auc_p",
        "q1_mean",
        "q5_mean",
        "spread",
        "spread_w1",
        "q1_tail",
        "q5_tail",
        "years",
        "year_same",
        "cat_same",
        "ic_short",
        "pass_ic",
        "pass_auc",
    ]
    print("alpha", alpha)
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
