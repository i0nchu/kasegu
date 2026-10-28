"""NLA-01 step 3: data-quality flags and the yearly listing table.

Usage: PYTHONPATH=research/nla uv run python research/nla/n3_dataset.py
Needs: data/research/nla/events_raw.csv and bars/ (n2_events.py)
Output: data/research/nla/events.csv, research/nla/results/N1_dataset/*.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from n2_events import LEVERAGED, STABLE

OUT = Path("data/research/nla")
RES = Path("research/nla/results/N1_dataset")
MIN, DAY = 60_000, 86_400_000
FIAT = {
    "EUR",
    "GBP",
    "AUD",
    "TRY",
    "BRL",
    "RUB",
    "UAH",
    "NGN",
    "ZAR",
    "PLN",
    "RON",
    "ARS",
    "JPY",
    "MXN",
    "COP",
    "CZK",
    "IDRT",
    "BIDR",
    "BKRW",
    "BVND",
}


def bar_stats(eid: str) -> dict:
    p = OUT / "bars" / f"{eid}.npz"
    if not p.exists():
        return {}
    k = np.load(p)["k"]
    t, o, h, lo, c, v, qv = (k[:, i] for i in range(7))
    t0 = t[0]
    d1 = t < t0 + DAY
    h1 = t < t0 + 3_600_000
    expect = int(min(DAY, t[-1] - t0 + MIN) // MIN)
    vwap1h = qv[h1].sum() / v[h1].sum() if v[h1].sum() > 0 else np.nan
    return dict(
        span_days=(t[-1] - t0 + MIN) / DAY,
        missing_min_d1=1 - d1.sum() / max(expect, 1),
        zero_vol_min_d1=float((v[d1] == 0).mean()),
        first_open=o[0],
        first_close=c[0],
        first_bar_range=h[0] / lo[0] - 1 if lo[0] > 0 else np.nan,
        first_open_vs_vwap1h=o[0] / vwap1h - 1 if vwap1h > 0 else np.nan,
        quote_vol_d1=qv[d1].sum(),
    )


def main():
    ev = pd.read_csv(OUT / "events_raw.csv")
    roster = json.loads((OUT / "roster.json").read_text())
    arch0 = {
        m: min(x[0] for x in roster[k].values()) for m, k in (("spot", "spot"), ("perp", "um"))
    }
    last = max(x[-1] for k in roster for x in roster[k].values())
    st = pd.DataFrame([dict(event_id=e, **bar_stats(e)) for e in ev["event_id"]])
    ev = ev.merge(st, on="event_id", how="left")
    t = pd.to_datetime(ev["first_trade_time"], unit="ms")
    ev["year"] = t.dt.year
    ev["delisted"] = ev["last_month"] < last
    life = pd.PeriodIndex(ev["last_month"], freq="M") - pd.PeriodIndex(
        t.dt.strftime("%Y-%m"), freq="M"
    )
    ev["life_months"] = [x.n for x in life]

    f = pd.DataFrame(index=ev.index)
    f["archive_start"] = t.dt.strftime("%Y-%m") == ev["market_type"].map(arch0)
    f["no_panel_pair"] = ev["panel_symbol"].isna()
    f["panel_quote_later"] = (ev["panel_first_trade"] - ev["first_trade_time"]) > 5 * MIN
    f["first_quote_non_usd"] = ~ev["first_quote"].isin(["USDT", "FDUSD", "USDC", "BUSD", "TUSD"])
    f["window_lt_30d"] = ev["span_days"].fillna(0) < 30
    f["missing_min_d1_gt5pct"] = ev["missing_min_d1"].fillna(1) > 0.05
    f["first_open_far_from_vwap1h"] = ev["first_open_vs_vwap1h"].abs() > 0.2
    f["microsecond_ts"] = (ev["market_type"] == "spot") & (t >= "2025-01-01")
    f["multiplier_contract"] = ev["mult"] > 1
    f["relist"] = ev["seg"] > 0
    f["leveraged_token"] = ev["asset_id"].astype(str).str.match(LEVERAGED.pattern)
    f["stable_or_fiat"] = ev["asset_id"].isin(STABLE | FIAT | {"U", "USD", "KGST"})
    # 2026 tokenized US stocks/ETFs (NVDAB, TSLAB...): spot USDT only, never a perp
    f["tokenized_stock"] = (
        ev["asset_id"].astype(str).str.endswith("B")
        & (t >= "2026-06-01")
        & ev["perp_listing_time"].isna()
        & (ev["quotes"] == "USDT")
    )
    f["synthetic_usd"] = ev["synthetic_usd"].fillna(False).astype(bool)
    ev["data_quality_flags"] = [";".join(c for c in f.columns if r[c]) for _, r in f.iterrows()]
    f["no_bars"] = ev["n_bars"].fillna(0) == 0
    excl = ["leveraged_token", "stable_or_fiat", "tokenized_stock", "archive_start", "no_bars"]
    ev["study_eligible"] = ~f[excl].any(axis=1)
    ev.to_csv(OUT / "events.csv", index=False)

    RES.mkdir(parents=True, exist_ok=True)
    ev["dead12"] = ev["delisted"] & (ev["life_months"] < 12)
    el = ev[ev["study_eligible"]]
    by_year = pd.crosstab(el["year"], el["listing_category"], margins=True)
    by_year.to_csv(RES / "events_by_year_category.csv")
    dl = el.groupby("year").agg(
        events=("event_id", "size"),
        delisted=("delisted", "mean"),
        dead_within_12m=("dead12", "mean"),
    )
    dl.to_csv(RES / "delisting_by_year.csv")
    flags = pd.DataFrame({"all": f.sum(), "eligible": f[ev["study_eligible"]].sum()})
    flags.to_csv(RES / "quality_flags.csv")
    lag = el[el["listing_category"] == "perp_after_spot"]
    lag_d = (lag["first_trade_time"] - lag["spot_listing_time"]) / DAY
    lag_d.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_csv(
        RES / "perp_after_spot_lag_days.csv"
    )
    cols = [
        "event_id",
        "asset_id",
        "market_type",
        "listing_category",
        "first_trade_time",
        "first_symbol",
        "panel_symbol",
        "spot_listing_time",
        "perp_listing_time",
        "prior_market_exists",
        "last_month",
        "delisted",
        "span_days",
        "data_quality_flags",
        "study_eligible",
    ]
    ev[cols].to_csv(RES / "listing_events.csv.gz", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(by_year, dl.round(2), flags, lag_d.describe().round(1), sep="\n\n")


if __name__ == "__main__":
    main()
