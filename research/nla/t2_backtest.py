"""NLA-T01 backtest: short new USDT perpetuals 24h after their first trade, hold 1d / 3d / 7d, as
pre-registered in LOG T1. Dev only; val and hold are refused.

Usage: PYTHONPATH=research/nla uv run python research/nla/t2_backtest.py [dev|val] [holds]
Needs: n3/n5 outputs, bars/, t1_download.py cache
Output: data/research/nla/t1_trades_dev.csv, research/nla/results/T1_backtest/*
"""

import functools
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from archive import CACHE, read_klines

OUT = Path("data/research/nla")
RES = Path("research/nla/results/T1_backtest")
MIN, H, DAY = 60_000, 3_600_000, 86_400_000
HOLDS = {"1d": 24, "3d": 72, "7d": 168}
STOP = 0.20
LEV = 3.0
MMR = 0.01
LIQ = 1 / LEV - MMR
FEE = 0.0005
RISK = 0.01
MAX_OPEN = 15
N_CTRL = 5
SIZE = RISK / (STOP + 0.006)  # notional per trade as a fraction of equity
UM = CACHE / "futures/um/monthly"


def slip_tier(qv24: float) -> float:
    return 0.001 if qv24 >= 50e6 else 0.002 if qv24 >= 10e6 else 0.004


@functools.cache
def funding(sym: str) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for p in sorted((UM / "fundingRate" / sym).glob("*.zip")):
        with zipfile.ZipFile(p) as z:
            d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        if not str(d.columns[0]).startswith("calc"):
            d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None)
            d.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][: d.shape[1]]
        parts.append(d[["calc_time", "last_funding_rate"]])
    if not parts:
        return np.array([]), np.array([])
    d = pd.concat(parts).drop_duplicates("calc_time").sort_values("calc_time")
    return d["calc_time"].to_numpy(np.int64), d["last_funding_rate"].to_numpy(np.float64)


def sim(t, o, h, c, ta: float, hold_h: int, slip: float, fund, tol: int = 5 * MIN) -> dict | None:
    """One short. t/o/h/c are bar arrays (1m or 1h); returns net P&L as a fraction of notional."""
    i0 = np.searchsorted(t, ta)
    if i0 >= len(t) or t[i0] - ta > tol:
        return None
    ta = t[i0]
    te = ta + hold_h * H
    i1 = np.searchsorted(t, te)  # bars [i0, i1) are held
    if i1 > len(t) or t[i1 - 1] < te - 2 * H:
        return dict(reason="no_data")
    e = o[i0] * (1 - slip)
    stop, liq = e * (1 + STOP), e * (1 + LIQ)
    oo, hh = o[i0:i1], h[i0:i1]
    gap = np.flatnonzero(oo >= liq)
    hit = np.flatnonzero(hh >= stop)
    j = min(gap[0] if len(gap) else 1 << 30, hit[0] if len(hit) else 1 << 30)
    if j < i1 - i0:
        fill = max(stop, oo[j]) * (1 + 2 * slip)
        reason, x = ("liquidated", None) if fill >= liq else ("stop", fill)
        tx = t[i0 + j]
    else:
        reason, x, tx = "time", c[i1 - 1] * (1 + slip), te
    ft, fr = fund
    m = (ft > ta) & (ft <= tx)
    fpay = 0.0
    if m.any():
        idx = np.clip(np.searchsorted(t, ft[m]) - 1, 0, len(t) - 1)
        fpay = float(np.sum(fr[m] * c[idx] / e))
    if reason == "liquidated":
        pnl = -LIQ - MMR - FEE + fpay
    else:
        pnl = (e - x) / e - FEE * (1 + x / e) + fpay
    # unrealized P&L at each day end, for the daily equity curve
    days = np.arange((ta // DAY + 1) * DAY, tx, DAY)
    di = np.clip(np.searchsorted(t, days) - 1, 0, len(t) - 1)
    marks = (e - c[di]) / e if len(days) else np.array([])
    return dict(
        reason=reason,
        pnl=pnl,
        funding=fpay,
        exit_time=tx,
        entry=e,
        slip=slip,
        mark_days=days.tolist(),
        marks=marks.tolist(),
    )


@functools.cache
def hourly(sym: str) -> pd.DataFrame:
    ps = sorted((UM / "klines" / sym / "1h").glob("*.zip"))
    return pd.concat([read_klines(p) for p in ps], ignore_index=True) if ps else pd.DataFrame()


def control_universe() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Daily quote volume and listing age of every USDT perpetual, from the 1h klines."""
    roster = json.loads((OUT / "roster.json").read_text())["um"]
    syms = [s for s in roster if s.endswith("USDT") and "_" not in s]
    days = np.arange(
        pd.Timestamp("2019-09-01").value // 10**6, pd.Timestamp("2026-10-01").value // 10**6, DAY
    )
    qv = np.zeros((len(days), len(syms)), np.float32)
    first = np.full(len(syms), np.inf)
    for j, s in enumerate(syms):
        d = hourly(s)
        if d.empty:
            continue
        first[j] = d["open_time"].iat[0]
        g = d.groupby(d["open_time"] // DAY * DAY)["quote_volume"].sum()
        i = np.searchsorted(days, g.index.to_numpy())
        ok = i < len(days)
        qv[i[ok], j] = g.to_numpy()[ok]
    hourly.cache_clear()
    return days, syms, qv, first


def pick_controls(when, exclude, uni, rng) -> list[str]:
    days, syms, qv, first = uni
    di = np.searchsorted(days, when) - 1
    q30 = qv[max(0, di - 29) : di + 1].sum(0)
    ok = (first <= when - 180 * DAY) & (q30 > 0)
    ok &= np.array([s.removesuffix("USDT").lstrip("0123456789") != exclude for s in syms])
    top = np.argsort(-np.where(ok, q30, -1))[:100]
    top = top[ok[top]]
    return [syms[j] for j in rng.choice(top, min(N_CTRL, len(top)), replace=False)]


def run_trades(
    slip_mult: float = 1.0, with_controls: bool = True, which: str = "dev"
) -> pd.DataFrame:
    assert which in ("dev", "val"), "the holdout is reserved for the final one-shot check"
    sp = pd.read_csv(RES.parent / "N1_dataset" / "split.csv")
    ev = pd.read_csv(OUT / "events.csv").merge(sp[["event_id", "split"]], on="event_id")
    ev = ev[(ev["split"] == which) & (ev["market_type"] == "perp")]
    ev = ev[ev["panel_symbol"].str.endswith("USDT", na=False)]
    uni = control_universe() if with_controls else None
    rng = np.random.default_rng(11)
    rows = []
    for e in ev.sort_values("first_trade_time").itertuples():
        k = np.load(OUT / "bars" / f"{e.event_id}.npz")["k"]
        t0 = k[0, 0]
        ta = t0 + 24 * H
        qv24 = k[k[:, 0] < ta, 6].sum()
        slip = slip_tier(qv24) * slip_mult
        fund = funding(e.panel_symbol)
        ctrl = pick_controls(ta, e.asset_id, uni, rng) if with_controls else []
        for hn, hh in HOLDS.items():
            r = sim(k[:, 0], k[:, 1], k[:, 2], k[:, 4], ta, hh, slip, fund)
            base = dict(
                event_id=e.event_id,
                symbol=e.panel_symbol,
                category=e.listing_category,
                year=e.year,
                entry_time=ta,
                hold=hn,
                qv24=qv24,
                has_funding=len(fund[0]) > 0,
            )
            rows.append(base | (r or dict(reason="no_entry")) | dict(leg="signal"))
            for cs in ctrl:
                d = hourly(cs)
                if d.empty:
                    continue
                dq = d[(d["open_time"] >= ta - DAY) & (d["open_time"] < ta)]["quote_volume"].sum()
                rc = sim(
                    d["open_time"].to_numpy(),
                    d["open"].to_numpy(),
                    d["high"].to_numpy(),
                    d["close"].to_numpy(),
                    ta,
                    hh,
                    slip_tier(dq) * slip_mult,
                    funding(cs),
                    H,
                )
                rows.append(base | (rc or dict(reason="no_entry")) | dict(leg="control", ctrl=cs))
    return pd.DataFrame(rows)


def portfolio(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Fixed-risk account: 1% risk per trade, at most MAX_OPEN open, daily mark-to-market."""
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    eq, open_, taken, events = 1.0, [], [], []
    for r in tr.itertuples():
        for o in [o for o in open_ if o[0] <= r.entry_time]:
            eq += o[1]
            open_.remove(o)
        if len(open_) >= MAX_OPEN:
            taken.append(False)
            continue
        n = SIZE * eq
        open_.append((r.exit_time, n * r.pnl))
        taken.append(True)
        events.append((r.entry_time, r.exit_time, n, r.pnl, r.mark_days, r.marks))
    tr["taken"] = taken
    start = int(tr["entry_time"].min() // DAY * DAY)
    end = int(tr["exit_time"].max() // DAY * DAY + DAY)
    days = np.arange(start, end + DAY, DAY)
    real = np.zeros(len(days))
    unreal = np.zeros(len(days))
    for _ta, tx, n, pnl, md, mk in events:
        real[np.searchsorted(days, tx) :] += n * pnl
        for d, m in zip(md, mk, strict=True):
            unreal[np.searchsorted(days, d)] += n * m
    curve = pd.Series(1 + real + unreal, index=pd.to_datetime(days, unit="ms"))
    return tr, curve


def cluster_p(x: pd.Series, month: pd.Series, rng, n: int = 2000) -> tuple[float, float]:
    """Mean and one-sided p (mean > 0) from a month-cluster bootstrap."""
    ms = month.unique()
    idx = {m: np.flatnonzero(month.to_numpy() == m) for m in ms}
    v = x.to_numpy()
    bs = [v[np.concatenate([idx[m] for m in rng.choice(ms, len(ms))])].mean() for _ in range(n)]
    se = np.std(bs)
    z = v.mean() / se if se > 0 else np.nan
    return v.mean(), 0.5 * math.erfc(z / math.sqrt(2)) if np.isfinite(z) else np.nan


def stats(tr: pd.DataFrame, curve: pd.Series, rng) -> dict:
    x = tr["pnl"]
    month = pd.to_datetime(tr["entry_time"], unit="ms").dt.strftime("%Y-%m")
    mean, p = cluster_p(x, month, rng)
    win, loss = x[x > 0], x[x <= 0]
    ret = curve.pct_change().dropna()
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    dd = (curve / curve.cummax() - 1).min()
    by_year = tr.groupby("year")["pnl"].agg(["size", "mean"])
    big = by_year[by_year["size"] >= 10]
    return dict(
        trades=len(tr),
        mean_pnl=mean,
        p_one_sided=p,
        median_pnl=x.median(),
        mean_R=mean / (STOP + 0.006),
        win_rate=(x > 0).mean(),
        payoff=win.mean() / -loss.mean() if len(loss) and loss.mean() < 0 else np.inf,
        profit_factor=win.sum() / -loss.sum() if loss.sum() < 0 else np.inf,
        ann_return=curve.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
        sharpe=ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else np.nan,
        max_dd=dd,
        worst=x.min(),
        cvar5=x[x <= x.quantile(0.05)].mean(),
        stops=int((tr["reason"] == "stop").sum()),
        liquidations=int((tr["reason"] == "liquidated").sum()),
        funding_mean=tr["funding"].mean(),
        years_pos=float((big["mean"] > 0).mean()),
        n_years=len(big),
        final_equity=curve.iloc[-1],
    )


def main():
    import sys

    global RES
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    holds = sys.argv[2].split(",") if len(sys.argv) > 2 else list(HOLDS)
    if which != "dev":
        RES = RES.parent / f"T1_backtest_{which}"
    rng = np.random.default_rng(5)
    RES.mkdir(parents=True, exist_ok=True)
    allt = run_trades(which=which)
    allt = allt[allt["hold"].isin(holds)]
    allt.drop(columns=["mark_days", "marks"]).to_csv(OUT / f"t1_trades_{which}.csv", index=False)
    sig = allt[(allt["leg"] == "signal")]
    print(sig["reason"].value_counts().to_dict())
    print(
        "signal events",
        sig["event_id"].nunique(),
        "without funding file",
        int((~sig.drop_duplicates("event_id")["has_funding"]).sum()),
    )
    dbl = run_trades(2.0, with_controls=False, which=which)
    out, years, months, cats = [], [], [], []
    for hn in holds:
        s = sig[(sig["hold"] == hn) & sig["pnl"].notna()]
        s, curve = portfolio(s)
        taken = s[s["taken"]]
        st = dict(hold=hn, skipped_max_open=int((~s["taken"]).sum())) | stats(taken, curve, rng)
        c = allt[(allt["leg"] == "control") & (allt["hold"] == hn) & allt["pnl"].notna()]
        cm = c.groupby("event_id")["pnl"].mean()
        ex = taken.set_index("event_id")["pnl"] - cm.reindex(taken["event_id"]).to_numpy()
        ex = ex.dropna()
        mon = pd.to_datetime(taken.set_index("event_id").loc[ex.index, "entry_time"], unit="ms")
        st["control_mean"] = cm.mean()
        st["excess_mean"], st["excess_p"] = cluster_p(ex, mon.dt.strftime("%Y-%m"), rng)
        d2 = dbl[(dbl["hold"] == hn) & dbl["pnl"].notna()]
        st["mean_pnl_2x_slip"] = d2["pnl"].mean()
        st["pass"] = bool(
            st["p_one_sided"] < 0.05
            and st["mean_pnl"] > 0
            and st["profit_factor"] >= 1.2
            and st["excess_mean"] > 0
            and st["excess_p"] < 0.10
            and st["years_pos"] >= 0.6
            and st["max_dd"] >= -0.30
            and st["mean_pnl_2x_slip"] > 0
        )
        out.append(st)
        y = taken.groupby("year")["pnl"].agg(["size", "mean", "median", lambda v: (v > 0).mean()])
        y.columns = ["trades", "mean", "median", "win"]
        years.append(y.assign(hold=hn))
        m = curve.resample("ME").last().pct_change()
        months.append(m.rename(hn))
        cats.append(
            taken.groupby("category")["pnl"].agg(["size", "mean", "median"]).assign(hold=hn)
        )
        curve.to_csv(RES / f"equity_{hn}.csv")
    res = pd.DataFrame(out)
    res.to_csv(RES / "summary.csv", index=False)
    pd.concat(years).to_csv(RES / "by_year.csv")
    pd.concat(months, axis=1).to_csv(RES / "by_month.csv")
    pd.concat(cats).to_csv(RES / "by_category.csv")
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string())
    print(pd.concat(years).round(4).to_string())
    print(pd.concat(cats).round(4).to_string())


if __name__ == "__main__":
    main()
