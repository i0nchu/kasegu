"""Strategy research experiments. Results go to data/research/results/.

Usage: uv run python research/experiments.py baseline drift grid factors breakout screen
"""

import itertools
import json
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from lab import OUT, UNIVERSE, forward, run, table
from sim import Cfg, summary

RES = OUT / "results"
CRASH = "2025-10"  # 10 Oct 2025 liquidation cascade dominates any average that includes it
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)


def save(name: str, df: pd.DataFrame | dict):
    RES.mkdir(parents=True, exist_ok=True)
    if isinstance(df, dict):
        (RES / f"{name}.json").write_text(json.dumps(df, indent=1, default=float))
    else:
        df.to_csv(RES / f"{name}.csv")


def baseline():
    """Current live config on 28 coins x 12 months."""
    tr = run(Cfg())
    tr.to_pickle(RES / "baseline_trades.pkl")
    by_sym = table(tr, "symbol").sort_values("avg_net%")
    by_month = table(tr, "month")
    reasons = table(tr, "reason")
    out = {
        "all": summary(tr),
        "ex_crash": summary(tr[tr.month != CRASH]),
        "is": summary(tr[~tr.oos]),
        "oos": summary(tr[tr.oos]),
        "coins_positive": int((tr.groupby("symbol").net.sum() > 0).sum()),
        "coins": int(tr.symbol.nunique()),
        "months_positive": int((tr.groupby("month").net.sum() > 0).sum()),
        "cost_per_trade": float((tr.gross - tr.net).mean()),
    }
    save("baseline", out)
    save("baseline_by_symbol", by_sym)
    save("baseline_by_month", by_month)
    save("baseline_by_reason", reasons)
    print(json.dumps(out, indent=1, default=float))
    print(by_sym, by_month, reasons, sep="\n\n")


def drift():
    """Raw price path after every fill, before any exit rule or cost."""
    rows = []
    for model, k, trend in itertools.chain(
        itertools.product(["atr_amplitude"], [1, 2, 3, 4, 6, 8], ["with", "against"]),
        itertools.product(["atr_only", "quantile"], [1, 2, 3, 4], ["with"]),
    ):
        fw = forward(Cfg(model=model, k=k, trend=trend))
        fw["month"] = pd.to_datetime(fw.t, unit="ms").dt.strftime("%Y-%m")
        for scope, d in (("all", fw), ("ex_crash", fw[fw.month != CRASH])):
            r = {"model": model, "k": k, "trend": trend, "scope": scope, "fills": len(d)}
            r["fills_per_coin_day"] = len(d) / d.symbol.nunique() / 365
            for h in (1, 5, 15, 30, 60):
                r[f"r{h}"] = d[f"r{h}"].mean()
            r["r30_median"] = d.r30.median()
            r["r30_pos_coins"] = int((d.groupby("symbol").r30.mean() > 0).sum())
            r["r30_pos_months"] = int((d.groupby("month").r30.mean() > 0).sum())
            rows.append(r)
            print(r)
    save("drift", pd.DataFrame(rows))


def grid():
    """Parameter search for the fade strategy: choose on Oct-Mar, check on Apr-Sep."""
    rows = []
    space = itertools.product(
        ["atr_amplitude", "quantile"],
        [2, 3, 4, 6],
        [5, 15, 30, 60],
        [0.6, 1.2, 3.0],
        ["with", "against"],
    )
    for model, k, hold, sl, trend in space:
        if model == "quantile" and k > 3:
            continue
        cfg = Cfg(model=model, k=k, max_hold=hold, sl=sl, trend=trend)
        tr = run(cfg)
        r = {"model": model, "k": k, "hold": hold, "sl": sl, "trend": trend}
        for scope, d in (("is", tr[~tr.oos]), ("oos", tr[tr.oos])):
            s = summary(d)
            r[f"{scope}_n"], r[f"{scope}_avg"] = s["trades"], s.get("avg_net", 0)
            r[f"{scope}_gross"] = s.get("avg_gross", 0)
            r[f"{scope}_usdt"] = s["net_usdt"]
        r["is_ex_crash_avg"] = tr[~tr.oos & (tr.month != CRASH)].net.mean()
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows).sort_values("is_avg", ascending=False)
    save("grid_fade", df)
    print(df.head(15))


def factors():
    """Which conditions at the signal make a fill better or worse (k=3 fills)."""
    fw = forward(Cfg())
    fw["month"] = pd.to_datetime(fw.t, unit="ms").dt.strftime("%Y-%m")
    fw = fw[fw.month != CRASH]
    fw["hour"] = pd.to_datetime(fw.t, unit="ms").dt.hour
    out = []
    names = {
        "vol_ratio": "成交量放大倍數",
        "atr_ratio": "波動率相對當日平均",
        "atr_pct": "每分鐘波動%",
        "ema_gap": "價格離趨勢線距離",
        "ret60": "過去 60 分鐘漲跌",
        "dist": "掛單距離%",
    }
    for f in names:
        x = fw[f] * (fw.side if f in ("ema_gap", "ret60") else 1)
        b = pd.qcut(x.rank(method="first"), 5, labels=False)
        for (q, oos), d in fw.groupby([b, fw.oos]):
            out.append(
                {
                    "factor": f,
                    "bucket": int(q) + 1,
                    "oos": bool(oos),
                    "lo": float(x[d.index].min()),
                    "hi": float(x[d.index].max()),
                    "n": len(d),
                    "r5": d.r5.mean(),
                    "r30": d.r30.mean(),
                    "r30_median": d.r30.median(),
                }
            )
    df = pd.DataFrame(out)
    save("factors", df)
    print(df.pivot_table(index=["factor", "bucket"], columns="oos", values="r30").round(3))


def breakout():
    """Same signal, opposite trade: stop order with the move, taker in and out."""
    rows = []
    space = itertools.product([3, 4, 6, 8], [5, 15, 30, 60], [0.6, 1.2, 3.0], ["with", "against"])
    for k, hold, sl, trend in space:
        cfg = Cfg(k=k, max_hold=hold, sl=sl, trend=trend, mode="breakout")
        tr = run(cfg)
        r = {"k": k, "hold": hold, "sl": sl, "trend": trend}
        for scope, d in (
            ("is", tr[~tr.oos]),
            ("is_xc", tr[~tr.oos & (tr.month != CRASH)]),
            ("oos", tr[tr.oos]),
        ):
            s = summary(d)
            r[f"{scope}_n"], r[f"{scope}_avg"] = s["trades"], s.get("avg_net", 0)
            r[f"{scope}_usdt"] = s["net_usdt"]
        r["oos_pos_months"] = int((tr[tr.oos].groupby("month").net.sum() > 0).sum())
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows).sort_values("is_xc_avg", ascending=False)
    save("grid_breakout", df)
    print(df.head(15))


def screen():
    """Does a coin's first half predict its second half? (baseline config)"""
    tr = pd.read_pickle(RES / "baseline_trades.pkl")
    tr = tr[tr.month != CRASH]
    a = tr[~tr.oos].groupby("symbol").net.agg(["mean", "size"])
    b = tr[tr.oos].groupby("symbol").net.agg(["mean", "size"])
    df = a.join(b, lsuffix="_is", rsuffix="_oos").dropna()
    # coin traits measured on the first half only
    df["atr_is"] = tr[~tr.oos].groupby("symbol").atr_pct.median()
    rho = df["mean_is"].rank().corr(df["mean_oos"].rank())
    top = df.sort_values("mean_is", ascending=False).head(5)
    out = {
        "spearman_is_vs_oos": rho,
        "top5_is": list(top.index),
        "top5_oos_avg": float(tr[tr.oos & tr.symbol.isin(top.index)].net.mean()),
        "all_oos_avg": float(tr[tr.oos].net.mean()),
        "coins_positive_both": list(df[(df.mean_is > 0) & (df.mean_oos > 0)].index),
    }
    save("screen", df)
    save("screen_summary", out)
    print(df.sort_values("mean_is"), json.dumps(out, indent=1, default=float), sep="\n")


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "config.json").write_text(json.dumps(asdict(replace(Cfg())), indent=1))
    for name in sys.argv[1:] or ["baseline"]:
        print(f"\n######## {name} (universe {len(UNIVERSE)} coins)")
        globals()[name]()
    np.seterr(all="ignore")
