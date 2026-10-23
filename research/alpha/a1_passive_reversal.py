"""A1: does a resting buy far below a falling price earn a real reversal, or just a risk premium?

Every passive limit order of the rule family (side x trend x depth k), on 40 coins and 24
months, with markouts after the fill. Controls use the same passive mechanics:
  depth curve       the same order nearer (k=3..9) and further (k=15, 20)
  mirror            sell limits above a rising price (S_up), and both sides against the trend
  fixed depth       the same depth in plain % instead of volatility units
The cost of the round trip is maker in + taker out + slippage = 0.09%.

Writes data/research/runs/A1_passive_reversal/ (fills.csv.gz is the full event log) and
copies the tables to research/results/A1_passive_reversal/.

Usage: uv run python research/alpha/a1_passive_reversal.py
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import fmt, folder, market, market_days, publish, save_json
from passive import FEATS, HORIZONS, events
from stats import boot_mean, summarize, tail_share

from data import GRID, UNIVERSE, Y2

EXP = "A1_passive_reversal"
COST = 0.02 + 0.05 + 0.02
KS = (3, 6, 9, 12, 15, 20)
RULES = [
    {"name": f"{nm}_k{k}", "side": sd, "trend": tr, "k": k}
    for nm, sd, tr in (
        ("L_dn", 1, "down"),
        ("L_up", 1, "up"),
        ("S_up", -1, "up"),
        ("S_dn", -1, "down"),
    )
    for k in KS
] + [{"name": "L_dn_fixed_k12", "side": 1, "trend": "down", "k": 12, "fixed": True}]
LEAD = "L_dn_k12"


def _one(sym):
    return events(sym, market(), RULES)


def depth_curve(fx: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    months = 24
    for rule, g in fx.groupby("rule", sort=False):
        r = {"rule": rule, "fills": len(g), "episodes": int(g["first"].sum())}
        r["fills_per_coin_month"] = len(g) / len(UNIVERSE) / months
        oc = counts[counts.rule == rule]
        r["fill_rate"] = oc.filled.sum() / max(oc.placed.sum(), 1)
        for H in ("m0", "m5", "m30", "m60", "m240", "x_m30", "mfe30", "mae30"):
            r[H] = g[H].mean()
        m, lo, hi = boot_mean(g.m30 - COST, g.day.to_numpy())
        r.update({"net30": m, "net30_lo": lo, "net30_hi": hi})
        net = g.m30 - COST
        r["net30_median"] = net.median()
        r["net30_day_mean"] = net.groupby(g.day).mean().mean()  # every day counts once
        r["net30_ex_crash"] = net[g.event != "crash"].mean()
        r["share_from_crash_days"] = (
            net[g.event == "crash"].sum() / net.sum() if net.sum() else np.nan
        )
        r["net30_y1"] = (g[~g.y2].m30 - COST).mean()
        r["net30_y2"] = (g[g.y2].m30 - COST).mean()
        m, lo, hi = boot_mean(g.x_m30, g.day.to_numpy())
        r.update({"after_bar_x30": m, "after_bar_lo": lo, "after_bar_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def risk_vs_reversal(g: pd.DataFrame) -> dict:
    """Separate the bounce inside the fill bar, the drift after it, and the market exposure."""
    out = {}
    x = g.m30.to_numpy()
    mk = g.mk30.to_numpy()  # signed market move from fill bar close to +30m
    ok = np.isfinite(x) & np.isfinite(mk)
    A = np.c_[np.ones(ok.sum()), mk[ok]]
    coef, *_ = np.linalg.lstsq(A, x[ok], rcond=None)
    out["ols_m30_on_market"] = {"alpha": coef[0], "beta": coef[1]}
    out["parts_mean"] = {
        "fill_to_bar_close_m0": g.m0.mean(),
        "bar_close_to_30m_raw": (g.m30 - g.m0).mean(),
        "bar_close_to_30m_market_hedged": g.x_m30.mean(),
        "market_part": (g.m30 - g.m0 - g.x_m30).mean(),
    }
    out["distribution_net30"] = tail_share(x - COST)
    # market wide dumps (the index fell at least half as much as the coin in the fill bar)
    wide = g.mkt_fill_bar <= 0.5 * g.fill_bar_ret
    for nm, s in (("market_wide", g[wide]), ("idiosyncratic", g[~wide])):
        m, lo, hi = boot_mean(s.m30 - COST, s.day.to_numpy())
        out[nm] = {
            "n": len(s),
            "net30": m,
            "ci": [lo, hi],
            "m0": s.m0.mean(),
            "after_bar_hedged": s.x_m30.mean(),
        }
    # when the market keeps falling after the fill, how bad is it? (risk premium signature)
    q = pd.qcut(g.mk30, 5, labels=False, duplicates="drop")
    out["by_market_after_quintile"] = (
        g.assign(q=q)
        .groupby("q")
        .agg(n=("m30", "size"), mk30=("mk30", "mean"), net30=("m30", "mean"))
        .assign(net30=lambda d: d.net30 - COST)
        .reset_index()
        .to_dict("records")
    )
    return out


def penetration(g: pd.DataFrame) -> pd.DataFrame:
    """How far the price traded through the limit (ex post): a queued maker gets the deep ones."""
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1, 100]
    d = g.assign(pen_bin=pd.cut(g.pen, bins, include_lowest=True).astype(str))
    t = summarize(d, "m30", "pen_bin", cost=COST)
    t2 = d.groupby("pen_bin").agg(m0=("m0", "mean"), x_m30=("x_m30", "mean")).reset_index()
    return t.merge(t2, on="pen_bin")


def feature_buckets(g: pd.DataFrame) -> pd.DataFrame:
    """Quintile cuts from year 1 fills, applied unchanged to year 2."""
    rows = []
    y1, y2 = g[~g.y2], g[g.y2]
    for feat in FEATS:
        a = y1[feat].to_numpy()
        if np.isfinite(a).sum() < 100 or np.nanstd(a) == 0:
            continue
        cuts = np.unique(np.nanpercentile(a, [20, 40, 60, 80]))
        for part, s in (("y1", y1), ("y2", y2)):
            b = np.digitize(s[feat].to_numpy(), cuts)
            b = np.where(np.isfinite(s[feat].to_numpy()), b, -1)
            for q in range(len(cuts) + 1):
                x = s[b == q]
                if not len(x):
                    continue
                rows.append(
                    {
                        "feature": feat,
                        "part": part,
                        "q": q,
                        "lo": -np.inf if q == 0 else cuts[q - 1],
                        "hi": np.inf if q == len(cuts) else cuts[q],
                        "n": len(x),
                        "net30": (x.m30 - COST).mean(),
                        "net30_ex_crash": (x[x.event != "crash"].m30 - COST).mean(),
                        "net30_median": (x.m30 - COST).median(),
                        "m0": x.m0.mean(),
                        "x_m30": x.x_m30.mean(),
                    }
                )
    t = pd.DataFrame(rows)
    # spread top minus bottom quintile per feature, both years: a stable factor keeps its sign
    # robust version: crash days removed, so one day cannot make a factor look good
    s = t.pivot_table(index=["feature", "q"], columns="part", values="net30_ex_crash").reset_index()
    sp = s.groupby("feature").apply(
        lambda d: pd.Series(
            {
                "y1_spread": d.y1.iloc[-1] - d.y1.iloc[0],
                "y2_spread": d.y2.iloc[-1] - d.y2.iloc[0],
                "y1_rank_corr": np.corrcoef(d.q, d.y1)[0, 1],
                "y2_rank_corr": np.corrcoef(d.q, d.y2)[0, 1],
            }
        ),
        include_groups=False,
    )
    return t, sp.reset_index().sort_values("y2_spread")


def main():
    out = folder(EXP)
    save_json(
        out / "config.json",
        {
            "exp": EXP,
            "rules": RULES,
            "cost_pct": COST,
            "horizons_min": HORIZONS,
            "symbols": UNIVERSE,
            "period": [
                str(pd.Timestamp(GRID[0], unit="ms")),
                str(pd.Timestamp(GRID[-1], unit="ms")),
            ],
            "year2_from": str(pd.Timestamp(GRID[Y2], unit="ms")),
            "fill_rule": "bar after the order trades through the limit by more than half a tick",
            "features_at": "close of the bar the order was placed in (before it can fill)",
        },
    )
    market()
    with ProcessPoolExecutor(5) as ex:
        res = list(ex.map(_one, UNIVERSE))
    fx = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    fx = fx.merge(market_days(), on="day", how="left")
    counts = pd.DataFrame([c for r in res for c in r[1]])
    fx.to_csv(out / "fills.csv.gz", index=False)
    counts.to_csv(out / "order_counts.csv", index=False)

    dc = depth_curve(fx, counts)
    dc.to_csv(out / "depth_curve.csv", index=False)
    print(fmt(dc))
    lead = fx[fx.rule == LEAD]
    rv = {
        "lead": risk_vs_reversal(lead),
        "mirror_S_up_k12": risk_vs_reversal(fx[fx.rule == "S_up_k12"]),
    }
    save_json(out / "risk_vs_reversal.json", rv)
    print(rv)
    pen = penetration(lead)
    pen.to_csv(out / "penetration.csv", index=False)
    print(fmt(pen))
    deep = fx[fx.rule.isin(["L_dn_k9", "L_dn_k12", "L_dn_k15"])]
    t, sp = feature_buckets(deep)
    t.to_csv(out / "feature_buckets.csv", index=False)
    sp.to_csv(out / "feature_spreads.csv", index=False)
    print(fmt(sp))
    prof = []
    for (rule, ev), g in fx[fx.rule.isin([LEAD, "S_up_k12", "L_up_k12"])].groupby(
        ["rule", "event"]
    ):
        r = {"rule": rule, "event": ev, "n": len(g), "days": g.day.nunique()}
        for H in HORIZONS:
            r[f"net{H}"] = g[f"m{H}"].mean() - COST
            r[f"hedged_after_bar{H}"] = g[f"x_m{H}"].mean()
        prof.append(r)
    prof = pd.DataFrame(prof)
    prof.to_csv(out / "profile_by_event.csv", index=False)
    print(fmt(prof))
    summarize(lead, "m30", "month", cost=COST).to_csv(out / "lead_by_month.csv", index=False)
    summarize(lead, "m30", "symbol", cost=COST).to_csv(out / "lead_by_coin.csv", index=False)
    publish(
        EXP,
        ["config.json", "order_counts.csv", "depth_curve.csv", "risk_vs_reversal.json",
         "penetration.csv", "profile_by_event.csv", "feature_buckets.csv", "feature_spreads.csv", "lead_by_month.csv",  # noqa: E501
         "lead_by_coin.csv"],
    )  # fmt: skip


if __name__ == "__main__":
    main()
