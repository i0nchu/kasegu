"""A3: the extreme passive reversal as a tradable rule, one position per coin, full order logs.

Rule fixed from A1 before this run (no search here): long only, buy limit k=12 volatility
units below the ask while the close is under EMA240, re-placed every minute, post-only.
Exit: market sell 15 minutes after the fill (the A1 reversal is complete by then; longer
holds mostly add market exposure), or the 3% exchange stop. Taker exit + 0.02% slippage.

Groups (one change each against EPR15):
  hold      5 or 30 minute exit
  slip      0.05 / 0.1 / 0.2% exit slippage, and 0.3% on the stop
  trend     no trend filter (any time)
  ctrl      random entries in the same regime, same count per coin, same exit
40 coins, 2024-10 to 2026-09. Output follows research/runs.py: config.json, trades.csv,
candidates/ (every order: filled / expired / rejected:<why>), summary.json.

Usage: uv run python research/alpha/a3_candidate.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import fmt, market_days  # noqa: E402
from download import month_range  # noqa: E402
from runs import expectancy, group, path, publish  # noqa: E402
from sim import Cfg  # noqa: E402
from stats import boot_mean  # noqa: E402

from data import UNIVERSE  # noqa: E402

EXP = "A3_candidate"
ALL = month_range("2024-10:2026-09")
BASE = dict(k=12, trend="against", sides="long", sl=3.0, exit="time", max_hold=15)


def cfg(**kw) -> Cfg:
    return Cfg(**{**BASE, **kw}, extra={"months": ALL})


GROUPS = [
    ("EPR15", "base", cfg(), "候選規則：k=12 掛買單、跌勢中、15 分鐘出場、3% 停損"),
    ("hold5", "hold", cfg(max_hold=5), "5 分鐘出場"),
    ("hold30", "hold", cfg(max_hold=30), "30 分鐘出場"),
    ("slip0.05", "slip", cfg(slip=0.05), "出場滑價 0.05%"),
    ("slip0.1", "slip", cfg(slip=0.1), "出場滑價 0.1%"),
    ("slip0.2", "slip", cfg(slip=0.2, level_slip=0.3), "出場滑價 0.2%、停損再多滑 0.3%"),
    ("no_trend", "trend", cfg(trend="none"), "不看趨勢，任何時候都掛"),
    ("ctrl_random", "ctrl", cfg(), "對照：同時段同方向隨機進場，同出場"),
]


def main():
    days = market_days()
    rows, n_ctrl = [], {}
    for name, kind, c, note in GROUPS:
        tr = group(
            EXP,
            name,
            c,
            symbols=UNIVERSE,
            control=kind == "ctrl",
            n_ctrl=n_ctrl,
            note=note,
            log=name == "EPR15",
        )
        if name == "EPR15":
            n_ctrl = tr.groupby("symbol").size().to_dict()
        tr["day"] = tr.t // 86_400_000
        tr = tr.merge(days, on="day", how="left")
        tr.to_csv(path(EXP) / name / "trades.csv", index=False)
        e = expectancy(tr.net)
        m, lo, hi = boot_mean(tr.net.to_numpy(), tr.day.to_numpy())
        x = tr[tr.event != "crash"]
        mx, lox, hix = boot_mean(x.net.to_numpy(), x.day.to_numpy())
        rows.append(
            {
                "group": name,
                "kind": kind,
                "note": note,
                "trades": len(tr),
                "per_coin_month": len(tr) / len(UNIVERSE) / 24,
                "avg_net%": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "avg_net_ex_crash%": mx,
                "ex_crash_lo": lox,
                "ex_crash_hi": hix,
                "median_net%": tr.net.median(),
                "y1_avg%": tr[tr.month < "2025-10"].net.mean(),
                "y2_avg%": tr[tr.month >= "2025-10"].net.mean(),
                "win_rate": e["win_rate"],
                "payoff": e["payoff"],
                "net_usdt": tr.net_usdt.sum(),
                "crash_share": tr[tr.event == "crash"].net.sum() / tr.net.sum()
                if tr.net.sum()
                else None,
                "pos_months": f"{int((tr.groupby('month').net.sum() > 0).sum())}/{tr.month.nunique()}",  # noqa: E501
                "pos_months_ex_crash": f"{int((x.groupby('month').net.sum() > 0).sum())}/{x.month.nunique()}",  # noqa: E501
                "pos_coins": f"{int((tr.groupby('symbol').net.sum() > 0).sum())}/{tr.symbol.nunique()}",  # noqa: E501
                "stop_exits": int((tr.reason == "stop_loss").sum()),
            }
        )
        print(fmt(pd.DataFrame(rows[-1:])), flush=True)
        if name == "EPR15":
            by_ev = tr.groupby("event").agg(
                trades=("net", "size"), avg_net=("net", "mean"), sum_net=("net", "sum")
            )
            by_ev.to_csv(path(EXP) / "EPR15_by_event.csv")
            bm = tr.groupby("month").agg(
                trades=("net", "size"), avg_net=("net", "mean"), sum_usdt=("net_usdt", "sum")
            )
            bm["avg_net_ex_crash"] = x.groupby("month").net.mean()
            bm.to_csv(path(EXP) / "EPR15_by_month.csv")
            tr.groupby("reason").agg(trades=("net", "size"), avg_net=("net", "mean")).to_csv(
                path(EXP) / "EPR15_exit_reasons.csv"
            )
    t = pd.DataFrame(rows)
    t.to_csv(path(EXP) / "table.csv", index=False)
    publish(EXP)
    for f in ("EPR15_by_event.csv", "EPR15_by_month.csv", "EPR15_exit_reasons.csv"):
        (Path("research/results") / EXP / f).write_text((path(EXP) / f).read_text())
    print(fmt(t))


if __name__ == "__main__":
    main()
