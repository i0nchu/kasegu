"""E2: does the extreme-dump long survive data it has never seen?

Holdout A: the same 28 coins, 2024-10 to 2025-09 (the 12 months before the research window).
Holdout B: 12 coins that were never in the research, 2024-10 to 2026-09.
Controls enter at random minutes on the same side (long while below the EMA) with the same exits,
so any edge must come from the timing of the dump, not from being long with a wide stop.

Usage: uv run python research/e2_holdout.py
"""

import json

import pandas as pd
from download import month_range
from lab import MONTHS, UNIVERSE
from runs import group, path, publish
from sim import Cfg

EXP = "E2_holdout"
HOLD = month_range("2024-10:2025-09")
ALL = month_range("2024-10:2026-09")
NEW = [
    "AAVEUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT", "ETCUSDT", "HBARUSDT",
    "XLMUSDT", "PENDLEUSDT", "JUPUSDT", "ONDOUSDT", "1000FLOKIUSDT", "CRVUSDT",
]  # fmt: skip


def lead(months, **kw) -> Cfg:
    # the lead from round 1: long only, below EMA 240, k=12, stop 3%, engine tiers, 30m
    base = dict(k=12, trend="against", sides="long", sl=3.0, max_hold=30)
    return Cfg(**{**base, **kw}, extra={"months": months})


def main():
    groups = [
        (
            "A_lead_holdout",
            lead(HOLD),
            UNIVERSE,
            False,
            "線索設定，28 幣，2024-10～2025-09（未看過）",
        ),
        ("A_ctrl_holdout", lead(HOLD), UNIVERSE, True, "對照：同方向隨機進場，同出場"),
        ("A_both_sides_holdout", lead(HOLD, sides="both"), UNIVERSE, False, "線索但多空都做"),
        ("A_baseline_holdout", Cfg(extra={"months": HOLD}), UNIVERSE, False, "原策略現行設定"),
        ("B_lead_new_coins", lead(ALL), NEW, False, "線索設定，12 個新幣，24 個月"),
        ("B_ctrl_new_coins", lead(ALL), NEW, True, "對照：新幣隨機進場"),
        ("R_lead_research", lead(MONTHS), UNIVERSE, False, "線索設定，原研究期間（參考）"),
        ("R_ctrl_research", lead(MONTHS), UNIVERSE, True, "對照：原研究期間隨機進場"),
    ]
    rows, n_ctrl = [], {}
    for name, cfg, syms, control, note in groups:
        tr = group(EXP, name, cfg, symbols=syms, control=control, n_ctrl=n_ctrl, note=note)
        if not control:
            n_ctrl = tr.groupby("symbol").size().to_dict()
        s = json.loads((path(EXP) / name / "summary.json").read_text())
        e = s["expectancy"]
        rows.append(
            {
                "group": name,
                "note": note,
                "trades": e["trades"],
                "avg_gross%": s["avg_gross_pct"],
                "avg_net%": s["avg_net_pct"],
                "win_rate": e["win_rate"],
                "payoff": e["payoff"],
                "breakeven_win": e["breakeven_win_rate"],
                "net_usdt": s["net_usdt"],
                "pos_months": f"{s['pos_months']}/{s['months']}",
                "pos_coins": f"{s['pos_coins']}/{s['coins']}",
                "worst_usdt": float(tr.net_usdt.min()),
            }
        )
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path(EXP) / "table.csv", index=False)
    publish(EXP)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
