"""E3: how fragile is the extreme-dump long?

All groups: 28 coins, 24 months (2024-10 to 2026-09). One knob changes per group against L0.
  fill  : the price must trade X% beyond the limit before it counts as filled (queue risk)
  slip  : extra loss on stop / trailing exits during the crash (stops slip in fast markets)
  param : neighbours of k, stop, hold, EMA and volatility window
  exit  : same entries, different exits; random-entry controls with the same exits

Usage: uv run python research/e3_robust.py
"""

import json

import pandas as pd
from download import month_range
from lab import UNIVERSE
from runs import group, path, publish
from sim import Cfg

EXP = "E3_robust"
ALL = month_range("2024-10:2026-09")
BASE = dict(k=12, trend="against", sides="long", sl=3.0, max_hold=30)


def cfg(**kw) -> Cfg:
    return Cfg(**{**BASE, **kw}, extra={"months": ALL})


GROUPS = [
    ("L0_lead", "base", cfg(), "線索設定（對照基準）"),
    ("F1_pen0.05", "fill", cfg(penetrate=0.05), "價格需再穿過掛單價 0.05% 才算成交"),
    ("F2_pen0.1", "fill", cfg(penetrate=0.1), "需穿過 0.1%"),
    ("F3_pen0.2", "fill", cfg(penetrate=0.2), "需穿過 0.2%"),
    ("F4_pen0.5", "fill", cfg(penetrate=0.5), "需穿過 0.5%"),
    ("S1_slip0.1", "slip", cfg(level_slip=0.1), "停損 / 移動停利出場多滑 0.1%"),
    ("S2_slip0.3", "slip", cfg(level_slip=0.3), "多滑 0.3%"),
    ("S3_slip0.3_pen0.2", "slip", cfg(level_slip=0.3, penetrate=0.2), "多滑 0.3% 且需穿過 0.2%"),
    ("P_k10", "param", cfg(k=10), "k=10"),
    ("P_k15", "param", cfg(k=15), "k=15"),
    ("P_sl2", "param", cfg(sl=2.0), "停損 2%"),
    ("P_sl5", "param", cfg(sl=5.0), "停損 5%"),
    ("P_hold15", "param", cfg(max_hold=15), "最長持倉 15 分"),
    ("P_hold60", "param", cfg(max_hold=60), "最長持倉 60 分"),
    ("P_ema120", "param", cfg(ema=120), "趨勢線 EMA 120"),
    ("P_ema480", "param", cfg(ema=480), "趨勢線 EMA 480"),
    ("P_vol30", "param", cfg(vol=30), "波動視窗 30 分"),
    ("P_vol120", "param", cfg(vol=120), "波動視窗 120 分"),
    ("P_no_trend", "param", cfg(trend="none"), "不看趨勢，任何時候都接極端急殺"),
    ("X_time", "exit", cfg(exit="time"), "出場：只留 3% 停損 + 30 分"),
    ("X_tp1", "exit", cfg(exit="bracket", tp=1.0), "出場：固定 +1% 停利 / -3% 停損"),
    ("X_tp2", "exit", cfg(exit="bracket", tp=2.0, max_hold=60), "出場：+2% 停利 / -3%，60 分"),
    ("X_hold120", "exit", cfg(exit="time", sl=50, max_hold=120), "出場：不停損，抱 120 分"),
    ("XC_lead_exit", "exit_ctrl", cfg(), "對照：隨機進場 + 線索出場"),
    ("XC_tp1", "exit_ctrl", cfg(exit="bracket", tp=1.0), "對照：隨機進場 + 固定 +1% / -3%"),
]


def main():
    rows, n_ctrl = [], {}
    for name, kind, c, note in GROUPS:
        control = kind == "exit_ctrl"
        tr = group(
            EXP,
            name,
            c,
            symbols=UNIVERSE,
            control=control,
            n_ctrl=n_ctrl,
            note=note,
            log=name == "L0_lead",
        )
        if name == "L0_lead":
            n_ctrl = tr.groupby("symbol").size().to_dict()
        s = json.loads((path(EXP) / name / "summary.json").read_text())
        e = s["expectancy"]
        h1 = tr[tr.month < "2025-10"].net.mean()
        h2 = tr[tr.month >= "2025-10"].net.mean()
        rows.append(
            {
                "group": name,
                "kind": kind,
                "note": note,
                "trades": e["trades"],
                "avg_net%": s["avg_net_pct"],
                "y1_avg%": h1,
                "y2_avg%": h2,
                "win_rate": e["win_rate"],
                "payoff": e["payoff"],
                "net_usdt": s["net_usdt"],
                "pos_months": f"{s['pos_months']}/{s['months']}",
                "pos_coins": f"{s['pos_coins']}/{s['coins']}",
            }
        )
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path(EXP) / "table.csv", index=False)
    publish(EXP)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
