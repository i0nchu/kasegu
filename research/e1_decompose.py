"""E1: is the loss from the entry, the exit, or the expectancy math itself?

Same 28 coins x 12 months. Treatment groups keep the engine entry and swap the exit;
control groups keep the exits but enter at random minutes (same coin, same trend side,
same number of entries), so whatever they earn is what the exit rules alone produce.

Usage: uv run python research/e1_decompose.py
"""

import json

import pandas as pd
from runs import group, path
from sim import Cfg

EXP = "E1_decompose"
BASE = Cfg()  # engine baseline: atr_amplitude k=3, EMA 240, SL 0.6, tiers, 30m


def main():
    groups = {
        "G0_baseline": (BASE, False, "引擎現行設定（對照基準）"),
        "G1_time_exit": (Cfg(exit="time"), False, "同進場，只留 0.6% 停損 + 30 分鐘時間出場"),
        "G2_bracket": (Cfg(exit="bracket", tp=0.6), False, "同進場，固定 +0.6% 停利 / -0.6% 停損"),
        "G3_hold_only": (Cfg(exit="time", sl=50), False, "同進場，不停損，固定抱 30 分鐘"),
        "G4_zero_cost": (
            Cfg(maker=0, taker=0, slip=0, level_slip=0),
            False,
            "現行設定但手續費、滑價全為 0（只看價格本身）",
        ),
        "C0_random_baseline_exit": (BASE, True, "隨機進場 + 現行出場（對照組）"),
        "C4_random_zero_cost": (
            Cfg(maker=0, taker=0, slip=0, level_slip=0),
            True,
            "隨機進場 + 現行出場，成本全為 0（對照組）",
        ),
        "C3_random_hold_only": (
            Cfg(exit="time", sl=50),
            True,
            "隨機進場 + 固定抱 30 分鐘（對照組）",
        ),
    }
    n_ctrl: dict = {}
    rows = []
    for name, (cfg, control, note) in groups.items():
        tr = group(EXP, name, cfg, control=control, n_ctrl=n_ctrl, note=note)
        if name == "G0_baseline":
            n_ctrl = tr.groupby("symbol").size().to_dict()
        s = json.loads((path(EXP) / name / "summary.json").read_text())
        e = s["expectancy"]
        rows.append(
            {
                "group": name,
                "note": note,
                "trades": e["trades"],
                "avg_gross%": s["avg_gross_pct"],
                "cost%": s["cost_pct"],
                "avg_net%": s["avg_net_pct"],
                "win_rate": e["win_rate"],
                "avg_win%": e["avg_win_pct"],
                "avg_loss%": e["avg_loss_pct"],
                "payoff": e["payoff"],
                "breakeven_win": e["breakeven_win_rate"],
                "net_usdt": s["net_usdt"],
                "pos_months": s["pos_months"],
            }
        )
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path(EXP) / "table.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
