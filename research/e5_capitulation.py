"""E5: inside the extreme-dump long, which conditions carry the edge? (capitulation filters)

Thresholds come from year 1 (2024-10 to 2025-09) lead trades only, tercile cuts:
  volume surge  vol_ratio >= 1.07 (5m volume vs 4h average before the order)
  already fell  ret60 <= -0.26%   (price change over the 60 minutes before the order)
  volatile coin atr_pct >= 0.081% (average 1m true range)
Year 2 (2025-10 to 2026-09) is the check. Filtered bars are logged as rejected:filter.

Usage: uv run python research/e5_capitulation.py
"""

import json

import pandas as pd
from download import month_range
from runs import group, path, publish
from sim import Cfg

EXP = "E5_capitulation"
ALL = month_range("2024-10:2026-09")
BASE = dict(k=12, trend="against", sides="long", sl=3.0, max_hold=30)
VOL, DROP, ATR = {"min_vol_ratio": 1.07}, {"max_ret60": -0.26}, {"min_atr_pct": 0.081}


def cfg(extra: dict | None = None, **kw) -> Cfg:
    return Cfg(**{**BASE, **kw}, extra={"months": ALL, **(extra or {})})


GROUPS = [
    ("L0_lead", cfg(), False, "線索設定（對照基準）"),
    ("V_volume", cfg(VOL), False, "加條件：下單前成交量已放大"),
    ("D_drop", cfg(DROP), False, "加條件：下單前 60 分鐘已下跌"),
    ("A_atr", cfg(ATR), False, "加條件：幣種平常波動夠大"),
    ("VD", cfg(VOL | DROP), False, "成交量放大 + 已下跌"),
    ("VDA", cfg(VOL | DROP | ATR), False, "三個條件都要"),
    ("VDA_ctrl", cfg(VOL | DROP | ATR), True, "對照：同樣三個條件成立的時段隨機進場"),
    ("VDA_k8", cfg(VOL | DROP | ATR, k=8), False, "三個條件 + 掛近一點 k=8"),
]


def main():
    rows, n_ctrl = [], {}
    for name, c, control, note in GROUPS:
        tr = group(
            EXP, name, c, control=control, n_ctrl=n_ctrl, note=note, log=name in ("L0_lead", "VDA")
        )
        if not control:
            n_ctrl = tr.groupby("symbol").size().to_dict()
        s = json.loads((path(EXP) / name / "summary.json").read_text())
        e = s["expectancy"]
        y1, y2 = tr[tr.month < "2025-10"], tr[tr.month >= "2025-10"]
        rows.append(
            {
                "group": name,
                "note": note,
                "trades": e["trades"],
                "avg_net%": s["avg_net_pct"],
                "y1_avg%": y1.net.mean(),
                "y2_avg%": y2.net.mean(),
                "y2_usdt": y2.net_usdt.sum(),
                "win_rate": e["win_rate"],
                "payoff": e["payoff"],
                "net_usdt": s["net_usdt"],
                "pos_months": f"{s['pos_months']}/{s['months']}",
                "pos_coins": f"{s['pos_coins']}/{s['coins']}",
                "max_dd_usdt": float(
                    (
                        tr.sort_values("t").net_usdt.cumsum().cummax()
                        - tr.sort_values("t").net_usdt.cumsum()
                    ).max()
                ),
            }
        )
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path(EXP) / "table.csv", index=False)
    publish(EXP)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
