"""Follow-up: fade only extreme moves in the trend direction (k >= 10, wide stop).

Usage: uv run python research/extreme.py
"""

import json

import pandas as pd
from experiments import CRASH, RES, save
from lab import run, table
from sim import Cfg, summary

CFG = Cfg(k=12, trend="against", max_hold=30, sl=3.0)


def main():
    rows = []
    for k in (6, 8, 10, 12, 15, 20):
        for sl in (0.6, 1.2, 3.0):
            tr = run(Cfg(k=k, trend="against", max_hold=30, sl=sl))
            x = tr[tr.month != CRASH]
            rows.append(
                {
                    "k": k,
                    "sl": sl,
                    "n": len(tr),
                    "is_avg": x[~x.oos].net.mean(),
                    "oos_avg": x[x.oos].net.mean(),
                    "crash_avg": tr[tr.month == CRASH].net.mean(),
                    "usdt": tr.net.sum() * CFG.notional / 100,
                    "pos_months": int((tr.groupby("month").net.sum() > 0).sum()),
                    "pos_coins": int((tr.groupby("symbol").net.sum() > 0).sum()),
                }
            )
            print(rows[-1], flush=True)
    save("extreme_grid", pd.DataFrame(rows))

    tr = run(CFG)
    tr.to_pickle(RES / "extreme_trades.pkl")
    by_coin = table(tr, "symbol").sort_values("avg_net%")
    by_month = table(tr, "month")
    usdt = CFG.notional / 100
    out = {
        "all": summary(tr),
        "is": summary(tr[~tr.oos]),
        "oos": summary(tr[tr.oos]),
        "by_side": table(tr, "side").to_dict(),
        "reasons": table(tr, "reason").to_dict(),
        "worst_trade_usdt": float(tr.net.min() * usdt),
        "best_trade_usdt": float(tr.net.max() * usdt),
        "trades_per_coin_month": len(tr) / tr.symbol.nunique() / 12,
        "usdt_per_coin_month": float(tr.net.sum() * usdt / tr.symbol.nunique() / 12),
    }
    save("extreme", out)
    save("extreme_by_symbol", by_coin)
    save("extreme_by_month", by_month)
    print(json.dumps(out, indent=1, default=float), by_coin, by_month, sep="\n\n")


if __name__ == "__main__":
    main()
