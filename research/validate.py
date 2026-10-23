"""Compare the fast 1m simulator with engine backtest reports on the same days.

Usage: uv run python research/validate.py data/backtests/ref_*.json
"""

import json
import sys
from datetime import date, timedelta

import pandas as pd
from sim import CACHE, Cfg, _read_zip, bars_from_trades, simulate, tick_of


def main(files: list[str]):
    rows = []
    for f in files:
        rep = json.load(open(f))
        s, k = rep["symbols"][0], rep["strategy"]["k"]
        d0, d1 = date.fromisoformat(rep["start"]), date.fromisoformat(rep["end"])
        warm = _read_zip(CACHE / f"daily/klines/{s}/1m/{s}-1m-{d0 - timedelta(1)}.zip")
        days = [str(d0 + timedelta(i)) for i in range((d1 - d0).days + 1)]
        df = bars_from_trades(s, days, warm)
        start = int(pd.Timestamp(d0, tz="UTC").timestamp() * 1000)
        tr = simulate(df, Cfg(k=k, tick=tick_of(s)), start_ms=start)
        eng = rep["trades"]
        e_net = sum(t["pnl"] - t["fees"] + t["funding"] for t in eng)
        notional = 297
        # match trades by entry minute
        em = {t["open_time"] // 60000 * 60000: t for t in eng}
        both = [t for t in tr.t if t in em]
        rows.append(
            {
                "symbol": s,
                "eng_n": len(eng),
                "sim_n": len(tr),
                "matched": len(both),
                "eng_net": round(e_net, 2),
                "sim_net": round(tr.net.sum() * notional / 100, 2) if len(tr) else 0,
                "eng_avg%": round(e_net / notional * 100 / max(len(eng), 1), 3),
                "sim_avg%": round(tr.net.mean(), 3) if len(tr) else 0,
            }
        )
        if "-v" in sys.argv:
            for t in eng:
                m = t["open_time"] // 60000 * 60000
                r = tr[tr.t == m]
                sim = f"{r.reason.iloc[0]:>10} {r.net.iloc[0]:+.3f}" if len(r) else "      --"
                en = (t["pnl"] - t["fees"]) / 297 * 100
                print(f"{s} {pd.Timestamp(m, unit='ms')} {t['reason']:>13} {en:+.3f} | {sim}")
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(out[["eng_n", "sim_n", "matched", "eng_net", "sim_net"]].sum().to_string())


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("-")])
