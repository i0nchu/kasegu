"""E9 position-tool charts: per signal timeframe, the typical (median-closest) winner and loser on large coins
in 2024-26, long and short; candle size picked per trade from the swing duration (chart disp="auto").
Usage: PYTHONPATH=research/divergence python research/divergence/e9_charts.py OUT_DIR TF X EXIT [TF X EXIT ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import e8_charts as G


def main(out, specs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle("data/research/divergence_e9.pkl")
    rows = []
    k = 0
    for tf, x, ex in specs:
        g = D[(D.tf == tf) & (D.x == x) & (D.exit == ex) & D.sym.isin(G.BIG) & (D.t >= C.T_OOS)]
        for side in (1, -1):
            for kind, sub in (("win", g[(g.side == side) & (g.net > 0)]), ("loss", g[(g.side == side) & (g.net <= 0)])):
                if not len(sub):
                    continue
                r = next(sub.iloc[np.argsort(np.abs(sub.net - sub.net.median()).to_numpy())].itertuples())
                k += 1
                p = out / f"{k}_{tf}_{'long' if side > 0 else 'short'}_{kind}_{r.sym}.png"
                G.chart(r, p, disp="auto")
                rows.append({"file": p.name, "t_in": G.ts(r.t_in), "px_in": r.px_in, "t_out": G.ts(r.t_out),
                             "px_out": r.px_out, "stop": r.stop, "tp1": r.tp1, "net": round(100 * r.net, 2),
                             "R": round(r.R, 2), "why": G.WHY.get(int(r.why))})  # fmt: skip
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    a = sys.argv[2:]
    main(sys.argv[1], [(a[i], float(a[i + 1]), a[i + 2]) for i in range(0, len(a), 3)])
