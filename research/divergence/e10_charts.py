"""E10 position-tool charts: typical (median-closest) 2024-26 trades on large coins for one configuration,
per (timeframe, side, win/loss) spec; candle size picked per trade from the swing duration.
Usage: PYTHONPATH=research/divergence python research/divergence/e10_charts.py OUT_DIR STOP RR TF:SIDE:KIND ...
  e.g. 0.5 1.5 1h:-1:win 4h:1:loss
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import e8_charts as G


def main(out, stop_k, rr, specs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle("data/research/divergence_e10_all.pkl")
    D = D[D.rule3 & (D.exit == "body") & (D.stop_k == stop_k) & (D.rr == rr) & D.sym.isin(G.BIG) & (D.t >= C.T_OOS)]
    rows = []
    for k, spec in enumerate(specs, 1):
        tf, side, kind = spec.split(":")
        g = D[(D.tf == tf) & (D.side == int(side))]
        g = g[g.net > 0] if kind == "win" else g[g.net <= 0]
        if not len(g):
            continue
        r = next(g.iloc[np.argsort(np.abs(g.net - g.net.median()).to_numpy())].itertuples())
        p = out / f"{k}_{tf}_{'long' if r.side > 0 else 'short'}_{kind}_{r.sym}.png"
        G.chart(r, p, f"理論盈虧比 {rr:g}", disp="auto")
        rows.append({"file": p.name, "t_in": G.ts(r.t_in), "px_in": r.px_in, "t_out": G.ts(r.t_out), "px_out": r.px_out,
                     "stop": r.stop, "tp1": r.tp1, "net": round(100 * r.net, 2), "R": round(r.R, 2),
                     "why": G.WHY.get(int(r.why))})  # fmt: skip
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
