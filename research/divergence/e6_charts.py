"""Charts of representative E6 trades (4h short, market entry at the trigger, stop H1 + 1 ATR,
half at L1 then breakeven, rest to the measured move): typical winners and losers on large coins.
Usage: PYTHONPATH=research/divergence python research/divergence/e6_charts.py OUT_DIR
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

BIG = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
       "AVAXUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT"]  # fmt: skip
WHY = {0: "time exit", 1: "stopped before L1", 2: "half at L1, rest breakeven", 3: "half at L1, rest at target",
       4: "half at L1, rest time exit"}  # fmt: skip


def chart(row, path):
    X = C.load(row.sym)
    B = C.resample(X, 48)
    t = B["t"]
    i0 = max(np.searchsorted(t, row.a) - 12, 0)
    i1 = min(np.searchsorted(t, row.t_out) + 12, len(t))
    sl = slice(i0, i1)
    tt = pd.to_datetime(t[sl], unit="ms")
    o, h, l, c = B["o"][sl], B["h"][sl], B["l"][sl], B["c"][sl]
    line = C.ema(B["c"], 12) - C.ema(B["c"], 26)
    sig = C.ema(line, 9)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    w = pd.Timedelta(hours=2.6)
    for k in range(len(tt)):
        col = "#26a69a" if c[k] >= o[k] else "#ef5350"
        ax.vlines(tt[k], l[k], h[k], color=col, lw=0.8)
        ax.bar(tt[k], abs(c[k] - o[k]) or 1e-12, bottom=min(o[k], c[k]), width=w, color=col)

    def ts(x):
        return pd.to_datetime(x, unit="ms")

    for name, tx, px, mk in (("L0", row.a, row.L0, "^"), ("H1", row.b, row.H1, "v"), ("L1", row.ci, row.L1, "^")):
        ax.annotate(name, (ts(tx) + pd.Timedelta(hours=2), px), textcoords="offset points",
                    xytext=(0, 10 if mk == "v" else -16), ha="center", fontsize=10, weight="bold")  # fmt: skip
    ax.axhline(row.stop, color="#c62828", ls="--", lw=1, label=f"stop {row.stop:.4g}")
    ax.axhline(row.L1, color="#1565c0", ls="--", lw=1, label=f"TP1 = L1 {row.L1:.4g}")
    mm = row.L1 - (row.H1 - row.L1)
    ax.axhline(mm, color="#6a1b9a", ls=":", lw=1, label=f"TP2 measured move {mm:.4g}")
    ax.axvline(ts(row.t) + pd.Timedelta(hours=2), color="grey", ls=":", lw=1)
    for x in (0.5, 0.75, 0.9):  # limit-entry levels tested in E6
        lv = row.L1 + x * (row.H1 - row.L1)
        ax.hlines(lv, ts(row.t), tt[-1], color="grey", ls="-.", lw=0.7)
        ax.text(tt[-1], lv, f" lim {x}", fontsize=7, va="center", color="grey")
    ax.scatter([ts(row.t_in)], [row.px_in], marker="v", s=120, color="black", zorder=5,
               label=f"short {ts(row.t_in):%m-%d %H:%M} @ {row.px_in:.4g}")  # fmt: skip
    ax.scatter([ts(row.t_out)], [row.px_out], marker="x", s=120, color="black", zorder=5,
               label=f"exit {ts(row.t_out):%m-%d %H:%M} @ {row.px_out:.4g}")  # fmt: skip
    ax.legend(loc="best", fontsize=8)
    res = "WIN" if row.net > 0 else "LOSS"
    ax.set_title(f"{row.sym} 4h  {res} {100 * row.net:+.2f}%  ({WHY.get(int(row.why), '')})")
    bx.plot(tt, line[sl], color="black", lw=1, label="MACD")
    bx.plot(tt, sig[sl], color="orange", lw=1, label="signal")
    bx.bar(tt, (line - sig)[sl], width=w, color="#90a4ae")
    bx.axvline(ts(row.t) + pd.Timedelta(hours=2), color="grey", ls=":", lw=1)
    bx.axhline(0, color="grey", lw=0.5)
    bx.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle("data/research/divergence_e6.pkl")
    g = D[(D.tf == "4h") & (D.side == -1) & (D.entry == "market") & (D.buf == 1.0) & (D.exit == "partial")
          & D.sym_ratio.between(0.33, 3.0) & D.net.notna() & D.sym.isin(BIG) & (D.t >= C.T_OOS)].copy()  # fmt: skip
    picks = []
    for kind, sub in (("win", g[g.net > 0]), ("loss", g[g.net <= 0])):
        # typical trades: closest to the median outcome of each group, different coins
        sub = sub.iloc[np.argsort(np.abs(sub.net - sub.net.median()).to_numpy())]
        seen = set()
        for r in sub.itertuples():
            if r.sym in seen:
                continue
            seen.add(r.sym)
            picks.append((kind, r))
            if len(seen) == 3:
                break
    rows = []
    for k, (kind, r) in enumerate(picks):
        p = out / f"{kind}_{k + 1}_{r.sym}.png"
        chart(r, p)
        rows.append({"file": p.name, "sym": r.sym, "t_in": pd.to_datetime(r.t_in, unit="ms"), "px_in": r.px_in,
                     "t_out": pd.to_datetime(r.t_out, unit="ms"), "px_out": r.px_out, "net": r.net,
                     "why": WHY.get(int(r.why))})  # fmt: skip
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
