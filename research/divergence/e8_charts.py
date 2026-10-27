"""E8 charts in a TradingView position-tool style: entry box with target (green) and stop (red) zones,
entry / exit markers with time and price, down leg H0 -> L0, neckline H1 and the entry zone.
Usage: PYTHONPATH=research/divergence python research/divergence/e8_charts.py OUT_DIR
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import common as C  # noqa: E402
import e8_wbottom as E8  # noqa: E402

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BIG = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
       "AVAXUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT"]  # fmt: skip
WHY = {0: "時間出場", 1: "止損", 2: "TP1 平半倉，餘倉保本", 3: "止盈", 4: "TP1 平半倉，餘倉時間出場",
       5: "跌破第一底且 MACD 創新低（無背離）出場"}  # fmt: skip
TFH = {"5m": 1 / 12, "15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24}


def ts(x):
    return pd.to_datetime(x, unit="ms")


def chart(r, path, note="", disp=None):
    """disp: candle timeframe for display (defaults to the signal timeframe)."""
    disp = disp or r.tf
    if disp == "auto":  # candle size from the swing's own duration: whole structure in about 60-90 candles
        span = (r.t_out - r.t0) / 3_600_000
        disp = next((k for k, v in TFH.items() if v >= TFH[r.tf] and span / v <= 70), "1d")
    X = C.load(r.sym)
    n5 = int(12 * TFH[disp])
    B = C.resample(X, n5)
    t = B["t"]
    i0 = max(np.searchsorted(t, r.t0) - 8, 0)
    i1 = min(np.searchsorted(t, r.t_out) + 10, len(t))
    sl = slice(i0, i1)
    tt = pd.to_datetime(t[sl], unit="ms")
    o, h, l, c = B["o"][sl], B["h"][sl], B["l"][sl], B["c"][sl]
    S = C.resample(X, int(12 * TFH[r.tf]))  # MACD on the signal timeframe (the one the rules use)
    ms = (S["t"] >= t[i0]) & (S["t"] < t[min(i1, len(t) - 1)] + n5 * C.M5)
    ts_ = pd.to_datetime(S["t"][ms], unit="ms")
    ml, _ = C.macd(S["c"])
    sig = C.ema(ml, 9)
    bw = pd.Timedelta(hours=TFH[disp])
    sb = pd.Timedelta(hours=TFH[r.tf])  # signal bar
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k in range(len(tt)):
        col = "#26a69a" if c[k] >= o[k] else "#ef5350"
        ax.vlines(tt[k] + bw / 2, l[k], h[k], color=col, lw=0.8)
        ax.bar(tt[k] + bw / 2, abs(c[k] - o[k]) or 1e-12, bottom=min(o[k], c[k]), width=bw * 0.7, color=col)

    s = r.side
    tp = getattr(r, "tp1", r.H1)
    # position tool: box from entry to exit
    x0, x1 = ts(r.t_in), max(ts(r.t_out), ts(r.t_in) + 4 * bw)
    w = x1 - x0
    ax.add_patch(Rectangle((x0, min(r.px_in, tp)), w, abs(tp - r.px_in), color="#26a69a", alpha=0.22, lw=0))
    ax.add_patch(Rectangle((x0, min(r.px_in, r.stop)), w, abs(r.px_in - r.stop), color="#ef5350", alpha=0.22, lw=0))
    ax.hlines(r.px_in, x0, x1, color="#455a64", lw=1.2)
    rr = abs(tp - r.px_in) / abs(r.px_in - r.stop)
    ax.text(x1, tp, f" TP1 {tp:.4g} ({100 * s * (tp / r.px_in - 1):+.2f}%)", va="center", fontsize=8, color="#00796b")
    ax.text(x1, r.stop, f" 止損 {r.stop:.4g} ({100 * s * (r.stop / r.px_in - 1):+.2f}%)", va="center", fontsize=8,
            color="#c62828")  # fmt: skip
    ax.text(x1, r.px_in, f" {'做多' if s > 0 else '做空'} {r.px_in:.4g}  盈虧比 {rr:.2f}", va="center", fontsize=8)
    if r.exit in ("partial", "body_half"):
        t2 = r.H1 + (r.H1 - r.L0)
        if min(np.nanmin(l), r.stop) <= t2 <= max(np.nanmax(h), r.stop):
            ax.hlines(t2, x0, x1, color="#00796b", ls=":", lw=1)
            ax.text(x1, t2, f" TP2 {t2:.4g}", va="center", fontsize=8, color="#00796b")
        else:  # far outside the chart: note it on the entry line instead
            ax.text(x1, r.px_in, f"\n\n TP2 {t2:.4g}（圖外）", va="center", fontsize=8, color="#00796b")
    # structure
    ax.plot([ts(r.t0) + bw / 2, ts(r.ta) + bw / 2, ts(r.tb) + bw / 2], [r.H0, r.L0, r.H1], color="#1565c0", lw=1,
            alpha=0.6)  # fmt: skip
    up = 10 if s > 0 else -16
    names = ("起點", "第一底 L0", "頸線 H1") if s > 0 else ("起點", "第一頂", "頸線")
    for name, tx, px, dy in zip(names, (r.t0, r.ta, r.tb), (r.H0, r.L0, r.H1), (up, -up - 6, up)):
        ax.annotate(name, (ts(tx) + bw / 2, px), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=9,
                    weight="bold", color="#1565c0")  # fmt: skip
    ax.hlines(r.H1, ts(r.tb), x1, color="#1565c0", ls="--", lw=0.8)
    ax.hlines(r.L0, ts(r.ta), x1, color="#1565c0", ls="--", lw=0.8)
    z0 = r.L0 - (0.15 if hasattr(r, "rr") else 0) * (r.H1 - r.L0)  # E10 entries may sit slightly beyond L0
    z1 = r.L0 + 0.5 * (r.H1 - r.L0)
    ax.add_patch(Rectangle((ts(r.tb), min(z0, z1)), x1 - ts(r.tb), abs(z1 - z0), color="#fff59d", alpha=0.35, lw=0))
    ax.text(ts(r.tb), z1, "入場區（中點～第一底/頂）", fontsize=7, color="#9e9d24", va="bottom" if s > 0 else "top")
    ax.axvline(ts(r.t) + sb, color="grey", ls=":", lw=0.8)
    ax.scatter([ts(r.t_in)], [r.px_in], marker="^" if s > 0 else "v", s=130, color="black", zorder=5)
    ax.scatter([ts(r.t_out)], [r.px_out], marker="x", s=130, color="black", zorder=5)
    ax.annotate(f"進場 {ts(r.t_in):%Y-%m-%d %H:%M}\n@ {r.px_in:.4g}", (ts(r.t_in), r.px_in), textcoords="offset points",
                xytext=(-10, -34 if s > 0 else 22), ha="right", fontsize=8,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})  # fmt: skip
    ax.annotate(f"離場 {ts(r.t_out):%Y-%m-%d %H:%M}\n@ {r.px_out:.4g}", (ts(r.t_out), r.px_out),
                textcoords="offset points", xytext=(10, 22 if r.net > 0 else -34), ha="left", fontsize=8,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})  # fmt: skip
    res = "獲利" if r.net > 0 else "虧損"
    why = WHY.get(int(r.why), "")
    if s < 0:
        why = why.replace("跌破第一底且 MACD 創新低", "突破第一頂且 MACD 創新高")
    ax.set_title(f"{r.sym} 訊號 {r.tf}／K 線 {disp} {'做多（雙底）' if s > 0 else '做空（雙頂）'}  {res} {100 * r.net:+.2f}%（含成本）  "
                 f"{why}  {note}", fontsize=11)  # fmt: skip
    lo, hi = min(np.nanmin(l), r.stop, tp), max(np.nanmax(h), r.stop, tp)
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    bx.plot(ts_ + sb / 2, ml[ms], color="black", lw=1, label=f"MACD（{r.tf}）")
    bx.plot(ts_ + sb / 2, sig[ms], color="orange", lw=1, label="signal")
    bx.bar(ts_ + sb / 2, (ml - sig)[ms], width=sb * 0.7, color="#90a4ae")
    bx.axhline(0, color="grey", lw=0.6)
    bx.axhline(r.M0, color="#1565c0", ls=":", lw=0.8, label="M0（第一底/頂附近的 MACD 極值）")
    bx.axvline(ts(r.t) + sb, color="grey", ls=":", lw=0.8)
    bx.axvline(ts(r.t_in), color="black", ls=":", lw=0.8)
    bx.legend(loc="upper left", fontsize=7)
    span = tt[-1] - tt[0]
    bx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d" if span > pd.Timedelta(days=4) else "%m-%d %H:%M"))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def select(D, cfg):
    g = D[(D.t_in >= 0)]
    for k, v in cfg.items():
        g = g[g[k] == v]
    return g[E8.rule(g)]


def main(out, K=3, **cfg):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle(f"data/research/divergence_e8_k{K}.pkl")
    g = select(D, cfg)
    g = g[g.sym.isin(BIG) & (g.t >= C.T_OOS)]
    picks = []
    for side in (1, -1):
        for kind, sub in (("win", g[(g.side == side) & (g.net > 0)]), ("loss", g[(g.side == side) & (g.net <= 0)])):
            sub = sub.iloc[np.argsort(np.abs(sub.net - sub.net.median()).to_numpy())]
            seen = set()
            for r in sub.itertuples():
                if r.sym in seen:
                    continue
                seen.add(r.sym)
                picks.append((kind, r))
                if len(seen) == 1:
                    break
    rows = []
    for k, (kind, r) in enumerate(picks):
        p = out / f"{k + 1}_{'long' if r.side > 0 else 'short'}_{kind}_{r.sym}.png"
        chart(r, p)
        rows.append({"file": p.name, "t_in": ts(r.t_in), "px_in": r.px_in, "t_out": ts(r.t_out), "px_out": r.px_out,
                     "stop": r.stop, "tp": r.H1, "net": round(100 * r.net, 2), "why": WHY.get(int(r.why))})  # fmt: skip
    # Kuri's annotated BCH double bottom (策略修正.pdf)
    g0 = select(D, cfg)
    ex = g0[(g0.sym == "BCHUSDT") & (g0.side == 1) & (g0.ta == int(pd.Timestamp("2025-12-15 16:00", tz="UTC").value // 10**6))]
    for r in ex.itertuples():
        p = out / "0_long_kuri_example_BCHUSDT.png"
        chart(r, p, "（你標註的範例）")
        rows.append({"file": p.name, "t_in": ts(r.t_in), "px_in": r.px_in, "t_out": ts(r.t_out), "px_out": r.px_out,
                     "stop": r.stop, "tp": r.H1, "net": round(100 * r.net, 2), "why": WHY.get(int(r.why))})  # fmt: skip
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main(sys.argv[1], tf="4h", x=0.25, stop_k=0.5, exit="full", divfail=1)
