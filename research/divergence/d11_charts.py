"""D11: representative DIV-S4 trades as 4h K-line charts with MACD and CVD panels.

Trades come from the DIV-S4 account ledger (max 3 positions) rebuilt from D9. Sampling as before:
for each group, the trade on a large coin whose net is closest to the group median. Groups:
take profit, stop, time exit with profit, time exit with loss (all 2024-2026), plus one 2023 trade
(alt rally year, the weak regime), one trade from the 10 best days (crash days) and one holdout
(2020) trade. The chart marks both pivot highs, the divergence lines on price / MACD / CVD, and
the position box (entry, stop zone, take-profit zone, exit).
Usage: PYTHONPATH=research/divergence python research/divergence/d11_charts.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import common as C  # noqa: E402
import d7_account as D7  # noqa: E402
import e8_charts as G  # noqa: E402

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path("research/divergence/charts/d11")
H4 = 4 * 3_600_000
WHY = {0: "時間到期出場", 1: "止損", 2: "止盈（3 ATR）", 3: "資料結束出場"}


def ts(x):
    return pd.to_datetime(x, unit="ms")


def ledger():
    D = pd.read_pickle("data/research/divergence_d9.pkl")
    H = pd.read_pickle("data/research/divergence_d9_holdout.pkl")
    H = H[H.t >= int(pd.Timestamp("2020-05-15", tz="UTC").value // 10**6)]
    isg = D[D.t < C.T_OOS]
    th = json.loads((C.RESULTS / "D7_account/thresholds.json").read_text())["macd_cvd_threshold"]
    A = pd.concat([H, D], ignore_index=True)
    pr = {f: np.sort(isg[f].dropna().to_numpy()) for f in ("macd", "cvd")}
    A["s2"] = sum(np.searchsorted(pr[f], A[f].to_numpy()) / len(pr[f]) for f in pr) / 2
    A = A.assign(side=-1, net=A.net_base)
    K = D7.account(A[A.s2 >= th], 3)
    K["stop"] = K.px * (1 + 1.5 * K.atr)
    K["tp"] = K.px * (1 - 3 * K.atr)
    return K


def exit_px(r, X):
    k = int(np.searchsorted(X["t"], r.tx))
    op = X["o"][k] if r.tx > r.t_in else r.px
    if r.why == 1:
        return max(r.stop, op)
    if r.why == 2:
        return min(r.tp, op)
    return X["c"][k]


def chart(r, path, note):
    X = C.load(r.sym)
    B = C.resample(X, 48)
    I = C.indicators(B)
    t = B["t"]
    i0 = max(int(r.q) - 10, 0)
    i1 = min(int(np.searchsorted(t, r.tx)) + 8, len(t))
    sl = slice(i0, i1)
    tt = ts(t[sl])
    o, h, l, c = B["o"][sl], B["h"][sl], B["l"][sl], B["c"][sl]
    bw = pd.Timedelta(hours=4)
    sig = C.ema(I["macd"], 9)
    cv = I["cvd"] - I["cvd"][i0]
    px_out = exit_px(r, X)
    fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    for k in range(len(tt)):
        col = "#26a69a" if c[k] >= o[k] else "#ef5350"
        ax.vlines(tt[k] + bw / 2, l[k], h[k], color=col, lw=0.8)
        ax.bar(tt[k] + bw / 2, abs(c[k] - o[k]) or 1e-12, bottom=min(o[k], c[k]), width=bw * 0.7, color=col)
    # position box
    x0, x1 = ts(r.t_in), max(ts(r.tx), ts(r.t_in) + 4 * bw)
    ax.add_patch(Rectangle((x0, r.tp), x1 - x0, r.px - r.tp, color="#26a69a", alpha=0.22, lw=0))
    ax.add_patch(Rectangle((x0, r.px), x1 - x0, r.stop - r.px, color="#ef5350", alpha=0.22, lw=0))
    ax.hlines(r.px, x0, x1, color="#455a64", lw=1.2)
    ax.text(x1, r.tp, f" 止盈 {r.tp:.4g} ({100 * (1 - r.tp / r.px):+.2f}%)", va="center", fontsize=8, color="#00796b")
    ax.text(x1, r.stop, f" 止損 {r.stop:.4g} ({100 * (1 - r.stop / r.px):+.2f}%)", va="center", fontsize=8, color="#c62828")
    ax.text(x1, r.px, f" 做空 {r.px:.4g}  盈虧比 2.0", va="center", fontsize=8)
    # divergence: two pivot highs
    tq, tp_ = ts(r.t_q) + bw / 2, ts(r.t_p) + bw / 2
    ax.plot([tq, tp_], [r.hi_q, r.hi_p], color="#1565c0", lw=1.4)
    for name, x, y in (("高點 1", tq, r.hi_q), ("高點 2（更高）", tp_, r.hi_p)):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10), ha="right" if "2" in name else "center", fontsize=9, weight="bold",
                    color="#1565c0")  # fmt: skip
    ax.axvline(ts(r.t) + bw, color="grey", ls=":", lw=0.8)
    ax.text(ts(r.t) + bw, ax.get_ylim()[0], " 確認（右邊 3 根收完）", fontsize=7, color="grey", va="bottom")
    ax.scatter([ts(r.t_in)], [r.px], marker="v", s=130, color="black", zorder=5)
    ax.scatter([ts(r.tx)], [px_out], marker="x", s=130, color="black", zorder=5)
    ax.annotate(f"進場 {ts(r.t_in):%Y-%m-%d %H:%M}\n@ {r.px:.4g}", (ts(r.t_in), r.px), textcoords="offset points",
                xytext=(-12, -40), ha="right", fontsize=8, bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})  # fmt: skip
    ax.annotate(f"離場 {ts(r.tx):%Y-%m-%d %H:%M}\n@ {px_out:.4g}", (ts(r.tx), px_out), textcoords="offset points",
                xytext=(10, -34 if r.net > 0 else 22), ha="left", fontsize=8,
                bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})  # fmt: skip
    res = "獲利" if r.net > 0 else "虧損"
    ax.set_title(f"{r.sym} 4h 看跌背離做空 {res} {100 * r.net:+.2f}%（含成本與資金費）  {WHY[int(r.why)]}  "
                 f"分數 {r.s2:.2f}  {note}", fontsize=11)  # fmt: skip
    lo, hi = min(np.nanmin(l), r.tp), max(np.nanmax(h), r.stop)
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    # MACD
    xs = tt + bw / 2
    bx.plot(xs, I["macd"][sl], color="black", lw=1, label="MACD 線（除以價格）")
    bx.plot(xs, sig[sl], color="orange", lw=1, label="signal")
    bx.bar(xs, (I["macd"] - sig)[sl], width=bw * 0.7, color="#90a4ae")
    bx.axhline(0, color="grey", lw=0.6)
    bx.plot([tq, tp_], [I["macd"][int(r.q)], I["macd"][int(r.p)]], color="#c62828", lw=1.4, marker="o", ms=4,
            label="高點 2 的 MACD 低於高點 1" if r.macd > 0 else "高點 2 的 MACD 沒有更低（分數主要來自 CVD）")  # fmt: skip
    bx.legend(loc="upper left", fontsize=7)
    # CVD
    cx.plot(xs, cv[sl], color="#6a1b9a", lw=1, label="CVD（主動買 − 主動賣，累計）")
    cx.plot([tq, tp_], [cv[int(r.q)], cv[int(r.p)]], color="#c62828", lw=1.4, marker="o", ms=4)
    cx.axvspan(tq, tp_, color="#ce93d8", alpha=0.15, label=f"兩高點間淨主動買 / 成交量 = {-r.cvd:+.3f}")
    cx.legend(loc="upper left", fontsize=7)
    for a in (bx, cx):
        a.axvline(ts(r.t_in), color="black", ls=":", lw=0.8)
    cx.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def median_pick(g):
    g = g[g.sym.isin(G.BIG)] if g.sym.isin(G.BIG).any() else g
    return g.iloc[int(np.argmin(np.abs(g.net - g.net.median()).to_numpy()))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    K = ledger()
    yr = pd.to_datetime(K.t, unit="ms").dt.year
    o = K[K.t >= C.T_OOS]
    day = K[K.t >= C.T_IS].groupby(K.tx // C.DAY).net.sum().sort_values(ascending=False)
    crash = K[(K.tx // C.DAY).isin(day.index[:10])]
    specs = [
        ("1_tp_win", o[o.why == 2], "2024-26 止盈的典型"),
        ("2_stop_loss", o[o.why == 1], "2024-26 止損的典型"),
        ("3_time_win", o[(o.why == 0) & (o.net > 0)], "2024-26 到期獲利的典型"),
        ("4_time_loss", o[(o.why == 0) & (o.net <= 0)], "2024-26 到期虧損的典型"),
        ("5_2023_loss", K[(yr == 2023) & (K.net <= 0)], "2023 山寨反彈年的虧損"),
        ("6_crash_day_win", crash[crash.net > 0], "最賺 10 天之一"),
        ("7_holdout", K[K.t < C.T_IS], "保留期 2020"),
    ]
    rows = []
    for name, g, note in specs:
        r = median_pick(g)
        p = OUT / f"{name}_{r.sym}.png"
        chart(r, p, note)
        rows.append({"file": p.name, "group_n": len(g), "group_median_net%": round(100 * g.net.median(), 2),
                     "t_in": ts(r.t_in), "px_in": r.px, "t_out": ts(r.tx), "stop": r.stop, "tp": r.tp,
                     "net%": round(100 * r.net, 2), "why": WHY[int(r.why)], "score": round(r.s2, 3)})  # fmt: skip
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "picks.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.to_string())


if __name__ == "__main__":
    main()
