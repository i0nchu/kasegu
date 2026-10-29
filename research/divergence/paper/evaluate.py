"""Pre-registered evaluation of the DIV-S4 paper ledgers (DIVS4_LOG.md, D12). Do not change the
metrics or the decision rule after the paper run starts.

Writes ledger_<name>.csv (one row per admitted trade, everything the run recorded), metrics.json
and report.md into the state directory.
Usage: PYTHONPATH=research/divergence:research/divergence/paper python research/divergence/paper/evaluate.py [STATE_DIR [START]]
START (YYYY-MM-DD) is only for testing on a replay; the paper evaluation starts at frozen paper_start.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from engine import DAY, F, H4, LEDGERS

START = int(pd.Timestamp(F["paper_start"]).value // 10**6)
RNG_SEED = 12
NBOOT = 2000
MIN_FINAL = 150  # DIV-S4 closed trades needed for the final verdict
NAMES = {"div_s4": "DIV-S4", "macd_only": "MACD-only", "all_highs": "所有更高高點做空"}


def load(d):
    S = pd.read_csv(d / "signals.csv")
    T = pd.read_csv(d / "trades.csv")
    bk = pd.read_csv(d / "basket.csv").sort_values("t_bar")
    T = T[(T.why >= 0) & (T.t_sig >= START)].merge(
        S.drop(columns=[c for c in S.columns if c.startswith("in_")] + ["sym", "t_bar", "t_sig"]), on="id")  # fmt: skip
    cum = np.log1p(bk.ret.fillna(0.0).to_numpy()).cumsum()
    tb = bk.t_bar.to_numpy()
    a = np.searchsorted(tb, T.t_bar.to_numpy())  # signal bar (cum through its close = entry)
    b = np.searchsorted(tb, (T.t_exit // H4 * H4).to_numpy())  # 4h bar holding the exit
    ok = (a < len(tb)) & (b < len(tb)) & (tb[np.minimum(a, len(tb) - 1)] == T.t_bar) & (
        tb[np.minimum(b, len(tb) - 1)] == T.t_exit // H4 * H4)  # fmt: skip
    T["basket_ret"] = np.where(ok, np.exp(cum[np.minimum(b, len(tb) - 1)] - cum[np.minimum(a, len(tb) - 1)]) - 1, np.nan)
    T["net_xs"] = T.net + T.basket_ret  # short the coin, long the equal-weight basket
    T["quote_slip"] = T.quote_bid / T.px_entry - 1  # real bid at the signal vs the simulated open (+ = better)
    T["regime_alt"] = np.where(T.basket_30d.isna(), "", np.where(T.basket_30d > 0, "山寨 30 天上漲", "山寨 30 天下跌"))
    T["regime_btc"] = np.where(T.btc_30d.isna(), "", np.where(T.btc_30d > 0, "BTC 30 天上漲", "BTC 30 天下跌"))
    T["regime_vol"] = np.where(T.btc_vol7.isna(), "",
                               np.where(T.btc_vol7 > F["regime_btc_vol7_median"], "BTC 波動高", "BTC 波動低"))  # fmt: skip
    return T


def account(K):
    """W = 10% per trade, booked on the exit day (same as the research account)."""
    if not len(K):
        return {"ret": 0.0, "maxdd": 0.0}
    day = K.t_exit // DAY * DAY
    r = (F["account_weight"] * K.net).groupby(day).sum()
    r = r.reindex(np.arange(START // DAY * DAY, int(day.max()) + DAY, DAY), fill_value=0.0)
    eq = (1 + r).cumprod()
    return {"ret": float(eq.iloc[-1] - 1), "maxdd": float((eq / eq.cummax() - 1).min()),
            "sharpe_ann": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan}  # fmt: skip


def stats(K):
    if not len(K):
        return {"n": 0}
    mu, t, nd = C.day_t(K.t_sig.to_numpy(), K.net.to_numpy())
    pnl = K.groupby(K.t_exit // DAY).net.sum().sort_values(ascending=False)
    top5 = pnl.index[:5]
    coin = K.groupby("sym").net.sum().sort_values(ascending=False)
    months = max((K.t_sig.max() - START) / (30.4 * DAY), 1e-9)
    return {"n": len(K), "per_month": len(K) / months, "win": float((K.net > 0).mean()), "bp": 1e4 * K.net.mean(),
            "median_bp": 1e4 * K.net.median(), "day_t": t, "signal_days": nd,
            "xs_bp": 1e4 * K.net_xs.mean(), "xs_n": int(K.net_xs.notna().sum()), "basket_bp": 1e4 * K.basket_ret.mean(),
            "funding_bp": 1e4 * K.funding.mean(), "quote_slip_bp": 1e4 * K.quote_slip.mean(),
            "live_quotes": int(K.quote_slip.notna().sum()),
            "exit_stop": float((K.why == 1).mean()), "exit_tp": float((K.why == 2).mean()), "exit_time": float((K.why == 0).mean()),
            "top5_days_share": float(pnl.head(5).clip(lower=0).sum() / pnl.clip(lower=0).sum()) if (pnl > 0).any() else np.nan,
            "bp_ex_top5_days": 1e4 * K[~(K.t_exit // DAY).isin(top5)].net.mean(),
            "top3_coins_share": float(coin.head(3).clip(lower=0).sum() / coin.clip(lower=0).sum()) if (coin > 0).any() else np.nan,
            "max_trades_one_day": int(K.groupby(K.t_sig // DAY).size().max()), **account(K)}  # fmt: skip


def diff_ci(A, B):
    """Mean net A - mean net B with a day-block bootstrap (calendar days of the signal resampled jointly)."""
    if not len(A) or not len(B):
        return {"diff_bp": np.nan}
    rng = np.random.default_rng(RNG_SEED)
    days = np.union1d(A.t_sig // DAY, B.t_sig // DAY)
    ga, gb = A.groupby(A.t_sig // DAY).net, B.groupby(B.t_sig // DAY).net
    sa, na, sb, nb = (ga.sum().reindex(days, fill_value=0).to_numpy(), ga.size().reindex(days, fill_value=0).to_numpy(),
                      gb.sum().reindex(days, fill_value=0).to_numpy(), gb.size().reindex(days, fill_value=0).to_numpy())  # fmt: skip
    k = rng.integers(0, len(days), (NBOOT, len(days)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sa[k].sum(1) / na[k].sum(1) - sb[k].sum(1) / nb[k].sum(1)
    d = d[np.isfinite(d)]
    return {"diff_bp": 1e4 * (A.net.mean() - B.net.mean()), "ci_lo_bp": 1e4 * np.quantile(d, 0.025),
            "ci_hi_bp": 1e4 * np.quantile(d, 0.975), "p_le_0": float((d <= 0).mean())}  # fmt: skip


def verdict(M):
    s, b, m = M["ledgers"]["div_s4"], M["ledgers"]["all_highs"], M["ledgers"]["macd_only"]
    if s.get("n", 0) < MIN_FINAL:
        return f"未到最終評估（DIV-S4 已平倉 {s.get('n', 0)} / {MIN_FINAL} 筆），只記錄、不下結論"
    support = (s["bp"] > 0 and M["vs"]["div_s4-all_highs"]["diff_bp"] > 0 and s["xs_bp"] > 0
               and s["bp_ex_top5_days"] > 0 and s["maxdd"] > -0.25)  # fmt: skip
    fail = s["bp"] <= 0 or M["vs"]["div_s4-all_highs"]["diff_bp"] <= 0
    v = "支持" if support else ("不支持" if fail else "不確定，繼續累積到 300 筆")
    cvd = "CVD 有加分" if M["vs"]["div_s4-macd_only"]["diff_bp"] > 0 else "CVD 沒有加分，考慮只用 MACD"
    return f"{v}；{cvd}（MACD-only {m['bp']:+.0f} bp，基準 {b['bp']:+.0f} bp）"


def main(d, quiet=False):
    T = load(d)
    M = {"asof": pd.Timestamp(int(T.t_exit.max()) if len(T) else START, unit="ms").isoformat(), "ledgers": {},
         "uncapped": {}, "vs": {}, "regimes": {}}  # fmt: skip
    L = {}
    for lg in LEDGERS:
        K = T[T[f"in_{lg}"]].sort_values("t_sig")
        L[lg] = K
        K.to_csv(d / f"ledger_{lg}.csv", index=False)
        M["ledgers"][lg] = stats(K)
        M["uncapped"][lg] = stats(T[T[f"rule_{lg}"]])
    for a, b in (("div_s4", "all_highs"), ("div_s4", "macd_only"), ("macd_only", "all_highs")):
        M["vs"][f"{a}-{b}"] = diff_ci(L[a], L[b])
    for reg in ("regime_alt", "regime_btc", "regime_vol"):
        for val, g in T.groupby(reg):
            if not val:
                continue
            row = {lg: {"n": int(g[f"in_{lg}"].sum()), "bp": 1e4 * g[g[f"in_{lg}"]].net.mean()} for lg in LEDGERS}
            row["div_s4-all_highs_bp"] = row["div_s4"]["bp"] - row["all_highs"]["bp"]
            M["regimes"][val] = row
    M["verdict"] = verdict(M)
    (d / "metrics.json").write_text(json.dumps(M, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    report(d, M)
    if not quiet:
        print((d / "report.md").read_text(encoding="utf-8"))


def report(d, M):
    f = lambda x, k=0: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{k}f}"  # noqa: E731
    out = [f"# DIV-S4 Paper 評估（資料到 {M['asof'][:16]} UTC）", "", f"判定：**{M['verdict']}**", "",
           "| 帳本 | 筆數 | 每月 | 勝率 | 每筆 bp | 日 t | 扣籃子 bp | 籃子 bp | 資金費 bp | 帳戶報酬 | 最大回撤 | 前 5 天占全部獲利日 | 去掉前 5 天 bp |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for lg in LEDGERS:
        s = M["ledgers"][lg]
        if not s.get("n"):
            out.append(f"| {NAMES[lg]} | 0 |" + " — |" * 11)
            continue
        out.append(f"| {NAMES[lg]} | {s['n']} | {s['per_month']:.1f} | {s['win']:.0%} | {f(s['bp'])} | {f(s['day_t'], 2)} | "
                   f"{f(s['xs_bp'])} | {f(s['basket_bp'])} | {f(s['funding_bp'], 1)} | {s['ret']:+.1%} | {s['maxdd']:.1%} | "
                   f"{100 * s['top5_days_share']:.0f}% | {f(s['bp_ex_top5_days'])} |")  # fmt: skip
    out += ["", "相對比較（每筆平均差，日區塊 bootstrap 95% 區間）：", "", "| 比較 | 差 bp | 95% 區間 | 差 ≤ 0 的機率 |", "|---|---|---|---|"]
    for k, v in M["vs"].items():
        a, b = k.split("-")
        out.append(f"| {NAMES[a]} − {NAMES[b]} | {f(v['diff_bp'])} | {f(v.get('ci_lo_bp'))} ~ {f(v.get('ci_hi_bp'))} | "
                   f"{v.get('p_le_0', float('nan')):.2f} |")  # fmt: skip
    out += ["", "市場狀態（訊號當下，定義事先固定）：", "", "| 狀態 | DIV-S4 筆數 / bp | MACD-only | 基準 | DIV-S4 − 基準 |", "|---|---|---|---|---|"]
    for k, v in M["regimes"].items():
        out.append(f"| {k} | {v['div_s4']['n']} / {f(v['div_s4']['bp'])} | {v['macd_only']['n']} / {f(v['macd_only']['bp'])} | "
                   f"{v['all_highs']['n']} / {f(v['all_highs']['bp'])} | {f(v['div_s4-all_highs_bp'])} |")  # fmt: skip
    s = M["ledgers"]["div_s4"]
    if s.get("n"):
        out += ["", f"DIV-S4 出場：止損 {s['exit_stop']:.0%}、止盈 {s['exit_tp']:.0%}、到期 {s['exit_time']:.0%}；"
                f"單日最多 {s['max_trades_one_day']} 筆；前 3 個幣占全部獲利幣 {100 * s['top3_coins_share']:.0f}%；"
                f"即時報價 {s['live_quotes']} 筆，實際買一價相對模擬開盤 {f(s['quote_slip_bp'], 1)} bp。"]  # fmt: skip
    u = M["uncapped"]
    out += ["", "不設持倉上限（所有符合規則的訊號，次要參考）：" + "、".join(
        f"{NAMES[lg]} {u[lg].get('n', 0)} 筆 {f(u[lg].get('bp', np.nan))} bp" for lg in LEDGERS)]  # fmt: skip
    (d / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        START = int(pd.Timestamp(sys.argv[2], tz="UTC").value // 10**6)
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/paper/divs4"))
