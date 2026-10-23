"""B3: cross-sectional relative value. Does a coin that moved away from the market come back?

At each rebalance time (bar close, every H minutes) for the 40 coins:
  residual move  e = r(L) - beta x m(L)    r: coin log return over the last L minutes,
                                           m: equal weight market, beta: last 3 days of 1m data
  portfolio      long the 8 coins with the lowest e, short the 8 with the highest, equal
                 weight, 0.5 gross each side (dollar neutral), held H minutes
  control        the same portfolio built from raw returns r(L) (no market adjustment), and
                 the same rule with the sign flipped (momentum), which must be the mirror
The forward P&L uses raw coin returns (what a dollar neutral book earns), plus the
beta-neutral version (forward residuals). Cost per unit of turnover: maker 0.02% or taker
plus slippage 0.07%, per side traded.

Usage: uv run python research/alpha/b3_relative_value.py
"""

import numpy as np
import pandas as pd
from common import fmt, folder, market, market_days, publish, save_json
from stats import boot_mean

from data import GRID, MIN, UNIVERSE, Y2, matrix

EXP = "B3_relative_value"
LS = (15, 60, 240, 1440)
HS = (15, 60, 240)
NQ = 8  # coins per side
COSTS = {"maker": 0.02, "taker": 0.07}
WARM = 3 * 1440 + 1440


def rolling_beta(r: np.ndarray, m: np.ndarray, n: int) -> np.ndarray:
    """Per coin rolling beta to the market (N x K), known at each bar close."""
    ok = np.isfinite(r) & np.isfinite(m)[:, None]
    r0 = np.where(ok, r, 0.0)
    m0 = np.where(ok, m[:, None], 0.0)

    def rs(x):
        c = np.cumsum(x, axis=0)
        out = np.full_like(c, np.nan)
        out[n - 1 :] = c[n - 1 :] - np.r_[np.zeros((1, x.shape[1])), c[:-n]]
        return out

    cnt = rs(ok.astype(float))
    sxy, sx, sy, sxx = rs(r0 * m0), rs(m0), rs(r0), rs(m0 * m0)
    cov = sxy / cnt - sx / cnt * sy / cnt
    var = sxx / cnt - (sx / cnt) ** 2
    return np.clip(cov / np.where(var > 0, var, np.nan), 0, 3)


def load():
    P = np.log(matrix("c", UNIVERSE))
    live = matrix("live", UNIVERSE, bool)
    mk = np.nan_to_num(market())
    r1 = np.vstack([np.full((1, P.shape[1]), np.nan), np.diff(P, axis=0)])
    r1[~live] = np.nan
    beta = rolling_beta(r1, mk, 3 * 1440)
    M = np.cumsum(mk)
    # a coin is tradable at i when it had live bars in the last hour (no listing gaps)
    lc = np.cumsum(live, axis=0)
    act = np.zeros_like(live)
    act[60:] = (lc[60:] - lc[:-60]) >= 55
    return P, M, beta, act


def book(e: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Weights: long the NQ lowest e, short the NQ highest, 0.5 gross per side."""
    w = np.zeros_like(e)
    idx = np.flatnonzero(ok & np.isfinite(e))
    if len(idx) < 2 * NQ + 4:
        return w
    order = idx[np.argsort(e[idx])]
    w[order[:NQ]] = 0.5 / NQ
    w[order[-NQ:]] = -0.5 / NQ
    return w


def run(P, M, beta, act, L, H, signal="residual", sign=1, delay=0):
    times = np.arange(WARM + (H - WARM % H) % H, len(GRID) - H - delay, H)  # bar index, close known
    rows, w_prev = [], np.zeros(P.shape[1])
    contrib = np.zeros(P.shape[1])
    for i in times:
        b = beta[i]
        r = P[i] - P[i - L]
        m = M[i] - M[i - L]
        e = r - b * m if signal == "residual" else r
        ok = act[i] & act[i + H + delay] & np.isfinite(b)
        w = sign * book(e, ok)
        # delay: enter `delay` minutes after the signal, so a bid / ask bounce in the signal
        # close cannot show up as reversal
        a, z = i + delay, i + delay + H
        fr = np.nan_to_num(P[z] - P[a])
        fx = np.nan_to_num(fr - b * (M[z] - M[a]))
        turn = np.abs(w - w_prev).sum()
        rows.append(
            {
                "i": i,
                "t": GRID[i] + MIN,
                "gross": float(w @ fr) * 100,
                "gross_beta_neutral": float(w @ fx) * 100,
                "turnover": float(turn),
                "n_coins": int(ok.sum()),
                "book_beta": float(w @ np.nan_to_num(b)),
            }
        )
        contrib += w * fr * 100
        w_prev = w
    d = pd.DataFrame(rows)
    for k, c in COSTS.items():
        d[f"net_{k}"] = d.gross - d.turnover * c
    d["day"] = d.t // 86_400_000
    d["y2"] = d.t >= GRID[Y2]
    d["month"] = pd.to_datetime(d.t, unit="ms").dt.strftime("%Y-%m")
    return d, contrib


def stats_row(d: pd.DataFrame, **key) -> dict:
    per_day = d.groupby("day")[["gross", "net_maker", "net_taker", "gross_beta_neutral"]].sum()
    r = dict(key)
    r["rebalances"] = len(d)
    r["turnover_per_rebalance"] = d.turnover.mean()
    for c in ("gross", "gross_beta_neutral", "net_maker", "net_taker"):
        m, lo, hi = boot_mean(d[c].to_numpy(), d.day.to_numpy())
        r[f"{c}_bp"] = m * 100  # basis points per rebalance on the gross book
        if c in ("gross", "net_taker", "net_maker"):
            r[f"{c}_lo_bp"], r[f"{c}_hi_bp"] = lo * 100, hi * 100
    r["daily_sharpe_net_maker"] = per_day.net_maker.mean() / per_day.net_maker.std() * np.sqrt(365)
    r["y1_gross_bp"] = d[~d.y2].gross.mean() * 100
    r["y2_gross_bp"] = d[d.y2].gross.mean() * 100
    r["pos_months_gross"] = int((d.groupby("month").gross.sum() > 0).sum())
    r["book_beta"] = d.book_beta.mean()
    return r


def main():
    out = folder(EXP)
    save_json(
        out / "config.json",
        {"exp": EXP, "lookbacks": LS, "holds": HS, "coins_per_side": NQ, "costs_pct_per_side": COSTS,  # noqa: E501
         "symbols": UNIVERSE, "beta_window_min": 3 * 1440},
    )  # fmt: skip
    P, M, beta, act = load()
    days = market_days()
    rows, keep = [], {}
    for L in LS:
        for H in HS:
            for sig, sign, nm in (
                ("residual", 1, "reversal"),
                ("raw", 1, "raw_reversal"),
                ("residual", -1, "momentum"),
            ):
                d, contrib = run(P, M, beta, act, L, H, sig, sign)
                d = d.merge(days, on="day", how="left")
                rows.append(stats_row(d, rule=nm, L=L, H=H))
                print(fmt(pd.DataFrame(rows[-1:])), flush=True)
                if nm == "reversal":
                    keep[(L, H)] = (d, contrib)
                    d1, _ = run(P, M, beta, act, L, H, sig, sign, delay=1)
                    rows.append(
                        stats_row(
                            d1.merge(days, on="day", how="left"), rule="reversal_delay1", L=L, H=H
                        )
                    )
                    print(fmt(pd.DataFrame(rows[-1:])), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "grid.csv", index=False)
    # detail for the best-looking reversal cell by year 1 gross only (no peeking at year 2)
    rev = t[t.rule == "reversal"].sort_values("y1_gross_bp", ascending=False)
    L, H = int(rev.iloc[0].L), int(rev.iloc[0].H)
    d, contrib = keep[(L, H)]
    d.to_csv(out / f"rebalances_L{L}_H{H}.csv.gz", index=False)
    bym = d.groupby("month")[["gross", "net_maker", "net_taker"]].sum().reset_index()
    bym.to_csv(out / "best_by_month.csv", index=False)
    bye = pd.DataFrame([stats_row(g, event=ev) for ev, g in d.groupby("event")])
    bye.to_csv(out / "best_by_event.csv", index=False)
    pd.DataFrame({"symbol": UNIVERSE, "gross_contrib_pct": contrib}).sort_values(
        "gross_contrib_pct"
    ).to_csv(out / "best_by_coin.csv", index=False)
    save_json(
        out / "summary.json", {"chosen_by_year1": {"L": L, "H": H}, "row": rev.iloc[0].to_dict()}
    )
    print(fmt(t))
    print(fmt(bye), fmt(bym), sep="\n")
    publish(
        EXP,
        [
            "config.json",
            "grid.csv",
            "best_by_month.csv",
            "best_by_event.csv",
            "best_by_coin.csv",
            "summary.json",
        ],
    )


if __name__ == "__main__":
    main()
