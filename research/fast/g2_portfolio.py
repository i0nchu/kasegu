"""G2: EPR15 trade ledger under the per-symbol risk model (PR #11), with fill-model scenarios.

Same frozen rules as G1 (1m klines, limit 12 vol units under the close when the close < EMA240,
15-minute time exit or 3% stop), but every trade is recorded in full and the account is run as
the engine of PR #11 runs it:
  - each symbol trades on its own with its own notional (one position per symbol, no account
    position cap); every symbol can rest an entry at the same time
  - account stop: realized net since the start + open pnl at minute closes <= -max_account_loss
    closes everything and stops the account (the path ends there)
  - max_daily_loss / max_consecutive_losses pauses as in the engine (realized net, new entries
    only); max_daily_loss on the UTC day, the streak pause lasts loss_pause_min minutes

Fill scenarios (entry fill rule, extra exit slippage, extra stop slippage), see FILL:
  base   the research rule: the next bar's low below the limit fills the whole order
  ...    more are added by G3 once the aggTrades / order book calibration is known

Ledger columns: sym, t_in, t_out (ms), px_in, px_out, notional, fee_in, fee_out, gross, net (USDT),
why (0 time, 1 stop), open_at_entry (positions open across the account when it filled),
mae (worst minute low during the hold, % of entry). Daily: portfolio net by UTC exit day,
worst intraday open pnl, trades, max concurrent positions.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/g2_portfolio.py
"""

import concurrent.futures as cf
import json

import numba as nb
import numpy as np
import pandas as pd

import fast as F
import g1_epr15_pit as G1

DEPLOY6 = ["SOLUSDT", "DOGEUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "SUIUSDT"]
UNIVERSES = {"deploy6": DEPLOY6, "surv40": G1.SURV40, "pit": None}  # None = every PIT coin
# entry penetration (fraction beyond the limit), extra exit slippage, extra stop slippage
FILL = {"base": (0.0, 0.0, 0.0)}
T0 = int(pd.Timestamp("2022-01-01", tz="UTC").value // 10**6)
NOTIONAL = 19.8  # configs/epr15.example.yaml: 10 margin x 2 x 0.99


@nb.njit(cache=True)
def sim_ledger(o, h, l, c, ema, vol, ok, pen, xslip, sslip):
    """Rows: entry bar, exit bar, entry px, exit px, why, mae, then 15 minute closes / entry."""
    n = len(c)
    out = np.full((n // 16 + 1, 21), np.nan)
    k = 0
    i = 0
    slip = 0.0002
    while i < n - 15:
        if not ok[i] or not (c[i] < ema[i]) or not (vol[i] > 0) or np.isnan(l[i + 1]):
            i += 1
            continue
        lim = c[i] * (1 - 12 * vol[i] / 100)
        j = i + 1
        if not (l[j] < lim * (1 - pen)):
            i += 1
            continue
        stop = lim * 0.97
        ex = np.nan
        why = 0
        m = j
        mae = 0.0
        while m < j + 15:
            if np.isnan(o[m]):
                ex = c[m - 1] * (1 - slip - xslip)
                m -= 1
                break
            mae = min(mae, l[m] / lim - 1)
            if l[m] <= stop:
                px = stop if m == j else min(stop, o[m])
                ex = px * (1 - 0.0003 - slip - sslip)
                why = 1
                break
            out[k, 6 + m - j] = c[m] / lim
            if m == j + 14:
                ex = c[m] * (1 - slip - xslip)
                break
            m += 1
        out[k, 0] = j
        out[k, 1] = max(m, j)
        out[k, 2] = lim
        out[k, 3] = ex
        out[k, 4] = why
        out[k, 5] = max(mae, ex / lim - 1)
        k += 1
        i = max(m, j)
    return out[:k]


def coin_ledger(sym: str, pit: bool, fills=FILL) -> list[pd.DataFrame]:
    X = G1.load1m(sym)
    if X is None:
        return []
    ema, vol, warm = G1.indicators(X)
    ok = warm & np.isfinite(X["c"]) & (X["t"] >= T0 - 15 * 60_000)
    if pit:
        ok &= X["e"]
    res = []
    for name, (pen, xs, ss) in fills.items():
        r = sim_ledger(X["o"], X["h"], X["l"], X["c"], ema, vol, ok, pen, xs, ss)
        d = pd.DataFrame(
            {
                "sym": sym,
                "t_in": X["t"][r[:, 0].astype(int)],
                "t_out": X["t"][r[:, 1].astype(int)] + 60_000,  # exit at that bar's close
                "px_in": r[:, 2],
                "px_out": r[:, 3],
                "why": r[:, 4].astype(int),
                "mae": r[:, 5],
            }
        )
        d["path"] = list(r[:, 6:21])
        d["fill"] = name
        res.append(d[d.t_in >= T0])
    return res


def ledger(universe: str, fills=FILL, notional=NOTIONAL) -> pd.DataFrame:
    syms = UNIVERSES[universe] or sorted(set(F.coins()))
    with cf.ProcessPoolExecutor(10) as ex:
        parts = [p for r in ex.map(coin_ledger, syms, [UNIVERSES[universe] is None] * len(syms), [fills] * len(syms)) for p in r]  # fmt: skip
    L = pd.concat(parts, ignore_index=True)
    L["notional"] = notional
    L["fee_in"] = notional * 0.0002
    L["fee_out"] = notional * L.px_out / L.px_in * 0.0005
    L["gross"] = notional * (L.px_out / L.px_in - 1)
    L["net"] = L.gross - L.fee_in - L.fee_out
    return L.sort_values(["fill", "t_in", "sym"]).reset_index(drop=True)


def run_account(L: pd.DataFrame, max_account_loss=0.0, max_daily_loss=0.0, max_streak=0,
                pause_min=120) -> tuple[pd.DataFrame, pd.DataFrame, dict]:  # fmt: skip
    """Replay one fill scenario's trades through the account rules in time order.

    Entries are taken when the account is not paused or stopped; a skipped entry does not
    free the symbol earlier (small undercount, see LOG). Open pnl for the account stop uses
    the minute closes of every open trade.
    """
    L = L.sort_values(["t_in", "sym"]).reset_index(drop=True)
    open_, taken = [], np.zeros(len(L), bool)
    concurrent = np.zeros(len(L), int)
    realized, day, day_net, streak, paused_until = 0.0, -1, 0.0, 0, 0
    stopped_at = None

    def settle(upto):
        nonlocal realized, day, day_net, streak, paused_until, open_
        keep = []
        for r in sorted(open_, key=lambda r: r["t_out"]):
            if r["t_out"] > upto:
                keep.append(r)
                continue
            realized += r["net"]
            d = r["t_out"] // F.DAY
            if d != day:
                day, day_net = d, 0.0
            day_net += r["net"]
            streak = streak + 1 if r["net"] < 0 else 0
            if max_daily_loss and day_net <= -max_daily_loss:
                paused_until = max(paused_until, (day + 1) * F.DAY)
            elif max_streak and streak >= max_streak:
                paused_until = max(paused_until, r["t_out"] + pause_min * 60_000)
                streak = 0
        open_ = keep

    for idx, row in enumerate(L.itertuples()):
        settle(row.t_in)
        if max_account_loss:
            # open pnl of the positions still open, at the worst minute close so far
            for r in open_:
                done = min(15, (row.t_in - r["t_in"]) // 60_000)
                if done > 0:
                    p = r["path"][:done]
                    p = p[np.isfinite(p)]
                    if len(p):
                        r["worst"] = min(r.get("worst", 0.0), r["notional"] * (p.min() - 1))
            fl = sum(r.get("worst", 0.0) for r in open_)
            if realized + fl <= -max_account_loss:
                stopped_at = row.t_in
                break
        if row.t_in < paused_until:
            continue
        taken[idx] = True
        concurrent[idx] = len(open_) + 1
        open_.append({"t_in": row.t_in, "t_out": row.t_out, "net": row.net, "path": row.path,
                      "notional": row.notional})  # fmt: skip
    settle(np.iinfo(np.int64).max if stopped_at is None else stopped_at + 16 * 60_000)
    T = L[taken].copy()
    T["open_at_entry"] = concurrent[taken]
    days = pd.DataFrame({
        "net": T.groupby(T.t_out // F.DAY).net.sum(),
        "trades": T.groupby(T.t_out // F.DAY).size(),
        "losers": T[T.net < 0].groupby(T.t_out // F.DAY).sym.nunique(),
        "max_open": T.groupby(T.t_in // F.DAY).open_at_entry.max(),
    })  # fmt: skip
    idx = np.arange(T0 // F.DAY, F.T_END // F.DAY)
    days = days.reindex(idx).fillna({"net": 0.0, "trades": 0, "losers": 0, "max_open": 0})
    days.index = pd.to_datetime(days.index * F.DAY, unit="ms")
    info = {"stopped_at": None if stopped_at is None else str(pd.to_datetime(stopped_at, unit="ms")),
            "trades": int(taken.sum()), "skipped_paused": int((~taken).sum())}  # fmt: skip
    return T, days, info


def main():
    out = F.RESULTS / "G2_portfolio"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for uni in UNIVERSES:
        L = ledger(uni)
        L.drop(columns="path").to_csv(out / f"ledger_{uni}.csv.gz", index=False, float_format="%.8g")
        L.to_pickle(F.OUT / f"G2_ledger_{uni}.pkl")
        for name, kw in {
            "none": {},
            "engine_example": dict(max_account_loss=10, max_daily_loss=3, max_streak=3),
        }.items():
            T, days, info = run_account(L[L.fill == "base"], **kw)
            days.to_csv(out / f"daily_{uni}_{name}.csv.gz", float_format="%.6g")
            eq = days.net.cumsum()
            rows.append({"universe": uni, "rules": name, **info,
                         "net_usdt": round(T.net.sum(), 2), "avg_net_pct": round(100 * T.net.mean() / NOTIONAL, 4),
                         "max_dd_usdt": round((eq - eq.cummax()).min(), 2),
                         "worst_day_usdt": round(days.net.min(), 2),
                         "max_losers_one_day": int(days.losers.max()),
                         "max_open": int(days.max_open.max())})  # fmt: skip
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    json.dump({"notional": NOTIONAL, "fill": FILL}, open(out / "config.json", "w"))


if __name__ == "__main__":
    main()
