"""5m data, point-in-time universe, trade simulator and metrics for the short-horizon experiments.

Data: per coin, 5m klines for the months it is in the E5 universe (download_5m.py), funding from
download_all.py. Bar i covers [t[i], t[i] + 5m); everything about bar i is known at its close.

Universe: a coin may open a trade on day d only if it was in the top-40-by-30-day-volume set
computed at the close of day d-1 (listed >= 60 days). Delisted coins trade until their last bar;
an open position on the last bar is closed at its close.

Execution: a signal at the close of bar i enters at the open of bar i+1 with a market order.
Exits are a stop (fills at the stop, or at the open when the bar gaps through it), a take profit
(fills at the level, or the open when gapping through), or a time exit at a bar's close. Stop and
take profit inside the same bar count as the stop. Every fill pays FEE + SLIP per side.

Sample split: IS = 2021-01 to 2023-12 (design), OOS = 2024-01 to 2026-09 (confirmation only).
"""

import io
import json
import zipfile
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

import e5_survivorship as E5

HIST = Path("data/history/monthly")
OUT = Path("data/research/fast")
RESULTS = Path("research/fast/results")
M5 = 300_000
DAY = 86_400_000
T_IS = int(pd.Timestamp("2021-01-01", tz="UTC").value // 10**6)
T_OOS = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)
T_END = int(pd.Timestamp("2026-10-01", tz="UTC").value // 10**6)

# Binance VIP0: taker 0.05%, maker 0.02%. Slippage 0.05% per side is the base case (E1-E7).
FEE = 0.0005
SLIP = 0.0005


def _read_zip(p: Path) -> pd.DataFrame:
    with zipfile.ZipFile(p) as z:
        raw = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(d.iloc[0, 0]).lstrip("-").isdigit():
        d = d.iloc[1:]
    return d


def coins() -> list[str]:
    return sorted(json.loads((OUT / "months.json").read_text()))


def _elig_days() -> tuple[np.ndarray, dict]:
    """Day start times and, per coin, a bool array: may open trades that day (decided the day before)."""
    D = E5.daily_panel()
    el = E5.eligible(D)
    el = np.vstack([np.zeros((1, el.shape[1]), bool), el[:-1]])
    return D["t"], {str(s): el[:, j] for j, s in enumerate(D["names"])}


_EL = None


def load(sym: str) -> dict | None:
    """5m arrays for one coin: t o h l c qv tb (taker-buy quote volume) fund elig. Cached as npz."""
    global _EL
    p = OUT / "5m" / f"{sym}.npz"
    if not p.exists():
        files = sorted((HIST / "klines" / sym / "5m").glob("*.zip"))
        if not files:
            return None
        k = pd.concat([_read_zip(f) for f in files]).iloc[:, [0, 1, 2, 3, 4, 7, 10]].astype(float)
        k.columns = ["t", "o", "h", "l", "c", "qv", "tb"]
        k["t"] = k.t.astype(np.int64)
        k.loc[k.t > 10**14, "t"] //= 1000
        k = k.drop_duplicates("t").sort_values("t")
        t0, t1 = k.t.iloc[0], k.t.iloc[-1]
        grid = np.arange(t0, t1 + M5, M5, dtype=np.int64)
        k = k.set_index("t").reindex(grid)  # gaps stay NaN (exchange outages, months not downloaded)
        f = np.zeros(len(grid))
        fz = sorted((HIST / "fundingRate" / sym).glob("*.zip"))
        if fz:
            d = pd.concat([_read_zip(z) for z in fz])
            ft = d.iloc[:, 0].astype(np.int64).to_numpy()
            i = (ft - t0) // M5  # funding stamped inside bar i is paid by a position held over bar i
            ok = (i >= 0) & (i < len(grid))
            np.add.at(f, i[ok], d.iloc[:, -1].astype(float).to_numpy()[ok])
        if _EL is None:
            _EL = _elig_days()
        dt, el = _EL
        di = np.searchsorted(dt, grid, side="right") - 1
        e = el[sym][np.clip(di, 0, len(dt) - 1)] & (di >= 0)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, t=grid, f=f, e=e, **{c: k[c].to_numpy() for c in ("o", "h", "l", "c", "qv", "tb")})
    z = np.load(p)
    X = {k: z[k] for k in z.files}
    X["fund"] = X.pop("f")
    X["elig"] = X.pop("e")
    return X


def resample(X: dict, n: int) -> dict:
    """Aggregate 5m bars to n*5m bars aligned to 00:00 UTC. Bars with any missing 5m are NaN."""
    t = X["t"]
    off = int(((-(t[0] // M5)) % n))  # first index aligned to the n-bar grid
    m = (len(t) - off) // n
    sl = slice(off, off + m * n)

    def r(a):
        return a[sl].reshape(m, n)

    out = {"t": t[sl][::n], "o": r(X["o"])[:, 0], "c": r(X["c"])[:, -1]}
    out["h"] = r(X["h"]).max(1)
    out["l"] = r(X["l"]).min(1)
    for c in ("qv", "tb", "fund"):
        out[c] = r(X[c]).sum(1)
    out["elig"] = r(X["elig"])[:, 0]
    return out


@nb.njit(cache=True)
def simulate(o, h, l, c, fund, elig, sig, stop, tp, hold, flat, cost, trail):
    """Trades for one coin. sig[i] in {-1, 0, 1} at bar i close -> enter at o[i+1] if flat.

    stop[i], tp[i]: distances (fractions of entry price) fixed at entry; NaN or <= 0 disables.
    hold: max bars held (time exit at the close of bar entry + hold - 1).
    flat[i]: True forces an exit at the close of bar i (session end).
    trail[i]: trailing stop distance (fraction) from the best close since entry; NaN or <= 0 disables.
    The trail only uses closes of bars already finished, so it never looks into the current bar.
    Returns rows (entry_i, exit_i, side, gross, funding, reason) with reason 0 time, 1 stop, 2 tp, 3 flat.
    """
    n = len(o)
    out = np.empty((n // 2 + 1, 6))
    k = 0
    i = 0
    while i < n - 1:
        s = sig[i]
        if s == 0 or not elig[i] or np.isnan(o[i + 1]) or np.isnan(c[i]):
            i += 1
            continue
        j = i + 1
        px = o[j]
        sd, td, tr = stop[i], tp[i], trail[i]
        sp = px * (1 - s * sd) if sd > 0 else np.nan
        best = px
        tq = px * (1 + s * td) if td > 0 else np.nan
        fsum = 0.0
        ex = np.nan
        why = 0
        m = j
        while m < n:
            if np.isnan(o[m]):  # data gap: exit at the last known close
                ex = c[m - 1]
                m -= 1
                why = 0
                break
            fsum += fund[m]
            op = o[m] if m > j else px
            if tr > 0:
                tl = best * (1 - s * tr)
                if sp != sp or (s > 0 and tl > sp) or (s < 0 and tl < sp):
                    sp = tl
            if s > 0:
                if sp == sp and (l[m] <= sp):
                    ex = min(sp, op)
                    why = 1
                    break
                if tq == tq and h[m] >= tq:
                    ex = max(tq, op)
                    why = 2
                    break
            else:
                if sp == sp and (h[m] >= sp):
                    ex = max(sp, op)
                    why = 1
                    break
                if tq == tq and l[m] <= tq:
                    ex = min(tq, op)
                    why = 2
                    break
            if s > 0:
                best = max(best, c[m])
            else:
                best = min(best, c[m])
            if flat[m]:
                ex = c[m]
                why = 3
                break
            if m - j + 1 >= hold or m == n - 1:
                ex = c[m]
                why = 0
                break
            m += 1
        g = s * (ex / px - 1)
        out[k, 0] = j
        out[k, 1] = m
        out[k, 2] = s
        out[k, 3] = g - 2 * cost
        out[k, 4] = -s * fsum
        out[k, 5] = why
        k += 1
        i = max(m, i + 1)  # the next entry can come from the exit bar's own close
    return out[:k]


def trades(X: dict, sig, stop, tp, hold, flat=None, cost=FEE + SLIP, trail=np.nan) -> pd.DataFrame:
    n = len(X["t"])
    flat = np.zeros(n, np.bool_) if flat is None else flat
    stop = np.broadcast_to(np.asarray(stop, float), (n,)).copy()
    tp = np.broadcast_to(np.asarray(tp, float), (n,)).copy()
    trail = np.broadcast_to(np.asarray(trail, float), (n,)).copy()
    r = simulate(X["o"], X["h"], X["l"], X["c"], X["fund"], X["elig"], sig.astype(np.int64),
                 stop, tp, int(hold), flat, cost, trail)  # fmt: skip
    d = pd.DataFrame(r, columns=["i", "x", "side", "net", "fund", "why"])
    d["t"] = X["t"][d.i.astype(int)]
    d["tx"] = X["t"][d.x.astype(int)]
    d["net"] += d.fund
    d["gross"] = d.net - d.fund + 2 * cost
    return d


def summarize(tr: pd.DataFrame, name: str, **kw) -> dict:
    """Per-trade stats on IS and OOS; returns in basis points per trade (1x notional)."""
    row = {"name": name, **kw}
    for tag, m in (("is", (tr.t >= T_IS) & (tr.t < T_OOS)), ("oos", tr.t >= T_OOS)):
        x = tr.net[m]
        row[f"{tag}_n"] = int(m.sum())
        row[f"{tag}_bp"] = 1e4 * x.mean() if len(x) else np.nan
        row[f"{tag}_gross_bp"] = 1e4 * tr.gross[m].mean() if len(x) else np.nan
        row[f"{tag}_t"] = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan
        row[f"{tag}_win"] = (x > 0).mean() if len(x) else np.nan
    return row


def daily(d: pd.DataFrame, w: float) -> pd.Series:
    """Daily portfolio return when each trade risks a fixed fraction w of equity (no compounding
    inside a day), booked on the exit day."""
    day = (d.tx // DAY) * DAY
    s = (w * d.net).groupby(day).sum()
    idx = np.arange(T_IS, T_END, DAY)
    return s.reindex(idx, fill_value=0.0)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": cagr, "sharpe": sh, "maxdd": dd}
