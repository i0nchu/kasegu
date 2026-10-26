"""Data, indicators, divergence detection, trade simulator and stats for the divergence research.

Data: the 5m panel built for research/fast (data/research/fast/5m/<SYM>.npz): t o h l c qv tb f e,
385 USDT perpetuals that were ever in the point-in-time top-40-by-volume universe, delisted coins
included. e[i] says the coin was in the universe on that day (decided the day before).
Bar i covers [t[i], t[i] + bar); everything about bar i is known at its close.

Sample split: IS = 2021-01 to 2023-12 (design), OOS = 2024-01 to 2026-09 (confirmation only).
"""

from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

DATA = Path("data/research/fast/5m")
RESULTS = Path("research/divergence/results")
M5 = 300_000
DAY = 86_400_000
T_IS = int(pd.Timestamp("2021-01-01", tz="UTC").value // 10**6)
T_OOS = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)
T_END = int(pd.Timestamp("2026-10-01", tz="UTC").value // 10**6)
FEE = 0.0005
SLIP = 0.0005
COST = FEE + SLIP  # per side


def coins() -> list[str]:
    return sorted(p.stem for p in DATA.glob("*.npz"))


def load(sym: str) -> dict:
    z = np.load(DATA / f"{sym}.npz")
    X = {k: z[k] for k in z.files}
    X["fund"] = X.pop("f")
    X["elig"] = X.pop("e")
    return X


def resample(X: dict, n: int) -> dict:
    """Aggregate 5m bars to n*5m bars aligned to 00:00 UTC. Bars with any missing 5m are NaN."""
    if n == 1:
        return X
    t = X["t"]
    off = int((-(t[0] // M5)) % n)
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


# ---------- indicators ----------


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, ignore_na=True).mean().to_numpy()


def macd(c: np.ndarray, f=12, s=26, g=9):
    line = ema(c, f) - ema(c, s)
    sig = ema(line, g)
    return line / c, (line - sig) / c  # scaled by price so levels compare across time


def rsi(c: np.ndarray, n=14) -> np.ndarray:
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


def obv(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = np.sign(np.diff(c, prepend=np.nan))
    return np.nancumsum(np.nan_to_num(d) * np.nan_to_num(v))


def cvd(qv: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """Cumulative taker delta: taker buy minus taker sell quote volume."""
    return np.nancumsum(np.nan_to_num(2 * tb - qv))


def indicators(B: dict) -> dict:
    c = B["c"]
    ml, mh = macd(c)
    vol = pd.Series(B["qv"]).rolling(3, center=False, min_periods=1).mean().to_numpy()
    return {
        "macd": ml,
        "hist": mh,
        "rsi": rsi(c),
        "obv": obv(c, B["qv"]),
        "cvd": cvd(B["qv"], B["tb"]),
        "vol": vol,
    }


# ---------- divergence detection ----------


@nb.njit(cache=True)
def pivots(x, k, low):
    """p[i] = index of the pivot confirmed at bar i (pivot bar i-k is the extreme of [i-2k, i]), else -1."""
    n = len(x)
    p = np.full(n, -1, np.int64)
    for i in range(2 * k, n):
        c = i - k
        v = x[c]
        if v != v:
            continue
        ok = True
        for j in range(i - 2 * k, i + 1):
            w = x[j]
            if w != w or (low and w < v) or ((not low) and w > v):
                ok = False
                break
            if j != c and w == v and j < c:
                ok = False  # ties: the first bar is the pivot
                break
        if ok:
            p[i] = c
    return p


@nb.njit(cache=True)
def pair(piv, minsep, maxsep):
    """For each confirmed pivot (piv[i] >= 0), the previous pivot q with minsep <= p - q <= maxsep."""
    n = len(piv)
    q = np.full(n, -1, np.int64)
    last = np.full(64, -1, np.int64)  # ring of recent pivots
    m = 0
    for i in range(n):
        p = piv[i]
        if p < 0:
            continue
        # most recent earlier pivot with enough separation
        for r in range(min(m, 64)):
            c = last[(m - 1 - r) % 64]
            if p - c > maxsep:
                break
            if p - c >= minsep:
                q[i] = c
                break
        last[m % 64] = p
        m += 1
    return q


def events(B: dict, I: dict, k=3, minsep=5, maxsep=60) -> dict:
    """Pivot-pair events. For lows (bullish side) and highs (bearish side): arrays over bars with
    p, q (pivot bars), and price change between the pivots. Signal bar = confirmation bar i."""
    out = {}
    for side, x, low in ((1, B["l"], True), (-1, B["h"], False)):
        piv = pivots(x, k, low)
        q = pair(piv, minsep, maxsep)
        i = np.nonzero(q >= 0)[0]
        p = piv[i]
        qq = q[i]
        out[side] = {"i": i, "p": p, "q": qq, "dp": x[p] / x[qq] - 1}
    return out


def fwd(B: dict, i: np.ndarray, H: int, side: int) -> np.ndarray:
    """Side-adjusted gross return: enter at open of i+1, exit at close of i+H."""
    n = len(B["o"])
    ok = i + H < n
    r = np.full(len(i), np.nan)
    ii = i[ok]
    r[ok] = side * (B["c"][ii + H] / B["o"][ii + 1] - 1)
    return r


# ---------- trade simulator (same rules as research/fast) ----------


@nb.njit(cache=True)
def simulate(o, h, l, c, fund, elig, sig, stop, tp, hold, cost, trail):
    """Trades for one coin. sig[i] in {-1, 0, 1} at bar i close -> enter at o[i+1] if flat.

    stop[i], tp[i]: distances (fractions of entry) fixed at entry; NaN or <= 0 disables.
    hold: max bars held (time exit at the close of bar entry + hold - 1).
    trail[i]: trailing stop distance from the best close since entry; NaN or <= 0 disables.
    Stop fills at the stop or the open if gapped through; stop and tp inside one bar count as stop.
    Returns rows (entry_i, exit_i, side, net, funding, reason) reason 0 time, 1 stop, 2 tp.
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
            if np.isnan(o[m]):
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
                if sp == sp and l[m] <= sp:
                    ex = min(sp, op)
                    why = 1
                    break
                if tq == tq and h[m] >= tq:
                    ex = max(tq, op)
                    why = 2
                    break
            else:
                if sp == sp and h[m] >= sp:
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
        i = max(m, i + 1)
    return out[:k]


def trades(X: dict, sig, stop, tp, hold, cost=COST, trail=np.nan) -> pd.DataFrame:
    n = len(X["t"])
    b = lambda a: np.broadcast_to(np.asarray(a, float), (n,)).copy()  # noqa: E731
    r = simulate(X["o"], X["h"], X["l"], X["c"], X["fund"], X["elig"], sig.astype(np.int64),
                 b(stop), b(tp), int(hold), cost, b(trail))  # fmt: skip
    d = pd.DataFrame(r, columns=["i", "x", "side", "net", "fund", "why"])
    d["t"] = X["t"][d.i.astype(int)]
    d["tx"] = X["t"][d.x.astype(int)]
    d["net"] += d.fund
    d["gross"] = d.net - d.fund + 2 * cost
    return d


# ---------- statistics ----------


def day_t(t: np.ndarray, r: np.ndarray) -> tuple[float, float, int]:
    """Mean per trade and t-stat with trades averaged per day first (same-day events are correlated)."""
    ok = ~np.isnan(r)
    if ok.sum() < 3:
        return np.nan, np.nan, 0
    s = pd.Series(r[ok]).groupby(t[ok] // DAY).mean()
    if len(s) < 3 or s.std() == 0:
        return r[ok].mean(), np.nan, len(s)
    return r[ok].mean(), s.mean() / s.std() * np.sqrt(len(s)), len(s)


def daily(d: pd.DataFrame, w: float, t0=T_IS, t1=T_END) -> pd.Series:
    """Daily account return when each trade uses a fixed fraction w of equity, booked on exit day."""
    day = (d.tx // DAY) * DAY
    s = (w * d.net).groupby(day).sum()
    return s.reindex(np.arange(t0, t1, DAY), fill_value=0.0)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": cagr, "sharpe": sh, "maxdd": dd}
