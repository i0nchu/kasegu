"""mgrid-regime: data loading, regime factors and martingale grid episode simulation.

Data: per coin 5m npz (t in 5m units since epoch, o h l c, f = funding rate stamped in the bar it is
paid, e = point-in-time top-40 eligibility decided the previous day). See HYPOTHESIS.md.
"""

import os
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("MGRID_DATA", "/home/user/data/mgrid"))
DAY5 = 288  # 5m bars per day
H = 864  # 72h episode
EXCLUDE = {
    "BTCUSDT", "ETHUSDT", "TSLAUSDT", "NVDAUSDT", "MSTRUSDT", "CRCLUSDT", "INTCUSDT", "QQQUSDT",
    "SOXLUSDT", "SOXSUSDT", "SPCXUSDT", "SAMSUNGUSDT", "SKHYNIXUSDT", "SKHYUSDT", "SNDKUSDT",
    "MRVLUSDT", "MUUSDT", "EWYUSDT", "XAUUSDT", "XAGUSDT", "PAXGUSDT", "DEFIUSDT",
}  # fmt: skip
M = 1.5
NLEV = 6
W = M ** np.arange(NLEV)
W = W / W.sum()  # notional share of each order; full grid = 1
MAKER, TAKER = 0.0002, 0.0005
SLIP_IN, SLIP_OUT = 0.0002, 0.0005
PEN = 0.0005
MARGIN = (100 / 3) / 104  # isolated margin as a share of slot notional
W_ = W
DAY5_ = DAY5


def day_ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value // 10**9 // 300)


SPLITS = {
    "dev": (day_ts("2020-07-01"), day_ts("2023-06-27")),
    "val": (day_ts("2023-07-08"), day_ts("2024-12-28")),
    "hold": (day_ts("2025-01-08"), day_ts("2026-09-27")),
}


def coins() -> list[str]:
    return sorted(p.stem for p in DATA.glob("*.npz") if p.stem not in EXCLUDE)


def load(sym: str) -> dict:
    z = np.load(DATA / f"{sym}.npz")
    X = {k: z[k] for k in z.files}
    X["t"] = X["t"].astype(np.int64)
    for k in ("o", "h", "l", "c", "f"):
        X[k] = X[k].astype(np.float64)
    return X


def hourly(X: dict) -> pd.DataFrame:
    """Closed 1h bars indexed by hour number (t // 12)."""
    hr = X["t"] // 12
    d = pd.DataFrame({"hr": hr, "o": X["o"], "h": X["h"], "l": X["l"], "c": X["c"]})
    g = d.groupby("hr")
    out = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "n": g.c.count()})
    out.loc[out.n < 12, ["o", "h", "l", "c"]] = np.nan
    full = np.arange(out.index.min(), out.index.max() + 1)
    return out.reindex(full)


def _adx(h, l, c, n=14):
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    a = 1 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.ewm(alpha=a, adjust=False).mean()


def _resample(hb: pd.DataFrame, k: int) -> pd.DataFrame:
    g = (hb.index // k) * k
    r = hb.groupby(g).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("c", "count"))
    r.loc[r.n < k, ["o", "h", "l", "c"]] = np.nan
    return r


def features(hb: pd.DataFrame) -> pd.DataFrame:
    """Factors known at the start of each UTC day, indexed by day-start hour (multiple of 24).
    Row for day-start hour s uses hours < s only."""
    c, h, l = hb.c, hb.h, hb.l
    r = np.log(c).diff()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    F = {}
    for L in (3, 14):
        n = 24 * L
        F[f"ER{L}"] = (c - c.shift(n)).abs() / c.diff().abs().rolling(n, min_periods=n).sum()
        r24 = np.log(c).diff(24)
        F[f"VR{L}"] = r24.rolling(n, min_periods=n).var() / (24 * r.rolling(n, min_periods=n).var())
        F[f"CHOP{L}"] = np.log10(
            tr.rolling(n, min_periods=n).sum() / (h.rolling(n, min_periods=n).max() - l.rolling(n, min_periods=n).min())
        ) / np.log10(n)
        F[f"AC1{L}"] = r.rolling(n, min_periods=n).corr(r.shift())
        F[f"VOLR{L}"] = r.rolling(24, min_periods=24).std() / r.rolling(n, min_periods=n).std()
        F[f"RV{L}"] = r.rolling(n, min_periods=n).std() * np.sqrt(24)
    # ADX: L=3 on 4h bars, L=14 on daily bars; value of the last closed bar
    for L, k in ((3, 4), (14, 24)):
        rb = _resample(hb, k)
        a = _adx(rb.h, rb.l, rb.c)
        a.index = a.index + k  # known at the end of the bar
        F[f"ADX{L}"] = a.reindex(hb.index).ffill(limit=k)
    F["SIG"] = r.rolling(168, min_periods=120).std() * np.sqrt(24)
    F = pd.DataFrame(F)
    # shift by one hour: row s must use hours < s
    F = F.shift(1)
    # forward 72h ER (target T1), hours s .. s+71
    n = 72
    fwd = (c.shift(-n + 1) - c.shift(1)).abs() / c.diff().abs().rolling(n, min_periods=n).sum().shift(-n + 1)
    F["fER"] = fwd
    F["fRET"] = c.shift(-n + 1) / c.shift(1) - 1
    starts = F.index[F.index % 24 == 0]
    return F.loc[starts]


@nb.njit(cache=True)
def episode(o, h, l, c, f, i0, step, side, pen, slip_out, maker, taker, slip_in, tp_same_bar, MARGIN):
    """One martingale grid episode starting at the open of bar i0. side +1 long, -1 short.
    Returns (net return on full grid notional, exit bar, reason, levels filled, funding, mtm d1, mtm d2).
    reason: 0 time, 1 tp, 2 stop, 3 liquidation, 4 data end."""
    n = len(o)
    p0 = o[i0]
    qty = 0.0
    cost = 0.0  # sum qty*price (notional at fill)
    fees = 0.0
    fund = 0.0
    filled = 0
    # base order (taker)
    px = p0 * (1 + side * slip_in)
    w0 = W_[0]
    q = w0 / px
    qty += q
    cost += q * px
    fees += w0 * taker
    filled = 1
    stop = p0 * (1 - side * NLEV * step)
    mtm1 = np.nan
    mtm2 = np.nan
    last = min(i0 + H, n) - 1
    j = i0
    while j <= last:
        if np.isnan(c[j]):
            j += 1
            continue
        avg = cost / qty
        tp_pre = avg * (1 + side * step)
        q_pre, c_pre, f_pre = qty, cost, fees
        new_fill = False
        # adverse extreme first: fills, then stop / liquidation
        ext = l[j] if side == 1 else h[j]
        while filled < NLEV:
            lv = p0 * (1 - side * filled * step)
            if (side == 1 and ext <= lv * (1 - pen)) or (side == -1 and ext >= lv * (1 + pen)):
                wq = W_[filled]
                q = wq / lv
                qty += q
                cost += q * lv
                fees += wq * maker
                filled += 1
                new_fill = True
            else:
                break
        avg = cost / qty
        # liquidation check at the adverse extreme
        upnl = side * (qty * ext - cost)
        if upnl - fees + fund <= -MARGIN:
            # if the stop would trigger before liquidation, stop takes precedence
            stop_hit = (side == 1 and ext <= stop) or (side == -1 and ext >= stop)
            if stop_hit:
                sp = stop
                if (side == 1 and o[j] < stop) or (side == -1 and o[j] > stop):
                    sp = o[j]
                sp = sp * (1 - side * slip_out)
                spnl = side * (qty * sp - cost)
                if spnl - fees + fund - qty * sp * taker > -MARGIN:
                    pnl = spnl - fees - qty * sp * taker + fund
                    return pnl, j, 2, filled, fund, mtm1, mtm2
            return -MARGIN - 0.001, j, 3, filled, fund, mtm1, mtm2
        if (side == 1 and ext <= stop) or (side == -1 and ext >= stop):
            sp = stop
            if (side == 1 and o[j] < stop) or (side == -1 and o[j] > stop):
                sp = o[j]
            sp = sp * (1 - side * slip_out)
            pnl = side * (qty * sp - cost) - fees - qty * sp * taker + fund
            return pnl, j, 2, filled, fund, mtm1, mtm2
        # favourable extreme: take profit with the order resting before this bar.
        # Default (tp_same_bar False): if this bar also filled new levels, assume the adverse move came
        # last and allow no take profit in this bar. tp_same_bar True: favourable move first.
        if j > i0:
            fav = h[j] if side == 1 else l[j]
            hit = (side == 1 and fav >= tp_pre * (1 + pen)) or (side == -1 and fav <= tp_pre * (1 - pen))
            if hit and (tp_same_bar or not new_fill):
                pnl = side * (q_pre * tp_pre - c_pre) - f_pre - q_pre * tp_pre * maker + fund
                return pnl, j, 1, filled, fund, mtm1, mtm2
        # funding paid over this bar
        if f[j] != 0.0:
            fund -= side * f[j] * qty * c[j]
        k = j - i0 + 1
        if k == DAY5_:
            mtm1 = side * (qty * c[j] - cost) - fees + fund
        elif k == 2 * DAY5_:
            mtm2 = side * (qty * c[j] - cost) - fees + fund
        j += 1
    # time exit at the close of the last bar (or data end)
    jj = last
    while jj > i0 and np.isnan(c[jj]):
        jj -= 1
    xp = c[jj] * (1 - side * slip_out)
    pnl = side * (qty * xp - cost) - fees - qty * xp * taker + fund
    reason = 0 if last == i0 + H - 1 else 4
    return pnl, jj, reason, filled, fund, mtm1, mtm2



@nb.njit(cache=True)
def run_days(o, h, l, c, f, starts, steps, side, pen, slip_out, maker, taker, slip_in, tp_same_bar, margin):
    m = len(starts)
    out = np.full((m, 7), np.nan)
    for k in range(m):
        i0 = starts[k]
        if steps[k] != steps[k] or i0 < 0 or np.isnan(o[i0]):
            continue
        r = episode(o, h, l, c, f, i0, steps[k], side, pen, slip_out, maker, taker, slip_in, tp_same_bar, margin)
        a0, a1, a2, a3, a4, a5, a6 = r
        out[k, 0] = a0
        out[k, 1] = a1
        out[k, 2] = a2
        out[k, 3] = a3
        out[k, 4] = a4
        out[k, 5] = a5
        out[k, 6] = a6
    return out


def coin_table(sym: str, split: str, kset=(0.5, 1.0), pen=PEN, slip_out=SLIP_OUT, maker=MAKER, taker=TAKER,
               slip_in=SLIP_IN, tp_same_bar=False, margins=(("", MARGIN), ("N", 25 / 104))) -> pd.DataFrame | None:
    """One row per eligible coin-day: factors, targets and grid episode results for every (side, k)."""
    X = load(sym)
    hb = hourly(X)
    F = features(hb)
    t = X["t"]
    day0 = F.index.to_numpy() * 12  # 5m index of day start
    pos = np.searchsorted(t, day0)
    ok = (pos < len(t)) & (t[np.clip(pos, 0, len(t) - 1)] == day0)
    F = F[ok].copy()
    pos = pos[ok]
    F["elig"] = X["e"][pos]
    F = F[F.elig & F.SIG.notna()]
    a, b = SPLITS[split]
    F = F[(F.index.to_numpy() * 12 >= a) & (F.index.to_numpy() * 12 <= b)]
    pos = np.searchsorted(t, F.index.to_numpy() * 12)
    F["t5"] = F.index.to_numpy() * 12
    F["sym"] = sym
    if F.empty:
        return None
    sig = F.SIG.clip(0.005, 0.10).to_numpy()
    for pre, mgn in margins:
        for k in kset:
            st = (k * sig).astype(np.float64)
            for side, nm in ((1, "L"), (-1, "S")):
                r = run_days(X["o"], X["h"], X["l"], X["c"], X["f"], pos.astype(np.int64), st, side,
                             pen, slip_out, maker, taker, slip_in, tp_same_bar, mgn)
                tag = f"{pre}{nm}{k}"
                F[f"{tag}_ret"] = r[:, 0]
                F[f"{tag}_end"] = r[:, 1] - pos  # bars from start to exit bar
                F[f"{tag}_why"] = r[:, 2]
                F[f"{tag}_lev"] = r[:, 3]
                F[f"{tag}_fund"] = r[:, 4]
                F[f"{tag}_m1"] = r[:, 5]
                F[f"{tag}_m2"] = r[:, 6]
    return F.reset_index(drop=True)
