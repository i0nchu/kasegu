# ruff: noqa: E741
"""Fast 1m bar approximation of the kasegu engine for parameter and factor research.

Same rules as the engine (EMA trend, limit at k x volatility from the close, fixed stop,
3 tier trailing stop, max hold), but exits are evaluated on an O-L-H-C / O-H-L-C path
inside each 1m bar instead of tick by tick. Validated against the engine in validate.py.
"""

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/history")
TICKS = Path("data/research/ticks.json")


def tick_of(symbol: str) -> float:
    import json

    return json.loads(TICKS.read_text()).get(symbol, 0.0) if TICKS.exists() else 0.0


MIN = 60_000


@dataclass
class Cfg:
    model: str = "atr_amplitude"  # atr_amplitude | atr_only | quantile
    k: float = 3.0
    ema: int = 240
    vol: int = 60
    q: float = 0.95
    lookback: int = 720
    sl: float = 0.6
    low_th: float = 0.3
    low_protect: float = 0.2
    th1: float = 0.8
    trail1: float = 0.35
    th2: float = 2.0
    trail2: float = 0.20
    max_hold: int = 30  # minutes
    maker: float = 0.02  # %
    taker: float = 0.05  # %
    slip: float = 0.02  # %
    notional: float = 297.0  # 30 margin x10 minus 1% cost reserve
    trend: str = "with"  # with | against
    level_slip: float = 0.03  # % extra loss on stop / trailing exits, ticks jump past the level
    sides: str = "both"  # both | long | short
    exit: str = "tiers"  # tiers (engine) | time: hard stop + max hold only | bracket: tp / sl
    tp: float = 0.0  # take profit % for exit = bracket
    penetrate: float = 0.0  # % the price must trade beyond the limit to count as filled
    mode: str = "fade"  # fade: limit order (engine) | breakout: stop order the other way
    tick: float = 0.0  # price tick; quotes are last trade +- tick / 2 like the replay
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- data


def _read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, header=None, usecols=range(6))
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.astype(float)
    df.columns = ["t", "o", "h", "l", "c", "v"]
    df["t"] = df["t"].astype(np.int64)
    df.loc[df.t > 10**14, "t"] //= 1000
    return df


def load_klines(symbol: str, months: list[str]) -> pd.DataFrame:
    parts = []
    for m in months:
        p = CACHE / "monthly/klines" / symbol / "1m" / f"{symbol}-1m-{m}.zip"
        if p.exists():
            parts.append(_read_zip(p))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    # fill missing minutes with the previous close, like the engine's trade_bars
    full = np.arange(df.t.iloc[0], df.t.iloc[-1] + MIN, MIN)
    df = df.set_index("t").reindex(full)
    c = df.c.ffill()
    for col in ("o", "h", "l"):
        df[col] = df[col].fillna(c)
    df["c"], df["v"] = c, df.v.fillna(0)
    return df.rename_axis("t").reset_index()


def load_funding(symbol: str, months: list[str]) -> np.ndarray:
    rows = []
    for m in months:
        p = CACHE / "monthly/fundingRate" / symbol / f"{symbol}-fundingRate-{m}.zip"
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
            d = pd.read_csv(f)
        rows.append(d.iloc[:, [0, 2]].to_numpy(float))
    return np.concatenate(rows) if rows else np.zeros((0, 2))


def bars_from_trades(symbol: str, days: list[str], warm: pd.DataFrame) -> pd.DataFrame:
    """1m bars built from cached aggTrades exactly like the engine replay does."""
    parts = []
    for d in days:
        p = CACHE / "daily/aggTrades" / symbol / f"{symbol}-aggTrades-{d}.zip"
        with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
            tr = pd.read_csv(f, usecols=[1, 2, 5], header=None, skiprows=1)
        tr.columns = ["p", "q", "t"]
        tr["m"] = tr.t - tr.t % MIN
        g = tr.groupby("m")
        parts.append(
            pd.DataFrame(
                {
                    "t": g.m.first(),
                    "o": g.p.first(),
                    "h": g.p.max(),
                    "l": g.p.min(),
                    "c": g.p.last(),
                    "v": g.q.sum(),
                }
            )
        )
    df = pd.concat([warm] + parts).drop_duplicates("t", keep="last").sort_values("t")
    full = np.arange(df.t.iloc[0], df.t.iloc[-1] + MIN, MIN)
    df = df.set_index("t").reindex(full)
    c = df.c.ffill()
    for col in ("o", "h", "l"):
        df[col] = df[col].fillna(c)
    df["c"], df["v"] = c, df.v.fillna(0)
    return df.rename_axis("t").reset_index()


# ---------------------------------------------------------------- signals


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x).rolling(n).mean()
    return s.to_numpy()


def signals(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Per bar: side (+1/-1/0), limit price and features, decided at the bar close."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    ema = pd.Series(c).ewm(span=cfg.ema, adjust=False).mean().to_numpy()
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    atr_pct = _roll_mean(tr, cfg.vol) / c * 100
    amp_pct = _roll_mean((h - l) / c * 100, cfg.vol)
    side = np.sign(c - ema)
    if cfg.trend == "against":
        side = -side
    if cfg.model == "atr_amplitude":
        dist = (atr_pct + amp_pct) / 2 * cfg.k
    elif cfg.model == "atr_only":
        dist = atr_pct * cfg.k
    elif cfg.model == "quantile":
        n = cfg.lookback
        dn = np.maximum((prev - l) / prev * 100, 0)  # next bar adverse move for longs
        up = np.maximum((h - prev) / prev * 100, 0)
        qd = pd.Series(dn).rolling(n).quantile(cfg.q).to_numpy()
        qu = pd.Series(up).rolling(n).quantile(cfg.q).to_numpy()
        dist = np.where(side > 0, qd, qu) * cfg.k
    else:
        raise ValueError(cfg.model)
    warm = max(cfg.ema * 3, cfg.vol + 1, cfg.lookback + 1 if cfg.model == "quantile" else 0)
    ok = np.isfinite(dist) & (dist > 0)
    ok[:warm] = False
    side = np.where(ok, side, 0)
    if cfg.sides == "long":
        side = np.where(side > 0, side, 0)
    elif cfg.sides == "short":
        side = np.where(side < 0, side, 0)
    half = cfg.tick / 2
    ref = c + half  # engine reference is the ask
    limit = ref * (1 - side * dist / 100)
    if cfg.tick:
        limit = np.where(
            side > 0,
            np.floor(limit / cfg.tick + 1e-9) * cfg.tick,
            np.ceil(limit / cfg.tick - 1e-9) * cfg.tick,
        )
    # post-only: a limit at or through the touch is rejected
    post = ((side > 0) & (limit >= c + half)) | ((side < 0) & (limit <= c - half))
    want = side.copy()
    side = np.where(post, 0, side)
    out = pd.DataFrame(
        {
            "t": df.t.to_numpy(),
            "side": side.astype(int),
            "want": want.astype(int),  # side before the post-only check
            "limit": limit,
            "dist": dist,
            "atr_pct": atr_pct,
            "amp_pct": amp_pct,
            "ema_gap": (c - ema) / ema * 100,
            "ema_slope": (ema - np.r_[np.full(60, np.nan), ema[:-60]]) / ema * 100,
            "ret15": (c / np.r_[np.full(15, np.nan), c[:-15]] - 1) * 100,
            "ret60": (c / np.r_[np.full(60, np.nan), c[:-60]] - 1) * 100,
            "ret240": (c / np.r_[np.full(240, np.nan), c[:-240]] - 1) * 100,
            "vol_ratio": _roll_mean(df.v.to_numpy(), 5) / _roll_mean(df.v.to_numpy(), 240),
            "atr_ratio": atr_pct / pd.Series(atr_pct).rolling(1440).mean().to_numpy(),
        }
    )
    return out


# ---------------------------------------------------------------- simulation


def _path(o, h, l, c):
    return (o, l, h, c) if c >= o else (o, h, l, c)


def simulate(
    df: pd.DataFrame,
    cfg: Cfg,
    funding: np.ndarray | None = None,
    sig: pd.DataFrame | None = None,
    start_ms: int = 0,
    log: bool = False,
    entries: np.ndarray | None = None,
):
    """Sequential one-position-per-symbol simulation. Returns one row per trade.

    log: also return every candidate order with its outcome (filled / expired / rejected:why).
    entries: control group, (bar, side) pairs entered at that bar's open instead of signals.
    """
    sig = signals(df, cfg) if sig is None else sig
    t = df.t.to_numpy()
    O, H, L, C = (df[x].to_numpy() for x in "ohlc")
    side = sig.side.to_numpy()
    limit = sig.limit.to_numpy()
    n = len(df)
    # candidate fill bars: order from bar i-1 trades through in bar i
    s_prev, lim_prev = np.r_[0, side[:-1]], np.r_[np.nan, limit[:-1]]
    half = cfg.tick / 2
    # simulated fills need the opposite quote (trade +- half tick) through the limit
    pen = lim_prev * cfg.penetrate / 100
    hit = ((s_prev > 0) & (L + half < lim_prev - pen)) | (
        (s_prev < 0) & (H - half > lim_prev + pen)
    )
    hit &= t >= start_ms
    cand = np.flatnonzero(hit)
    if entries is not None:
        cand = entries[:, 0].astype(int)
        s_prev = np.zeros(n, int)
        s_prev[cand] = entries[:, 1]
        lim_prev = O.copy()
    ft = funding[:, 0] if funding is not None and len(funding) else np.zeros(0)
    fr = funding[:, 1] if funding is not None and len(funding) else np.zeros(0)

    rows, spans = [], []
    free_from = 0
    for j in cand:
        if j < free_from or j < 1:
            continue
        sd, px = int(s_prev[j]), lim_prev[j]
        if cfg.mode == "breakout":
            # stop market the other way at the same level: taker fill with slippage
            ps = -sd
            entry = (px - half * sd) * (1 + cfg.slip / 100 * ps)
            fee_in = cfg.taker
        else:
            ps, entry, fee_in = sd, px, cfg.maker
        x, reason, exit_px, peak, trough, stop = _run_exit(
            j, sd, px, O, H, L, C, n, cfg, ps, entry, at_open=entries is not None
        )
        free_from = x + 1
        spans.append((j, x))
        gross = (exit_px - entry) / entry * 100 * ps
        fee_out = cfg.maker if reason == "take_profit" else cfg.taker
        fees = fee_in + fee_out * exit_px / entry
        fund = 0.0
        if len(ft):
            a, b = np.searchsorted(ft, [t[j], t[x] + MIN])
            fund = float(-ps * fr[a:b].sum() * 100)
        sg = sig.iloc[j - 1]
        rows.append(
            {
                "t": t[j],
                "exit_t": t[x],
                "side": ps,
                "entry": entry,
                "exit": exit_px,
                "reason": reason,
                "gross": gross,
                "net": gross - fees + fund,
                "funding": fund,
                "mfe": peak,
                "mae": trough,
                "hold": x - j + 1,
                **{k: sg[k] for k in sig.columns if k not in ("t", "side", "want", "limit")},
            }
        )
    tr = pd.DataFrame(rows)
    if not log:
        return tr
    return tr, _candidates(sig, spans, t >= start_ms)


def _candidates(sig: pd.DataFrame, spans, live) -> pd.DataFrame:
    """One row per bar close where the strategy wanted an order, with what happened to it."""
    want = sig.want.to_numpy()
    n = len(want)
    out = np.full(n, "expired", dtype=object)
    out[(want != 0) & (sig.side.to_numpy() == 0)] = "rejected:post_only"
    trade_id = np.full(n, -1)
    for k, (j, x) in enumerate(spans):
        out[j:x] = "rejected:in_position"  # bar closes while the position is open
        out[j - 1] = "filled"
        trade_id[j - 1] = k
    keep = (want != 0) & live
    c = sig.loc[keep, ["t", "want", "limit", "dist"]].rename(columns={"want": "side"})
    c["outcome"] = out[keep]
    c["trade_id"] = trade_id[keep]
    return c.reset_index(drop=True)


def _run_exit(j, sd, px, O, H, L, C, n, cfg: Cfg, ps=None, entry=None, at_open=False):
    """Walk the bar path from the fill. Returns (exit bar, reason, exit price, mfe, mae, stop).

    sd / px: the order side and price that trigger the fill.
    ps / entry: position side and entry price (default: the limit order itself).
    """
    ps = sd if ps is None else ps
    entry = px if entry is None else entry
    peak, trough, stop, tier = 0.0, 0.0, None, -1
    slip = cfg.slip / 100

    def prof(p):
        return (p - entry) / entry * 100 * ps

    half = cfg.tick / 2

    def out(p):  # taker exit at the bid/ask with slippage against us
        return (p - half * ps) * (1 - slip * ps)

    def level_px(pct):
        return entry * (1 + ps * pct / 100)

    filled = False
    last = None
    for b in range(j, n):
        pts = _path(O[b], H[b], L[b], C[b])
        if not filled:
            # find the fill on the path, then continue from the limit price
            if not at_open:
                i = next((i for i, a in enumerate(pts) if (a + half * sd - px) * sd < 0), 3)
                pts = pts if i == 0 else (entry,) + pts[i:]
            filled = True
        elif last is not None:
            pts = (last,) + pts  # gap from the previous close to this open
        for i in range(1, len(pts)):
            a0, a1 = pts[i - 1], pts[i]
            p0, p1 = prof(a0), prof(a1)
            if p1 >= p0:
                # rising: only the peak and the trailing stop move
                trough = min(trough, p0)
                if cfg.exit == "bracket" and cfg.tp and p1 >= cfg.tp:
                    # resting take-profit limit: maker fill at the level, no slippage
                    return (
                        b,
                        "take_profit",
                        level_px(max(cfg.tp, p0)),
                        max(peak, cfg.tp),
                        trough,
                        stop,
                    )
                peak = max(peak, p1)
                t2, s2 = _tier(peak, cfg)
                tier = max(tier, t2)
                if s2 is not None:
                    stop = s2 if stop is None else max(stop, s2)
                continue
            trough = min(trough, p1)
            lvl = -cfg.sl if stop is None else max(stop, -cfg.sl)
            if p1 <= lvl if lvl == -cfg.sl else p1 < lvl:
                hit_px = a0 if p0 <= lvl else level_px(lvl - cfg.level_slip)
                why = "stop_loss" if lvl == -cfg.sl else f"trail_t{tier}"
                return b, why, out(hit_px), peak, max(trough, lvl), stop
        last = C[b]
        if b - j + 1 >= cfg.max_hold:
            return b, "max_hold", out(C[b]), peak, trough, stop
    return n - 1, "end_of_data", out(C[n - 1]), peak, trough, stop


def _tier(peak, cfg: Cfg):
    if cfg.exit != "tiers":
        return -1, None
    if peak >= cfg.th2:
        return 2, peak * (1 - cfg.trail2)
    if peak >= cfg.th1:
        return 1, peak * (1 - cfg.trail1)
    if peak >= cfg.low_th:
        return 0, cfg.low_protect
    return -1, None


def summary(tr: pd.DataFrame, cfg: Cfg | None = None) -> dict:
    notional = (cfg or Cfg()).notional
    if tr.empty:
        return {"trades": 0, "net_pct": 0.0, "net_usdt": 0.0}
    net = tr.net.to_numpy()
    eq = np.cumsum(net)
    wins, loss = net[net > 0], net[net <= 0]
    return {
        "trades": len(tr),
        "net_pct": float(net.sum()),
        "net_usdt": float(net.sum() * notional / 100),
        "avg_net": float(net.mean()),
        "avg_gross": float(tr.gross.mean()),
        "win": float((net > 0).mean()),
        "pf": float(wins.sum() / -loss.sum()) if loss.sum() < 0 else None,
        "max_dd_pct": float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()),
        "mfe": float(tr.mfe.mean()),
        "mae": float(tr.mae.mean()),
        "hold": float(tr.hold.mean()),
    }
