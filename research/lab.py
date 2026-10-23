"""Shared helpers for the experiments: cached bar loading and multi-coin runs."""

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd
from sim import Cfg, load_funding, load_klines, simulate, tick_of

MONTHS = [f"2025-{m:02d}" for m in (10, 11, 12)] + [f"2026-{m:02d}" for m in range(1, 10)]
IS_END = pd.Timestamp("2026-04-01", tz="UTC").value // 10**6  # in-sample: before April
OUT = Path("data/research")
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "SUIUSDT",
    "1000PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "TIAUSDT",
    "INJUSDT", "SEIUSDT", "ENAUSDT", "1000BONKUSDT", "FETUSDT", "ORDIUSDT", "WLDUSDT",
]  # fmt: skip


def bars(symbol: str, months: list[str] | None = None) -> tuple[pd.DataFrame, object]:
    months = months or MONTHS
    tag = "" if months == MONTHS else f"_{months[0]}_{months[-1]}"
    p = OUT / "bars" / f"{symbol}{tag}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    df, fund = load_klines(symbol, months), load_funding(symbol, months)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps((df, fund)))
    return df, fund


def _one(args) -> pd.DataFrame:
    symbol, cfg = args
    df, fund = bars(symbol)
    if df.empty:
        return pd.DataFrame()
    tr = simulate(df, replace(cfg, tick=tick_of(symbol)), fund)
    tr.insert(0, "symbol", symbol)
    return tr


def run(cfg: Cfg, symbols: list[str] = UNIVERSE, workers: int = 8) -> pd.DataFrame:
    """Trades of every symbol, one independent account per symbol."""
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_one, [(s, cfg) for s in symbols]))
    tr = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    tr["month"] = pd.to_datetime(tr.t, unit="ms").dt.strftime("%Y-%m")
    tr["oos"] = tr.t >= IS_END
    return tr


def table(tr: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    g = tr.groupby(by)
    return pd.DataFrame(
        {
            "n": g.size(),
            "avg_net%": g.net.mean().round(3),
            "avg_gross%": g.gross.mean().round(3),
            "win": g.net.apply(lambda x: (x > 0).mean()).round(3),
            "sum_net%": g.net.sum().round(1),
        }
    )


def _fwd_one(args) -> pd.DataFrame:
    import numpy as np
    from sim import signals

    symbol, cfg, horizons = args
    df, _ = bars(symbol)
    if df.empty:
        return pd.DataFrame()
    cfg = replace(cfg, tick=tick_of(symbol))
    sig = signals(df, cfg)
    side, limit = sig.side.to_numpy(), sig.limit.to_numpy()
    H, L, C = (df[x].to_numpy() for x in "hlc")
    sp, lp = np.r_[0, side[:-1]], np.r_[np.nan, limit[:-1]]
    half = cfg.tick / 2
    j = np.flatnonzero(((sp > 0) & (L + half < lp)) | ((sp < 0) & (H - half > lp)))
    j = j[j + max(horizons) < len(df)]
    sd, px = sp[j], lp[j]
    out = {"symbol": symbol, "t": df.t.to_numpy()[j], "side": sd}
    for h in horizons:
        # close h minutes after the fill bar, from the limit price
        out[f"r{h}"] = (C[j + h] - px) / px * 100 * sd
    feats = sig.iloc[j - 1].reset_index(drop=True)
    res = pd.DataFrame(out)
    for c in ("dist", "atr_pct", "ema_gap", "ema_slope", "ret60", "vol_ratio", "atr_ratio"):
        res[c] = feats[c].to_numpy()
    return res


def forward(cfg: Cfg, symbols: list[str] = UNIVERSE, horizons=(1, 5, 15, 30, 60)):
    """Every fill (ignoring one-position limits) with its raw forward returns in %."""
    with ProcessPoolExecutor(8) as ex:
        parts = list(ex.map(_fwd_one, [(s, cfg, horizons) for s in symbols]))
    fw = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    fw["oos"] = fw.t >= IS_END
    return fw
