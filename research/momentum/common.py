"""Hourly panel, cost model and metrics shared by the breakout / momentum experiments.

Panel: 1h USDⓈ-M klines and funding for 40 coins on one hourly grid, 2020-01-01 to 2026-10-01 UTC.
Bar i covers [GRID[i], GRID[i] + 1h); everything about bar i is known at its close.
A coin is NaN before its listing and after any gap (no forward fill of prices).

Execution convention for weight-based strategies: weights decided at the close of bar i are held
over bar i+1, so the position earns c[i+1] / c[i] - 1 and pays COST on every unit of turnover.
Funding: a position held across a funding timestamp pays w * rate (longs pay positive rates).

Sample split: IS = 2020-01 to 2023-12 (design), OOS = 2024-01 to 2026-09 (only for confirmation).
The universe is today's 40 liquid coins, so the sample has survivorship bias in favour of longs.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HIST = Path("data/history/monthly")
OUT = Path("data/research/momentum")
RESULTS = Path("research/momentum/results")
H = 3_600_000
START = int(pd.Timestamp("2020-01-01", tz="UTC").value // 10**6)
END = int(pd.Timestamp("2026-10-01", tz="UTC").value // 10**6)
GRID = np.arange(START, END, H, dtype=np.int64)
N = len(GRID)
OOS = int(np.searchsorted(GRID, int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)))

# Binance VIP0 taker 0.05% plus 0.05% slippage per side, the base case for market / stop orders.
FEE = 0.0005
SLIP = 0.0005
COST = FEE + SLIP

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "SUIUSDT",
    "1000PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "TIAUSDT",
    "INJUSDT", "SEIUSDT", "ENAUSDT", "1000BONKUSDT", "FETUSDT", "ORDIUSDT", "WLDUSDT",
    "1000FLOKIUSDT", "AAVEUSDT", "ATOMUSDT", "CRVUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT",
    "JUPUSDT", "ONDOUSDT", "PENDLEUSDT", "UNIUSDT", "XLMUSDT",
]  # fmt: skip


def _read_zip(p: Path) -> pd.DataFrame:
    with zipfile.ZipFile(p) as z:
        raw = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(d.iloc[0, 0]).lstrip("-").isdigit():
        d = d.iloc[1:]
    return d


def _build(symbol: str) -> dict:
    parts = [_read_zip(p) for p in sorted((HIST / "klines" / symbol / "1h").glob("*.zip"))]
    k = pd.concat(parts).iloc[:, [0, 1, 2, 3, 4, 7]].astype(float)
    k.columns = ["t", "o", "h", "l", "c", "qv"]
    k["t"] = k.t.astype(np.int64)
    k.loc[k.t > 10**14, "t"] //= 1000
    k = k.drop_duplicates("t").set_index("t").sort_index().reindex(GRID)
    out = {c: k[c].to_numpy() for c in ("o", "h", "l", "c", "qv")}

    f = np.zeros(N)
    fparts = [_read_zip(p) for p in sorted((HIST / "fundingRate" / symbol).glob("*.zip"))]
    if fparts:
        d = pd.concat(fparts)
        t = d.iloc[:, 0].astype(np.int64).to_numpy()
        r = d.iloc[:, -1].astype(float).to_numpy()
        i = (t - START) // H  # funding stamped inside bar i is paid by the position held over bar i
        ok = (i >= 0) & (i < N)
        np.add.at(f, i[ok], r[ok])
    out["fund"] = f
    return out


def panel() -> dict:
    """Arrays of shape (N, n_coins): o h l c qv fund. Cached in data/research/momentum."""
    p = OUT / "panel_1h.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = [_build(s) for s in UNIVERSE]
        np.savez(p, **{k: np.stack([c[k] for c in cols], axis=1) for k in cols[0]})
    z = np.load(p)
    return {k: z[k] for k in z.files}


def resample(P: dict, hours: int) -> dict:
    """Aggregate the hourly panel to `hours`-hour bars aligned to 00:00 UTC (NaN if any hour missing)."""
    n = N // hours
    sl = slice(0, n * hours)

    def r(a):
        return a[sl].reshape(n, hours, a.shape[1])

    import warnings

    with warnings.catch_warnings():  # all-NaN slices are coins not listed yet
        warnings.simplefilter("ignore", RuntimeWarning)
        h, l = np.nanmax(r(P["h"]), axis=1), np.nanmin(r(P["l"]), axis=1)
    # first / last traded price inside the bar (an exchange gap of a few hours is not a missing bar)
    o = pd.DataFrame(P["o"][sl]).bfill(limit=hours - 1).to_numpy().reshape(n, hours, -1)[:, 0]
    c = pd.DataFrame(P["c"][sl]).ffill(limit=hours - 1).to_numpy().reshape(n, hours, -1)[:, -1]
    out = {
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "qv": r(P["qv"]).sum(axis=1),
        "fund": r(P["fund"]).sum(axis=1),
    }
    out["t"] = GRID[sl][::hours]
    return out


def run_weights(W: np.ndarray, B: dict, cost: float = COST) -> pd.Series:
    """Per-bar portfolio return of weights W (decided at bar close, held over the next bar).

    W is signed exposure as a fraction of equity per coin; sum(|W|) is the gross leverage.
    Returns net of trading cost and funding, indexed by bar start time.
    """
    c = pd.DataFrame(B["c"]).ffill(limit=3).to_numpy()  # bridge short exchange outages
    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1
    ret = np.nan_to_num(ret)
    W = np.nan_to_num(W)
    held = np.zeros_like(W)
    held[1:] = W[:-1]
    # a coin that stops trading is closed at its last price
    gross = (held * ret).sum(axis=1)
    fund = (held * np.nan_to_num(B["fund"])).sum(axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0)).sum(axis=1)
    turn = np.concatenate([[0], turn[:-1]])  # trade at the open of the bar where it is held
    r = gross - fund - turn * cost
    return pd.Series(r, index=pd.to_datetime(B["t"], unit="ms", utc=True))


def metrics(r: pd.Series, per_year: float | None = None) -> dict:
    """Standard summary of a periodic return series (compounded)."""
    r = r.dropna()
    if per_year is None:
        dt = (r.index[1] - r.index[0]).total_seconds()
        per_year = 365 * 86400 / dt
    eq = (1 + r).cumprod()
    years = len(r) / per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = r.std() * np.sqrt(per_year)
    dd = (eq / eq.cummax() - 1).min()
    return {
        "cagr": round(float(cagr), 4),
        "vol": round(float(sd), 4),
        "sharpe": round(float(r.mean() * per_year / sd), 2) if sd > 0 else 0.0,
        "mdd": round(float(dd), 4),
        "total": round(float(eq.iloc[-1] - 1), 4),
    }


def split_metrics(r: pd.Series) -> dict:
    cut = pd.Timestamp("2024-01-01", tz="UTC")
    out = {"all": metrics(r), "is": metrics(r[r.index < cut]), "oos": metrics(r[r.index >= cut])}
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    out["by_year"] = {int(k): round(float(v), 4) for k, v in yearly.items()}
    return out


def flat(name: str, m: dict) -> dict:
    row = {"name": name}
    for part in ("all", "is", "oos"):
        for k, v in m[part].items():
            row[f"{part}_{k}"] = v
    for y, v in m["by_year"].items():
        row[f"y{y}"] = v
    return row
