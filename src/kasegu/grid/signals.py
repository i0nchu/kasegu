"""Daily signals of the trend-aligned grid, from closed bars only (research mgrid-trend).

All functions take bars that closed before the decision time (00:00 UTC)."""

import math

import numpy as np

EXCLUDE = {
    "BTCUSDT", "ETHUSDT", "TSLAUSDT", "NVDAUSDT", "MSTRUSDT", "CRCLUSDT", "INTCUSDT", "QQQUSDT",
    "SOXLUSDT", "SOXSUSDT", "SPCXUSDT", "SAMSUNGUSDT", "SKHYNIXUSDT", "SKHYUSDT", "SNDKUSDT",
    "MRVLUSDT", "MUUSDT", "EWYUSDT", "XAUUSDT", "XAGUSDT", "PAXGUSDT", "DEFIUSDT",
}  # fmt: skip
DAY_MS = 86_400_000


def _logret(closes) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    return np.diff(np.log(c))


def sigma_7d(hourly_closes) -> float | None:
    """Daily vol from the last 168 hourly returns (needs >= 120), sample std * sqrt(24)."""
    r = _logret(hourly_closes[-169:])
    r = r[~np.isnan(r)]
    if len(r) < 120:
        return None
    return float(np.std(r, ddof=1) * math.sqrt(24))


def rv_3d(hourly_closes) -> float | None:
    """Daily vol from the last 72 hourly returns, sample std * sqrt(24)."""
    if len(hourly_closes) < 73:
        return None
    r = _logret(hourly_closes[-73:])
    if np.isnan(r).any():
        return None
    return float(np.std(r, ddof=1) * math.sqrt(24))


def above_sma(daily_closes, n: int) -> bool | None:
    """Last close above the mean of the last n closes (the close itself included)."""
    if len(daily_closes) < n:
        return None
    c = np.asarray(daily_closes[-n:], dtype=float)
    return bool(c[-1] > c.mean())


def direction(btc_daily, coin_daily, btc_n: int = 100, coin_n: int = 50) -> int:
    """+1 long grid if BTC > SMA100 and coin > SMA50, -1 if both below, 0 otherwise."""
    b, c = above_sma(btc_daily, btc_n), above_sma(coin_daily, coin_n)
    if b is None or c is None:
        return 0
    if b and c:
        return 1
    if not b and not c:
        return -1
    return 0


def universe(
    daily_qv: dict[str, list[float]],
    onboard: dict[str, int],
    now: int,
    top: int = 40,
    min_age_days: int = 60,
) -> list[str]:
    """Top `top` symbols by mean quote volume of the last 30 closed days, listed >= min_age_days.
    BTC / ETH count for the ranking (they are excluded later), as in the research universe."""
    ranked = []
    for s, qv in daily_qv.items():
        if len(qv) < 30 or now - onboard.get(s, now) < min_age_days * DAY_MS:
            continue
        ranked.append((float(np.mean(qv[-30:])), s))
    ranked.sort(reverse=True)
    return [s for _, s in ranked[:top]]
