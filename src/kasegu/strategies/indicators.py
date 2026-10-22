import numpy as np
import pandas as pd


def ema(close: np.ndarray, period: int) -> float:
    return float(pd.Series(close).ewm(span=period, adjust=False).mean().iloc[-1])


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    # TR for bars[1:], needs previous close
    prev = close[:-1]
    h, lo = high[1:], low[1:]
    return np.maximum.reduce([h - lo, np.abs(h - prev), np.abs(lo - prev)])


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    tr = true_range(high[-period - 1 :], low[-period - 1 :], close[-period - 1 :])
    return float(tr.mean())


def amplitude_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    h, lo, c = high[-period:], low[-period:], close[-period:]
    return float(((h - lo) / c * 100).mean())
