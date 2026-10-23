"""Small statistics helpers: day-clustered bootstrap and grouped tables."""

import numpy as np
import pandas as pd

DAY = 86_400_000


def boot_mean(x: np.ndarray, cluster: np.ndarray, b: int = 2000, seed: int = 0) -> tuple:
    """Mean and 95% interval resampling whole clusters (days), so crash days count once."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, cluster = x[ok], np.asarray(cluster)[ok]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    _, cid = np.unique(cluster, return_inverse=True)
    s = np.bincount(cid, x)
    n = np.bincount(cid).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=b)
    m = (w @ s) / (w @ n)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(
    df: pd.DataFrame, col: str, by, cost: float = 0.0, day_col: str = "day"
) -> pd.DataFrame:
    """Per group: n, mean, median, win rate, day-clustered 95% CI of the mean, after cost."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        x = g[col].to_numpy() - cost
        m, lo, hi = boot_mean(x, g[day_col].to_numpy(), b=1000)
        rows.append(
            {
                **dict(
                    zip(
                        [by] if isinstance(by, str) else by,
                        key if isinstance(key, tuple) else (key,),
                        strict=False,
                    )
                ),
                "n": len(g),
                "days": g[day_col].nunique(),
                "mean": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "median": float(np.nanmedian(x)),
                "win": float((x > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def tail_share(x: np.ndarray, q: float = 0.05) -> dict:
    """How much of the total comes from the best / worst q of outcomes (risk premium signature)."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    k = max(1, int(len(x) * q))
    tot = x.sum()
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "skew": float(((x - x.mean()) ** 3).mean() / x.std() ** 3) if x.std() > 0 else 0.0,
        "worst_q_sum": float(x[:k].sum()),
        "best_q_sum": float(x[-k:].sum()),
        "total": float(tot),
        "p05": float(x[k - 1]),
        "p95": float(x[-k]),
    }
