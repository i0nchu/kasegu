"""Hourly panel (time x coin) of o h l c qv tb fund elig, built from the 5m files. Cached."""

from pathlib import Path

import numpy as np

import common as C

P = Path("data/research/divergence_panel_{}.npz")


def build(n: int = 12) -> dict:
    p = Path(str(P).format(n))
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return {k: z[k] for k in z.files}
    syms = C.coins()
    step = n * C.M5
    t0 = int(np.datetime64("2020-06-01", "ms").astype(np.int64))
    T = np.arange(t0, C.T_END, step, dtype=np.int64)
    out = {k: np.full((len(T), len(syms)), np.nan, np.float32) for k in ("o", "h", "l", "c", "qv", "tb", "fund")}
    out["elig"] = np.zeros((len(T), len(syms)), bool)
    for j, s in enumerate(syms):
        B = C.resample(C.load(s), n)
        idx = (B["t"] - t0) // step
        ok = (idx >= 0) & (idx < len(T)) & ((B["t"] - t0) % step == 0)
        for k in ("o", "h", "l", "c", "qv", "tb", "fund"):
            out[k][idx[ok], j] = B[k][ok]
        out["elig"][idx[ok], j] = B["elig"][ok]
    out["t"] = T
    out["names"] = np.array(syms)
    np.savez(p, **out)
    return out
