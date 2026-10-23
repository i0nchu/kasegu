"""Shared run helpers: cached market index, run folders and publishing small results to git."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from data import OUT, UNIVERSE

RUNS = Path("data/research/runs")
RESULTS = Path("research/results")


def market() -> np.ndarray:
    p = OUT / "market_ew.npy"
    if p.exists():
        return np.load(p)
    from features import market_returns

    m = market_returns(UNIVERSE)
    np.save(p, m)
    return m


def folder(exp: str) -> Path:
    f = RUNS / exp
    f.mkdir(parents=True, exist_ok=True)
    return f


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


def publish(exp: str, files: list[str]) -> None:
    """Copy the small, reviewable outputs of an experiment into research/results/<exp>/."""
    src, dst = RUNS / exp, RESULTS / exp
    dst.mkdir(parents=True, exist_ok=True)
    for name in files:
        p = src / name
        if p.is_dir():
            shutil.copytree(p, dst / name, dirs_exist_ok=True)
        elif p.exists():
            shutil.copy(p, dst / name)


def fmt(df: pd.DataFrame, nd: int = 3) -> str:
    return df.round(nd).to_string(index=False)


def market_days() -> pd.DataFrame:
    """Each UTC day's worst 60 minute move of the equal weight market, and its event class.

    crash: the market fell more than 5% within an hour that day; stress: 2-5%; calm: less.
    """
    from data import GRID

    m = np.nan_to_num(market())
    cm = np.cumsum(m)
    r60 = np.full(len(m), np.nan)
    r60[60:] = (np.exp(cm[60:] - cm[:-60]) - 1) * 100
    d = pd.DataFrame({"day": GRID // 86_400_000, "r60": r60}).groupby("day").r60.min()
    out = d.rename("mkt_worst60").to_frame()
    out["event"] = np.select(
        [out.mkt_worst60 <= -5, out.mkt_worst60 <= -2], ["crash", "stress"], "calm"
    )
    return out.reset_index()
