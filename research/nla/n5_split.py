"""Freeze the dev / validation / holdout split (user's choice 2026-10-28): within each listing
category, events sorted by first trade time, the earliest 50% are dev, the next 25% validation and
the latest 25% holdout. Categories with fewer than 8 events go to dev. Events of one
asset inside a category stay together. The holdout is not to be read
before NLA-06.

Usage: PYTHONPATH=research/nla uv run python research/nla/n5_split.py
Needs: data/research/nla/events.csv (n3_dataset.py)
Output: research/nla/results/N1_dataset/split.csv (tracked in git, so the split cannot drift)
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/research/nla")
RES = Path("research/nla/results/N1_dataset")


def main():
    ev = pd.read_csv(OUT / "events.csv")
    ev = ev[ev["study_eligible"]].sort_values(["first_trade_time", "event_id"])
    split = []
    for _, g in ev.groupby("listing_category"):
        # spot and perp of a same-day listing share one asset key, so they never straddle a boundary
        key = g["asset_id"].astype(str)
        order = g.groupby(key, sort=False)["first_trade_time"].transform("min")
        rank = order.rank(method="dense").to_numpy()
        n = rank.max()
        r = (rank - 1) / n
        s = (
            np.where(r < 0.5, "dev", np.where(r < 0.75, "val", "hold"))
            if n >= 8
            else ["dev"] * len(g)
        )
        split.append(
            pd.DataFrame(
                {
                    "event_id": g["event_id"],
                    "split": s,
                    "listing_category": g["listing_category"],
                    "first_trade_time": g["first_trade_time"],
                }
            )
        )
    sp = pd.concat(split).sort_values("first_trade_time")
    sp["first_trade_utc"] = pd.to_datetime(sp.pop("first_trade_time"), unit="ms")
    sp.to_csv(RES / "split.csv", index=False)
    rng = sp.groupby(["listing_category", "split"])["first_trade_utc"].agg(["size", "min", "max"])
    print(rng.to_string())


if __name__ == "__main__":
    main()
