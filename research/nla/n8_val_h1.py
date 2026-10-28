"""NLA-03V: test the single pre-registered hypothesis H1 on val (LOG N3V).

H1: vwap_dev at the 24h anchor is positively related to the 7d simple return in excess of the alt
basket. Same code path as NLA-03 (n7_factors.py), only the split changes. The holdout is not read.

Usage: PYTHONPATH=research/nla uv run python research/nla/n8_val_h1.py
Output: data/research/nla/factors_val.csv, research/nla/results/N3V_val/h1.json
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from n7_factors import ANCHORS, OUT, H, build, factors, test

RES = Path("research/nla/results/N3V_val")


def leakage_check(split: str, n: int = 200) -> dict:
    """Factors computed from bars cut at the anchor must equal the ones from the full window."""
    sp = pd.read_csv("research/nla/results/N1_dataset/split.csv")
    ids = sp.loc[sp["split"] == split, "event_id"].to_numpy()
    bad, units, checked = 0, 0, 0
    for eid in ids[:n]:
        k = np.load(OUT / "bars" / f"{eid}.npz")["k"]
        t = k[:, 0]
        units += int(
            not (1.4e12 < t.min() and t.max() < 1.9e12 and np.all(np.diff(t) % 60_000 == 0))
        )
        t0 = t[0]
        for ah in ANCHORS.values():
            ta = t0 + ah * H
            if ta > t[-1]:
                continue
            a, b = factors(k, t0, ta), factors(k[t < ta], t0, ta)
            checked += 1
            bad += int(
                any(not (np.isclose(a[x], b[x]) or (np.isnan(a[x]) and np.isnan(b[x]))) for x in a)
            )
    return dict(
        split=split,
        events=min(n, len(ids)),
        anchors_checked=checked,
        mismatches=bad,
        bad_timestamp_events=units,
    )


def main():
    checks = [leakage_check("dev"), leakage_check("val")]
    print(checks)
    f = build("val")
    rng = np.random.default_rng(7)
    r = test(f, "vwap_dev", "24h", "7d", rng)
    z = r["ic_ctrl"] / r["ic_se"]
    r["z"] = z
    r["p_one_sided"] = 0.5 * math.erfc(z / math.sqrt(2))
    r["ci90"] = [r["ic_ctrl"] - 1.645 * r["ic_se"], r["ic_ctrl"] + 1.645 * r["ic_se"]]
    r["ci95"] = [r["ic_ctrl"] - 1.96 * r["ic_se"], r["ic_ctrl"] + 1.96 * r["ic_se"]]
    d = f[f["anchor"] == "24h"].dropna(subset=["vwap_dev", "ex_7d"])
    r["months"] = int(d["month"].nunique())
    r["by_category_n"] = d["listing_category"].value_counts().to_dict()
    r["by_year_n"] = {int(k): int(v) for k, v in d["year"].value_counts().sort_index().items()}
    r["passed_primary"] = bool(r["ic_ctrl"] > 0 and r["p_one_sided"] < 0.05)
    r["checks"] = checks
    RES.mkdir(parents=True, exist_ok=True)
    clean = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}
    (RES / "h1.json").write_text(json.dumps(clean, indent=1, default=float))
    for k, v in clean.items():
        print(k, v if not isinstance(v, float) else round(v, 4))


if __name__ == "__main__":
    main()
