"""Replay the paper engine on the research 5m panel and compare it with the backtest (D9 events,
D7 account rule). This checks that the paper ledgers trade the same rules the research tested.
Usage: PYTHONPATH=research/divergence:research/divergence/paper python research/divergence/paper/replay.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import d7_account as D7
from engine import Engine
from sources import HistSource

OUT = Path("data/paper/replay")
T0 = int(pd.Timestamp("2026-03-01", tz="UTC").value // 10**6)
T1 = int(pd.Timestamp("2026-04-01", tz="UTC").value // 10**6)  # signals recorded from here
T2 = C.T_END


def account(T, N=3):
    """D7.account with ties at the same 4h close taken in symbol order (the engine's order)."""
    open_, keep = [], []
    for r in T.sort_values(["t", "sym"], kind="stable").itertuples():
        entry = r.t + 4 * 3_600_000
        open_ = [o for o in open_ if o[0] > entry]
        if len(open_) >= N or any(o[1] == r.sym for o in open_):
            continue
        open_.append((r.tx, r.sym))
        keep.append(r.Index)
    return T.loc[keep]


def main():
    if "--reuse" not in sys.argv:
        shutil.rmtree(OUT, ignore_errors=True)
        eng = Engine(HistSource(), OUT, T0, T1 + 4 * 3_600_000, log=lambda m: None)  # first recorded bar starts at T1
        eng.run_until(T2)
    S = pd.read_csv(OUT / "signals.csv")
    T = pd.read_csv(OUT / "trades.csv")
    D = pd.read_pickle("data/research/divergence_d9.pkl")
    R, meta = D7.rules(D.assign(side=-1, net=D.net_base))  # IS percentiles need the full sample
    R = {k: v[(v.t >= T1) & (v.t < T2 - 4 * 3_600_000)]for k, v in R.items()}
    D = D[(D.t >= T1) & (D.t < T2 - 4 * 3_600_000)]
    S = S[S.t_bar < T2 - 4 * 3_600_000]
    a = S.set_index(["sym", "t_bar"])
    b = D.rename(columns={"t": "t_bar"}).set_index(["sym", "t_bar"])
    both = a.index.intersection(b.index)
    m = a.loc[both].join(T.set_index(["sym", "t_bar"])[["net", "why"]], rsuffix="_tr").join(
        b.loc[both][["net_base", "macd", "cvd"]], rsuffix="_bt")  # fmt: skip
    res = {"engine_events": len(S), "backtest_events": len(D), "matched": len(both),
           "only_engine": len(a.index.difference(b.index)), "only_backtest": len(b.index.difference(a.index)),
           "net_equal": int(np.isclose(m.net, m.net_base, atol=1e-12).sum()), "net_compared": int(m.net.notna().sum()),
           "max_abs_net_diff": float((m.net - m.net_base).abs().max()),
           "macd_max_rel_diff": float(((m.macd - m.macd_bt).abs() / m.macd_bt.abs()).max()),
           "rule_div_s4_flips": int((m.rule_div_s4 != b.loc[both].index.isin(R["macd_cvd"].set_index(["sym", "t"]).index)).sum()),
           "rule_macd_flips": int((m.rule_macd_only != b.loc[both].index.isin(R["macd"].set_index(["sym", "t"]).index)).sum())}  # fmt: skip
    for lg, rule in (("div_s4", "macd_cvd"), ("macd_only", "macd"), ("all_highs", "ctl")):
        K = account(R[rule])
        e = set(zip(S[S[f"in_{lg}"]].sym, S[S[f"in_{lg}"]].t_bar))
        k = set(zip(K.sym, K.t))
        res[f"{lg}_engine"] = len(e)
        res[f"{lg}_backtest"] = len(k)
        res[f"{lg}_same"] = len(e & k)
    # basket vs D9 definition
    bk = pd.read_csv(OUT / "basket.csv")
    res["basket_bars"] = len(bk)
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0)


if __name__ == "__main__":
    main()
