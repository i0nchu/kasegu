"""G6: the coin-selection rule from G5 as a live estimate, and checks on simultaneous losses.

1. Simultaneous losses in the G2 deploy6 ledger: concurrent positions at each entry, losing coins
   per day, and the share of the worst days' loss that came from 3+ coins losing together.
2. The rule chosen on the G5 design half (2022-01 to 2024-06): trade a coin only while its 30-day
   return (known at the previous daily close) is above +10%. Reported on the check half, on the
   whole point-in-time universe and on the six deployed coins (base fills); the G3 fill models
   are applied as expected values to the deploy6 ledger (fill probability by penetration
   bucket, mean measured slippage).
3. Natural frequency: trades per month for the universe and per coin, with and without the rule.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/g6_selection.py
"""

import json

import numpy as np
import pandas as pd

import fast as F
import g5_factors as G5

DEPLOY6 = ["SOLUSDT", "DOGEUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "SUIUSDT"]
RULE = 0.10  # ret30 threshold chosen on the design half


def expected(d: pd.DataFrame, cal: dict, model: str) -> tuple[float, float]:
    """Expected net per trade (%) under a fill model: P(fill) weighted, mean slippage applied."""
    edges, probs = np.array(cal["pen_edges"]), np.array(cal["fill_prob"][model])
    b = np.clip(np.searchsorted(edges, d.pen.to_numpy(), side="right") - 1, 0, len(probs) - 1)
    p = probs[b]
    xs, ss = np.mean(cal["exit_slip"]), np.mean(cal["stop_slip"])
    if model == "queue_p90":
        xs = np.quantile(cal["exit_slip"], 0.875)
        ss += cal["stop_slip_stress_add"]
    adj = np.where(d.why == 1, ss - 0.0005, xs - 0.0002) * d.notional
    net = (d.net - adj).to_numpy() / d.notional.to_numpy()
    return 100 * (p * net).sum() / max(p.sum(), 1e-9), float(p.mean())


def main():
    out = F.RESULTS / "G6_selection"
    out.mkdir(parents=True, exist_ok=True)
    rep = {}

    # 1. simultaneous losses, deploy6 ledger under the per-symbol model
    L = pd.read_pickle(F.OUT / "G3_ledger_deploy6.pkl").sort_values("t_in")
    t_in, t_out = L.t_in.to_numpy(), L.t_out.to_numpy()
    conc = np.array([((t_in <= a) & (t_out > a)).sum() for a in t_in])
    L["open_at_entry"] = conc
    day = L.t_out // F.DAY
    D = pd.DataFrame({"net": L.net.groupby(day).sum(), "losers": L[L.net < 0].sym.groupby(day[L.net < 0]).nunique()}).fillna(0)
    worst = D.nsmallest(20, "net")
    rep["simultaneous"] = {
        "trades": len(L), "max_open_at_entry": int(conc.max()),
        "entries_with_3plus_open": int((conc >= 3).sum()),
        "days_with_3plus_losing_coins": int((D.losers >= 3).sum()),
        "worst20_days_net_usdt": round(float(worst.net.sum()), 2),
        "worst20_days_share_with_3plus_losers": round(float((worst.losers >= 3).mean()), 2),
        "worst_day": {"date": str(pd.to_datetime(worst.index[0] * F.DAY, unit="ms").date()),
                      "net_usdt": round(float(worst.net.iloc[0]), 2), "losing_coins": int(worst.losers.iloc[0])},
    }  # fmt: skip
    L.drop(columns="path").to_csv(out / "ledger_deploy6_with_concurrency.csv.gz", index=False, float_format="%.8g")
    D.to_csv(out / "daily_deploy6.csv.gz", float_format="%.6g")

    # 2-3. the rule on the PIT universe and on deploy6
    cal = json.loads((F.RESULTS / "G3_fills" / "calibration.json").read_text())
    # fill models on the deploy6 ledger (it has the limit and the fill-minute low); the G5
    # trades do not store the limit, so the rule results below are base fills
    rep["fill_models_deploy6"] = {
        m: dict(zip(("net_pct_per_filled_trade", "fill_rate"), expected(L, cal, m)))
        for m in ("touch", "queue_med", "queue_p90")
    }
    T = pd.read_pickle(F.OUT / "G5_trades.pkl")
    rows = []
    for name, sel in {"all coins": T.sym.notna(), "deploy6": T.sym.isin(DEPLOY6)}.items():
        for rule, m in {"no rule": True, f"ret30>{RULE:.0%}": T.ret30 > RULE}.items():
            for per, pm in (("design 2022-01..2024-06", T.t < G5.SPLIT), ("check 2024-07..2026-09", T.t >= G5.SPLIT)):
                d = T[sel & m & pm]
                if len(d) < 10:
                    continue
                g = d.groupby(d.t // F.DAY).net.sum()
                months = (pm.sum() and (d.t.max() - d.t.min()) / F.DAY / 30.4) or np.nan
                rows.append({
                    "coins": name, "rule": rule, "period": per, "trades": len(d),
                    "trades_per_month": round(len(d) / months, 1),
                    "coin_months": int(d.groupby([d.sym, pd.to_datetime(d.t, unit="ms").dt.to_period("M")]).ngroups),
                    "net_pct_base": round(100 * d.net.mean(), 3),
                    "day_t": round(g.mean() / g.std() * np.sqrt(len(g)), 2),
                    "worst_day_pct": round(100 * g.min(), 1),
                    "stop_rate": round(d.stop.mean(), 3),
                })  # fmt: skip
    R = pd.DataFrame(rows)
    R.to_csv(out / "rule_summary.csv", index=False)
    rep["rule"] = rows
    json.dump(rep, open(out / "summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps({k: rep[k] for k in ("simultaneous", "fill_models_deploy6")}, indent=1))
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
