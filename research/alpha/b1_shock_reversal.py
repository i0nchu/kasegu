"""B1: liquidity shock reversal. After a one minute burst of one-sided aggressive flow, does
the price give part of the move back once the book refills?

Shock at the close of bar i (everything about bar i is known then):
  z      = r1 / sd1d             1m log return in units of the coin's last day 1m volatility
  vmult  = v / mean(v, 1 day)    volume multiple of the bar
  flow   = imb1                  (taker buys - taker sells) / volume of the bar
  idio   = r1 - beta x market r1 (the part not explained by the market that minute)
A shock is |z| >= Z, vmult >= 5 and flow agreeing with the move (|imb1| >= 0.2).
Trade: fade it at the close (taker), exit H minutes later. Signed return from the close.
Controls with the same mechanics:
  quiet    same |z| but vmult < 2 (a move without the volume burst: no liquidity shock)
  market   the market itself moved at least half as much (z of the market move >= Z / 2)
  random   the same number of random bars per coin, same fade side distribution
Costs: taker in and out 0.05% each + 0.03% slippage each side = 0.16% round trip; a maker
exit variant at 0.02% + 0.05% + 0.03% = 0.10% (still taker entry) is reported as well.

Usage: uv run python research/alpha/b1_shock_reversal.py
"""

import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import fmt, folder, market, market_days, publish, save_json
from features import coin
from passive import beta1d
from stats import boot_mean

from data import GRID, MIN, UNIVERSE, Y2

EXP = "B1_shock_reversal"
HS = (1, 2, 5, 15, 30, 60)
ZS = (4, 6, 8, 12)
COST_TAKER = 0.16
COST_MAKER_EXIT = 0.10
WARM = 1440 * 2


def _one(sym: str) -> pd.DataFrame:
    mk = market()
    f = coin(sym, mk)
    c = f["c"]
    n = len(c)
    beta = beta1d(f["r1"], mk)
    msd = pd.Series(mk).rolling(1440).std().to_numpy()
    sd = f["rv1d"] / 100
    z = f["r1"] / sd
    mz = mk / msd
    ok = f["live"] & np.isfinite(z) & (sd > 0)
    ok[:WARM] = False
    ok[n - max(HS) - 1 :] = False
    vm = f["v1_ratio"]
    imb = f["imb1"]
    base = ok & (np.abs(z) >= min(ZS))
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    sets = {
        "shock": base & (vm >= 5) & (np.sign(imb) == np.sign(z)) & (np.abs(imb) >= 0.2),
        "quiet": base & (vm < 2),
    }
    i_sh = np.flatnonzero(sets["shock"])
    pool = np.flatnonzero(ok)
    rows = []
    for name, m in sets.items():
        i = np.flatnonzero(m)
        rows.append(pd.DataFrame({"i": i, "group": name}))
    # random bars, same count as shocks, fade side copied from the shocks
    if len(i_sh):
        ir = np.sort(rng.choice(pool, size=len(i_sh), replace=False))
        rows.append(pd.DataFrame({"i": ir, "group": "random", "rside": -np.sign(z[i_sh])}))
    d = pd.concat(rows, ignore_index=True)
    if d.empty:
        return d
    i = d.i.to_numpy()
    side = np.where(
        d.group == "random", d.get("rside", pd.Series(np.nan, index=d.index)), -np.sign(z[i])
    )
    d["side"] = side
    d["symbol"] = sym
    d["t"] = GRID[i] + MIN
    d["z"] = z[i]
    d["absz"] = np.abs(z[i])
    d["mz"] = mz[i]
    d["vmult"] = vm[i]
    d["imb1"] = imb[i]
    d["r1_pct"] = f["r1"][i] * 100
    d["idio_pct"] = (f["r1"][i] - np.nan_to_num(beta[i]) * mk[i]) * 100
    d["ret15_before"] = f["ret15"][i - 1]  # the 15 minutes before the shock bar
    d["oi15"] = f["oi15"][i]
    d["atr_pct"] = f["atr_pct"][i]
    d["market_move"] = np.abs(mz[i]) >= np.abs(z[i]) / 2
    M = np.exp(np.cumsum(np.nan_to_num(mk)))
    for H in HS:
        d[f"f{H}"] = (c[i + H] / c[i] - 1) * 100 * side
        d[f"x{H}"] = d[f"f{H}"] - side * np.nan_to_num(beta[i]) * (M[i + H] / M[i] - 1) * 100
    return d


def cell(d: pd.DataFrame, **key) -> dict:
    r = dict(key)
    r["n"] = len(d)
    r["coin_months"] = d.groupby(["symbol", "month"]).ngroups
    r["per_coin_month"] = len(d) / (len(UNIVERSE) * 24)
    for H in HS:
        r[f"g{H}"] = d[f"f{H}"].mean()
    for H in (5, 15):
        m, lo, hi = boot_mean(d[f"f{H}"] - COST_TAKER, d.day.to_numpy(), b=1000)
        r.update({f"net{H}": m, f"net{H}_lo": lo, f"net{H}_hi": hi})
        r[f"hedged{H}"] = d[f"x{H}"].mean()
    r["net5_maker_exit"] = (d.f5 - COST_MAKER_EXIT).mean()
    r["net5_y1"] = (d[~d.y2].f5 - COST_TAKER).mean()
    r["net5_y2"] = (d[d.y2].f5 - COST_TAKER).mean()
    r["net5_ex_crash"] = (d[d.event != "crash"].f5 - COST_TAKER).mean()
    return r


def main():
    out = folder(EXP)
    save_json(
        out / "config.json",
        {"exp": EXP, "z": ZS, "horizons": HS, "cost_taker_round_trip": COST_TAKER,
         "cost_maker_exit_round_trip": COST_MAKER_EXIT, "shock": "vmult>=5, imb1 same sign, |imb1|>=0.2",  # noqa: E501
         "quiet": "vmult<2", "symbols": UNIVERSE},
    )  # fmt: skip
    market()
    with ProcessPoolExecutor(5) as ex:
        d = pd.concat([x for x in ex.map(_one, UNIVERSE) if len(x)], ignore_index=True)
    d["day"] = d.t // 86_400_000
    d["y2"] = d.t >= GRID[Y2]
    d["month"] = pd.to_datetime(d.t, unit="ms").dt.strftime("%Y-%m")
    d = d.merge(market_days(), on="day", how="left")
    d.to_csv(out / "events.csv.gz", index=False)
    rows = []
    for Z in ZS:
        s = d[d.absz >= Z]
        for g in ("shock", "quiet"):
            x = s[s.group == g]
            for dirn, y in (("down", x[x.z < 0]), ("up", x[x.z > 0])):
                for scope, w in (("idio", y[~y.market_move]), ("market", y[y.market_move])):
                    if len(w) >= 30:
                        rows.append(cell(w, Z=Z, group=g, shock_dir=dirn, scope=scope))
        r = d[d.group == "random"]
        rows.append(cell(r, Z=Z, group="random", shock_dir="any", scope="any"))
    t = pd.DataFrame(rows)
    t.to_csv(out / "grid.csv", index=False)
    print(fmt(t[["Z", "group", "shock_dir", "scope", "n", "per_coin_month", "g1", "g5", "g15", "g60",  # noqa: E501
                 "net5", "net5_lo", "net5_hi", "hedged5", "net5_y1", "net5_y2", "net5_ex_crash"]]))  # fmt: skip  # noqa: E501
    # where does it work: by coin and month for the down / idio shocks at Z=6
    s = d[(d.group == "shock") & (d.absz >= 6) & ~d.market_move]
    by = (
        s.groupby(["symbol"])
        .agg(n=("f5", "size"), g5=("f5", "mean"))
        .assign(net5=lambda x: x.g5 - COST_TAKER)
    )
    by.to_csv(out / "z6_idio_by_coin.csv")
    s.groupby("month").agg(n=("f5", "size"), g5=("f5", "mean")).to_csv(out / "z6_idio_by_month.csv")
    publish(EXP, ["config.json", "grid.csv", "z6_idio_by_coin.csv", "z6_idio_by_month.csv"])


if __name__ == "__main__":
    main()
