"""E5: E1 / E2 on a point-in-time universe of every USDT perp, delisted coins included.

The 40-coin list was picked in 2024-2026 from coins that are still big, which flatters long
momentum: in 2020-2023 it already "knows" SOL, DOGE or PEPE will survive and grow, and it leaves
out LUNA, FTT and hundreds of coins that faded. Here the universe on each day is the TOP coins by
30-day average quote volume among all USDT perps listed at least 60 days, using only data known
that day. Delisted coins are held until their last daily close.

Strategies, with the E1 / E2 rules unchanged:
  tsmom_L         long-only TSMOM ensemble, vol-scaled, 1/TOP of equity per coin
  donchian_L      long-only Donchian ensemble, same sizing
Each is run on the base universe (top 40, listed 60+ days) and a stricter one (top 20, a year+).

  rot_f_L{L}_H7   top-5 rotation by L-day return with the BTC 100-day filter, every phase
                  (weekday the weekly rebalance falls on) reported, since E2 swung with timing

Usage: PYTHONPATH=research/momentum uv run python research/momentum/e5_survivorship.py
"""

import numpy as np
import pandas as pd

import common as C
import e1_trend as E1

TOP = 40
DAY = 86_400_000
GRID_D = np.arange(C.START, C.END, DAY, dtype=np.int64)


def daily_panel() -> dict:
    p = C.OUT / "panel_1d_all.npz"
    if not p.exists():
        syms = (C.HIST / "all_usdt_symbols.txt").read_text().split()
        cols, names = [], []
        for s in syms:
            files = sorted((C.HIST / "klines" / s / "1d").glob("*.zip"))
            if not files:
                continue
            k = pd.concat([C._read_zip(f) for f in files]).iloc[:, [0, 1, 2, 3, 4, 7]].astype(float)
            k.columns = ["t", "o", "h", "l", "c", "qv"]
            k["t"] = k.t.astype(np.int64)
            k.loc[k.t > 10**14, "t"] //= 1000
            k = k.drop_duplicates("t").set_index("t").sort_index().reindex(GRID_D)
            if k.c.notna().sum() < 30:
                continue
            f = np.zeros(len(GRID_D))
            fz = sorted((C.HIST / "fundingRate" / s).glob("*.zip"))
            if fz:
                d = pd.concat([C._read_zip(z) for z in fz])
                t = d.iloc[:, 0].astype(np.int64).to_numpy()
                i = (t - C.START) // DAY
                ok = (i >= 0) & (i < len(GRID_D))
                np.add.at(f, i[ok], d.iloc[:, -1].astype(float).to_numpy()[ok])
            cols.append({**{c: k[c].to_numpy() for c in ("o", "h", "l", "c", "qv")}, "fund": f})
            names.append(s)
        np.savez(p, names=np.array(names), **{k: np.stack([c[k] for c in cols], 1) for k in cols[0]})
    z = np.load(p)
    D = {k: z[k] for k in z.files}
    D["t"] = GRID_D
    return D


def eligible(D, top=TOP, min_age=60) -> np.ndarray:
    """True where the coin is in the day's top-`top` by trailing 30-day quote volume (known at close)."""
    qv = pd.DataFrame(D["qv"]).rolling(30, min_periods=30).mean()
    age = pd.DataFrame(D["c"]).notna().cumsum()
    qv = qv.where((age >= min_age) & pd.DataFrame(D["c"]).notna())
    rank = qv.rank(axis=1, ascending=False)
    return (rank <= top).to_numpy()


def rotation(D, elig, L, phase, k=5):
    c = pd.DataFrame(D["c"])
    mom = (c / c.shift(L) - 1).to_numpy().copy()
    mom[~elig] = np.nan
    btc = int(np.nonzero(D["names"] == "BTCUSDT")[0][0])
    btc_ok = (c[btc] > c[btc].rolling(100).mean()).to_numpy()
    W = np.zeros_like(mom)
    cur = np.zeros(mom.shape[1])
    for i in range(len(mom)):
        if i % 7 == phase:
            cur = np.zeros(mom.shape[1])
            ok = np.where(np.isfinite(mom[i]))[0]
            if btc_ok[i] and len(ok) >= 2 * k:
                cur[ok[np.argsort(mom[i][ok])][-k:]] = 1.0 / k
        cur = np.where(np.isfinite(D["c"][i]), cur, 0)
        W[i] = cur
    return W


def main():
    D = daily_panel()
    out = C.RESULTS / "E5_survivorship"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}

    def add(name, W):
        r = C.run_weights(W, D)
        rows.append(C.flat(name, C.split_metrics(r)))
        curves[name] = r
        m = rows[-1]
        print(name, m["is_cagr"], m["is_sharpe"], m["oos_cagr"], m["oos_sharpe"], m["all_mdd"], flush=True)

    sd = pd.DataFrame(D["c"]).pct_change().rolling(60, min_periods=30).std().to_numpy() * np.sqrt(365)
    lev = np.minimum(E1.TV / sd, 2.0)
    s_ts = np.clip(E1.tsmom(D, False), 0, None)
    s_dc = np.mean([E1.donchian_state(D, L, False) for L in E1.LOOKBACKS], axis=0)
    # base universe (top 40, 60 days listed) plus a stricter one (top 20, a year listed)
    for top, age, tag in ((TOP, 60, ""), (20, 365, "_top20")):
        elig = eligible(D, top, age)
        print(len(D["names"]), "symbols; eligible per day (median)", int(np.median(elig.sum(1))))
        add(f"tsmom_L{tag}", np.nan_to_num(np.where(elig, s_ts * lev, 0)) / top)
        add(f"donchian_L{tag}", np.nan_to_num(np.where(elig, s_dc * lev, 0)) / top)
        for L in (7, 14, 28):
            for ph in range(7):
                add(f"rot_f_L{L}_H7_p{ph}{tag}", rotation(D, elig, L, ph))
        W = elig / np.maximum(elig.sum(1, keepdims=True), 1)
        add(f"equal_weight{tag}", W.astype(float))
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "daily_returns.csv.gz")
    rot = df[df.name.str.startswith("rot")].copy()
    rot["L"] = rot.name.str.extract(r"_L(\d+)_")[0].astype(int)
    rot["u"] = np.where(rot.name.str.endswith("_top20"), "top20", "top40")
    print(rot.groupby(["u", "L"])[["is_cagr", "is_sharpe", "oos_cagr", "oos_sharpe", "all_mdd"]].agg(["min", "median", "max"]).T)


if __name__ == "__main__":
    main()
