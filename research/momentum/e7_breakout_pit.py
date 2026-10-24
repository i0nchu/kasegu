"""E7: the E4 4h breakouts on the point-in-time universe (2024-01 to 2026-09).

Same rules, sizing (1/40 of equity per trade) and costs as E4, but a coin may only be entered
on a day when it is in the E5 top-40-by-volume universe (decided with the previous day's data).
Positions already open are kept to their normal exit if the coin drops out. 310 coins were in
that universe at some point after 2024-01; 1h klines for all of them come from
`download.py data/research/momentum/oos_universe.txt 2024-01`.

The 40-coin numbers for the same period are printed alongside, so the gap is the survivorship
effect on these short-horizon strategies. Long and short legs are also reported separately.

The 2024 warm-up uses the first 120 4h bars of the sample (20 days) only for indicators.

Usage: PYTHONPATH=research/momentum uv run python research/momentum/e7_breakout_pit.py
"""

import numpy as np
import pandas as pd

import common as C
import e3_intraday as E3
import e4_portfolio as E4
import e5_survivorship as E5

T0 = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)
GRID = np.arange(T0, C.END, C.H, dtype=np.int64)


def panel_4h(names) -> dict:
    p = C.OUT / "panel_4h_pit.npz"
    if not p.exists():
        cols = {k: [] for k in ("o", "h", "l", "c", "qv", "fund")}
        for s in names:
            files = sorted((C.HIST / "klines" / s / "1h").glob("*.zip"))
            files = [f for f in files if f.stem[-7:] >= "2024-01"]
            if files:
                k = pd.concat([C._read_zip(f) for f in files]).iloc[:, [0, 1, 2, 3, 4, 7]].astype(float)
                k.columns = ["t", "o", "h", "l", "c", "qv"]
                k["t"] = k.t.astype(np.int64)
                k.loc[k.t > 10**14, "t"] //= 1000
                k = k.drop_duplicates("t").set_index("t").sort_index().reindex(GRID)
            else:
                k = pd.DataFrame(np.nan, index=GRID, columns=["o", "h", "l", "c", "qv"])
            f = np.zeros(len(GRID))
            fz = [z for z in sorted((C.HIST / "fundingRate" / s).glob("*.zip")) if z.stem[-7:] >= "2024-01"]
            if fz:
                d = pd.concat([C._read_zip(z) for z in fz])
                t = d.iloc[:, 0].astype(np.int64).to_numpy()
                i = (t - T0) // C.H
                ok = (i >= 0) & (i < len(GRID))
                np.add.at(f, i[ok], d.iloc[:, -1].astype(float).to_numpy()[ok])
            for c in ("o", "h", "l", "c", "qv"):
                cols[c].append(k[c].to_numpy())
            cols["fund"].append(f)
        np.savez(p, **{k: np.stack(v, 1) for k, v in cols.items()})
    z = np.load(p)
    P = {k: z[k] for k in z.files}
    n = len(GRID) // 4
    B = {}
    r4 = lambda a: a[: n * 4].reshape(n, 4, a.shape[1])  # noqa: E731
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        B["h"] = np.nanmax(r4(P["h"]), 1)
        B["l"] = np.nanmin(r4(P["l"]), 1)
    B["o"] = pd.DataFrame(P["o"][: n * 4]).bfill(limit=3).to_numpy().reshape(n, 4, -1)[:, 0]
    B["c"] = pd.DataFrame(P["c"][: n * 4]).ffill(limit=3).to_numpy().reshape(n, 4, -1)[:, -1]
    B["qv"] = r4(P["qv"]).sum(1)
    B["fund"] = r4(P["fund"]).sum(1)
    B["t"] = GRID[: n * 4][::4]
    B["atr"] = E3.atr(B)
    return B


def run(B, sig, names, M, stop_atr, exit_x, exit_fn):
    E3.EXIT_STATE["x"] = exit_x
    rows = []
    for j, s in enumerate(names):
        rows += E3.sim_4h(B, sig[:, j], exit_fn, s, j, stop_atr)
    return pd.DataFrame(rows)


def signals(B, kind, N=20, mult=2):
    c, h, l = B["c"], B["h"], B["l"]
    if kind == "dc":
        hi = pd.DataFrame(h).rolling(N).max().shift(1).to_numpy()
        lo = pd.DataFrame(l).rolling(N).min().shift(1).to_numpy()
        hi2 = pd.DataFrame(h).rolling(N // 2).max().shift(1).to_numpy()
        lo2 = pd.DataFrame(l).rolling(N // 2).min().shift(1).to_numpy()
        sig = np.where(c > hi, 1, 0) + np.where(c < lo, -1, 0)
        qv = pd.DataFrame(B["qv"])
        sig = np.where((qv > mult * qv.rolling(20).mean()).to_numpy(), sig, 0)
        return sig, np.where(c < lo2, -1, 0) + np.where(c > hi2, 1, 0)
    cd = pd.DataFrame(c)
    mid, sd = cd.rolling(20).mean(), cd.rolling(20).std()
    bw = 4 * sd / mid
    recent = (bw <= bw.rolling(120).quantile(0.1)).rolling(6).max().astype(bool)
    sig = ((cd > mid + 2 * sd) & recent).to_numpy().astype(int) - ((cd < mid - 2 * sd) & recent).to_numpy().astype(int)
    m = mid.to_numpy()
    return sig, np.where(c < m, -1, 0) + np.where(c > m, 1, 0)


def main():
    names = (C.OUT / "oos_universe.txt").read_text().split()
    D = E5.daily_panel()
    elig_d = E5.eligible(D)
    col = {s: i for i, s in enumerate(D["names"])}
    # day index of each 4h bar; the universe of the previous day is the one known at the bar
    B = panel_4h(names)
    di = np.searchsorted(D["t"], B["t"], side="right") - 2
    elig = elig_d[np.clip(di, 0, None)][:, [col[s] for s in names]]
    out = C.RESULTS / "E7_breakout_pit"
    out.mkdir(parents=True, exist_ok=True)
    B40 = C.resample(C.panel(), 4)
    B40["atr"] = E3.atr(B40)
    k40 = np.searchsorted(B40["t"], B["t"][0])
    B40 = {k: v[k40:] for k, v in B40.items()}
    rows = []
    specs = {
        "dc4h_N20_M3_v2": ("dc", 20, 2, 3, "chand"),
        "dc4h_N20_M2_v2": ("dc", 20, 2, 2, "chand"),
        "dc4h_N40_M4_v2": ("dc", 40, 2, 4, "chand"),
        "squeeze": ("sq", 0, 0, 2, "none"),
    }
    for name, (kind, N, mult, M, ex) in specs.items():
        fn = (lambda i, pos, best, a, M=M: best - pos * M * a) if ex == "chand" else (
            lambda i, pos, best, a: -np.inf if pos > 0 else np.inf)
        for uni, BB, gate, nm in (("pit", B, elig, names), ("40coins", B40, None, C.UNIVERSE)):
            sig, x = signals(BB, kind, N, mult)
            if gate is not None:
                sig = np.where(gate, sig, 0)
            t = run(BB, sig, nm, M, M, x, fn)
            for leg, tt in (("LS", t), ("L", t[t.side > 0]), ("S", t[t.side < 0])):
                r = E4.trades_to_returns(tt, BB)
                r = r[r.index >= pd.Timestamp("2024-01-21", tz="UTC")]
                m = C.metrics(r)
                rows.append({"name": name, "universe": uni, "leg": leg, "trades": len(tt),
                             "net_bp": round(float(tt.net.mean() * 1e4), 1), **m})  # fmt: skip
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
