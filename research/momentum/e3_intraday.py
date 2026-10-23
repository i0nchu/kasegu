"""E3: short-horizon breakouts, the "fast doubling" family most bot repos and TradingView scripts sell.

Per-trade simulation on the 40-coin hourly panel, one position per coin at a time, 1x notional.
Entries are stop or market orders (taker), exits are stops or market: every trade pays
2 x (0.05% fee + 0.05% slippage) = 0.20% round trip, plus funding while held. Stop fills gap
through: a bar that opens beyond the stop fills at its open. Stop and trigger in the same bar
count as a loss.

  vbo     Larry Williams volatility breakout. Day = 00:00-24:00 UTC. Long trigger = day open +
          k * previous day range, short trigger = open - k * range. First trigger hit (on 1h bars)
          is taken, stop at the day open, exit at the day close. k in {0.3, 0.5, 0.7}.
  dc4h    Donchian breakout on 4h bars: close above the highest high of the previous N bars ->
          buy next open; exit on a chandelier stop (highest close since entry - M * ATR14) or the
          opposite N/2 channel. Short mirror. Optional filter: bar volume > 2x its 20-bar mean.
  squeeze Bollinger squeeze on 4h: bandwidth(20, 2) in its lowest 10% of the last 120 bars within
          the previous 6 bars, then close outside the band -> enter next open in that direction;
          exit when close crosses the middle band, or a 2 ATR stop.

Usage: PYTHONPATH=research/momentum uv run python research/momentum/e3_intraday.py
"""

import numpy as np
import pandas as pd

import common as C

RT = 2 * C.COST  # round-trip cost per trade
OOS_T = int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6)


def atr(B, n=14):
    h, l, c = B["h"], B["l"], B["c"]
    pc = np.vstack([c[:1], c[:-1]])
    tr = np.nanmax(np.stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.DataFrame(tr).rolling(n, min_periods=n).mean().to_numpy()


def trade(rows, sym, t_in, side, px_in, px_out, fund, **where):
    g = side * (px_out / px_in - 1)
    rows.append({"sym": sym, "t": t_in, "side": side, "gross": g, "net": g - RT - side * fund,
                 "px_in": px_in, "px_out": px_out, **where})  # fmt: skip


# ---------------------------------------------------------------- volatility breakout (1h bars)
def vbo(P, k, both):
    rows = []
    o, h, l, c, f = P["o"], P["h"], P["l"], P["c"], P["fund"]
    nd = C.N // 24
    for j, sym in enumerate(C.UNIVERSE):
        oh = o[: nd * 24, j].reshape(nd, 24)
        hh = h[: nd * 24, j].reshape(nd, 24)
        lh = l[: nd * 24, j].reshape(nd, 24)
        ch = c[: nd * 24, j].reshape(nd, 24)
        fh = f[: nd * 24, j].reshape(nd, 24)
        rng = np.nanmax(hh, axis=1) - np.nanmin(lh, axis=1)
        for d in range(1, nd):
            if np.isnan(oh[d]).any() or np.isnan(rng[d - 1]):
                continue
            op = oh[d, 0]
            up, dn = op + k * rng[d - 1], op - k * rng[d - 1]
            hit_u = np.nonzero(hh[d] >= up)[0]
            hit_d = np.nonzero(lh[d] <= dn)[0] if both else np.array([], int)
            iu = hit_u[0] if len(hit_u) else 99
            idn = hit_d[0] if len(hit_d) else 99
            if iu == idn == 99 or iu == idn:
                continue  # nothing, or both in one hour (ambiguous: skip)
            side, i, lvl = (1, iu, up) if iu < idn else (-1, idn, dn)
            px = max(lvl, oh[d, i]) if side > 0 else min(lvl, oh[d, i])
            out = ch[d, -1]
            # stop at the day open, checked from the hour after entry (entry hour: if it closes
            # beyond the open again we treat it as stopped)
            for m in range(i, 24):
                if side > 0 and lh[d, m] <= op and (m > i or ch[d, m] <= op):
                    out = min(op, oh[d, m]) if m > i else op
                    break
                if side < 0 and hh[d, m] >= op and (m > i or ch[d, m] >= op):
                    out = max(op, oh[d, m]) if m > i else op
                    break
            trade(rows, sym, C.GRID[d * 24], side, px, out, fh[d, i + 1 :].sum())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4h channel / squeeze
def sim_4h(B, entry_sig, exit_fn, sym, j, stop_atr):
    """entry_sig[i] in {-1,0,1} decided at close of bar i -> enter at open of i+1."""
    rows = []
    o, h, l, c, f = (B[k][:, j] for k in ("o", "h", "l", "c", "fund"))
    a = B["atr"][:, j]
    pos, px, stop, best, fund, t_in, i_in = 0, 0.0, 0.0, 0.0, 0.0, 0, 0
    for i in range(1, len(c)):
        if pos:
            fund += f[i]
            hit = (pos > 0 and l[i] <= stop) or (pos < 0 and h[i] >= stop)
            if hit:
                out = min(stop, o[i]) if pos > 0 else max(stop, o[i])
                trade(rows, sym, t_in, pos, px, out, fund, j=j, i_in=i_in, i_out=i)
                pos = 0
            else:
                best = max(best, c[i]) if pos > 0 else min(best, c[i])
                new_stop = exit_fn(i, pos, best, a[i])
                stop = max(stop, new_stop) if pos > 0 else min(stop, new_stop)
                if exit_now(B, i, j, pos):
                    if i + 1 < len(c) and np.isfinite(o[i + 1]):
                        trade(rows, sym, t_in, pos, px, o[i + 1], fund, j=j, i_in=i_in, i_out=i + 1)
                    pos = 0
                    continue
        if not pos and entry_sig[i] != 0 and i + 1 < len(c) and np.isfinite(o[i + 1]) and np.isfinite(a[i]):
            pos = int(entry_sig[i])
            px, best, fund, t_in, i_in = o[i + 1], o[i + 1], 0.0, B["t"][i + 1], i + 1
            stop = px - pos * stop_atr * a[i]
    return rows


EXIT_STATE: dict = {}


def exit_now(B, i, j, pos):
    x = EXIT_STATE["x"]
    return bool(x[i, j] * pos < 0) if x is not None else False


def dc4h(B, N, M, vol_filter, both):
    c, h, l = B["c"], B["h"], B["l"]
    hi = pd.DataFrame(h).rolling(N).max().shift(1).to_numpy()
    lo = pd.DataFrame(l).rolling(N).min().shift(1).to_numpy()
    hi2 = pd.DataFrame(h).rolling(N // 2).max().shift(1).to_numpy()
    lo2 = pd.DataFrame(l).rolling(N // 2).min().shift(1).to_numpy()
    sig = np.where(c > hi, 1, 0) + (np.where(c < lo, -1, 0) if both else 0)
    if vol_filter:
        qv = pd.DataFrame(B["qv"])
        sig = np.where((qv > 2 * qv.rolling(20).mean()).to_numpy(), sig, 0)
    # opposite half channel exit: long exits on close below the N/2 low, short above the N/2 high
    EXIT_STATE["x"] = np.where(c < lo2, -1, 0) + np.where(c > hi2, 1, 0)
    rows = []
    for j, sym in enumerate(C.UNIVERSE):
        rows += sim_4h(B, sig[:, j], lambda i, pos, best, a: best - pos * M * a, sym, j, M)
    return pd.DataFrame(rows)


def squeeze(B, both):
    c = pd.DataFrame(B["c"])
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bw = (4 * sd / mid)
    low = bw <= bw.rolling(120).quantile(0.1)
    recent = low.rolling(6).max().astype(bool)
    up = (c > mid + 2 * sd) & recent
    dn = (c < mid - 2 * sd) & recent
    sig = up.to_numpy().astype(int) - (dn.to_numpy().astype(int) if both else 0)
    m = mid.to_numpy()
    EXIT_STATE["x"] = np.where(B["c"] < m, -1, 0) + np.where(B["c"] > m, 1, 0)
    rows = []
    for j, sym in enumerate(C.UNIVERSE):
        rows += sim_4h(B, sig[:, j], lambda i, pos, best, a: -np.inf if pos > 0 else np.inf, sym, j, 2)
    return pd.DataFrame(rows)


def summarize(name, t):
    if not len(t):
        return {"name": name, "n": 0}
    oos = t.t >= OOS_T
    out = {"name": name, "n": len(t), "trades_per_day": round(len(t) / (C.N / 24), 2)}
    for tag, part in (("is", t[~oos]), ("oos", t[oos])):
        n = part.net
        out[f"{tag}_n"] = len(n)
        out[f"{tag}_win"] = round(float((n > 0).mean()), 3)
        out[f"{tag}_gross_bp"] = round(float(part.gross.mean() * 1e4), 1)
        out[f"{tag}_net_bp"] = round(float(n.mean() * 1e4), 1)
        out[f"{tag}_t"] = round(float(n.mean() / n.std() * np.sqrt(len(n))), 2) if len(n) > 2 else 0
    return out


def main():
    P = C.panel()
    rows = []
    out = C.RESULTS / "E3_intraday"
    out.mkdir(parents=True, exist_ok=True)
    for k in (0.3, 0.5, 0.7):
        for both in (False, True):
            name = f"vbo_k{k}_{'LS' if both else 'L'}"
            t = vbo(P, k, both)
            rows.append(summarize(name, t))
            print(rows[-1], flush=True)
    B = C.resample(P, 4)
    B["atr"] = atr(B)
    for N in (20, 55):
        for M in (2, 3):
            for vf in (False, True):
                name = f"dc4h_N{N}_M{M}{'_vol' if vf else ''}"
                t = dc4h(B, N, M, vf, True)
                rows.append(summarize(name, t))
                t.to_csv(out / f"trades_{name}.csv.gz", index=False) if (N, M, vf) == (20, 3, False) else None
                print(rows[-1], flush=True)
    for both in (False, True):
        name = f"squeeze_{'LS' if both else 'L'}"
        rows.append(summarize(name, squeeze(B, both)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    cols = ["name", "trades_per_day", "is_n", "is_win", "is_gross_bp", "is_net_bp", "is_t",
            "oos_n", "oos_win", "oos_gross_bp", "oos_net_bp", "oos_t"]  # fmt: skip
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
