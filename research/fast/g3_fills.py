"""G3: calibrate EPR15 entry fills and exit / stop slippage on real trades and order books.

Sources:
  aggTrades  data.binance.vision daily files for a sample of G2 trades (deploy6 coins). For each
             simulated fill (limit L alive in minute [t, t+60s)):
               through   quantity traded strictly below L (our bid at L was consumed: a sure fill)
               at        quantity traded exactly at L (fills us only after the queue ahead)
               first_ms  ms after the minute start of the first trade at or below L
             time exit: first sell-aggressor trade after the exit bar's close vs that close
             stop: first trade at or below the stop, then the first sell after it vs the stop
  order book the shadow collector (research/validation/shadow_collector.py, 100 ms diff depth) on
             the deploy6 coins: resting quantity at the exact tick L would be posted at, for
             distances below the mid like EPR15's, sampled once a minute. That is the queue
             ahead of a new Post-Only bid (it is cancelled and re-posted every minute, so it
             always joins the back).

Fill models (applied per sampled trade):
  touch      the research rule (low < L): every sampled trade fills
  queue_med  fill if through > 0, or at > median queue ahead + own size, and the touch came
             after `lat` ms of cancel / replace latency (300 ms)
  queue_p90  the same with the 90th percentile queue and 1,000 ms latency
Output: per-trade calibration table, fill probability by penetration bucket ((L - low) / L) for
each model, slippage quantiles. G4 applies these to the full ledger.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/g3_fills.py
"""

import concurrent.futures as cf
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import fast as F

AGG = Path("data/history/daily/aggTrades")
BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
BOOK = Path("data/research/shadow")
OUT = F.RESULTS / "G3_fills"
PEN_BUCKETS = [0, 0.0005, 0.001, 0.002, 0.005, 1.0]
RNG = np.random.default_rng(11)


def fetch(sym: str, day: str) -> Path | None:
    p = AGG / sym / f"{sym}-aggTrades-{day}.zip"
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}/{sym}/{p.name}", timeout=300) as r:
                data = r.read()
            break
        except urllib.error.HTTPError:
            return None
        except OSError:
            if i == 4:
                return None
            time.sleep(2**i)
    p.with_suffix(".part").write_bytes(data)
    p.with_suffix(".part").rename(p)
    return p


def agg_day(p: Path) -> pd.DataFrame:
    with zipfile.ZipFile(p) as z:
        raw = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(raw), header=None, usecols=[1, 2, 5, 6], dtype=str)
    if not d.iloc[0, 0].replace(".", "").isdigit():
        d = d.iloc[1:]
    d.columns = ["px", "qty", "t", "bm"]
    return pd.DataFrame({
        "px": d.px.astype(float).to_numpy(), "qty": d.qty.astype(float).to_numpy(),
        "t": d.t.astype(np.int64).to_numpy(), "sell": (d.bm.str.lower() == "true").to_numpy(),
    })  # fmt: skip


def measure(args) -> list[dict]:
    sym, day, rows = args
    p = fetch(sym, day)
    if p is None:
        return []
    A = agg_day(p)
    out = []
    for r in rows:
        t0 = r["t_in"]  # start of the fill minute
        L, own = r["px_in"], r["notional"] / r["px_in"]
        w = A[(A.t >= t0) & (A.t < t0 + 60_000)]
        tick = r["tick"]
        below = w[w.px < L - tick / 2]
        at = w[(w.px >= L - tick / 2) & (w.px <= L + tick / 2)]
        touch = w[w.px <= L + tick / 2]
        rec = {**{k: r[k] for k in ("sym", "t_in", "px_in", "net", "why", "low")},
               "own": own, "through": below.qty.sum(), "at": at.qty.sum(),
               "first_ms": int(touch.t.iloc[0] - t0) if len(touch) else -1,
               "first_through_ms": int(below.t.iloc[0] - t0) if len(below) else -1,
               "trades_min": len(w)}  # fmt: skip
        # time exit: market sell after the exit bar closes; the first sell-aggressor trade is the bid
        if r["why"] == 0:
            after = A[(A.t >= r["t_out"]) & A.sell]
            if len(after):
                rec["exit_slip"] = 1 - after.px.iloc[0] / r["close_out"]
                rec["exit_lag_ms"] = int(after.t.iloc[0] - r["t_out"])
        else:
            stop = L * 0.97
            hit = A[(A.t >= t0) & (A.px <= stop)]
            if len(hit):
                after = A[(A.t >= hit.t.iloc[0] + 200) & A.sell]  # trigger + ~200 ms to the market order
                if len(after):
                    rec["stop_slip"] = 1 - after.px.iloc[0] / stop
        out.append(rec)
    return out


def queue_from_book(path: Path, dist_pct: np.ndarray, every_ms=60_000) -> pd.DataFrame:
    """Resting bid quantity at the exact tick a new bid at mid x (1 - d) would join."""
    rows, books, last = [], {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            k, p = x["kind"], x["payload"]
            if k == "snapshot":
                books[p["symbol"]] = ({float(a): float(b) for a, b in p["bids"]}, {float(a): float(b) for a, b in p["asks"]})
            elif k == "depth" and p["s"] in books:
                bids, asks = books[p["s"]]
                for a, b in p["b"]:
                    a, b = float(a), float(b)
                    bids.pop(a, None) if b == 0 else bids.__setitem__(a, b)
                for a, b in p["a"]:
                    a, b = float(a), float(b)
                    asks.pop(a, None) if b == 0 else asks.__setitem__(a, b)
                s, t = p["s"], p["E"]
                if t - last.get(s, 0) < every_ms or not bids or not asks:
                    continue
                last[s] = t
                bb, ba = max(bids), min(asks)
                mid = (bb + ba) / 2
                ticks = sorted(bids)
                tick = min(np.diff(ticks[-50:])) if len(ticks) > 50 else 0
                for d in dist_pct:
                    px = mid * (1 - d / 100)
                    lvl = round(np.floor(px / tick) * tick, 10) if tick else px
                    q = max((v for kk, v in bids.items() if abs(kk - lvl) < tick / 2), default=0.0)
                    rows.append({"sym": s, "t": t, "dist_pct": d, "queue": q, "queue_usdt": q * lvl,
                                 "depth_to_lvl_usdt": sum(v * kk for kk, v in bids.items() if kk >= lvl)})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = pd.read_pickle(F.OUT / "G2_ledger_deploy6.pkl")
    L = L[L.fill == "base"].copy()
    ticks = json.loads(Path("data/research/ticks.json").read_text()) if Path("data/research/ticks.json").exists() else {}
    L["tick"] = L.sym.map(lambda s: ticks.get(s, 0.0)).fillna(0.0)
    # minute low and exit close from the 1m data (for the penetration buckets and exit slippage)
    import g1_epr15_pit as G1

    lows, closes = [], []
    for s, g in L.groupby("sym"):
        X = G1.load1m(s)
        i = np.searchsorted(X["t"], g.t_in.to_numpy())
        j = np.searchsorted(X["t"], g.t_out.to_numpy() - 60_000)
        lows.append(pd.Series(X["l"][i], index=g.index))
        closes.append(pd.Series(X["c"][j], index=g.index))
    L["low"], L["close_out"] = pd.concat(lows), pd.concat(closes)
    L["pen"] = 1 - L.low / L.px_in
    # sample: every stopped trade (the costly tail) plus a random third of the rest, by coin-day
    L["day"] = pd.to_datetime(L.t_in, unit="ms").dt.strftime("%Y-%m-%d")
    days = L.groupby(["sym", "day"]).why.max().reset_index()
    pick = pd.concat([days[days.why == 1], days[days.why == 0].sample(frac=1 / 3, random_state=3)])
    S = L.merge(pick[["sym", "day"]], on=["sym", "day"])
    print(len(L), "trades,", len(S), "sampled on", len(pick), "coin-days", flush=True)
    jobs = [(s, d, g.to_dict("records")) for (s, d), g in S.groupby(["sym", "day"])]
    res = []
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(measure, jobs):
            res += r
    C = pd.DataFrame(res)
    C.to_csv(OUT / "calibration_trades.csv", index=False, float_format="%.8g")
    print(C.describe().T.to_string(), flush=True)




def book():
    """Queue ahead from the collector file(s): EPR15 distances are 12 vol units, 1-8% below."""
    parts = [queue_from_book(p, np.array([1.0, 2.0, 3.0, 5.0, 8.0])) for p in sorted(BOOK.glob("*.jsonl"))]
    q = pd.concat(parts)
    q.to_csv(OUT / "queue_by_distance.csv", index=False, float_format="%.6g")
    print(q.groupby(["sym", "dist_pct"])[["queue", "queue_usdt", "depth_to_lvl_usdt"]].median().to_string())


def models():
    """Fill and slippage models from the measured sample, saved for G4.

    The sample shows 97.7% of simulated fills had trades strictly below the limit inside the
    minute (a sure fill whatever the queue), all of them seconds after the order went live, so
    queue position matters only for the shallow touches. Models:
      queue_med   fill = trade-through, or a touch with more traded at the limit than the
                  median queue ahead measured by the collector + own size
      queue_p90   adverse-selection stress: only fills where the low ran >= 0.2% through the
                  limit (as if a long queue meant we were filled only in a real flush)
    Exit slippage: measured sell-aggressor price after the exit bar's close vs that close.
    Stop slippage: measured, floored at 0 (no price improvement); the stress adds the 0.46%
    worse stop fills the engine's mark-price replay saw (docs/epr15.md section 4).
    """
    C = pd.read_csv(OUT / "calibration_trades.csv")
    C["pen"] = 1 - C.low / C.px_in
    q = pd.read_csv(OUT / "queue_by_distance.csv") if (OUT / "queue_by_distance.csv").exists() else None
    # median resting quantity (base asset) at the tick a new bid would join, per coin
    qmed = q.groupby("sym")["queue"].median() if q is not None and len(q) else pd.Series(dtype=float)
    C["fill_med"] = (C.through > 0) | (C["at"] > C.sym.map(qmed).fillna(np.inf) + C.own)
    C["fill_p90"] = C.pen >= 0.002
    edges = PEN_BUCKETS
    b = pd.cut(C.pen, edges, labels=False, include_lowest=True)
    prob = {
        "touch": [1.0] * (len(edges) - 1),
        "queue_med": [float(C.fill_med[b == i].mean()) if (b == i).any() else 1.0 for i in range(len(edges) - 1)],
        "queue_p90": [float(C.fill_p90[b == i].mean()) if (b == i).any() else 1.0 for i in range(len(edges) - 1)],
    }  # fmt: skip
    xs = C.exit_slip.dropna().to_numpy()
    ss = np.maximum(C.stop_slip.dropna().to_numpy(), 0.0)
    cal = {
        "pen_edges": edges, "fill_prob": prob,
        "exit_slip": xs.tolist(), "stop_slip": ss.tolist(), "stop_slip_stress_add": 0.0046,
        "sample_trades": len(C), "share_trade_through": float((C.through > 0).mean()),
        "queue_median_base_qty": {k: float(v) for k, v in qmed.items()},
        "exit_slip_q": {str(k): float(np.quantile(xs, k)) for k in (0.5, 0.75, 0.9, 0.99)},
        "stop_slip_raw_q": {str(k): float(np.quantile(C.stop_slip.dropna(), k)) for k in (0.1, 0.5, 0.9, 0.99)},
    }  # fmt: skip
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1))
    # the whole deploy6 ledger with its penetration, for G4
    L = pd.read_pickle(F.OUT / "G2_ledger_deploy6.pkl")
    L = L[L.fill == "base"].copy()
    import g1_epr15_pit as G1

    lows = []
    for s, g in L.groupby("sym"):
        X = G1.load1m(s)
        lows.append(pd.Series(X["l"][np.searchsorted(X["t"], g.t_in.to_numpy())], index=g.index))
    L["pen"] = 1 - pd.concat(lows) / L.px_in
    L.to_pickle(F.OUT / "G3_ledger_deploy6.pkl")
    print(json.dumps({k: v for k, v in cal.items() if k not in ("exit_slip", "stop_slip")}, indent=1))


if __name__ == "__main__":
    import sys

    {"measure": main, "book": lambda: book(), "models": models}[sys.argv[1] if len(sys.argv) > 1 else "measure"]()
