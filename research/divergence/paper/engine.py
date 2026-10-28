"""DIV-S4 paper signal engine: three frozen ledgers on the same 4h higher-high pivot events.

Ledgers (same events, same exits, same cost model; each ledger has its own account cap):
  div_s4     mean IS percentile of MACD and CVD divergence >= frozen threshold (main candidate)
  macd_only  MACD divergence >= frozen threshold (does CVD add anything?)
  all_highs  every higher-high pivot (baseline: plain short after a new high)
Rules are copied from D7 / D9 and read from frozen.json; nothing here may be tuned.

Data come from a Source (sources.py): Binance REST for paper trading, the 5m npz panel for replay.
State lives in one directory: signals.csv (every event, with ledger admission), trades.csv (closed
outcomes), basket.csv (equal-weight universe, 4h), universe.csv, open.json, state.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3

HERE = Path(__file__).parent
F = json.loads((HERE / "frozen.json").read_text())
REF = {k: v for k, v in np.load(HERE / "is_ref.npz").items()}
H4 = 4 * 3_600_000
M5 = 300_000
DAY = 86_400_000
LEDGERS = ("div_s4", "macd_only", "all_highs")
COST = F["fee_per_side"] + F["slip_per_side"]


def rank(x, ref):
    # same as D7: NaN sorts last and gets rank 1
    return np.searchsorted(ref, x) / len(ref)


def scan(sym, B, elig, t_from, t_to):
    """Events confirmed on closed 4h bars whose start is in [t_from, t_to). B: 4h arrays (closed bars
    only, last F window); elig: bool per bar (coin in the universe that day)."""
    n = len(B["t"])
    if n < 2 * F["pivot_k"] + F["min_sep"] + 2:
        return []
    I = C.indicators(B)
    e = C.events(B, I, F["pivot_k"], F["min_sep"], F["max_sep"])[-1]
    a = D3.atr(B)
    vol = pd.Series(B["qv"]).cumsum().to_numpy()
    rows = []
    for i, p, q, dp in zip(e["i"], e["p"], e["q"], e["dp"]):
        t = int(B["t"][i])
        if not (t_from <= t < t_to) or dp <= 0 or not elig[i]:
            continue
        macd = -(I["macd"][p] - I["macd"][q])
        cvd = -(I["cvd"][p] - I["cvd"][q]) / (vol[p] - vol[q])
        rm, rc = rank(macd, REF["macd"]), rank(cvd, REF["cvd"])
        rows.append({"id": f"{sym}-{t}", "sym": sym, "t_bar": t, "t_sig": t + H4, "t_p": int(B["t"][p]),
                     "t_q": int(B["t"][q]), "hi_p": B["h"][p], "hi_q": B["h"][q], "close": B["c"][i],
                     "atr": a[i] / B["c"][i], "macd": macd, "cvd": cvd, "rank_macd": rm, "rank_cvd": rc,
                     "score": (rm + rc) / 2, "rule_all_highs": True, "rule_macd_only": bool(macd >= F["macd_threshold"]),
                     "rule_div_s4": bool((rm + rc) / 2 >= F["div_s4_threshold"])})  # fmt: skip
    return rows


def simulate(ev, k5, fund):
    """Short trade for one event on 5m bars (same rules as D9 outcome).

    k5: DataFrame t o h l c on a full 5m grid from the entry bar (start = t_sig), closed bars only,
    missing bars as NaN rows. fund: list of (time, rate). Returns None while the trade is still open.
    why: 0 time, 1 stop, 2 take profit, 3 data ended (gap or delisting), -1 no entry bar."""
    o, h, l, c, t = (k5[x].to_numpy() for x in ("o", "h", "l", "c", "t"))
    if not len(o):
        return None
    if np.isnan(o[0]):
        return {"why": -1}
    px = o[0]
    sp = px * (1 + F["stop_atr"] * ev["atr"])
    tq = px * (1 - F["tp_atr"] * ev["atr"])
    brk = (h >= sp) | (l <= tq) | np.isnan(o)  # first bar that ends the trade
    hit = np.nonzero(brk)[0]
    m = int(hit[0]) if len(hit) else len(o)
    if m >= F["hold_5m_bars"] or (not len(hit) and len(o) >= F["hold_5m_bars"]):
        m = F["hold_5m_bars"] - 1
        ex, why = c[m], 0
    elif not len(hit):
        return None
    elif np.isnan(o[m]):
        m -= 1
        ex, why = c[m], 3
    elif h[m] >= sp:
        ex, why = max(sp, o[m] if m > 0 else px), 1
    else:
        ex, why = min(tq, o[m] if m > 0 else px), 2
    t_exit = int(t[m])
    f = sum(r for ft, r in fund if t[0] <= ft < t_exit + M5)
    gross = -(ex / px - 1)
    return {"why": why, "t_entry": int(t[0]), "px_entry": px, "stop": sp, "tp": tq, "t_exit": t_exit, "px_exit": ex,
            "px_sim_entry": px * (1 - F["slip_per_side"]), "px_sim_exit": ex * (1 + F["slip_per_side"]),
            "bars": m + 1, "gross": gross, "cost": 2 * COST, "funding": f, "net": gross - 2 * COST + f,
            "mae": np.nanmax(h[: m + 1]) / px - 1, "mfe": 1 - np.nanmin(l[: m + 1]) / px}  # fmt: skip


def grid5m(df, start, end):
    """Reindex 5m bars onto the full grid [start, end) so missing bars become NaN rows."""
    g = np.arange(start, end, M5, dtype=np.int64)
    if not len(g):
        return pd.DataFrame(columns=["t", "o", "h", "l", "c"])
    return df.set_index("t").reindex(g).rename_axis("t").reset_index()[["t", "o", "h", "l", "c"]]


class Engine:
    def __init__(self, src, state_dir, start, record_from, log=print):
        self.src = src
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.record_from = int(record_from)  # 4h closes before this only build the basket history
        st = self._json("state.json", {})
        self.last_T = st.get("last_T", int(start) // H4 * H4)
        self.open = self._json("open.json", {})  # event id -> event dict (still open trades)
        self.pos = st.get("pos", {lg: [] for lg in LEDGERS})  # ledger -> [event id] admitted and open

    def _json(self, name, default):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else default

    def _append(self, name, rows):
        if not rows:
            return
        p = self.dir / name
        pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False)

    def _save(self):
        (self.dir / "open.json").write_text(json.dumps(self.open, default=float))
        (self.dir / "state.json").write_text(json.dumps({"last_T": self.last_T, "pos": self.pos}))

    def update(self, now):
        """Close every open trade whose exit is decided by closed 5m bars before `now`."""
        done = []
        for eid, ev in list(self.open.items()):
            end = min(now // M5 * M5, ev["t_sig"] + F["hold_5m_bars"] * M5)
            k5 = grid5m(self.src.bars5m(ev["sym"], ev["t_sig"], end), ev["t_sig"], end)
            ok = np.nonzero(k5.o.notna().to_numpy())[0]
            k5 = k5.iloc[: ok[-1] + 1] if len(ok) else k5.iloc[:0]  # trailing bars not served yet are not a gap
            if len(k5) and len(k5) < F["hold_5m_bars"] and self.src.delisted(ev["sym"], now):
                k5 = pd.concat([k5, pd.DataFrame({"t": [int(k5.t.iloc[-1]) + M5], "o": [np.nan]})], ignore_index=True)
            fund = self.src.funding(ev["sym"], ev["t_sig"], end) if len(k5) else []
            r = simulate(ev, k5, fund)
            if r is None:
                continue
            row = {"id": eid, "sym": ev["sym"], "t_bar": ev["t_bar"], "t_sig": ev["t_sig"], **r,
                   **{f"in_{lg}": ev[f"in_{lg}"] for lg in LEDGERS}}  # fmt: skip
            if ev.get("quote_bid"):
                row["net_quote"] = r["net"] + (ev["quote_bid"] / r["px_entry"] - 1) if r["why"] >= 0 else np.nan
            done.append(row)
            del self.open[eid]
        self._append("trades.csv", done)
        if done:
            self.log(f"closed {len(done)} trades")
        return done

    def universe(self, day):
        return self.src.universe(day)

    def step(self, T, now):
        """Process the 4h bar that closed at T."""
        bar = T - H4
        uni = sorted(self.universe(bar // DAY * DAY))
        self._append("universe.csv", [{"t_bar": bar, "n": len(uni), "syms": " ".join(uni)}] if bar % DAY == 0 else [])
        rets, events, btc = [], [], None
        warm = T < self.record_from
        for sym in uni:
            B = self.src.bars4h(sym, bar)
            if B is None or not len(B["t"]) or B["t"][-1] != bar:
                continue
            if len(B["t"]) >= 2 and B["t"][-2] == bar - H4:
                rets.append(B["c"][-1] / B["c"][-2] - 1)
            if not warm:
                events += scan(sym, B, np.ones(len(B["t"]), bool), bar, bar + H4)
        Bb = self.src.bars4h("BTCUSDT", bar)
        if Bb is not None and len(Bb["t"]) > 181 and Bb["t"][-1] == bar:
            lr = np.diff(np.log(Bb["c"][-43:]))
            btc = {"btc_30d": Bb["c"][-1] / Bb["c"][-181] - 1, "btc_vol7": float(np.std(lr, ddof=1))}
        lo, hi = F["basket_clip"]
        bret = float(np.mean(np.clip(rets, lo, hi))) if rets else np.nan
        self._append("basket.csv", [{"t_bar": bar, "n": len(rets), "ret": bret}])
        bk = self.basket()
        b30 = bk[(bk.t_bar > bar - 180 * H4) & (bk.t_bar <= bar)]
        reg = {"basket_30d": float(np.prod(1 + b30.ret.fillna(0)) - 1) if len(b30) >= 150 else np.nan,
               "basket_bars": len(b30), **(btc or {"btc_30d": np.nan, "btc_vol7": np.nan})}  # fmt: skip
        # current ledger positions: drop the ones already closed
        for lg in LEDGERS:
            self.pos[lg] = [e for e in self.pos[lg] if e in self.open]  # update(T) closed exits before T
        rows = []
        for ev in sorted(events, key=lambda x: x["sym"]):
            q = self.src.quote(ev["sym"], now) if now - T < 10 * 60_000 else None
            ev |= reg | {"quote_bid": q["bid"] if q else np.nan, "quote_ask": q["ask"] if q else np.nan,
                         "quote_delay_s": (q["time"] - T) / 1000 if q else np.nan,
                         "mode": "live" if q else "backfill"}  # fmt: skip
            for lg in LEDGERS:
                ok = ev[f"rule_{lg}"]
                held = [self.open[e]["sym"] for e in self.pos[lg]]
                why = "" if not ok else ("full" if len(held) >= F["account_max_positions"] else
                                         ("same_coin" if ev["sym"] in held else ""))  # fmt: skip
                ev[f"in_{lg}"] = bool(ok and not why)
                ev[f"skip_{lg}"] = why
                if ev[f"in_{lg}"]:
                    self.pos[lg].append(ev["id"])
            self.open[ev["id"]] = ev
            rows.append(ev)
        self._append("signals.csv", rows)
        if rows:
            self.log(f"{pd.Timestamp(T, unit='ms')}: {len(rows)} events, " + ", ".join(
                f"{lg} +{sum(r[f'in_{lg}'] for r in rows)}" for lg in LEDGERS))  # fmt: skip

    def basket(self):
        p = self.dir / "basket.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["t_bar", "n", "ret"])

    def run_until(self, now):
        T = self.last_T + H4
        while T <= now:
            self.update(T)  # positions closed before this 4h close free their slot
            self.step(T, now)
            self.last_T = T
            self._save()
            T += H4
        self.update(now)
        self._save()
