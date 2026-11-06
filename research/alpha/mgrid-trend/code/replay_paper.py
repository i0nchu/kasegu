"""Replay kasegu.grid.paper.GridPaper on history (val period, already consumed) and compare with the
research ledger: same coins chosen, same direction, same PnL?  Usage: uv run python ... [months]"""
import asyncio, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from kasegu.core.types import Kline
from kasegu.grid.config import GridConfig
from kasegu.grid.paper import GridPaper, DAY_MS

DATA = Path('/home/user/data/mgrid')
Z = np.load('/home/user/data/panel_1d_all.npz')
NAMES = list(Z['names']); D0 = int(pd.Timestamp('2020-01-01').value // 10**6)
PC, PQ = Z['c'], Z['qv']

class Hist:
    def __init__(self, clock): self.clock = clock; self.cache = {}
    def _x(self, s):
        if s not in self.cache:
            p = DATA / f'{s}.npz'
            if not p.exists(): self.cache[s] = None
            else:
                z = np.load(p); t = z['t'].astype(np.int64) * 300_000
                d = pd.DataFrame({'o': z['o'], 'h': z['h'], 'l': z['l'], 'c': z['c'], 'f': z['f']}, index=t).astype(float)
                hr = d.groupby(d.index // 3_600_000 * 3_600_000).agg(c=('c', 'last'), n=('c', 'count'))
                hr.loc[hr.n < 12, 'c'] = np.nan
                self.cache[s] = (d, hr.c)
        return self.cache[s]
    async def perpetuals(self):
        i = (self.clock() - D0) // DAY_MS
        out = {}
        for j, s in enumerate(NAMES):
            col = PC[:i, j]; ok = ~np.isnan(col)
            if ok.any() and not np.isnan(PC[i - 1, j]):
                out[s] = D0 + int(np.argmax(ok)) * DAY_MS
        return out
    async def klines(self, s, interval, limit, start=None):
        now = self.clock()
        if interval == '1d':
            j = NAMES.index(s); i = (now - D0) // DAY_MS
            out = []
            for d in range(max(0, i - limit), i):
                c = PC[d, j]
                if np.isnan(c): continue
                t = D0 + d * DAY_MS; out.append(Kline(s, t, t + DAY_MS - 1, c, c, c, c, float(PQ[d, j])))
            return out
        x = self._x(s)
        if x is None: return []
        d, hc = x
        if interval == '1h':
            h = hc[hc.index + 3_600_000 <= now].iloc[-limit:]
            return [Kline(s, int(t), int(t) + 3_599_999, c, c, c, c, 0) for t, c in h.items() if c == c]
        b = d[(d.index >= start) & (d.index + 300_000 <= now)].iloc[:limit]
        b = b[b.c.notna()]
        return [Kline(s, int(t), int(t) + 299_999, r.o, r.h, r.l, r.c, 0) for t, r in b.iterrows()]
    async def funding(self, s, start, end):
        d, _ = self._x(s)
        f = d.f[(d.index >= start) & (d.index < end)]
        return [(int(t), v) for t, v in f.items() if v != 0]

async def main(months):
    t0 = int(pd.Timestamp('2023-07-08').value // 10**6); t1 = int((pd.Timestamp('2023-07-08') + pd.DateOffset(months=months)).value // 10**6)
    clock = {'now': t0 + 30_000}
    out = Path(tempfile.mkdtemp())
    gp = GridPaper(GridConfig(out_dir=out, max_account_loss=0), Hist(lambda: clock['now']), now=lambda: clock['now'])
    while clock['now'] < t1:
        await gp.step()
        clock['now'] += 300_000
    tr = pd.read_csv(out / 'trades.csv')
    tr['date'] = pd.to_datetime(tr.start).dt.normalize()
    ref = pd.read_csv('research/alpha/mgrid-trend/results/val_ledger.csv', parse_dates=['date'])
    ref = ref[ref.date < pd.Timestamp(t1, unit='ms')]
    m = ref.merge(tr, left_on=['date', 'sym'], right_on=['date', 'symbol'], how='outer', indicator=True)
    both = m[m._merge == 'both']
    print(f'research episodes {len(ref)}, paper {len(tr)}, matched {len(both)}, only research {(m._merge=="left_only").sum()}, only paper {(m._merge=="right_only").sum()}')
    print('matched: same side %.3f, pnl abs diff max %.4f, sum research %.2f paper %.2f' % ((both.dir == both.side).mean(), (both.pnl_usd - both.pnl).abs().max(), both.pnl_usd.sum(), both.pnl.sum()))
    print('totals: research %.2f, paper %.2f (paper still open %d)' % (ref.pnl_usd.sum(), tr.pnl.sum(), len(gp.active)))
    print(m[m._merge != 'both'][['date', 'sym', 'symbol', 'pnl_usd', 'pnl']].head(12).to_string())

asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
