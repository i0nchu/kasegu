"""Bold-play check: single-coin compounding trend grid (MGT-BOTH-k1.0-RV3, K=1) at leverage 1-4,
historical rolling starts over 2020-07..2026-09. P(reach target within horizon), P(bust), median end.
Bust = equity too small to place the 6-level grid (equity * L < 104 USDT notional)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '../../mgrid-trend/code'); sys.argv = ['x', 'dev']
import trend as R  # noqa: E402

PRE = {1: 'K1', 2: 'K2', 3: '', 4: 'N'}  # margin columns: liquidation at ~1/L of notional
frames = []
for split in ('dev', 'val', 'hold'):
    T = R.load(split, '')
    T = T.assign(dir=R.direction(T, 'BOTH'))
    T = T[(T.dir != 0) & (T.RV3 >= R.RV3_CUT)]
    frames.append(T)
T = pd.concat(frames, ignore_index=True)
by_day = {d: g.sort_values('RV3', ascending=False) for d, g in T.groupby('day')}
days = np.arange(T.day.min(), T.day.max() + 1)
# K=1 sequence: next free day after each episode (uses the 1x path for timing)
seq, d = [], days[0]
while d <= days[-1]:
    g = by_day.get(d)
    if g is None: d += 1; continue
    r = g.iloc[0]; side = 'L' if r.dir == 1 else 'S'
    rets = {L: r[f'{PRE[L]}{side}1.0_ret'] for L in PRE}
    if any(v != v for v in rets.values()): d += 1; continue
    end = int(r[f'K1{side}1.0_end'] // 288)
    seq.append(dict(day=d, sym=r.sym, **{f'r{L}': v for L, v in rets.items()}))
    d += end + 1
S = pd.DataFrame(seq); S['date'] = pd.to_datetime(S.day * 86400, unit='s')
print('episodes', len(S), S.date.min().date(), S.date.max().date())
rows = []
for L in (1, 2, 3, 4):
    for target, months in ((2, 1), (5, 3), (10, 6)):
        hits = busts = 0; ends = []
        starts = S.date[S.date <= S.date.max() - pd.DateOffset(months=months)]
        for s0 in starts.iloc[::3]:  # every 3rd episode as a start
            e = 100.0; win = S[(S.date >= s0) & (S.date < s0 + pd.DateOffset(months=months))]
            out = 'none'
            for r in win[f'r{L}']:
                e *= 1 + r * L
                if e >= 100 * target: out = 'hit'; break
                if e * L < 104: out = 'bust'; break
            hits += out == 'hit'; busts += out == 'bust'; ends.append(e)
        n = len(ends)
        rows.append(dict(leverage=L, target=f'{target}x in {months}m', starts=n, p_hit=hits / n, p_bust=busts / n,
                         median_end=np.median(ends), p_end_below_50=np.mean(np.array(ends) < 50)))
Rr = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(Rr.round(3).to_string()); Rr.to_csv('../results/bold_table.csv', index=False); S.to_csv('../results/k1_sequence.csv', index=False)
# by year for the 2x/1m, L=3
