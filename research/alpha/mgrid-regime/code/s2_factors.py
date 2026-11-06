"""S2 (dev only): do regime factors predict the next 72h grid PnL (T2) and forward ER (T1)?"""
import numpy as np, pandas as pd
from scipy import stats

T = pd.read_parquet('/home/user/data/mgrid_dev.parquet')
T['T2'] = T['L0.5_ret'] + T['S0.5_ret']
T['T1'] = T['fER']
FAC = [f'{n}{L}' for n in ('ER', 'VR', 'ADX', 'CHOP', 'AC1', 'VOLR', 'BTCER', 'RV') for L in (3, 14)]

def nw_t(x, lag=3):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); m = x.mean(); e = x - m
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(v / n), n

def sub(d):
    return np.where(d < '2022-01-01', 'A_2020H2-21', np.where(d < '2023-01-01', 'B_2022', 'C_2023H1'))
T['per'] = sub(T.date.astype(str))
rows = []
for f in FAC:
    for tgt in ('T2', 'T1'):
        if f.startswith('BTCER'):
            # same value for all coins on a day: time-series IC on daily cross-sectional mean target
            g = T.groupby('date').agg(x=(f, 'first'), y=(tgt, 'mean')).dropna()
            # rolling-free: use daily rank correlation over the whole sample + per-period, NW on product of ranks
            zx = (g.x.rank() - g.x.rank().mean()) / g.x.rank().std(); zy = (g.y.rank() - g.y.rank().mean()) / g.y.rank().std()
            ic_series = zx * zy
        else:
            ic_series = T.dropna(subset=[f, tgt]).groupby('date').apply(lambda d: stats.spearmanr(d[f], d[tgt])[0] if len(d) >= 10 else np.nan, include_groups=False)
        t, n = nw_t(ic_series.values, lag=3 if not f.startswith('BTCER') else 10)
        per = {}
        for p in ('A_2020H2-21', 'B_2022', 'C_2023H1'):
            idx = ic_series.index
            mask = sub(pd.Series(idx).astype(str).values) == p
            per[p] = np.nanmean(ic_series.values[mask])
        rows.append(dict(factor=f, target=tgt, IC=np.nanmean(ic_series.values), t_NW=t, days=n, p=2 * stats.norm.sf(abs(t)), **per))
R = pd.DataFrame(rows)
# Holm over all 32
R = R.sort_values('p').reset_index(drop=True)
m = len(R); R['holm_p'] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * R.p))
R['holm_sig'] = R.holm_p < 0.05
pd.set_option('display.width', 200)
print(R.round(4).to_string())
R.to_csv('../results/S2_ic.csv', index=False)

# pooled quintiles of T2 by factor (dev-wide quintile edges), plus per period
Q = []
for f in FAC:
    d = T.dropna(subset=[f, 'T2']).copy()
    d['q'] = pd.qcut(d[f], 5, labels=False, duplicates='drop')
    g = d.groupby('q').T2.mean() * 100
    gp = d.groupby(['per', 'q']).T2.mean().unstack() * 100
    Q.append(dict(factor=f, **{f'q{int(k)+1}': v for k, v in g.items()}, spread_q5_q1=g.iloc[-1] - g.iloc[0],
                  **{f'{p}_spread': gp.loc[p].iloc[-1] - gp.loc[p].iloc[0] for p in gp.index}))
Q = pd.DataFrame(Q)
print('\nT2 (neutral k=0.5 grid net return per episode, % of grid notional) by factor quintile')
print(Q.round(3).to_string())
Q.to_csv('../results/S2_quintiles.csv', index=False)
