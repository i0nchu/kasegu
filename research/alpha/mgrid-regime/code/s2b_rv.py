"""S2b (dev): is the RV effect real? decomposition by side, step units, stop rate; incremental IC over RV3."""
import numpy as np, pandas as pd
from scipy import stats
T = pd.read_parquet('/home/user/data/mgrid_dev.parquet')
T['T2'] = T['L0.5_ret'] + T['S0.5_ret']
T['step'] = 0.5 * T.SIG.clip(0.005, 0.10)
T['ratio'] = T.RV3 / T.SIG
T['q'] = pd.qcut(T.RV3, 5, labels=False)
g = T.groupby('q').agg(RV3=('RV3', 'median'), step=('step', 'median'), T2=('T2', 'mean'), L=('L0.5_ret', 'mean'), S=('S0.5_ret', 'mean'),
                       stopL=('L0.5_why', lambda x: (x >= 2).mean()), stopS=('S0.5_why', lambda x: (x >= 2).mean()),
                       winN=('T2', lambda x: (x > 0).mean()), fret=('fRET', 'mean'), n=('T2', 'size'))
g['T2_in_steps'] = T.groupby('q').apply(lambda d: (d.T2 / d.step).mean(), include_groups=False)
for c in ('T2', 'L', 'S', 'fret'): g[c] *= 100
print('RV3 quintiles (returns in %):'); print(g.round(3).to_string())
# ratio of short-term vol to the 7d vol used for the step
T['qr'] = pd.qcut(T.ratio, 5, labels=False)
print('\nRV3/SIG quintiles'); print((T.groupby('qr').agg(ratio=('ratio', 'median'), T2=('T2', 'mean')).assign(T2=lambda d: d.T2 * 100)).round(3).to_string())
# 2-way: RV3 quintile x ER14 tercile
T['er'] = pd.qcut(T.ER14, 3, labels=['chop', 'mid', 'trend'])
print('\nT2 % by RV3 quintile x ER14 tercile'); print((T.pivot_table(index='q', columns='er', values='T2', aggfunc='mean', observed=True) * 100).round(3).to_string())
# incremental IC over RV3: daily rank-residualize factor and target on RV3
FAC = [f'{n}{L}' for n in ('ER', 'VR', 'ADX', 'CHOP', 'AC1', 'VOLR') for L in (3, 14)]
def resid(d, col):
    x = d['RV3'].rank(); y = d[col].rank()
    b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
rows = []
for f in FAC:
    ics = []
    for dt, d in T.dropna(subset=[f, 'T2', 'RV3']).groupby('date'):
        if len(d) < 10: continue
        ics.append(stats.pearsonr(resid(d, f), resid(d, 'T2'))[0])
    ics = np.array(ics); m = ics.mean(); e = ics - m; v = e @ e / len(e)
    for k in range(1, 4): v += 2 * (1 - k / 4) * (e[k:] @ e[:-k]) / len(e)
    rows.append(dict(factor=f, partial_IC=m, t_NW=m / np.sqrt(v / len(e))))
P = pd.DataFrame(rows); print('\nPartial IC vs T2 controlling RV3 (12 tests)'); print(P.round(4).to_string())
P.to_csv('../results/S2b_partial_ic.csv', index=False); g.to_csv('../results/S2b_rv_quintiles.csv')
