"""S3b (dev): how much of the long-grid + RV3 result is long beta to high-vol alts? and selection noise."""
import numpy as np, pandas as pd, statsmodels.api as sm
import s3_account as A
T = A.T
T['per'] = np.where(T.date < '2022-01-01', 'A_2020H2-21', np.where(T.date < '2023-01-01', 'B_2022', 'C_2023H1'))
hi = T[T.RV3 >= A.RV3_CUT]
print('episode-level (all eligible coin-days, % of grid notional): mean grid ret and mean coin 3d return')
for nm, d in (('RV3 top tercile', hi), ('all', T)):
    print(nm); print((d.groupby('per')[['L0.5_ret', 'S0.5_ret', 'NL0.5_ret', 'NS0.5_ret', 'fRET']].mean() * 100).round(3).to_string())
# regression of long grid episode return on coin 3d return, day-clustered
for tag in ('L0.5', 'L1.0'):
    d = hi.dropna(subset=[f'{tag}_ret', 'fRET'])
    X = sm.add_constant(d.fRET.clip(-0.5, 0.5))
    m = sm.OLS(d[f'{tag}_ret'], X).fit(cov_type='cluster', cov_kwds={'groups': d.day})
    print(f'\n{tag} on fRET (RV3 top tercile): alpha {m.params.const*100:.3f}% (t {m.tvalues.const:.2f}), beta {m.params.fRET:.3f} (t {m.tvalues.fRET:.1f}), n {len(d)}')
    for p, g in d.groupby('per'):
        mm = sm.OLS(g[f'{tag}_ret'], sm.add_constant(g.fRET.clip(-0.5, 0.5))).fit(cov_type='cluster', cov_kwds={'groups': g.day})
        print(f'   {p}: alpha {mm.params.const*100:.3f}% (t {mm.tvalues.const:.2f}) beta {mm.params.fRET:.3f}')
# selection noise: no-filter long 0.5 with 10 seeds
tots = []
for seed in range(10):
    s, led = A.run('long', 0.5, 'none', seed); tots.append(s.mean() * 365)
print('\nlong 0.5 no filter, 10 random selection seeds, USD/yr:', np.round(tots, 1), 'mean', np.mean(tots).round(1), 'sd', np.std(tots).round(1))
tots = []
for seed in range(10):
    s, led = A.run('neutral', 0.5, 'none', seed); tots.append(s.mean() * 365)
print('neutral 0.5 no filter, 10 seeds, USD/yr:', np.round(tots, 1), 'mean', np.mean(tots).round(1), 'sd', np.std(tots).round(1))
