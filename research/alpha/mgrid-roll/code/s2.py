"""S2 on dev only: Q1 selection IC (with partial vs RV3), Q2 market timing terciles."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
from common import table, nw_t, SUBP, RV3_CUT

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'dev'
assert SPLIT == 'dev'
T = table('dev'); A = T[(T.dir != 0) & T.A_ret.notna()].copy()
A['date'] = pd.to_datetime(A.day * 86400, unit='s')
SEL = ['RV3', 'RV14', 'MOM7a', 'MOM30a', 'DIST50a', 'DD90a', 'FUND3a', 'QVR', 'LQV', 'BETA30', 'AGE']

def rk(x): return x.rank(pct=True)

rows = []
for f in SEL:
    ic, pic, dts = [], [], []
    for d, g in A.groupby('day'):
        g = g[list(dict.fromkeys([f, 'RV3', 'A_ret']))].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) < 5 or g[f].nunique() < 3: continue
        y, x, z = rk(g.A_ret), rk(g[f]), rk(g.RV3)
        ic.append(np.corrcoef(x, y)[0, 1])
        if f != 'RV3':
            bx = np.polyfit(z, x, 1); by = np.polyfit(z, y, 1)
            ex, ey = x - np.polyval(bx, z), y - np.polyval(by, z)
            pic.append(np.corrcoef(ex, ey)[0, 1] if ex.std() > 0 and ey.std() > 0 else np.nan)
        else: pic.append(np.nan)
        dts.append(d)
    s = pd.Series(ic, index=pd.to_datetime(np.array(dts) * 86400, unit='s')); p = pd.Series(pic, index=s.index)
    t = nw_t(s.values)
    r = dict(feature=f, days=len(s), IC=s.mean(), t=t, p=2 * norm.sf(abs(t)), pIC=p.mean(), p_t=nw_t(p.values) if f != 'RV3' else np.nan)
    for nm, a, b in SUBP: r[nm] = s[(s.index >= a) & (s.index < b)].mean()
    # quintile spread of mean episode return (pooled by daily rank)
    q = A.groupby('day')[f].rank(pct=True)
    r['Q5_ret%'] = A.A_ret[q > .8].mean() * 100; r['Q1_ret%'] = A.A_ret[q <= .2].mean() * 100
    rows.append(r)
R = pd.DataFrame(rows).sort_values('p')
m = len(R); R['holm_ok'] = [p <= 0.05 / (m - i) for i, p in enumerate(R.p)]
R['holm_ok'] = R.holm_ok.cummin()
R['consistent'] = np.sign(R[[n for n, _, _ in SUBP]]).nunique(axis=1) == 1
R['PASS'] = R.holm_ok & R.consistent & (((R.p_t.abs() > 2) & (np.sign(R.p_t) == np.sign(R.IC))) | (R.feature == 'RV3'))
pd.set_option('display.width', 250)
print('Q1 selection (dev)'); print(R.round(4).to_string(index=False))
R.to_csv('../results/S2_selection_dev.csv', index=False)

# Q2 timing: daily pool quality
P = A[A.RV3 >= RV3_CUT]
y = P.groupby('day').A_ret.mean()
Mf = A.groupby('day')[['BTCTR', 'BREADTH', 'MRV', 'MFUND', 'BTCRV7']].first()
Mf = Mf.loc[y.index]
half = y.index[: len(y) // 2]
rows = []
for f in Mf.columns:
    x = Mf[f]; lo, hi = x.loc[half].quantile([1 / 3, 2 / 3])
    terc = np.where(x <= lo, 1, np.where(x > hi, 3, 2))
    t1, t2, t3 = (y[terc == k] for k in (1, 2, 3))
    # OLS y ~ 1 + D3 on T1 u T3 days with NW se  (difference T3 - T1)
    sel = terc != 2; yy = y[sel].values; dd = (terc[sel] == 3).astype(float)
    X = np.column_stack([np.ones_like(dd), dd]); beta = np.linalg.lstsq(X, yy, rcond=None)[0]; e = yy - X @ beta
    n = len(yy); S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for j in range(1, 6):
        G = (X[j:] * e[j:, None]).T @ (X[:-j] * e[:-j, None]) / n; S += (1 - j / 6) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n); V = Q @ S @ Q / n; tt = beta[1] / np.sqrt(V[1, 1])
    r = dict(feature=f, cut_lo=lo, cut_hi=hi, T1_ret=t1.mean() * 100, T2_ret=t2.mean() * 100, T3_ret=t3.mean() * 100,
             n1=len(t1), n3=len(t3), diff_t=tt)
    idx = pd.to_datetime(y.index * 86400, unit='s')
    for nm, a, b in SUBP:
        mk = (idx >= a) & (idx < b)
        r[nm] = (y[mk & (terc == 3)].mean() - y[mk & (terc == 1)].mean()) * 100
    rows.append(r)
Q2 = pd.DataFrame(rows)
Q2['PASS'] = (Q2.diff_t.abs() > 2.58) & ((np.sign(Q2[[n for n, _, _ in SUBP]]).eq(np.sign(Q2.diff_t), axis=0)).sum(axis=1) >= 2)
print('\nQ2 timing (dev), returns in % per episode'); print(Q2.round(4).to_string(index=False))
Q2.to_csv('../results/S2_timing_dev.csv', index=False)
