"""S2 dev: HIT by factor tercile, NW lag-30 t, Holm across 8 factors, ETH and sub-period consistency."""
import numpy as np, pandas as pd, statsmodels.api as sm
from entry import outcomes, features, panel, SIGN, SUB

P = panel()
rows, terc = [], {}
data = {}
for sym in ('BTCUSDT', 'ETHUSDT'):
    Y = outcomes(sym, 'dev'); F = features(sym, P).reindex(Y.index)
    data[sym] = Y.join(F)
    data[sym].to_csv(f'../results/dev_{sym}.csv')
res = []
for cfg in ('L10', 'L20'):
    for fac, sg in SIGN.items():
        D = data['BTCUSDT']
        x, y = D[fac], D[f'{cfg}_hit']
        m = x.notna() & y.notna()
        q1, q2 = x[m].quantile([1 / 3, 2 / 3])
        tr = pd.cut(x, [-np.inf, q1, q2, np.inf], labels=[0, 1, 2]).astype(float)
        base = y[m].mean()
        hr = [y[m & (tr == k)].mean() for k in range(3)]
        fav = (2 if sg > 0 else 0) if sg != 0 else (2 if hr[2] >= hr[0] else 0)
        # NW: y on (fav tercile) dummy vs rest
        X = sm.add_constant((tr[m] == fav).astype(float))
        ols = sm.OLS(y[m], X).fit(cov_type='HAC', cov_kwds={'maxlags': 30})
        tstat, p = ols.tvalues.iloc[1], ols.pvalues.iloc[1]
        # ETH with BTC-thresholds-style: ETH own terciles of same factor
        E = data['ETHUSDT']; xe, ye = E[fac], E[f'{cfg}_hit']; me = xe.notna() & ye.notna()
        e1, e2 = xe[me].quantile([1 / 3, 2 / 3])
        tre = pd.cut(xe, [-np.inf, e1, e2, np.inf], labels=[0, 1, 2]).astype(float)
        eth_d = ye[me & (tre == fav)].mean() - ye[me].mean()
        subs = []
        for name, a, b in SUB['dev']:
            s = m & (D.index >= a) & (D.index <= b)
            subs.append(y[s & (tr == fav)].mean() - y[s].mean())
        res.append(dict(cfg=cfg, factor=fac, n=int(m.sum()), base=base, T0=hr[0], T1=hr[1], T2=hr[2], fav=fav,
                        fav_minus_base=hr[fav] - base, nw_t=tstat, p=p, eth_diff=eth_d,
                        sub_2020H2_21=subs[0], sub_2022=subs[1], sub_2023H1=subs[2], thr1=q1, thr2=q2))
R = pd.DataFrame(res)
for cfg in ('L10', 'L20'):
    k = R.cfg == cfg
    ps = R.loc[k, 'p'].to_numpy(); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, ps[i] * (m - r))); adj[i] = run
    R.loc[k, 'p_holm'] = adj
R['pass'] = (R.fav_minus_base >= 0.05) & (R.p_holm < 0.05) & (R.eth_diff > 0) & (R[['sub_2020H2_21', 'sub_2022', 'sub_2023H1']] > 0).all(axis=1)
pd.set_option('display.width', 250)
print(R.drop(columns=['thr1', 'thr2']).round(3).to_string())
R.to_csv('../results/S2_dev.csv', index=False)
for sym in data:
    D = data[sym]
    print(sym, 'L10 hit', round(D.L10_hit.mean(), 3), 'median days', round(D.L10_days.median(), 1), 'L20 hit', round(D.L20_hit.mean(), 3), 'days', round(D.L20_days.median(), 1))
