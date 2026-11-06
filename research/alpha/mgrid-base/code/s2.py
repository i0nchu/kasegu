"""S2 (dev): forward 30-day return / max drawdown by regime state, BTC and ETH."""
import numpy as np, pandas as pd, base as B
rng = np.random.default_rng(0)
rows = []
for sym in ('BTCUSDT', 'ETHUSDT'):
    dc = B.panel_close(sym)
    R = B.regimes(dc)
    prev = dc.shift(1)  # entry ~ open of day d = close of d-1
    fwd = dc.shift(-29) / prev - 1
    mdd = dc[::-1].rolling(30, min_periods=30).min()[::-1] / prev - 1
    a, b = B.SPLITS['dev']
    for clf in R.columns:
        df = pd.DataFrame({'s': R[clf], 'f': fwd, 'm': mdd}).loc[a:'2023-06-01'].dropna()
        df['mon'] = df.index.to_period('M')
        for s, g in df.groupby('s'):
            rows.append(dict(sym=sym, clf=clf, state={-1: 'DOWN', 0: 'RANGE', 1: 'UP'}[s], days=len(g), share=len(g) / len(df),
                             fwd30=g.f.mean(), fwd30_med=g.f.median(), mdd30=g.m.mean(), p_up=(g.f > 0).mean()))
        # block bootstrap by month: DOWN minus non-DOWN fwd30
        mons = df.mon.unique(); diffs = []
        for _ in range(2000):
            pick = rng.choice(mons, len(mons)); x = pd.concat([df[df.mon == m] for m in pick[:]]) if False else df[df.mon.isin(pick)]
            # weight by multiplicity
            cnt = pd.Series(pick).value_counts(); w = df.mon.map(cnt).fillna(0).to_numpy()
            dn = w * (df.s == -1); nd = w * (df.s != -1)
            if dn.sum() == 0 or nd.sum() == 0: continue
            diffs.append((dn * df.f).sum() / dn.sum() - (nd * df.f).sum() / nd.sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(dict(sym=sym, clf=clf, state='DOWN-minus-rest', fwd30=np.mean(diffs), fwd30_med=lo, mdd30=hi))
R = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(R.round(3).to_string()); R.to_csv('../results/S2_states.csv', index=False)
