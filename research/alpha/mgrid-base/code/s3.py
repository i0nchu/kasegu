"""S3 (dev, exploratory after S2 failed): B0/B1/G1/G2 x 3 classifiers, BTC+ETH 50 USD sleeves."""
import sys
import numpy as np, pandas as pd, base as B
split = sys.argv[1] if len(sys.argv) > 1 else 'dev'
a, b = B.SPLITS[split]
data = {}
for sym in ('BTCUSDT', 'ETHUSDT'):
    h, _ = B.load(sym)
    R = B.regimes(B.panel_close(sym))
    data[sym] = (h.loc[a:b].iloc[:-1], R, B.sigma7(h.c).loc[a:b].iloc[:-1])
rows = []
for clf in ('MA50_200', 'MA20_100', 'TSMOM'):
    for mode in ('B0', 'B1', 'G1', 'G2'):
        if mode == 'B0' and clf != 'MA50_200': continue
        eqs, ntr, inmk = [], 0, []
        for sym, (h, R, sg) in data.items():
            eq, tr = B.simulate(h, R[clf], sg, mode)
            eqs.append(eq); ntr += tr
        eq = eqs[0] + eqs[1]
        st = B.stats(eq, 100.0)
        d = eq.resample('1D').last()
        per = {p: d.loc[x:y].iloc[-1] / d.loc[x:y].iloc[0] - 1 for p, x, y in B.PER[split] if len(d.loc[x:y])}
        rows.append(dict(clf=clf if mode != 'B0' else '-', mode=mode, **st, trades=ntr, **{f'ret_{k}': v for k, v in per.items()}))
R = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(split); print(R.round(3).to_string()); R.to_csv(f'../results/S3_{split}.csv', index=False)
