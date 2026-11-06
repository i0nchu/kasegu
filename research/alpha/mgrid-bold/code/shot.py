"""One-shot bold bets on BTC/ETH perps: at 00:00 UTC enter with leverage L, take profit when equity
reaches the target multiple, bust at -90% equity (stop just before liquidation). Direction: always
long, or by daily state (UP long, DOWN short, RANGE skip; MA50/200 stack). Entry every day of
2020-07..2026-09 (overlapping bets, each one a separate 100 USD try)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '../../mgrid-base/code')
import base as B  # noqa: E402

TAKER, SLIP = 0.0005, 0.0005
rows = []
for sym in ('BTCUSDT', 'ETHUSDT'):
    z = np.load(f'/home/user/data/mgrid/{sym}.npz')
    t = pd.to_datetime(z['t'].astype(np.int64) * 300, unit='s')
    d = pd.DataFrame({'o': z['o'], 'h': z['h'], 'l': z['l'], 'c': z['c'], 'f': z['f']}, index=t).astype(float)
    h = d.resample('1h').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'f': 'sum'}).loc['2020-07-01':]
    st = B.regimes(B.panel_close(sym))['MA50_200']
    o, hi, lo, c, f = (h[x].to_numpy() for x in 'ohlcf')
    day_idx = np.nonzero(h.index.hour == 0)[0]
    for L in (5, 10, 20):
        for target in (2, 5, 10):
            for rule in ('always_long', 'state'):
                res = []
                for i0 in day_idx:
                    day = h.index[i0].normalize()
                    s = st.get(day, np.nan)
                    if rule == 'always_long': side = 1
                    elif s == 1: side = 1
                    elif s == -1: side = -1
                    else: continue
                    if np.isnan(o[i0]): continue
                    p0 = o[i0] * (1 + side * SLIP)
                    cost = 2 * L * (TAKER + SLIP)  # equity fraction for entry + exit
                    # equity multiple m = 1 + side*L*(p/p0-1) - cost - funding
                    up = (target - 1 + cost) / L; dn = (0.9 - cost) / L  # price move for target / bust (before funding)
                    fund = 0.0; out = None
                    for j in range(i0, min(i0 + 24 * 180, len(o))):
                        if np.isnan(c[j]): continue
                        adv = lo[j] if side == 1 else hi[j]; fav = hi[j] if side == 1 else lo[j]
                        if side * (adv / p0 - 1) <= -(dn - fund / L): out = 0; break  # bust first (conservative)
                        if side * (fav / p0 - 1) >= up + fund / L: out = 1; break
                        fund += side * f[j] * L  # equity fraction paid
                    if out is None: continue
                    res.append((day, out))
                r = pd.DataFrame(res, columns=['day', 'hit'])
                for per, a, b in (('all', '2020', '2027'), ('2020H2-21', '2020', '2022'), ('2022-23', '2022', '2024'), ('2024', '2024', '2025'), ('2025-26', '2025', '2027')):
                    x = r[(r.day >= a) & (r.day < b)]
                    if len(x): rows.append(dict(sym=sym, L=L, target=target, rule=rule, period=per, n=len(x), p_hit=x.hit.mean()))
R = pd.DataFrame(rows)
piv = R[R.period == 'all'].pivot_table(index=['target', 'L'], columns=['sym', 'rule'], values='p_hit').round(3)
pd.set_option('display.width', 200); print(piv.to_string())
print(R[(R.target == 2) & (R.L == 10)].pivot_table(index='period', columns=['sym', 'rule'], values='p_hit').round(3).to_string())
R.to_csv('../results/shot_table.csv', index=False)
