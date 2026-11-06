"""mgrid-trend: trend-aligned martingale grid, account backtest on the mgrid-regime coin-day tables.
Usage: python trend.py dev [stress]   |   python trend.py val [stress] --frozen   (val only with the frozen config)"""
import sys, json
import numpy as np, pandas as pd

SPLIT = sys.argv[1]
TAG = '_stress' if 'stress' in sys.argv else ''
NOTIONAL = 104.0
RV3_CUT = 0.0696  # frozen in mgrid-regime (dev-wide top tercile)
MPRE = {1: 'K1', 2: 'K2', 3: ''}

def regimes():
    z = np.load('/home/user/data/panel_1d_all.npz')
    names = list(z['names']); c = pd.DataFrame(z['c'], columns=names)
    day0 = int(pd.Timestamp('2020-01-01').value // 10**9 // 86400)
    c.index = np.arange(day0, day0 + len(c))
    # decision on day d uses closes up to day d-1
    up50 = (c > c.rolling(50, min_periods=50).mean()).where(c.rolling(50, min_periods=50).mean().notna()).shift(1)
    b = c['BTCUSDT']; btc = (b > b.rolling(100).mean()).where(b.rolling(100).mean().notna()).shift(1)
    return up50, btc

def load(split, tag):
    T = pd.read_parquet(f'/home/user/data/mgrid_{split}{tag}.parquet')
    T['day'] = (T.t5 // 288).astype(int)
    up50, btc = regimes()
    T['btc_up'] = T.day.map(btc)
    s = up50.stack(); s.index.names = ['day', 'sym']
    T = T.merge(s.rename('coin_up').reset_index(), on=['day', 'sym'], how='left')
    return T

def direction(T, regime, seed=0):
    if regime == 'BTC100': d = np.where(T.btc_up == 1, 1, np.where(T.btc_up == 0, -1, 0))
    elif regime == 'COIN50': d = np.where(T.coin_up == 1, 1, np.where(T.coin_up == 0, -1, 0))
    elif regime == 'BOTH': d = np.where((T.btc_up == 1) & (T.coin_up == 1), 1, np.where((T.btc_up == 0) & (T.coin_up == 0), -1, 0))
    elif regime == 'LONG': d = np.ones(len(T), int)
    elif regime == 'SHORT': d = -np.ones(len(T), int)
    elif regime == 'RANDOM': d = np.random.default_rng(seed + 99).choice([-1, 1], len(T))
    return d

def run(T, regime, k, filt, K, seed=0):
    rng = np.random.default_rng(seed)
    T = T.assign(dir=direction(T, regime, seed))
    T = T[T.dir != 0]
    pre = MPRE[K]
    days = np.arange(DAYS[0], DAYS[1] + 1)
    by_day = {d: g for d, g in T.groupby('day')}
    pnl = np.zeros(len(days) + 3); busy = {}; led = []
    for di, d in enumerate(days):
        busy = {s: e for s, e in busy.items() if e >= d}
        free = K - len(busy); g = by_day.get(d)
        if free <= 0 or g is None: continue
        g = g[~g.sym.isin(busy)]
        if filt == 'RV3': g = g[g.RV3 >= RV3_CUT].sort_values('RV3', ascending=False)
        else: g = g.iloc[rng.permutation(len(g))]
        for _, r in g.head(free).iterrows():
            tag = f"{pre}{'L' if r.dir == 1 else 'S'}{k}"
            ret, end, m1, m2 = r[f'{tag}_ret'], r[f'{tag}_end'], r[f'{tag}_m1'], r[f'{tag}_m2']
            if ret != ret: continue
            dd = int(end // 288)
            parts = {0: ret} if dd == 0 else ({0: m1, 1: ret - m1} if dd == 1 else {0: m1, 1: m2 - m1, 2: ret - m2})
            for off, v in parts.items(): pnl[di + off] += v * NOTIONAL
            busy[r.sym] = d + dd
            led.append(dict(date=pd.to_datetime(d * 86400, unit='s'), sym=r.sym, dir=int(r.dir), RV3=r.RV3, step=k * min(max(r.SIG, .005), .1),
                            pnl_usd=ret * NOTIONAL, why=int(r[f'{tag}_why']), levels=int(r[f'{tag}_lev']), funding_usd=r[f'{tag}_fund'] * NOTIONAL, days=dd + 1))
    return pd.Series(pnl[:len(days)], index=pd.to_datetime(days * 86400, unit='s')), pd.DataFrame(led)

def nw_t(x, lag=5):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m; v = e @ e / n
    for j in range(1, lag + 1): v += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return m / np.sqrt(v / n)

PERIODS = {'dev': (('A_2020H2-21', '2020', '2022'), ('B_2022', '2022', '2023'), ('C_2023H1', '2023', '2024')),
           'val': (('2023H2', '2023', '2024'), ('2024H1', '2024-01', '2024-07'), ('2024H2', '2024-07', '2025'))}

def summ(s, led):
    eq = s.cumsum()
    out = dict(usd_per_year=s.mean() * 365, t_NW=nw_t(s.values), maxDD=(eq - eq.cummax()).min(), worst_day=s.min(),
               episodes=len(led), win=(led.pnl_usd > 0).mean() if len(led) else np.nan, worst_ep=led.pnl_usd.min() if len(led) else np.nan,
               long_share=(led.dir == 1).mean() if len(led) else np.nan)
    for p, a, b in PERIODS[SPLIT]: out[p] = s[(s.index >= a) & (s.index < b)].sum()
    return out

if __name__ == '__main__':
    import os
    T = load(SPLIT, TAG)
    DAYS = (T.day.min(), T.day.max())
    res = '../results'
    if SPLIT == 'dev':
        rows, ser = [], {}
        for regime in ('BTC100', 'COIN50', 'BOTH', 'LONG', 'SHORT', 'RANDOM'):
            for k in (0.5, 1.0):
                for filt in ('RV3', 'none'):
                    s, led = run(T, regime, k, filt, 3); ser[(regime, k, filt)] = s
                    rows.append(dict(regime=regime, k=k, filter=filt, **summ(s, led)))
        R = pd.DataFrame(rows)
        for i, r in R.iterrows():
            dlt = ser[(r.regime, r.k, r['filter'])] - ser[('LONG', r.k, r['filter'])]
            R.loc[i, 'diff_vs_long_yr'] = dlt.mean() * 365; R.loc[i, 'diff_t'] = nw_t(dlt.values)
        pd.set_option('display.width', 250)
        print(f'dev{TAG} K=3'); print(R.round(2).to_string())
        R.to_csv(f'{res}/dev{TAG}_K3_summary.csv', index=False)
    else:
        cfg = json.load(open(f'{res}/FROZEN.json'))
        rows = []
        for regime in (cfg['regime'], 'LONG', 'SHORT', 'RANDOM'):
            s, led = run(T, regime, cfg['k'], cfg['filter'], cfg['K'])
            rows.append(dict(regime=regime, **summ(s, led)))
            if regime == cfg['regime']:
                main = s; led.to_csv(f'{res}/{SPLIT}{TAG}_ledger.csv', index=False); s.to_csv(f'{res}/{SPLIT}{TAG}_daily.csv')
            if regime == 'LONG': base = s
        R = pd.DataFrame(rows); dlt = main - base
        print(f'{SPLIT}{TAG} frozen {cfg}'); print(R.round(2).to_string())
        print('paired diff vs LONG USD/yr %.1f t %.2f' % (dlt.mean() * 365, nw_t(dlt.values)))
        R.to_csv(f'{res}/{SPLIT}{TAG}_summary.csv', index=False)
