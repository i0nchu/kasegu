"""mgrid-roll shared: coin-day table with BOTH direction and selection / timing features (all known at day start)."""
import numpy as np, pandas as pd

DATA = '/home/user/data'
RV3_CUT = 0.0696
SUBP = (('A_2020H2-21', '2020', '2022'), ('B_2022', '2022', '2023'), ('C_2023H1', '2023', '2024'))
SUBP_VAL = (('2023H2', '2023', '2024'), ('2024H1', '2024-01', '2024-07'), ('2024H2', '2024-07', '2025'))
SUBP_HOLD = (('2025H1', '2025', '2025-07'), ('2025H2', '2025-07', '2026'), ('2026', '2026', '2027'))


def panel():
    z = np.load(f'{DATA}/panel_1d_all.npz')
    names = list(z['names'])
    day0 = int(pd.Timestamp('2020-01-01').value // 10**9 // 86400)
    idx = np.arange(day0, day0 + len(z['c']))
    P = {k: pd.DataFrame(z[k], columns=names, index=idx) for k in ('c', 'qv', 'fund')}
    return P


def daily_features():
    """Per (day, sym) features using closes up to day-1 (rows shifted by one)."""
    P = panel(); c, qv, fd = P['c'], P['qv'], P['fund'].fillna(0.0)
    r = np.log(c).diff()
    sma50 = c.rolling(50, min_periods=50).mean()
    b = c['BTCUSDT']; bs = b.rolling(100, min_periods=100).mean()
    rb = r['BTCUSDT']
    F = {
        'up50': (c > sma50).astype(float).where(sma50.notna()),
        'MOM7': c / c.shift(7) - 1, 'MOM30': c / c.shift(30) - 1,
        'DIST50': c / sma50 - 1,
        'DDH': c / c.rolling(90, min_periods=30).max() - 1,
        'DDL': c / c.rolling(90, min_periods=30).min() - 1,
        'FUND3': fd.rolling(3, min_periods=3).mean(),
        'QVR': qv.rolling(7, min_periods=7).mean() / qv.rolling(30, min_periods=30).mean(),
        'LQV': np.log(qv.rolling(7, min_periods=7).mean()),
        'BETA30': r.rolling(30, min_periods=30).cov(rb).div(rb.rolling(30, min_periods=30).var(), axis=0),
        'AGE': c.notna().cumsum().where(c.notna()),
    }
    out = []
    for k, v in F.items():
        s = v.shift(1).stack(future_stack=True); s.index.names = ['day', 'sym']; out.append(s.rename(k))
    D = pd.concat(out, axis=1).reset_index()
    M = pd.DataFrame({'btc_up': ((b > bs).astype(float).where(bs.notna())).shift(1),
                      'BTCTR': (b / bs - 1).shift(1),
                      'BTCRV7': (rb.rolling(7, min_periods=7).std()).shift(1)})
    M.index.name = 'day'
    return D, M.reset_index()


def table(split, tag=''):
    T = pd.read_parquet(f'{DATA}/mgrid_{split}{tag}.parquet')
    T['day'] = (T.t5 // 288).astype(int)
    D, M = daily_features()
    T = T.merge(D, on=['day', 'sym'], how='left').merge(M, on='day', how='left')
    T['dir'] = np.where((T.btc_up == 1) & (T.up50 == 1), 1, np.where((T.btc_up == 0) & (T.up50 == 0), -1, 0))
    for pre in ('', 'K1', 'K2', 'N'):
        T[f'{pre}A_ret'] = np.where(T.dir == 1, T[f'{pre}L1.0_ret'], T[f'{pre}S1.0_ret'])
    T['MOM7a'] = T.dir * T.MOM7; T['MOM30a'] = T.dir * T.MOM30; T['DIST50a'] = T.dir * T.DIST50
    T['DD90a'] = np.where(T.dir == 1, T.DDH, -T.DDL); T['FUND3a'] = T.dir * T.FUND3
    # market features from the eligible universe of the day
    g = T.groupby('day')
    T['BREADTH'] = g.up50.transform('mean'); T['MRV'] = g.RV3.transform('median'); T['MFUND'] = g.FUND3.transform('median')
    return T


def nw_t(x, lag=5):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); m = x.mean(); e = x - m; v = e @ e / n
    for j in range(1, lag + 1): v += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return m / np.sqrt(v / n)
