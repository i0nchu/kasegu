"""bold-entry: HIT outcomes for the bold shot (same rules as src/kasegu/bold/shot.py) and daily entry factors."""
import numpy as np, pandas as pd
from numba import njit

DATA = '/home/user/data/mgrid'
SPLITS = {'dev': ('2020-07-01', '2023-05-31'), 'val': ('2023-07-01', '2024-11-30'), 'hold': ('2025-01-01', '2026-08-31')}
SUB = {'dev': (('2020H2-21', '2020-07-01', '2021-12-31'), ('2022', '2022-01-01', '2022-12-31'), ('2023H1', '2023-01-01', '2023-05-31')),
       'val': (('2023H2', '2023-07-01', '2023-12-31'), ('2024', '2024-01-01', '2024-11-30')),
       'hold': (('2025', '2025-01-01', '2025-12-31'), ('2026', '2026-01-01', '2026-08-31'))}
CONFIGS = {'L10': (10.0, 2.0, 0.1), 'L20': (20.0, 2.0, 0.1)}
TAKER = SLIP = 0.0005
MAX_BARS = 30 * 288


def load5(sym):
    z = np.load(f'{DATA}/{sym}.npz')
    t = z['t'].astype(np.int64) * 300_000
    return t, *(z[k].astype(np.float64) for k in 'ohlcf')


@njit(cache=True)
def shot(o, h, l, c, f, i0, side, L, target, bust):
    """returns (reason 1 target / 0 bust / 2 time / -1 no data, equity, bars)"""
    cost = 2 * L * (TAKER + SLIP)
    p0 = np.nan
    fund = 0.0
    n = len(o)
    for j in range(i0, min(i0 + MAX_BARS, n)):
        if np.isnan(o[j]):
            continue
        if np.isnan(p0):
            p0 = o[j] * (1 + side * SLIP)
        up = (target - 1 + cost + fund) / L
        dn = (1 - bust - cost - fund) / L
        bpx = p0 * (1 - side * dn)
        tpx = p0 * (1 + side * up)
        adv = l[j] if side == 1 else h[j]
        fav = h[j] if side == 1 else l[j]
        if (side == 1 and adv <= bpx) or (side == -1 and adv >= bpx):
            gap = (o[j] < bpx) if side == 1 else (o[j] > bpx)
            eq = 1 + side * L * (o[j] / p0 - 1) - cost - fund if gap else bust
            return 0, max(eq, 0.0), j - i0 + 1
        if (side == 1 and fav >= tpx) or (side == -1 and fav <= tpx):
            gap = (o[j] > tpx) if side == 1 else (o[j] < tpx)
            eq = 1 + side * L * (o[j] / p0 - 1) - cost - fund if gap else target
            return 1, eq, j - i0 + 1
        if f[j] != 0:
            fund += side * f[j] * L
        if j - i0 >= MAX_BARS - 1:
            return 2, 1 + side * L * (c[j] / p0 - 1) - cost - fund, j - i0 + 1
    return -1, np.nan, 0


def outcomes(sym, split, side=1):
    t, o, h, l, c, f = load5(sym)
    a, b = SPLITS[split]
    days = pd.date_range(a, b, freq='D')
    idx = np.searchsorted(t, days.as_unit('ms').asi8)
    assert abs(t[idx[0]] - days.as_unit('ms').asi8[0]) < 300_000
    rows = []
    for d, i0 in zip(days, idx):
        r = {'day': d}
        for name, (L, tg, bu) in CONFIGS.items():
            reason, eq, bars = shot(o, h, l, c, f, int(i0), side, L, tg, bu)
            r[f'{name}_reason'], r[f'{name}_eq'], r[f'{name}_days'] = reason, eq, bars / 288
        rows.append(r)
    df = pd.DataFrame(rows).set_index('day')
    for name in CONFIGS:
        df[f'{name}_hit'] = (df[f'{name}_reason'] == 1).astype(float)
        df.loc[df[f'{name}_reason'] < 0, f'{name}_hit'] = np.nan
    return df


def panel():
    z = np.load('/home/user/data/panel_1d_all.npz')
    idx = pd.date_range('2020-01-01', periods=z['c'].shape[0], freq='D')
    names = list(z['names'])
    P = {k: pd.DataFrame(z[k], index=idx, columns=names) for k in ('c', 'qv', 'fund')}
    return P


def breadth(P, n=40):
    c, qv = P['c'], P['qv']
    alts = [x for x in c.columns if x not in ('BTCUSDT', 'ETHUSDT')]
    c, qv = c[alts], qv[alts]
    above = (c > c.rolling(20, min_periods=20).mean()).astype(float).where(c.rolling(20, min_periods=20).mean().notna())
    rank = qv.rolling(30, min_periods=20).mean().rank(axis=1, ascending=False)
    top = rank <= n
    return (above.where(top)).mean(axis=1)


def features(sym, P):
    """value at day D uses data through D-1 close"""
    c, qv, fu = P['c'][sym], P['qv'][sym], P['fund'][sym]
    r = np.log(c).diff()
    F = pd.DataFrame({
        'FUND3': fu.rolling(3).mean(),
        'RET3': c / c.shift(3) - 1,
        'RET30': c / c.shift(30) - 1,
        'DIST200': c / c.rolling(200).mean() - 1,
        'RV7': r.rolling(7).std(),
        'DD30': c / c.rolling(30).max() - 1,
        'VOLR': qv.rolling(3).mean() / qv.rolling(30).mean(),
        'BREADTH': breadth(P),
    })
    return F.shift(1)


SIGN = {'FUND3': -1, 'RET3': -1, 'RET30': 1, 'DIST200': 1, 'RV7': 0, 'DD30': 0, 'VOLR': 0, 'BREADTH': 1}
