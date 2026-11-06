"""S3 (dev): account-level backtest of the 6 grid configs x {no filter, RV3 top tercile}.
100 USD, no compounding, slot = 104 USDT grid notional. long/short: K=3 slots (margin 33 each);
neutral: K=2 coins x (long + short grid), margin 25 per grid. Episodes start at 00:00 UTC only;
a slot is free again the day after its episode ends. PnL marked to market at each day end."""
import sys
import numpy as np, pandas as pd

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'dev'
TAG = sys.argv[2] if len(sys.argv) > 2 else ''
NOTIONAL = 104.0
T = pd.read_parquet(f'/home/user/data/mgrid_{SPLIT}{TAG}.parquet')
RV3_CUT = pd.read_parquet('/home/user/data/mgrid_dev.parquet').RV3.quantile(2 / 3)  # dev-wide edge, frozen
T['day'] = (T.t5 // 288).astype(int)
days = np.arange(T.day.min(), T.day.max() + 1)

def contrib(row, tag):
    """daily MTM PnL (USD) of one grid episode as {day offset: pnl}"""
    ret, end, m1, m2 = row[f'{tag}_ret'], row[f'{tag}_end'], row[f'{tag}_m1'], row[f'{tag}_m2']
    dd = int(end // 288)
    if dd == 0: return {0: ret}, 0
    if dd == 1: return {0: m1, 1: ret - m1}, 1
    return {0: m1, 1: m2 - m1, 2: ret - m2}, 2

def run(direction, k, filt, seed=0):
    rng = np.random.default_rng(seed)
    if direction == 'neutral':
        K, tags = 2, [f'NL{k}', f'NS{k}']
    else:
        K, tags = 3, [('L' if direction == 'long' else 'S') + str(k)]
    by_day = {d: g for d, g in T.groupby('day')}
    pnl = np.zeros(len(days) + 3); busy = {}  # sym -> last busy day
    led = []
    for di, d in enumerate(days):
        busy = {s: e for s, e in busy.items() if e >= d}
        free = K - len(busy)
        g = by_day.get(d)
        if free <= 0 or g is None: continue
        g = g[~g.sym.isin(busy) & g[f'{tags[0]}_ret'].notna()]
        if filt == 'RV3':
            g = g[g.RV3 >= RV3_CUT].sort_values('RV3', ascending=False)
        else:
            g = g.iloc[rng.permutation(len(g))]
        for _, row in g.head(free).iterrows():
            last = 0; tot = 0.0
            for tag in tags:
                c, dd = contrib(row, tag)
                for off, v in c.items(): pnl[di + off] += v * NOTIONAL
                last = max(last, dd); tot += row[f'{tag}_ret'] * NOTIONAL
            busy[row.sym] = d + last
            led.append(dict(day=d, sym=row.sym, RV3=row.RV3, pnl=tot, why=max(row[f'{t}_why'] for t in tags), days_held=last + 1))
    return pd.Series(pnl[:len(days)], index=pd.to_datetime(days * 86400, unit='s')), pd.DataFrame(led)

def nw_t(x, lag=5):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m; v = e @ e / n
    for k in range(1, lag + 1): v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(v / n)

def summ(s, led):
    eq = s.cumsum(); dd = (eq - eq.cummax()).min()
    per = {p: s[(s.index >= a) & (s.index < b)].sum() for p, a, b in (('A', '2020', '2022'), ('B', '2022', '2023'), ('C', '2023', '2024'))}
    return dict(total_usd=s.sum(), usd_per_year=s.mean() * 365, t_NW=nw_t(s.values), maxDD=dd, worst_day=s.min(),
                episodes=len(led), win=(led.pnl > 0).mean(), worst_ep=led.pnl.min(), stops=(led.why >= 2).mean(),
                liq=(led.why == 3).mean(), A_2020H2_21=per['A'], B_2022=per['B'], C_2023H1=per['C'])

if __name__ == '__main__':
    rows, series = [], {}
    for direction in ('long', 'short', 'neutral'):
        for k in (0.5, 1.0):
            for filt in ('none', 'RV3'):
                s, led = run(direction, k, filt)
                series[(direction, k, filt)] = s
                rows.append(dict(direction=direction, k=k, filter=filt, **summ(s, led)))
                led.to_csv(f'../results/S3_{SPLIT}{TAG}_ledger_{direction}_{k}_{filt}.csv.gz', index=False)
    R = pd.DataFrame(rows)
    # paired difference RV3 - none (same config)
    for i, r in R[R['filter'] == 'RV3'].iterrows():
        d = series[(r.direction, r.k, 'RV3')] - series[(r.direction, r.k, 'none')]
        R.loc[i, 'diff_vs_none_usd_yr'] = d.mean() * 365; R.loc[i, 'diff_t'] = nw_t(d.values)
    pd.set_option('display.width', 250)
    print(f'{SPLIT}{TAG} RV3 cut {RV3_CUT:.4f}'); print(R.round(3).to_string())
    R.to_csv(f'../results/S3_{SPLIT}{TAG}_summary.csv', index=False)
    pd.DataFrame({f'{a}_{b}_{c}': v for (a, b, c), v in series.items()}).to_csv(f'../results/S3_{SPLIT}{TAG}_daily.csv.gz')
