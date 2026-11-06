"""S3 compounding account backtest. Usage: python s3.py dev   |   python s3.py val --frozen"""
import sys, json, numpy as np, pandas as pd
from common import table, nw_t, RV3_CUT, SUBP, SUBP_VAL, SUBP_HOLD

PRE = {1.04: 'K1', 2.08: 'K2', 3.12: '', 4.16: 'N'}  # account leverage L -> margin column prefix (margin share 1/L)
SIZING = [(1, 1.04), (1, 2.08), (1, 3.12), (1, 4.16), (2, 2.08), (2, 3.12), (2, 4.16), (3, 3.12), (3, 4.16)]
SCORE_F = {'RV3': 1, 'RV14': 1, 'AGE': -1, 'DIST50a': 1, 'MOM30a': 1, 'BETA30': 1, 'DD90a': -1, 'FUND3a': -1}  # S2 dev passes
BTCRV7_CUT = 0.0291  # dev first-half bottom tercile (S2)
MIN_NOTIONAL = 104.0
E0 = 100.0


def prep(split, tag=''):
    T = table(split, tag)
    T = T[(T.dir != 0) & (T.RV3 >= RV3_CUT)].copy()
    g = T.groupby('day')
    T['SCORE'] = sum(s * g[f].rank(pct=True) for f, s in SCORE_F.items()) / len(SCORE_F)
    T['LONGdir'] = 1
    return T


def cols(T, pre, side):
    nm = 'L' if side == 1 else 'S'
    return {k: T[f'{pre}{nm}1.0_{k}'].to_numpy() for k in ('ret', 'end', 'm1', 'm2', 'why')}


def run(T, sig, K, L, days, timing=False, long_only=False):
    pre = PRE[L]
    T = T.copy()
    if long_only:  # baseline: always-long grids on the same pool (dir ignored)
        T['dir'] = 1
    key = 'SCORE' if sig == 'SCORE' else 'RV3'
    T = T.sort_values(['day', key], ascending=[True, False])
    if timing: T = T[T.BTCRV7 > BTCRV7_CUT]
    CL, CS = cols(T, pre, 1), cols(T, pre, -1)
    sym, dirs, dd_ = T.sym.to_numpy(), T.dir.to_numpy(), T.day.to_numpy()
    rows_by_day = {}
    for i, d in enumerate(dd_): rows_by_day.setdefault(d, []).append(i)
    E = E0; eq = np.zeros(len(days)); fr = np.zeros(len(days)); pend = {}; busy = {}; led = []
    for di, d in enumerate(days):
        busy = {s: e for s, e in busy.items() if e >= d}
        K_eff = min(K, int(L * E // MIN_NOTIONAL))
        free = K_eff - len(busy)
        if free > 0 and d in rows_by_day:
            for i in rows_by_day[d]:
                if free <= 0: break
                if sym[i] in busy: continue
                C = CL if dirs[i] == 1 else CS
                ret = C['ret'][i]
                if ret != ret: continue
                N = L * E / K_eff
                end = int(C['end'][i] // 288)
                parts = [ret] if end == 0 else ([C['m1'][i], ret - C['m1'][i]] if end == 1 else [C['m1'][i], C['m2'][i] - C['m1'][i], ret - C['m2'][i]])
                for off, v in enumerate(parts): pend[di + off] = pend.get(di + off, 0.0) + v * N
                busy[sym[i]] = d + end; free -= 1
                led.append((d, sym[i], int(dirs[i]), N, ret * N, int(C['why'][i]), end + 1))
        p = pend.pop(di, 0.0)
        fr[di] = p / E if E > 0 else 0.0
        E = max(E + p, 0.0); eq[di] = E
    idx = pd.to_datetime(days * 86400, unit='s')
    led = pd.DataFrame(led, columns=['day', 'sym', 'dir', 'notional', 'pnl', 'why', 'days'])
    return pd.Series(eq, idx), pd.Series(fr, idx), led


def boot(fr, n=2000, block=30, horizon=365, seed=0):
    rng = np.random.default_rng(seed); x = fr.to_numpy(); m = len(x)
    out = np.empty(n)
    for k in range(n):
        st = rng.integers(0, m - block, horizon // block + 1)
        path = np.concatenate([x[s:s + block] for s in st])[:horizon]
        e = E0
        for r in path:
            e *= 1 + r
            if e < 30: break
        out[k] = e
    return dict(b_med=np.median(out), b_p5=np.percentile(out, 5), b_double=(out >= 200).mean(), b_ruin=(out < 30).mean())


def summ(eq, fr, led, periods):
    yrs = len(eq) / 365
    o = dict(final=eq.iloc[-1], CAGR=(eq.iloc[-1] / E0) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1,
             maxDD=(eq / eq.cummax() - 1).min(), worst_day=fr.min(), t_NW=nw_t(fr.values), eps=len(led),
             win=(led.pnl > 0).mean() if len(led) else np.nan, long_share=(led.dir == 1).mean() if len(led) else np.nan,
             stops=(led.why >= 2).mean() if len(led) else np.nan)
    for nm, a, b in periods:
        s = fr[(fr.index >= a) & (fr.index < b)]
        o[nm] = (1 + s).prod() - 1
    return o


if __name__ == '__main__':
    split = sys.argv[1]
    periods = {'dev': SUBP, 'val': SUBP_VAL, 'hold': SUBP_HOLD}[split]
    res = '../results'
    if split == 'dev':
        allrows = []
        for tag in ('', '_stress'):
            T = prep('dev', tag); days = np.arange(T.day.min(), T.day.max() + 1)
            for K, L in SIZING:
                base = None
                for name, sig, tm, lo in (('LONG', 'RV3', False, True), ('MGT', 'RV3', False, False),
                                          ('SCORE', 'SCORE', False, False), ('SCORE_T', 'SCORE', True, False)):
                    eq, fr, led = run(T, sig, K, L, days, tm, lo)
                    r = dict(cost='stress' if tag else 'base', K=K, L=L, variant=name, **summ(eq, fr, led, periods), **boot(fr))
                    if name == 'LONG': flong = fr
                    if name == 'MGT': fmgt = fr
                    r['diff_vs_LONG_t'] = nw_t((fr - flong).values) if name != 'LONG' else np.nan
                    r['diff_vs_MGT_t'] = nw_t((fr - fmgt).values) if name in ('SCORE', 'SCORE_T') else np.nan
                    r['diff_vs_SCORE_t'] = nw_t((fr - fsc).values) if name == 'SCORE_T' else np.nan
                    if name == 'SCORE': fsc = fr
                    allrows.append(r); print(r['cost'], K, L, name, round(r['CAGR'], 3), round(r['maxDD'], 3), round(r['b_med'], 1), round(r['b_ruin'], 3), flush=True)
                    if K == 3 and L == 3.12 and tag == '':
                        led.to_csv(f'{res}/S3_dev_ledger_{name}_K3L3.12.csv.gz', index=False); eq.to_csv(f'{res}/S3_dev_equity_{name}_K3L3.12.csv')
        pd.DataFrame(allrows).to_csv(f'{res}/S3_dev_summary.csv', index=False)
    else:
        assert '--frozen' in sys.argv
        cfg = json.load(open(f'{res}/FROZEN.json'))
        rows = []
        for tag in ('', '_stress'):
            T = prep(split, tag); days = np.arange(T.day.min(), T.day.max() + 1)
            for K, L, cid in ((cfg['K'], cfg['L'], cfg['id']), (cfg['secondary']['K'], cfg['secondary']['L'], cfg['secondary']['id'])):
                out = {}
                for name, sig, tm, lo in (('FROZEN', cfg['sig'], cfg['timing'], False), ('MGT', 'RV3', False, False), ('LONG', 'RV3', False, True)):
                    eq, fr, led = run(T, sig, K, L, days, tm, lo); out[name] = fr
                    rows.append(dict(cfg=cid, cost='stress' if tag else 'base', variant=name, **summ(eq, fr, led, periods), **boot(fr)))
                    if name == 'FROZEN':
                        led.to_csv(f'{res}/S4_{split}{tag}_{cid}_ledger.csv.gz', index=False); eq.to_csv(f'{res}/S4_{split}{tag}_{cid}_equity.csv')
                rows[-3]['diff_vs_MGT_t'] = nw_t((out['FROZEN'] - out['MGT']).values); rows[-3]['diff_vs_MGT_mean'] = (out['FROZEN'] - out['MGT']).mean()
                rows[-3]['diff_vs_LONG_t'] = nw_t((out['FROZEN'] - out['LONG']).values); rows[-3]['diff_vs_LONG_mean'] = (out['FROZEN'] - out['LONG']).mean()
        R = pd.DataFrame(rows); pd.set_option('display.width', 250)
        print(f'{split} frozen {cfg}'); print(R.round(4).to_string(index=False))
        R.to_csv(f'{res}/S4_{split}_summary.csv', index=False)
