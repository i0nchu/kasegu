"""Execution ablation on dev (not a factor test): maker entry + maker TP vs taker on both legs.
Entry: buy limit at the 00:00 open, filled if price trades 0.05% below within 60 min, else market at the
next bar open (+slip). TP: resting limit (maker 0.02%), needs 0.05% penetration. Stop: market taker+slip."""
import numpy as np, pandas as pd
from numba import njit
from entry import load5, SPLITS

MAKER, TAKER, SLIP, PEN = 0.0002, 0.0005, 0.0005, 0.0005
MAX_BARS = 30 * 288


@njit(cache=True)
def shot2(o, h, l, c, f, i0, L, maker_entry, maker_tp):
    n = len(o); j = i0
    if maker_entry:
        lim = o[i0]; p0 = np.nan
        for k in range(i0, min(i0 + 12, n)):
            if l[k] <= lim * (1 - PEN):
                p0 = lim; j = k; break
        fe = MAKER
        if np.isnan(p0):
            j = i0 + 12; p0 = o[j] * (1 + SLIP); fe = TAKER + SLIP  # same convention as Shot
    else:
        p0 = o[i0] * (1 + SLIP); fe = TAKER + SLIP
    ft = MAKER if maker_tp else TAKER + SLIP
    fs = TAKER + SLIP
    fund = 0.0; first = True
    for jj in range(j, min(i0 + MAX_BARS, n)):
        up = (1 + L * (fe + ft) + fund) / L
        dn = (0.9 - L * (fe + fs) - fund) / L
        bpx = p0 * (1 - dn); tpx = p0 * (1 + up)
        lo = l[jj]; hi = h[jj]
        if first and maker_entry and jj == j:
            first = False  # entry bar: only bars after the fill can trigger exits
            continue
        if lo <= bpx:
            px = o[jj] if o[jj] < bpx else bpx
            return 0, max(1 + L * (px / p0 - 1) - L * (fe + fs) - fund, 0.0)
        if (hi >= tpx * (1 + PEN)) if maker_tp else (hi >= tpx):
            return 1, 2.0
        if f[jj] != 0:
            fund += f[jj] * L
        if jj - i0 >= MAX_BARS - 1:
            return 2, 1 + L * (c[jj] / p0 - 1) - L * (fe + fs) - fund
    return -1, np.nan


rows = []
for sym in ('BTCUSDT', 'ETHUSDT'):
    t, o, h, l, c, f = load5(sym)
    days = pd.date_range(*SPLITS['dev'], freq='D')
    idx = np.searchsorted(t, days.as_unit('ms').asi8)
    for L in (10.0, 20.0):
        for me, mt, name in ((False, False, 'taker/taker'), (False, True, 'taker in, maker TP'), (True, True, 'maker in, maker TP')):
            r = np.array([shot2(o, h, l, c, f, int(i), L, me, mt) for i in idx])
            ok = r[:, 0] >= 0
            rows.append(dict(sym=sym, L=int(L), exec=name, hit=(r[ok, 0] == 1).mean(), time_exit=(r[ok, 0] == 2).mean(), mean_eq=r[ok, 1].mean()))
R = pd.DataFrame(rows)
print(R.round(3).to_string())
R.to_csv('../results/exec_ablation_dev.csv', index=False)
