"""numba shot == kasegu.bold.shot.Shot on random dev days"""
import sys; sys.path.insert(0, '/home/user/kasegu/src')
import numpy as np, pandas as pd
from entry import load5, shot
from kasegu.bold.shot import Shot, ShotParams
t, o, h, l, c, f = load5('BTCUSDT')
rng = np.random.default_rng(0)
days = pd.date_range('2020-07-01', '2023-05-31', freq='D')
bad = 0
for d in rng.choice(days, 40, replace=False):
    i0 = int(np.searchsorted(t, pd.Timestamp(d).value // 1_000_000))
    for L in (10.0, 20.0):
        r, eq, bars = shot(o, h, l, c, f, i0, 1, L, 2.0, 0.1)
        s = Shot('BTCUSDT', 1, int(t[i0]), ShotParams(leverage=L))
        j = i0
        while s.open and j < len(t):
            s.on_bar(int(t[j]), o[j], h[j], l[j], c[j], f[j]); j += 1
        ok = {'target': 1, 'bust': 0, 'time': 2}[s.reason] == r and abs(s.equity - eq) < 1e-9
        bad += not ok
print('mismatches', bad, 'of 80')
