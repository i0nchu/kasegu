"""Shot(entry=maker, tp_maker) == exec_ablation.shot2 on random dev days"""
import sys; sys.path.insert(0, '/home/user/kasegu/src')
import numpy as np, pandas as pd
from entry import load5
import importlib.util
spec = importlib.util.spec_from_file_location('ea', 'exec_ablation.py')
src = open('exec_ablation.py').read().split('\nrows = []')[0]
ns = {}; exec(compile(src, 'exec_ablation.py', 'exec'), ns)
shot2 = ns['shot2']
from kasegu.bold.shot import Shot, ShotParams
t, o, h, l, c, f = load5('BTCUSDT')
days = pd.date_range('2020-07-01', '2023-05-31', freq='D')
bad = 0; n = 0
for d in np.random.default_rng(1).choice(days, 60, replace=False):
    i0 = int(np.searchsorted(t, pd.Timestamp(d).value // 1_000_000))
    r, eq = shot2(o, h, l, c, f, i0, 10.0, True, True)
    s = Shot('BTCUSDT', 1, int(t[i0]), ShotParams(leverage=10, entry='maker', tp_maker=True))
    j = i0
    while s.open and j < len(t):
        s.on_bar(int(t[j]), o[j], h[j], l[j], c[j], f[j]); j += 1
    n += 1
    ok = {'target': 1, 'bust': 0, 'time': 2}[s.reason] == r and abs(s.equity - eq) < 1e-6
    if not ok: bad += 1; print(d, s.reason, s.equity, r, eq)
print('mismatches', bad, 'of', n)
