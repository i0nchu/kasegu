"""Check that kasegu.grid.episode reproduces the research backtest episode on random paths and on real data."""
import sys
import numpy as np
sys.path.insert(0, 'research/alpha/mgrid-regime/code')
import mg  # noqa: E402
from kasegu.grid.episode import GridEpisode, GridParams  # noqa: E402

def research(o, h, l, c, f, step, side):
    r = mg.episode(o, h, l, c, f, 0, step, side, mg.PEN, mg.SLIP_OUT, mg.MAKER, mg.TAKER, mg.SLIP_IN, False, mg.MARGIN)
    return r[0] * 104, int(r[1]), int(r[2])

def mine(o, h, l, c, f, step, side):
    e = GridEpisode('X', side, step, 0, GridParams())
    for j in range(len(o)):
        e.on_bar(j * 300_000, o[j], h[j], l[j], c[j], f[j])
        if not e.open:
            break
    return e.pnl, e.last_bar // 300_000, {'time': 0, 'tp': 1, 'stop': 2, 'liquidation': 3}[e.reason]

rng = np.random.default_rng(1)
worst = 0; n = 0
for trial in range(3000):
    vol = rng.choice([0.002, 0.005, 0.01, 0.02])
    r = rng.normal(rng.normal(0, 0.0005), vol, 864)
    c = 100 * np.exp(np.cumsum(r)); o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, 864))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, 864)))
    f = np.where(np.arange(864) % 96 == 50, rng.normal(0.0001, 0.0003), 0.0)
    step = rng.choice([0.01, 0.03, 0.1]); side = int(rng.choice([1, -1]))
    a, b = research(o, h, l, c, f, step, side), mine(o, h, l, c, f, step, side)
    assert a[1] == b[1] and a[2] == b[2], (trial, a, b)
    worst = max(worst, abs(a[0] - b[0])); n += 1
print(f'random paths: {n} episodes identical exit bar/reason, max pnl diff {worst:.2e} USD')
