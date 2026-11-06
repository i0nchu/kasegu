"""mgrid-base: regime classification on BTC/ETH daily bars and long-only spot grid simulation on 1h bars."""
import numpy as np, pandas as pd

DATA = '/home/user/data/mgrid'
H = 3_600_000
SPLITS = {'dev': ('2020-07-01', '2023-07-01'), 'val': ('2023-07-01', '2025-01-01'), 'hold': ('2025-01-01', '2026-10-01')}
PER = {'dev': (('2020H2-21', '2020', '2022'), ('2022', '2022', '2023'), ('2023H1', '2023', '2024')),
       'val': (('2023H2', '2023', '2024'), ('2024', '2024', '2025')), 'hold': (('2025', '2025', '2026'), ('2026', '2026', '2027'))}

def load(sym):
    z = np.load(f'{DATA}/{sym}.npz')
    t = pd.to_datetime(z['t'].astype(np.int64) * 300, unit='s')
    d = pd.DataFrame({'o': z['o'], 'h': z['h'], 'l': z['l'], 'c': z['c']}, index=t).astype(float)
    h = d.resample('1h').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'})
    dd = d.resample('1D').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'})
    return h, dd

def panel_close(sym):
    z = np.load('/home/user/data/panel_1d_all.npz')
    j = list(z['names']).index(sym)
    return pd.Series(z['c'][:, j], index=pd.date_range('2020-01-01', periods=z['c'].shape[0], freq='D'))

def regimes(dc: pd.Series) -> pd.DataFrame:
    """state for day d (known at 00:00 of d) from closes up to d-1: 1 UP, 0 RANGE, -1 DOWN, nan unknown"""
    out = {}
    for name, (a, b) in (('MA50_200', (50, 200)), ('MA20_100', (20, 100))):
        sa, sb = dc.rolling(a).mean(), dc.rolling(b).mean()
        s = np.where((dc > sa) & (sa > sb), 1, np.where((dc < sa) & (sa < sb), -1, 0)).astype(float)
        s[sb.isna().to_numpy()] = np.nan
        out[name] = s
    r30, r90 = dc / dc.shift(30) - 1, dc / dc.shift(90) - 1
    s = np.where((r30 > 0) & (r90 > 0), 1, np.where((r30 < 0) & (r90 < 0), -1, 0)).astype(float)
    s[r90.isna().to_numpy()] = np.nan
    out['TSMOM'] = s
    return pd.DataFrame(out, index=dc.index).shift(1)

def sigma7(hc: pd.Series) -> pd.Series:
    r = np.log(hc).diff()
    return (r.rolling(168, min_periods=120).std() * np.sqrt(24)).shift(1)  # known at the start of the hour

MAKER = TAKER = 0.001
SLIP, PEN = 0.0005, 0.0005
W = np.array([1, 1.5, 2.25, 3.375]); W = W / W.sum()

def simulate(h: pd.DataFrame, state_d: pd.Series, sig_h: pd.Series, mode: str, cap=50.0, k=1.0, lo=0.01, hi=0.10):
    """Spot long-only sleeve. Returns hourly equity series and a trade log.
    mode: B0 buy&hold, B1 trend hold, G1 grid accumulate, G2 grid with per-level take profit."""
    o, hh, ll, c = (h[x].to_numpy() for x in 'ohlc')
    idx = h.index
    st = state_d.reindex(idx.normalize()).to_numpy()
    sg = sig_h.to_numpy()
    cash, qty = cap, 0.0
    lots = []  # G2: list of (qty, buy price, level)
    orders = []  # resting buys: (price, usdt, level)
    eq = np.empty(len(idx)); trades = 0
    def mbuy(usdt, px):
        nonlocal cash, qty, trades
        if usdt < 1: return 0
        p = px * (1 + SLIP); q = usdt * (1 - TAKER) / p; cash -= usdt; qty += q; trades += 1; return q
    def msell_all(px):
        nonlocal cash, qty, trades, lots
        if qty <= 0: return
        cash += qty * px * (1 - SLIP) * (1 - TAKER); qty = 0.0; lots = []; trades += 1
    first = True
    for i in range(len(idx)):
        if np.isnan(c[i]):
            eq[i] = eq[i - 1] if i else cap; continue
        newday = idx[i].hour == 0 or first
        s = st[i]
        if mode == 'B0':
            if first: mbuy(cash, o[i])
        elif newday:
            orders = []
            if s != s:  # unknown: stay as is
                pass
            elif s == -1:
                msell_all(o[i])
            elif s == 1 or mode == 'B1':
                mbuy(cash, o[i])
            else:  # RANGE, grid modes: ladder below today's open
                step = k * min(max(sg[i], lo), hi) if sg[i] == sg[i] else np.nan
                if step == step and cash > 1:
                    base = cap  # level sizes from the sleeve capital, limited by cash
                    for j in range(4):
                        orders.append([o[i] * (1 - (j + 1) * step), base * W[j], j, step])
        first = False
        # adverse first: resting buys
        filled_now = []
        if orders:
            rest = []
            for px, usdt, j, step in orders:
                if ll[i] <= px * (1 - PEN) and cash > 1:
                    usdt = min(usdt, cash); q = usdt * (1 - MAKER) / px
                    cash -= usdt; qty += q; trades += 1
                    if mode == 'G2': filled_now.append((q, px, j, step))
                else:
                    rest.append([px, usdt, j, step])
            orders = rest
        # G2 take profit on lots bought in earlier bars, only in RANGE
        if mode == 'G2' and lots and s == 0:
            keep = []
            for q, px, j, step in lots:
                tp = px * (1 + step)
                if hh[i] >= tp * (1 + PEN):
                    cash += q * tp * (1 - MAKER); qty -= q; trades += 1
                    orders.append([px, q * px, j, step])  # re-arm the level
                else:
                    keep.append((q, px, j, step))
            lots = keep
        if mode == 'G2': lots += filled_now
        eq[i] = cash + qty * c[i]
    return pd.Series(eq, index=idx), trades

def stats(eq: pd.Series, cap: float):
    d = eq.resample('1D').last().dropna()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    ret = d.iloc[-1] / cap - 1
    cagr = (d.iloc[-1] / cap) ** (1 / yrs) - 1
    dd = (d / d.cummax() - 1).min()
    return dict(final=d.iloc[-1], ret=ret, cagr=cagr, maxDD=dd, calmar=cagr / -dd if dd < 0 else np.nan)
