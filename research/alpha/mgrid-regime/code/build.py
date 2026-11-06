"""Build the coin-day table for one split: factors, targets, grid episodes. Usage: python build.py dev"""
import sys, concurrent.futures as cf
import pandas as pd, mg

KW = {}
if len(sys.argv) > 2 and sys.argv[2] == 'stress':
    KW = dict(pen=0.002, maker=0.0004, taker=0.001, slip_in=0.0004, slip_out=0.001)
if len(sys.argv) > 2 and sys.argv[2] == 'tpsame':
    KW = dict(tp_same_bar=True)
TAG = '' if len(sys.argv) < 3 or sys.argv[2] == 'base' else '_' + sys.argv[2]
MARGINS = (("", mg.MARGIN), ("N", 25 / 104), ("K2", 50 / 104), ("K1", 100 / 104))

def one(sym):
    try:
        return mg.coin_table(sym, SPLIT, margins=MARGINS, **KW)
    except Exception as e:
        print('ERR', sym, e); return None

SPLIT = sys.argv[1]
if __name__ == '__main__':
    btc = mg.features(mg.hourly(mg.load('BTCUSDT')))[['ER3', 'ER14']].rename(columns={'ER3': 'BTCER3', 'ER14': 'BTCER14'})
    with cf.ProcessPoolExecutor(4) as ex:
        parts = [p for p in ex.map(one, mg.coins()) if p is not None and len(p)]
    T = pd.concat(parts, ignore_index=True)
    T['hr'] = T.t5 // 12
    T = T.merge(btc, left_on='hr', right_index=True, how='left')
    T['date'] = pd.to_datetime(T.t5 * 300, unit='s')
    T.to_parquet(f'/home/user/data/mgrid_{SPLIT}{TAG}.parquet')
    print(T.shape, T.sym.nunique(), T.date.min(), T.date.max())
