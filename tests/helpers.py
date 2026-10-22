import numpy as np

from kasegu.strategies.base import MarketState


def make_state(close, spread=0.0, bid=None, ask=None, hl=0.5, symbol="BTCUSDT"):
    close = np.asarray(close, dtype=float)
    high = close + hl
    low = close - hl
    open_ = np.r_[close[0], close[:-1]]
    last = close[-1]
    return MarketState(
        symbol=symbol,
        time=len(close) * 60_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        bid=bid if bid is not None else last - spread / 2,
        ask=ask if ask is not None else last + spread / 2,
    )
