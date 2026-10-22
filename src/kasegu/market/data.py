from collections import deque
from dataclasses import dataclass

import numpy as np

from kasegu.core.types import Kline, MarkPrice, Quote
from kasegu.strategies.base import MarketState

MINUTE = 60_000


@dataclass(slots=True)
class SymbolData:
    bars: deque[Kline]
    quote: Quote | None = None
    mark: MarkPrice | None = None


class MarketData:
    """Closed 1m bars plus latest quote and mark price per symbol."""

    def __init__(self, symbols: list[str], maxlen: int = 1500):
        self.data = {s: SymbolData(deque(maxlen=maxlen)) for s in symbols}

    def __getitem__(self, symbol: str) -> SymbolData:
        return self.data[symbol]

    def last_open(self, symbol: str) -> int | None:
        bars = self.data[symbol].bars
        return bars[-1].open_time if bars else None

    def load(self, bars: list[Kline]):
        for k in bars:
            self.add(k)

    def add(self, k: Kline) -> str:
        """Add a closed bar. Returns 'ok', 'dup', 'gap' or 'open'."""
        if not k.closed:
            return "open"
        bars = self.data[k.symbol].bars
        if bars and k.open_time <= bars[-1].open_time:
            return "dup"
        gap = bool(bars) and k.open_time > bars[-1].open_time + MINUTE
        if gap:
            return "gap"
        bars.append(k)
        return "ok"

    def on_quote(self, q: Quote):
        self.data[q.symbol].quote = q

    def on_mark(self, m: MarkPrice):
        self.data[m.symbol].mark = m

    def stale(self, symbol: str, now: int, max_age_ms: float) -> bool:
        d = self.data[symbol]
        if d.quote is None or now - d.quote.time > max_age_ms:
            return True
        return not d.bars or now - d.bars[-1].close_time > MINUTE + max_age_ms

    def state(self, symbol: str) -> MarketState | None:
        d = self.data[symbol]
        if not d.bars or d.quote is None:
            return None
        bars = d.bars
        return MarketState(
            symbol=symbol,
            time=bars[-1].close_time,
            open=np.fromiter((b.open for b in bars), float, len(bars)),
            high=np.fromiter((b.high for b in bars), float, len(bars)),
            low=np.fromiter((b.low for b in bars), float, len(bars)),
            close=np.fromiter((b.close for b in bars), float, len(bars)),
            bid=d.quote.bid,
            ask=d.quote.ask,
        )
