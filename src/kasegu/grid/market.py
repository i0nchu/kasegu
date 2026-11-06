"""Public Binance USDS-M data the grid paper runner needs."""

import time
from dataclasses import replace

from kasegu.config import ExchangeConfig
from kasegu.core.types import Kline
from kasegu.exchange.binance import BinanceMarket, kline_from_rest


class GridMarket:
    def __init__(self, cfg: ExchangeConfig):
        self.feed = BinanceMarket(cfg)

    async def perpetuals(self) -> dict[str, int]:
        """USDT perpetuals that are trading, symbol -> onboard time (ms)."""
        info = await self.feed._rest(self.feed.client.rest_api.exchange_information)
        out = {}
        for s in info.get("symbols", []):
            if (
                s.get("contractType") == "PERPETUAL"
                and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
            ):
                out[s["symbol"]] = int(s.get("onboardDate") or 0)
        return out

    async def klines(
        self, symbol: str, interval: str, limit: int, start: int | None = None
    ) -> list[Kline]:
        """Closed klines, oldest first. Quote volume is put in Kline.volume."""
        rows = await self.feed._rest(
            self.feed.client.rest_api.kline_candlestick_data,
            symbol=symbol,
            interval=interval,
            start_time=start,
            limit=limit,
        )
        now = int(time.time() * 1000)
        out = []
        for r in rows:
            if int(r[6]) >= now:
                continue
            out.append(replace(kline_from_rest(symbol, r), volume=float(r[7])))
        return out

    async def funding(self, symbol: str, start: int, end: int) -> list[tuple[int, float]]:
        return await self.feed.funding_history(symbol, start, end)

    async def close(self):
        await self.feed.close()
