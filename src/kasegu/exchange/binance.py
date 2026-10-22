import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from binance_common.configuration import ConfigurationRestAPI, ConfigurationWebSocketStreams
from binance_sdk_derivatives_trading_usds_futures import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL as REST_PROD,
)
from binance_sdk_derivatives_trading_usds_futures import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL as REST_TEST,
)
from binance_sdk_derivatives_trading_usds_futures import (
    DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL as WS_PROD,
)
from binance_sdk_derivatives_trading_usds_futures import (
    DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL as WS_TEST,
)
from binance_sdk_derivatives_trading_usds_futures import DerivativesTradingUsdsFutures
from binance_sdk_derivatives_trading_usds_futures.websocket_streams import (
    DerivativesTradingUsdsFuturesWebSocketStreams,
)

from kasegu.config import ExchangeConfig, Secrets
from kasegu.core.types import Kline, MarkPrice, Quote
from kasegu.exchange.rules import SymbolRules, parse_rules

log = logging.getLogger(__name__)

MINUTE = 60_000


def plain(x: Any) -> Any:
    """Convert SDK response models to plain python objects."""
    if isinstance(x, list):
        return [plain(i) for i in x]
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    inst = getattr(x, "actual_instance", None)
    if inst is not None:
        return plain(inst)
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return x


# raw payload converters (Binance field names)


def kline_from_rest(symbol: str, row: list) -> Kline:
    return Kline(
        symbol=symbol,
        open_time=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time=int(row[6]),
    )


def kline_from_ws(d: dict) -> Kline:
    k = d["k"]
    return Kline(
        symbol=k["s"],
        open_time=int(k["t"]),
        close_time=int(k["T"]),
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        closed=bool(k["x"]),
    )


def quote_from_ws(d: dict) -> Quote:
    return Quote(
        symbol=d["s"],
        bid=float(d["b"]),
        ask=float(d["a"]),
        bid_qty=float(d["B"]),
        ask_qty=float(d["A"]),
        time=int(d.get("T") or d.get("E") or time.time() * 1000),
    )


def quote_from_rest(d: dict) -> Quote:
    return Quote(
        symbol=d["symbol"],
        bid=float(d["bidPrice"]),
        ask=float(d["askPrice"]),
        bid_qty=float(d["bidQty"]),
        ask_qty=float(d["askQty"]),
        time=int(d.get("time") or time.time() * 1000),
    )


def mark_from_ws(d: dict) -> MarkPrice:
    return MarkPrice(
        symbol=d["s"],
        price=float(d["p"]),
        funding_rate=float(d.get("r") or 0),
        next_funding_time=int(d.get("T") or 0),
        time=int(d["E"]),
    )


class BinanceMarket:
    """Market data via REST (sync SDK in a thread) and WebSocket streams."""

    def __init__(self, cfg: ExchangeConfig, secrets: Secrets | None = None):
        self.cfg = cfg
        secrets = secrets or Secrets()
        key = secrets.api_key.get_secret_value() if secrets.api_key else None
        sec = secrets.api_secret.get_secret_value() if secrets.api_secret else None
        test = cfg.data_testnet
        self.client = DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=key,
                api_secret=sec,
                base_path=REST_TEST if test else REST_PROD,
                timeout=cfg.rest_timeout_ms,
            ),
        )
        self.ws_url = WS_TEST if test else WS_PROD
        self.has_key = bool(key and sec)
        self.ws = None
        self.last: dict[str, float] = {}  # stream -> last message (monotonic)
        self.lag: dict[str, float] = {}  # stream -> max event delay since last report (s)
        self.closed = ""

    async def _rest(self, fn: Callable, *args, **kw) -> Any:
        resp = await asyncio.to_thread(fn, *args, **kw)
        return plain(resp.data())

    async def klines(self, symbol: str, limit: int = 1000, start: int | None = None) -> list[Kline]:
        """Closed 1m klines, oldest first."""
        rows = await self._rest(
            self.client.rest_api.kline_candlestick_data,
            symbol=symbol,
            interval="1m",
            start_time=start,
            limit=limit,
        )
        now = int(time.time() * 1000)
        return [kline_from_rest(symbol, r) for r in rows if int(r[6]) < now]

    async def klines_since(self, symbol: str, start: int) -> list[Kline]:
        out: list[Kline] = []
        while True:
            batch = await self.klines(symbol, 1500, start)
            out += batch
            if len(batch) < 1500:
                return out
            start = batch[-1].open_time + MINUTE

    async def book_ticker(self, symbol: str) -> Quote:
        d = await self._rest(self.client.rest_api.symbol_order_book_ticker, symbol=symbol)
        return quote_from_rest(d)

    async def rules(self, symbols: list[str]) -> dict[str, SymbolRules]:
        info = await self._rest(self.client.rest_api.exchange_information)
        r = parse_rules(info)
        missing = set(symbols) - set(r)
        if missing:
            raise ValueError(f"unknown symbols: {sorted(missing)}")
        return {s: r[s] for s in symbols}

    async def commission(self, symbol: str) -> tuple[float, float] | None:
        """(maker, taker) rate of the account, needs API key."""
        if not self.has_key:
            return None
        d = await self._rest(self.client.rest_api.user_commission_rate, symbol=symbol)
        return float(d["makerCommissionRate"]), float(d["takerCommissionRate"])

    async def stream(
        self,
        symbols: list[str],
        on_kline: Callable[[Kline], None],
        on_quote: Callable[[Quote], None],
        on_mark: Callable[[MarkPrice], None],
    ):
        """Subscribe kline_1m, bookTicker and markPrice@1s for all symbols."""
        # fresh client per connect, the SDK keeps stale subscriptions after close
        ws = DerivativesTradingUsdsFuturesWebSocketStreams(
            ConfigurationWebSocketStreams(stream_url=self.ws_url)
        )
        ws.url_paths = ["market", "public"]  # private path is unused without a listen key
        self.ws = await ws.create_connection()
        if self.ws is None:
            raise ConnectionError("websocket connect failed")
        born = time.monotonic()
        for c in list(self.ws.connections):
            self.ws.on_connection(
                "close", lambda c=c: self._on_close(ws, c.url_path, born), connection=c
            )
        self.closed = ""
        self.last = {}
        self.lag = {}
        for s in symbols:
            sym = s.lower()
            subs = [
                (
                    "kline",
                    self.ws.kline_candlestick_streams(symbol=sym, interval="1m"),
                    kline_from_ws,
                    on_kline,
                ),
                (
                    "bookTicker",
                    self.ws.individual_symbol_book_ticker_streams(symbol=sym),
                    quote_from_ws,
                    on_quote,
                ),
                (
                    "markPrice",
                    self.ws.mark_price_stream(symbol=sym, update_speed="1s"),
                    mark_from_ws,
                    on_mark,
                ),
            ]
            for name, sub, conv, cb in subs:
                key = f"{sym}@{name}"
                (await sub).on("message", self._wrap(key, conv, cb))
                self.last[key] = time.monotonic()

    def _on_close(self, ws, path, born):
        # ignore close events of a replaced connection
        if ws is self.ws and not self.closed:
            self.closed = f"{path} connection closed after {time.monotonic() - born:.0f}s"

    def _wrap(self, key, conv, cb):
        def handle(msg):
            self.last[key] = time.monotonic()
            try:
                ev = conv(plain(msg))
                if isinstance(ev, Quote | MarkPrice):
                    lag = time.time() - ev.time / 1000
                    self.lag[key] = max(self.lag.get(key, 0.0), lag)
                cb(ev)
            except Exception:
                log.exception("stream message failed")

        return handle

    def pop_lag(self) -> dict[str, float]:
        """Max receive delay per stream since the previous call."""
        lag, self.lag = self.lag, {}
        return lag

    def stale(self) -> list[str]:
        """Closed connection or streams silent longer than the heartbeat."""
        if self.closed:
            return [self.closed]
        now = time.monotonic()
        return [k for k, t in self.last.items() if now - t > self.cfg.heartbeat_s]

    async def close(self):
        if self.ws is not None:
            ws, self.ws = self.ws, None
            try:
                await ws.close_connection(close_session=True)
            except Exception:
                log.exception("close failed")
