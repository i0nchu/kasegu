from kasegu.exchange.binance import (
    kline_from_rest,
    kline_from_ws,
    mark_from_ws,
    plain,
    quote_from_rest,
    quote_from_ws,
)


def test_kline_rest():
    row = [0, "1", "2", "0.5", "1.5", "10", 59_999, "0", 1, "0", "0", "0"]
    k = kline_from_rest("BTCUSDT", row)
    assert (k.open, k.high, k.low, k.close, k.close_time) == (1, 2, 0.5, 1.5, 59_999)


def test_kline_ws():
    d = {
        "e": "kline",
        "k": {
            "t": 0,
            "T": 59_999,
            "s": "BTCUSDT",
            "o": "1",
            "c": "2",
            "h": "3",
            "l": "0.5",
            "v": "9",
            "x": False,
        },
    }
    k = kline_from_ws(d)
    assert k.symbol == "BTCUSDT" and not k.closed and k.high == 3


def test_quote_and_mark():
    q = quote_from_ws({"s": "BTCUSDT", "b": "99", "B": "1", "a": "101", "A": "2", "T": 5})
    assert q.mid == 100 and q.time == 5
    q = quote_from_rest(
        {"symbol": "X", "bidPrice": "1", "bidQty": "2", "askPrice": "3", "askQty": "4", "time": 7}
    )
    assert q.ask == 3
    m = mark_from_ws({"s": "BTCUSDT", "p": "100.5", "r": "0.0001", "T": 10, "E": 3})
    assert m.price == 100.5 and m.funding_rate == 0.0001 and m.next_funding_time == 10


def test_plain():
    class M:
        def to_dict(self):
            return {"a": 1}

    assert plain([M(), {"b": M()}]) == [{"a": 1}, {"b": {"a": 1}}]


class FakeHandle:
    def __init__(self, ws, key):
        self.ws, self.key = ws, key

    def on(self, event, cb):
        self.ws.cbs[self.key] = cb


class FakeWS:
    def __init__(self, cfg):
        self.cbs, self.close_cbs, self.url_paths = {}, [], None

    async def create_connection(self):
        self.connections = [type("C", (), {"url_path": "public"})()]
        return self

    def on_connection(self, event, cb, connection=None):
        self.close_cbs.append(cb)

    async def kline_candlestick_streams(self, symbol, interval):
        return FakeHandle(self, f"{symbol}@kline")

    async def individual_symbol_book_ticker_streams(self, symbol):
        return FakeHandle(self, f"{symbol}@bookTicker")

    async def mark_price_stream(self, symbol, update_speed):
        return FakeHandle(self, f"{symbol}@markPrice")

    async def close_connection(self, close_session=True):
        for cb in self.close_cbs:
            cb()


async def test_stream_routing_and_stale(monkeypatch):
    import kasegu.exchange.binance as bn
    from kasegu.config import ExchangeConfig, Secrets

    monkeypatch.setattr(bn, "DerivativesTradingUsdsFuturesWebSocketStreams", FakeWS)
    m = bn.BinanceMarket(ExchangeConfig(heartbeat_s=10), Secrets(api_key=None, api_secret=None))
    got = []
    await m.stream(["BTCUSDT"], got.append, got.append, got.append)
    ws = m.ws
    assert ws.url_paths == ["market", "public"]
    ws.cbs["btcusdt@bookTicker"]({"s": "BTCUSDT", "b": "1", "B": "1", "a": "2", "A": "1", "T": 1})
    assert got[-1].ask == 2
    assert m.pop_lag()["btcusdt@bookTicker"] > 0
    assert m.pop_lag() == {}
    assert m.stale() == []

    m.last["btcusdt@markPrice"] -= 11
    assert m.stale() == ["btcusdt@markPrice"]

    # our own close on reconnect does not flag the new connection
    await m.close()
    await m.stream(["BTCUSDT"], got.append, got.append, got.append)
    assert not m.closed
    for cb in ws.close_cbs:
        cb()
    assert not m.closed
    for cb in m.ws.close_cbs:
        cb()
    assert m.stale()[0].startswith("public connection closed after")
