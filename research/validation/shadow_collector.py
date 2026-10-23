"""Record Binance public depth/aggTrade data; no keys or order endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp

from research.validation.orderbook import BookGap, DepthDelta, LocalBook

REST = "https://fapi.binance.com/fapi/v1/depth"
WSS = "wss://fstream.binance.com/public/stream?streams="


class JsonlRecorder:
    def __init__(self, path: Path, fsync_every: int = 100):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8")
        self.fsync_every = fsync_every
        self.count = 0

    def write(self, kind: str, payload: dict) -> None:
        row = {"recorded_ns": time.time_ns(), "kind": kind, "payload": payload}
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % self.fsync_every == 0:
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


class ShadowCollector:
    def __init__(self, symbols: list[str], out: Path):
        self.symbols = [symbol.upper() for symbol in symbols]
        self.books = {symbol: LocalBook(symbol) for symbol in self.symbols}
        self.buffers: dict[str, list[DepthDelta]] = {symbol: [] for symbol in self.symbols}
        self.recorder = JsonlRecorder(out)

    async def snapshot(self, session: aiohttp.ClientSession, symbol: str) -> None:
        async with session.get(REST, params={"symbol": symbol, "limit": 1000}) as response:
            response.raise_for_status()
            snap = await response.json()
        self.recorder.write("snapshot", {"symbol": symbol, **snap})
        book = self.books[symbol]
        book.install_snapshot(snap["lastUpdateId"], snap["bids"], snap["asks"])
        buffered, self.buffers[symbol] = self.buffers[symbol], []
        for delta in buffered:
            if delta.final_id < snap["lastUpdateId"]:
                continue
            try:
                book.apply(delta)
            except BookGap:
                self.recorder.write("book_unknown", {"symbol": symbol, "reason": "bridge"})
                await self.snapshot(session, symbol)
                return

    async def depth(self, session: aiohttp.ClientSession, data: dict) -> None:
        symbol = data["s"]
        delta = DepthDelta(
            symbol,
            int(data["E"]),
            int(data["U"]),
            int(data["u"]),
            int(data["pu"]),
            tuple((float(p), float(q)) for p, q in data["b"]),
            tuple((float(p), float(q)) for p, q in data["a"]),
        )
        self.recorder.write("depth", data)
        book = self.books[symbol]
        if book.last_update_id is None:
            self.buffers[symbol].append(delta)
            return
        try:
            book.apply(delta)
        except BookGap as error:
            self.recorder.write("book_unknown", {"symbol": symbol, "reason": str(error)})
            self.buffers[symbol] = [delta]
            await self.snapshot(session, symbol)

    async def consume(self, session, websocket, started: float, duration_s: int | None) -> None:
        async for message in websocket:
            if duration_s is not None and time.monotonic() - started >= duration_s:
                break
            if message.type is not aiohttp.WSMsgType.TEXT:
                continue
            wrapped = json.loads(message.data)
            data = wrapped.get("data", wrapped)
            if data.get("e") == "depthUpdate":
                await self.depth(session, data)
            elif data.get("e") == "aggTrade":
                self.recorder.write("agg_trade", data)

    async def run(self, duration_s: int | None = None) -> None:
        streams = "/".join(
            stream
            for symbol in self.symbols
            for stream in (
                f"{symbol.lower()}@depth@100ms",
                f"{symbol.lower()}@aggTrade",
            )
        )
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
        started = time.monotonic()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(WSS + streams, heartbeat=180) as websocket:
                    consumer = asyncio.create_task(
                        self.consume(session, websocket, started, duration_s)
                    )
                    await asyncio.gather(*(self.snapshot(session, s) for s in self.symbols))
                    await consumer
        finally:
            self.recorder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-s", type=int)
    args = parser.parse_args()
    asyncio.run(ShadowCollector(args.symbols, args.out).run(args.duration_s))


if __name__ == "__main__":
    main()
