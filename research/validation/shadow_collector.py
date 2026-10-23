"""Public-market-data-only shadow collector for USD-M Futures.

The collector records raw snapshot, diff-depth, aggTrade, book-health and candidate
events as append-only JSONL.  It does not read API keys and has no order endpoint.
Run explicitly; validation tests never connect to Binance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp

from .maker import BookGap, DepthDelta, LocalBook

REST = "https://fapi.binance.com/fapi/v1/depth"
WSS = "wss://fstream.binance.com/public/stream?streams="


class JsonlRecorder:
    def __init__(self, path: Path, fsync_every: int = 100):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        self.fsync_every = fsync_every
        self.count = 0

    def write(self, kind: str, payload: dict) -> None:
        row = {"recorded_ns": time.time_ns(), "kind": kind, "payload": payload}
        self.file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % self.fsync_every == 0:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()


class ShadowCollector:
    def __init__(self, symbols: list[str], out: Path):
        self.symbols = [s.upper() for s in symbols]
        self.books = {s: LocalBook(s) for s in self.symbols}
        self.buffers: dict[str, list[DepthDelta]] = {s: [] for s in self.symbols}
        self.recorder = JsonlRecorder(out)

    def candidate(self, payload: dict) -> None:
        """Record the exact strategy order intent without submitting it anywhere."""
        required = {"candidate_id", "symbol", "side", "price", "qty", "created_ms", "cancel_ms"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"candidate missing fields: {sorted(missing)}")
        self.recorder.write("candidate", payload)

    async def _snapshot(self, session: aiohttp.ClientSession, symbol: str) -> None:
        async with session.get(REST, params={"symbol": symbol, "limit": 1000}) as res:
            res.raise_for_status()
            snap = await res.json()
        self.recorder.write("snapshot", {"symbol": symbol, **snap})
        book = self.books[symbol]
        book.install_snapshot(snap["lastUpdateId"], snap["bids"], snap["asks"], snap.get("E"))
        buffered = self.buffers[symbol]
        self.buffers[symbol] = []
        for delta in buffered:
            if delta.final_id < snap["lastUpdateId"]:
                continue
            try:
                book.apply(delta)
            except BookGap:
                self.recorder.write("book_unknown", {"symbol": symbol, "reason": "snapshot_bridge"})
                return await self._snapshot(session, symbol)

    async def _handle_depth(self, session: aiohttp.ClientSession, d: dict) -> None:
        symbol = d["s"]
        delta = DepthDelta(
            symbol,
            int(d["E"]),
            int(d["U"]),
            int(d["u"]),
            int(d["pu"]),
            tuple((float(p), float(q)) for p, q in d["b"]),
            tuple((float(p), float(q)) for p, q in d["a"]),
        )
        self.recorder.write("depth", d)
        book = self.books[symbol]
        if book.last_update_id is None:
            self.buffers[symbol].append(delta)
            return
        try:
            book.apply(delta)
        except BookGap as exc:
            self.recorder.write("book_unknown", {"symbol": symbol, "reason": str(exc)})
            book.synchronized = False
            self.buffers[symbol] = [delta]
            await self._snapshot(session, symbol)
            return
        if book.synchronized:
            bid, ask = book.best_bid, book.best_ask
            self.recorder.write(
                "bbo",
                {
                    "symbol": symbol,
                    "event_time": d["E"],
                    "bid": bid,
                    "bid_qty": book.bids.get(bid, 0),
                    "ask": ask,
                    "ask_qty": book.asks.get(ask, 0),
                    "spread_bps": (ask / bid - 1) * 10_000 if bid and ask else None,
                },
            )

    async def run(self, duration_s: int | None = None) -> None:
        streams = "/".join(
            x for s in self.symbols for x in (f"{s.lower()}@depth@100ms", f"{s.lower()}@aggTrade")
        )
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
        started = time.monotonic()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(WSS + streams, heartbeat=180) as ws:
                # Consume immediately so depth updates are buffered while REST
                # snapshots are in flight.
                consumer = asyncio.create_task(self._consume(session, ws, started, duration_s))
                snapshot_tasks = [self._snapshot(session, s) for s in self.symbols]
                await asyncio.gather(*snapshot_tasks)
                await consumer
        self.recorder.close()

    async def _consume(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        started: float,
        duration_s: int | None,
    ) -> None:
        async for msg in ws:
            if duration_s is not None and time.monotonic() - started >= duration_s:
                break
            if msg.type is not aiohttp.WSMsgType.TEXT:
                continue
            wrapped = json.loads(msg.data)
            d = wrapped.get("data", wrapped)
            if d.get("e") == "depthUpdate":
                await self._handle_depth(session, d)
            elif d.get("e") == "aggTrade":
                self.recorder.write("agg_trade", d)


def main() -> None:
    p = argparse.ArgumentParser(description="Record public shadow market data; never sends orders")
    p.add_argument("symbols", nargs="+", help="USD-M symbols, for example BTCUSDT")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration-s", type=int)
    args = p.parse_args()
    asyncio.run(ShadowCollector(args.symbols, args.out).run(args.duration_s))


if __name__ == "__main__":
    main()
