"""Structured events for alerts and outside consumers (Telegram, webhook, a future WebUI).

The engine publishes and never waits: sinks run in a background task and a slow or
failing sink cannot block trading.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

LEVELS = {"info": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True, slots=True)
class Event:
    kind: str  # e.g. trade_closed, stop_rejected, loss_limit
    level: str  # info | warning | critical
    message: str
    symbol: str | None = None
    ts: int = field(default_factory=lambda: int(time.time() * 1000))
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


Sink = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self, maxsize: int = 1000):
        self.sinks: list[tuple[Sink, int]] = []
        self.q: asyncio.Queue[Event] = asyncio.Queue(maxsize)
        self.task: asyncio.Task | None = None

    def add(self, sink: Sink, min_level: str = "info"):
        self.sinks.append((sink, LEVELS[min_level]))

    def publish(self, ev: Event):
        if not self.sinks:
            return
        try:
            self.q.put_nowait(ev)
        except asyncio.QueueFull:
            log.warning("event queue full, dropped %s", ev.kind)

    def start(self):
        if self.sinks and self.task is None:
            self.task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self):
        while True:
            ev = await self.q.get()
            await self._send(ev)

    async def _send(self, ev: Event):
        lvl = LEVELS.get(ev.level, 0)
        for sink, min_lvl in self.sinks:
            if lvl < min_lvl:
                continue
            try:
                await asyncio.wait_for(sink(ev), timeout=10)
            except Exception as e:
                log.warning("event sink failed for %s: %s", ev.kind, e)

    async def close(self, timeout: float = 5.0):
        """Flush what is queued, then stop."""
        if self.task is None:
            return
        self.task.cancel()
        self.task = None
        try:
            async with asyncio.timeout(timeout):
                while not self.q.empty():
                    await self._send(self.q.get_nowait())
        except TimeoutError:
            pass
