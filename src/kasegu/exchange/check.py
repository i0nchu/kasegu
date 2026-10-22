"""Testnet check of the live plumbing and of the auto-cancel countdown.

1. Places a far post-only entry and a far reduce-only stop, starts a short countdown
   and reports which of them the exchange cancelled.
2. Places another stop, arms the countdown and disarms it the way kasegu does before
   protecting a position; the stop must survive.

Usage: uv run python -m kasegu.exchange.check [--symbol BTCUSDT] [--config configs/config.yaml]
"""

import argparse
import asyncio
import logging
import sys

from kasegu.config import Secrets, load_config
from kasegu.core.execution import new_id
from kasegu.core.types import Order, OrderType, Purpose, Side, TimeInForce
from kasegu.exchange.binance import BinanceMarket
from kasegu.exchange.trade import LiveBroker

log = logging.getLogger("kasegu.check")


async def check(cfg_path: str, symbol: str, wait_s: int) -> int:
    cfg = load_config(cfg_path)
    ex = cfg.exchange
    if not ex.testnet:
        log.error("this check only runs on testnet (exchange.testnet: true)")
        return 2
    ex.data_testnet = True
    feed = BinanceMarket(ex)
    broker = LiveBroker(ex, cfg.fee, Secrets())
    events: list[str] = []
    broker.listener = lambda o, f: events.append(
        f"{o.client_id[:3]} {o.status}" + (f" fill {f.qty}@{f.price}" if f else "")
    )

    r = (await feed.rules([symbol]))[symbol]
    q = await feed.book_ticker(symbol)
    await broker.prepare([symbol], cfg.risk.leverage)
    await broker.connect()
    if await broker.position_qty(symbol):
        log.error("%s has an open position on testnet, close it first", symbol)
        return 2

    px = r.price_down(q.bid * 0.97)  # inside PERCENT_PRICE, far from the touch
    qty = r.qty(r.min_notional * 1.5 / px + r.step)
    entry = Order(
        new_id(Purpose.ENTRY),
        symbol,
        Side.BUY,
        OrderType.LIMIT,
        qty,
        Purpose.ENTRY,
        price=px,
        tif=TimeInForce.GTX,
    )

    def make_stop():
        return Order(
            new_id(Purpose.STOP),
            symbol,
            Side.SELL,
            OrderType.STOP_MARKET,
            qty,
            Purpose.STOP,
            stop_price=r.price_down(q.bid * 0.9),
            reduce_only=True,
        )

    stop, stop2 = make_stop(), make_stop()
    ok = True
    try:
        e = await broker.place(entry)
        s = await broker.place(stop)
        print(f"entry {e.status} {e.reason}  stop {s.status} {s.reason}")
        if not (e.is_open and s.is_open):
            print("order placement failed, see reasons above")
            return 1
        await broker.arm(symbol, 5)
        print(f"[1] countdown 5s started, waiting {wait_s}s ...")
        await asyncio.sleep(wait_s)
        e = await broker.query(symbol, entry.client_id)
        s = await broker.query(symbol, stop.client_id)
        print(f"[1] after countdown: entry {e.status}  stop {s.status}")
        if e.is_open:
            print("[1] countdown did not cancel the entry: auto-cancel is not working")
            ok = False
        else:
            print("[1] countdown cancels entries" + ("" if s.is_open else " and stops too"))

        s = await broker.place(stop2)
        await broker.arm(symbol, 5)
        await broker.before_protect(symbol)
        print(f"[2] stop {s.status}, countdown armed then disarmed, waiting {wait_s}s ...")
        await asyncio.sleep(wait_s)
        s = await broker.query(symbol, stop2.client_id)
        print(f"[2] after disarm: stop {s.status}")
        if not s.is_open:
            print("[2] disarm did not protect the stop: keep live.auto_cancel_s = 0")
            ok = False
        elif ok:
            print("OK: live.auto_cancel_s can be enabled (armed only while flat)")
    finally:
        await broker.auto_cancel(symbol, 0)
        for o in (entry, stop, stop2):
            try:
                await broker.cancel(symbol, o.client_id)
            except Exception:
                log.exception("cleanup %s failed", o.client_id)
        await asyncio.sleep(1)
        print("order events:", "; ".join(events) or "none")
        await broker.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kasegu.exchange.check")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--wait", type=int, default=12)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(check(args.config, args.symbol.upper(), args.wait))


if __name__ == "__main__":
    sys.exit(main())
