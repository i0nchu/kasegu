import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime

from kasegu.backtest.data import Archive
from kasegu.backtest.replay import Replay
from kasegu.backtest.report import run_name
from kasegu.config import AppConfig, Secrets
from kasegu.core.engine import Engine
from kasegu.core.execution import PaperBroker
from kasegu.exchange.binance import BinanceMarket
from kasegu.exchange.rules import SymbolRules
from kasegu.exchange.trade import LiveBroker
from kasegu.market.data import MINUTE, MarketData
from kasegu.storage.journal import Journal
from kasegu.strategies import create

log = logging.getLogger(__name__)


async def load_market(feed: BinanceMarket, cfg: AppConfig, md: MarketData):
    for s in cfg.symbols:
        md.load(await feed.klines(s, cfg.exchange.history_bars))
        md.on_quote(await feed.book_ticker(s))


async def resync(feed: BinanceMarket, eng: Engine):
    """Fill missing bars and refresh quotes after a reconnect.

    Paper orders live in the local broker, so open entries are kept.
    """
    for s in eng.cfg.symbols:
        last = eng.market.last_open(s)
        if last is not None:
            for k in await feed.klines_since(s, last + MINUTE):
                eng.market.add(k)
        await eng.on_quote(await feed.book_ticker(s))


async def run_paper(cfg: AppConfig):
    feed = BinanceMarket(cfg.exchange)
    rules = await feed.rules(cfg.symbols)
    if cfg.fee.use_account_rate:
        rate = await feed.commission(cfg.symbols[0])
        if rate:
            cfg.fee.maker, cfg.fee.taker = rate
    log.info("fees maker=%s taker=%s", cfg.fee.maker, cfg.fee.taker)

    md = MarketData(cfg.symbols, maxlen=max(cfg.exchange.history_bars, 1500))
    await load_market(feed, cfg, md)

    journal = Journal(cfg.db_path, run_id=f"paper-{uuid.uuid4().hex[:8]}")
    journal.event("start", mode="paper", config=cfg.model_dump(mode="json"))
    broker = PaperBroker(cfg.fee)
    for s in cfg.symbols:
        broker.on_quote(md[s].quote)
    eng = Engine(cfg, create(cfg.strategy), broker, md, rules, journal, backfill=feed.klines_since)
    # paper positions do not survive a restart, sync resets local state
    eng.restore()
    await eng.sync()
    try:
        await serve(cfg, feed, eng, journal)
    finally:
        await feed.close()
        journal.event("stop")
        journal.close()


async def run_live(cfg: AppConfig):
    ex = cfg.exchange
    venue = "testnet" if ex.testnet else "mainnet"
    if ex.data_testnet != ex.testnet:
        # strategy prices must come from the book the orders go to
        log.info("live on %s: market data from %s too", venue, venue)
        ex.data_testnet = ex.testnet
    feed = BinanceMarket(ex)
    broker = LiveBroker(ex, cfg.fee, Secrets())
    rules = await feed.rules(cfg.symbols)
    await broker.prepare(cfg.symbols, cfg.risk.leverage)
    if cfg.fee.use_account_rate:
        cfg.fee.maker, cfg.fee.taker = await broker.commission(cfg.symbols[0])
    log.info("live on %s, fees maker=%s taker=%s", venue, cfg.fee.maker, cfg.fee.taker)

    md = MarketData(cfg.symbols, maxlen=max(ex.history_bars, 1500))
    await load_market(feed, cfg, md)

    # own state db per venue so paper or testnet state never leaks into mainnet
    db = cfg.db_path.with_name(f"{cfg.db_path.stem}-{venue}{cfg.db_path.suffix}")
    journal = Journal(db, run_id=f"live-{venue}-{uuid.uuid4().hex[:8]}")
    journal.event("start", mode="live", venue=venue, config=cfg.model_dump(mode="json"))
    eng = Engine(cfg, create(cfg.strategy), broker, md, rules, journal, backfill=feed.klines_since)
    eng.restore()
    # stream first so no fill between sync and subscribe is missed
    await broker.connect()
    await eng.sync()
    try:
        await serve(cfg, feed, eng, journal, broker)
    finally:
        await feed.close()
        await broker.close()
        journal.event("stop")
        journal.close()


async def check_positions(eng: Engine, broker: LiveBroker):
    """Exchange position wins: block a symbol whose local size disagrees."""
    for s in eng.cfg.symbols:
        pos, qty = eng.positions[s], await broker.position_qty(s)
        local = pos.side.sign * pos.qty if pos.has_qty else 0.0
        if abs(qty - local) > 1e-9:
            eng.blocked.add(s)
            eng.journal.event("position_mismatch", s, exchange=qty, local=local)
            log.error("%s exchange position %s != local %s, entries blocked", s, qty, local)


async def serve(
    cfg: AppConfig,
    feed: BinanceMarket,
    eng: Engine,
    journal: Journal,
    broker: LiveBroker | None = None,
):
    """Main event loop shared by paper and live."""
    q: asyncio.Queue = asyncio.Queue()

    async def subscribe():
        await feed.stream(
            cfg.symbols,
            on_kline=lambda k: q.put_nowait(("bar", k)),
            on_quote=lambda x: q.put_nowait(("quote", x)),
            on_mark=lambda m: q.put_nowait(("mark", m)),
        )

    async def drain(_):
        await eng.drain()

    await subscribe()
    handlers = {"bar": eng.on_bar, "quote": eng.on_quote, "mark": eng.on_mark, "drain": drain}
    if broker is not None:
        broker.wake = lambda: q.put_nowait(("drain", None))
    live = cfg.live
    due = {"keepalive": time.monotonic() + live.keepalive_min * 60, "countdown": 0.0}

    while True:
        try:
            kind, ev = await asyncio.wait_for(q.get(), timeout=1.0)
            if kind == "bar" and not ev.closed:
                continue
            await handlers[kind](ev)
            if kind == "bar" and ev.symbol == cfg.symbols[-1]:
                lag = feed.pop_lag()
                if lag:
                    worst = max(lag, key=lag.get)
                    log.info("max stream delay %.2fs (%s)", lag[worst], worst)
        except TimeoutError:
            pass

        now = time.monotonic()
        if broker is not None:
            try:
                if now >= due["keepalive"]:
                    await broker.keepalive()
                    due["keepalive"] = now + live.keepalive_min * 60
                if live.auto_cancel_s and now >= due["countdown"]:
                    for s in cfg.symbols:
                        await broker.auto_cancel(s, live.auto_cancel_s)
                    due["countdown"] = now + live.auto_cancel_s / 3
            except Exception:
                log.exception("live timer failed")

        bad = feed.stale() + (broker.stale() if broker else [])
        if not bad:
            continue
        log.warning("stream stale (%s), reconnecting", ", ".join(bad))
        eng.halted = True
        journal.event("reconnect", reason=bad)
        try:
            await feed.close()
            await subscribe()
            if broker is not None:
                await broker.close()
                await broker.connect()
            await resync(feed, eng)
            if broker is not None:
                await broker.reconcile([o for o in eng.exec.orders.values() if o.is_open])
                await eng.drain()
                await check_positions(eng, broker)
            eng.halted = False
        except Exception:
            log.exception("reconnect failed")
            await asyncio.sleep(5)


async def load_rules(cfg: AppConfig) -> dict[str, SymbolRules]:
    """Exchange rules, cached so a backtest can be repeated offline."""
    path = cfg.backtest.cache_dir / "rules.json"
    cached = json.loads(path.read_text()) if path.exists() else {}
    if not set(cfg.symbols) <= set(cached):
        rules = await BinanceMarket(cfg.exchange).rules(cfg.symbols)
        cached.update({s: asdict(r) for s, r in rules.items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=2))
    return {s: SymbolRules(**cached[s]) for s in cfg.symbols}


async def run_backtest(cfg: AppConfig, label: str = "") -> dict:
    bt = cfg.backtest
    if bt.start is None or bt.end is None or bt.end < bt.start:
        raise ValueError("backtest needs start <= end")
    rules = await load_rules(cfg)
    feed = BinanceMarket(cfg.exchange)
    replay = Replay(
        cfg, rules, Archive(bt.cache_dir), bt.start, bt.end, funding_api=feed.funding_history
    )
    t0 = time.monotonic()
    result = await replay.run()
    start, end = bt.start.isoformat(), bt.end.isoformat()
    st = cfg.strategy
    name = run_name(cfg.symbols, start, end, st.id, st.k, label, params=st.params)
    out = bt.out_dir / f"{name}.json"
    report = {
        "name": name,
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "source": "aggTrades",
        "strategy": cfg.strategy.model_dump(mode="json"),
        "symbols": cfg.symbols,
        "start": start,
        "end": end,
        "file": str(out),
        "result": result,
        "config": cfg.model_dump(mode="json"),
        "trades": [asdict(t) for t in replay.engine.trades],
    }
    bt.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("backtest done in %.0fs, saved to %s", report["elapsed_s"], out)
    return report
