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
from kasegu.config import AppConfig
from kasegu.core.engine import Engine
from kasegu.core.execution import PaperBroker
from kasegu.exchange.binance import BinanceMarket
from kasegu.exchange.rules import SymbolRules
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

    q: asyncio.Queue = asyncio.Queue()

    async def subscribe():
        await feed.stream(
            cfg.symbols,
            on_kline=lambda k: q.put_nowait(("bar", k)),
            on_quote=lambda x: q.put_nowait(("quote", x)),
            on_mark=lambda m: q.put_nowait(("mark", m)),
        )

    await subscribe()
    handlers = {"bar": eng.on_bar, "quote": eng.on_quote, "mark": eng.on_mark}
    try:
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
            bad = feed.stale()
            if bad:
                log.warning("stream stale (%s), reconnecting", ", ".join(bad))
                eng.halted = True
                journal.event("reconnect", reason=bad)
                try:
                    await feed.close()
                    await subscribe()
                    await resync(feed, eng)
                    eng.halted = False
                except Exception:
                    log.exception("reconnect failed")
                    await asyncio.sleep(5)
    finally:
        await feed.close()
        journal.event("stop")
        journal.close()


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
    name = run_name(cfg.symbols, start, end, cfg.strategy.id, cfg.strategy.k, label)
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
