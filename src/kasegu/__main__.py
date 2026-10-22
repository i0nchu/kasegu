import argparse
import asyncio
import logging
import os
import sys
from datetime import date
from logging.handlers import RotatingFileHandler
from pathlib import Path

from kasegu.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # env defaults make docker compose setup a matter of .env
    env = os.environ.get
    p = argparse.ArgumentParser(prog="kasegu")
    p.add_argument(
        "--mode", choices=["backtest", "paper", "live"], default=env("KASEGU_MODE", "paper")
    )
    p.add_argument("--config", default=env("KASEGU_CONFIG", "configs/config.yaml"))
    p.add_argument("--symbols", help="comma separated, overrides config symbols")
    p.add_argument("--strategy", help="overrides strategy.id")
    p.add_argument("--k", type=float, help="overrides strategy.k")
    p.add_argument(
        "--confirm-live",
        action="store_true",
        default=env("KASEGU_CONFIRM_LIVE") == "1",
        help="required together with --mode live (or KASEGU_CONFIRM_LIVE=1)",
    )
    p.add_argument("--start", type=date.fromisoformat, help="backtest start (UTC day)")
    p.add_argument("--end", type=date.fromisoformat, help="backtest end (UTC day)")
    p.add_argument("--label", default="", help="backtest name prefix, e.g. exp1")
    p.add_argument("--each", action="store_true", help="backtest each symbol separately")
    p.add_argument("--log-level", default=env("KASEGU_LOG_LEVEL", "INFO"))
    p.add_argument("--log-file", default=env("KASEGU_LOG_FILE"), help="rotating log file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(args.log_file, maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(logging.getLogger().handlers[0].formatter)
        logging.getLogger().addHandler(fh)
    if args.log_level.upper() != "DEBUG":
        # the Binance SDK logs every frame on the root logger at INFO
        for h in logging.getLogger().handlers:
            h.addFilter(lambda r: r.name != "root" or r.levelno >= logging.WARNING)
    log = logging.getLogger("kasegu")

    if args.mode == "live" and not args.confirm_live:
        log.error("live mode must be enabled explicitly with --confirm-live")
        return 2

    if not Path(args.config).exists():
        log.error(
            "%s not found, copy configs/config.example.yaml to %s first", args.config, args.config
        )
        return 2
    cfg = load_config(args.config)
    cfg.mode = args.mode
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.strategy and args.strategy != cfg.strategy.id:
        from kasegu.strategies import available

        if args.strategy not in available():
            log.error("unknown strategy %s (have %s)", args.strategy, available())
            return 2
        # params belong to the configured model, do not carry them over
        cfg.strategy.id, cfg.strategy.params = args.strategy, {}
    if args.k is not None:
        if args.k <= 0:
            log.error("--k must be > 0")
            return 2
        cfg.strategy.k = args.k
    log.info("mode=%s strategy=%s symbols=%s", cfg.mode, cfg.strategy.id, cfg.symbols)

    if cfg.mode == "paper":
        from kasegu.app import run_paper, until_stopped

        try:
            asyncio.run(until_stopped(run_paper(cfg)))
        except KeyboardInterrupt:
            pass
        return 0

    if cfg.mode == "backtest":
        from kasegu.app import run_backtest
        from kasegu.backtest.report import row, summary, table

        cfg.backtest.start = args.start or cfg.backtest.start
        cfg.backtest.end = args.end or cfg.backtest.end
        # --each runs every symbol on its own so results are comparable
        groups = [[s] for s in cfg.symbols] if args.each else [cfg.symbols]
        reports = []
        for syms in groups:
            run = cfg.model_copy(deep=True)
            run.symbols = syms
            rep = asyncio.run(run_backtest(run, args.label))
            print("\n" + summary(rep) + "\n")
            reports.append(rep)
        if len(reports) > 1:
            rows = sorted((row(r) for r in reports), key=lambda r: -r["net"])
            print(table(rows))
        return 0

    from kasegu.app import run_live, until_stopped

    if not cfg.exchange.testnet:
        log.warning("LIVE ON MAINNET: real orders with real funds")
    try:
        asyncio.run(until_stopped(run_live(cfg)))
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError) as e:
        log.error("%s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
