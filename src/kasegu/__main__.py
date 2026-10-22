import argparse
import asyncio
import json
import logging
import sys
from datetime import date

from kasegu.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="kasegu")
    p.add_argument("--mode", choices=["backtest", "paper", "live"], default="paper")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--confirm-live",
        action="store_true",
        help="required together with --mode live",
    )
    p.add_argument("--start", type=date.fromisoformat, help="backtest start (UTC day)")
    p.add_argument("--end", type=date.fromisoformat, help="backtest end (UTC day)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.log_level.upper() != "DEBUG":
        # the Binance SDK logs every frame on the root logger at INFO
        for h in logging.getLogger().handlers:
            h.addFilter(lambda r: r.name != "root" or r.levelno >= logging.WARNING)
    log = logging.getLogger("kasegu")

    if args.mode == "live" and not args.confirm_live:
        log.error("live mode must be enabled explicitly with --confirm-live")
        return 2

    cfg = load_config(args.config)
    cfg.mode = args.mode
    log.info("mode=%s strategy=%s symbols=%s", cfg.mode, cfg.strategy.id, cfg.symbols)

    if cfg.mode == "paper":
        from kasegu.app import run_paper

        try:
            asyncio.run(run_paper(cfg))
        except KeyboardInterrupt:
            pass
        return 0

    if cfg.mode == "backtest":
        from kasegu.app import run_backtest

        cfg.backtest.start = args.start or cfg.backtest.start
        cfg.backtest.end = args.end or cfg.backtest.end
        r = asyncio.run(run_backtest(cfg))["result"]
        print(json.dumps({k: v for k, v in r.items() if k != "by_group"}, indent=2))
        for g, v in r["by_group"].items():
            print(f"{g:16} trades={v['trades']:4} net={v['net']:.4f} win={v['win_rate']}")
        return 0

    log.error("mode %s is not implemented yet", cfg.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
