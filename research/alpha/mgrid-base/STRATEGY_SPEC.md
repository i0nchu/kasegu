# Frozen executable strategy contract — mgrid-base

> 此檔只有在策略符合階段資格時才作為可部署候選；文件存在不代表允許 Live。

## Identity / governance

- Strategy ID / version / Git commit / config SHA：
- Evidence tier：`TRADABLE_CANDIDATE / VALIDATED_CANDIDATE / FORWARD_SUPPORTED / LIVE_MONITORED`
- Auth：`SHADOW_ONLY`（除非使用者另外明確授權）
- Existing repo location / runner / dependency versions：
- Research results and baseline / holdout audit：

## Data contract

- Symbol universe / listing-delisting / contract multiplier：
- Source, schema, timestamp, timezone, late data & missing bars：
- Max staleness, candle closing, indicator warmup, state replay：
- Decision-time availability and point-in-time checks：

## Pure signal function

- Exact conditions, defaults and versioned thresholds：
- Signal timestamp, earliest executable timestamp：
- Entry side / candidate priority / cancellation / no-trade filters：
- Deterministic replay, no revised historical signal：

## Order lifecycle

- Market/limit; slippage; penetration / partial fill; min notional：
- Client order IDs; idempotence; exchange reconciliation：
- Entry / protective stop / TP / timeout / reduce-only：
- Gap handling, simultaneous TP-SL, funding schedule：
- Restart recovery / disconnected states：

## Risk contract

- Position notional vs account risk / margin / leverage：
- Maximum concurrent positions / gross and net exposure / per-symbol caps：
- Daily loss limit / DD kill switch / volatility & liquidity freeze：
- Liquidation threshold tests / emergency exit / funding guard：
- Capital allocation and capacity assumptions：

## Verification contract

- Unit tests: no lookahead, signal parity, PnL units, funding sign, venue units：
- Integration tests: gaps, double fills, partial fills, cancel/retry, reconnect：
- Backtest-vs-forward parity, realized fees/slippage dashboard：
- Paper result / thresholds for Small Live request：
- Explicit user authorization before placing real orders：

## Monitoring / retirement

- Reference `MONITORING.md` and emergency-disable runbook：
