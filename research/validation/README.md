# Independent validation tooling

This directory is research-only. It has no authenticated endpoint and never sends orders.

## Reproduce the frozen candidate and capacity study

The validation input is a copy-on-write snapshot under `data/validation/`. Its hashes are written to
`research/validation/results/data_manifest.csv`.

```bash
PYTHONPATH=research .venv/bin/python -m validation.run_validation --reproduce
.venv/bin/python -m pytest tests/test_validation.py -q
.venv/bin/python -m ruff check research/validation tests/test_validation.py
```

The frozen alpha definition is `frozen_vda.json`. Validation may worsen execution and portfolio
assumptions; it must not alter the strategy thresholds to recover profitability.

## Record future public shadow data

This command records public market data only. It does not read API credentials. It is intentionally
not run as part of tests or historical validation.

```bash
PYTHONPATH=research .venv/bin/python -m validation.shadow_collector \
  BTCUSDT ETHUSDT --out data/validation/shadow/market.jsonl --duration-s 3600
```

Strategy code can call `ShadowCollector.candidate()` with the exact intended order:

```json
{
  "candidate_id": "stable-unique-id",
  "symbol": "BTCUSDT",
  "side": "buy",
  "price": 65000.0,
  "qty": 0.001,
  "created_ms": 1790000000000,
  "cancel_ms": 1790000060000
}
```

The collector appends snapshot, diff-depth, BBO, depth, spread, aggTrade, book-gap and candidate
records. A sequence gap marks the book unknown and triggers a new snapshot. Historical OHLC or
aggTrades are never substituted for a missing book.

After candidate events exist, replay fixed and jittered latency assumptions:

```bash
PYTHONPATH=research .venv/bin/python -m validation.analyze_shadow \
  data/validation/shadow/market.jsonl \
  --out research/validation/results/shadow_latency.csv
```

The queue model starts behind displayed quantity at the candidate price. It ignores cancellations
ahead of us and decrements the queue only with opposite-aggressor trades at or through the order
price. This is deliberately conservative and supports partial fills. It remains a model; only the
recorded snapshot and deltas are observations.

## Evidence levels

- `sim.py`: one-minute OHLC with a deterministic intrabar path assumption.
- E4 replay: aggTrades with synthetic last-trade ± half-tick quotes.
- Shadow data: synchronized public order-book observation plus aggTrades.
- Real fills: unavailable until a separately authorized, non-shadow order experiment exists.

Do not describe the first two levels as real maker execution.
