# EPR15 Independent Reliability Review

## Scope and evidence

- Repository: `i0nchu/kasegu`
- Review baseline: `1026d8f872dbb88abb166832f8d5059a15b66fae`
- Independent branch: `test/epr15-reliability`
- Production Engine/Broker changes in this branch: none
- Credentials, authenticated endpoints, deployments, and real orders: none
- Baseline suite before adding independent tests: **124 passed**
- Independent baseline result: **3 passed, 6 failed** in
  `tests/test_epr15_reliability.py`

The broker fixtures keep exchange orders/positions separate from the Engine's local
references. Event order and endpoint failures are scripted, deterministic, and do
not use wall-clock sleeps or network access.

## Findings at the reviewed baseline

| ID | Scenario and evidence | Baseline result | Classification | Severity |
|---|---|---|---|---|
| EPR15-R01 | `test_stop_replacement_timeout_keeps_last_confirmed_stop_active`: a late fill requires a larger stop; placement times out before acceptance and query returns unknown. The Engine cancels the smaller confirmed stop because the unconfirmed local replacement still has status `NEW`. Exchange stop count becomes zero. | Failed | Confirmed bug | P0 |
| EPR15-R02 | `test_reconcile_keeps_global_block_when_entry_cancel_is_unconfirmed`: cancel and query both fail. `sync()` removes the block although the entry remains open on the simulated exchange. | Failed | Confirmed bug | P0 |
| EPR15-R03 | `test_reconcile_does_not_trust_an_incomplete_open_order_snapshot`: a stale empty open-order response hides a known entry. `sync()` removes the global block and would permit another symbol to enter. | Failed | Unsafe behavior under simulation; snapshot freshness contract is also unspecified | P0 |
| EPR15-R04 | `test_reconcile_rest_failure_cannot_clear_an_existing_block`: open-orders returns, the block is cleared, then the position endpoint fails. The exception escapes with the block already removed. | Failed | Confirmed bug | P0 |
| EPR15-R05 | `test_partial_fill_cancel_race_converges_to_exposure_and_full_stop`: 40% fills, another 20% fills during cancel, cumulative cancel is delivered before the incremental fill. Final local/exchange exposure agrees and one full-sized stop remains. | Passed | Test limitation: simulated broker cannot prove Binance user-stream delivery guarantees | P0 scenario, no bug reproduced |
| EPR15-R06 | `test_flat_sync_clears_stale_stop_before_same_size_reentry`: an external close is reconciled and the exchange stop is canceled, but `Engine.stop` still references the old open-looking object. A same-sized re-entry reuses it and places no exchange stop. | Failed | Confirmed bug | P0 |
| EPR15-R07 | `test_restart_cancels_pending_entry_before_entries_can_resume`: a pending entry left at the crash boundary is found and canceled on restart. | Passed | Test limitation: confirmed-cancel path only | P0 scenario, no bug reproduced |
| EPR15-R08 | `test_restart_with_protected_position_executes_overdue_time_exit`: restart adopts the matching stop, preserves the original open time, and exits at the first post-restart mark after 20 minutes. | Passed | Test limitation: journal and exchange agree | P0 scenario, no bug reproduced |
| EPR15-R09 | `test_restart_disagreement_places_resized_stop_before_canceling_old_stop`: journal and old stop cover 40%, while exchange exposure is 60%. Startup cancels the old stop before placing the resized stop, creating an avoidable zero-coverage interval. | Failed | Confirmed bug | P0 |

Passing tests are not proof of live safety. They establish only the exact modeled
sequence. In particular, WebSocket duplication, multi-stream ordering, exchange
snapshot consistency, and Binance conditional-order behavior still require Testnet
evidence after the P0 failures are fixed.

## Safety invariants required for revalidation

1. A stop is considered confirmed only after an exchange acknowledgement or an
   unambiguous query result for the same client id.
2. An unconfirmed replacement must never cause the last confirmed stop to be
   canceled. Local `NEW` is not exchange coverage.
3. Stop coverage must be measured against current exchange exposure. Partial
   coverage must remain explicit and must globally block new entries.
4. Reconciliation is transactional from a safety perspective: an existing block is
   removed only after all required order, cancellation, and position reads succeed.
5. A known open/unknown entry must be resolved by client id. One empty open-order
   snapshot is not sufficient evidence that it is gone.
6. Local stop references are cleared or replaced whenever the exchange position is
   flat, regardless of whether the next position has the same quantity.
7. During restart resize, a replacement stop is confirmed before an older,
   partially covering stop is canceled.

## Missing specifications and observability gaps

- The freshness/consistency guarantee expected from `open_orders()` is not defined.
  Reconciliation needs a documented rule for resolving a previously known client id.
- The minimum acceptable behavior while only partial stop coverage exists is not
  specified. This review requires global entry blocking and retention of the last
  confirmed stop.
- Journal order rows retain creation time but not cancel-request, cancel-ack, or each
  state-transition time. The conservative fill replay therefore uses a documented
  60,000 ms EPR15 candidate lifetime rather than claiming observed cancel latency.
- Market exit orders do not journal the decision-time mark/book reference. Exit
  latency is available, but time-exit slippage cannot be measured reliably from the
  current artifacts. The daily report flags this rather than inventing a price.
- Paper positions intentionally do not survive restart. Paper crash-recovery results
  must not be presented as evidence for LiveBroker recovery.

## Minimal Paper / Shadow workflow

Paper Mode already produces the source of truth needed for signals, orders, fills,
trades, alerts, and current positions:

- `data/journal-epr15.db`: candidates, orders, fills, trades, state, and events
- `data/events-epr15.jsonl`: alert stream
- `data/status-epr15.json`: five-second status/PnL snapshot

The added research collector records only public Binance depth and aggregate trades.
It has no credential loading and no order API. A depth sequence gap marks the book
unknown and forces a new snapshot. The analyzer begins behind displayed quantity,
does not infer queue cancellations, supports partial fills, and runs deterministic
latency sensitivity cases.

Run Paper and the public collector in separate terminals:

```bash
uv run python -m kasegu --mode paper --config configs/epr15.yaml \
  --log-file data/logs/epr15.log

uv run python -m research.validation.shadow_collector BTCUSDT ETHUSDT \
  --out data/validation/epr15-market.jsonl --duration-s 86400
```

At the UTC day boundary, produce conservative estimates and a daily report:

```bash
uv run python -m research.validation.analyze_shadow \
  data/validation/epr15-market.jsonl \
  --journal data/journal-epr15.db \
  --out data/validation/epr15-fill-estimates.csv

uv run python scripts/epr15_shadow_report.py \
  --db data/journal-epr15.db \
  --status data/status-epr15.json \
  --estimates data/validation/epr15-fill-estimates.csv \
  --date 2026-10-23 \
  --json-out data/reports/epr15-2026-10-23.json \
  --md-out data/reports/epr15-2026-10-23.md
```

The report contains candidate signal acceptance/rejection, order status, rejected or
expired orders, Paper fill quantities, first-fill latency by purpose, adverse
slippage when a defensible reference exists, conservative book-fill estimates,
position duration/exit reason, stop and critical events, WebSocket recovery events,
and realized/unrealized Paper PnL.

### Shadow readiness checklist

- [ ] Use `mode: paper`, `exchange.data_testnet: false`, and separate EPR15 Journal,
      JSONL, status, and log paths. No Binance keys are required.
- [ ] Confirm the frozen EPR15 parameters and `time_only` exit at startup.
- [ ] Run the public depth collector for exactly the Paper symbols and retain raw
      JSONL immutably.
- [ ] Treat `unknown_book` periods as unavailable evidence, never as non-fills.
- [ ] Review every `backfill`, `reconnect`, `reconnected`, and `disconnected` event.
- [ ] Require zero unexplained critical events (`order_unknown`, `stop_lost`,
      `stop_rejected`, `exit_rejected`, `disconnected`, `loss_limit`).
- [ ] Compare Paper fill rate/quantity with every conservative latency model. Paper
      is expected to be optimistic; quantify the gap by symbol and day.
- [ ] Review rejected and expired orders, entry/exit first-fill latency, available
      slippage, 15-minute holding duration, stop events, and daily PnL.
- [ ] Fail the daily review if the status snapshot is stale/missing, conservative
      estimates are missing, or the report declares another data gap.
- [ ] Accumulate at least two weeks before drawing readiness conclusions. Do not use
      Shadow results as evidence of actual maker queue position or live stop slippage.

## Independent revalidation

Revalidation must be run without weakening these assertions:

```bash
uv run pytest tests/test_epr15_reliability.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Revalidated against `b5d0c14b5512efb73e475cff111ee4a320a25c90`, fetched from
`origin/feat/epr15-integration` after the concurrent fixes were pushed.

- Independent P0 suite: **8 passed, 1 failed**.
- Updated integration suite, excluding only the one independent file containing the
  known blocker: **141 upstream tests passed**. The five added report/order-book tests
  also passed, for **146 passed** in that run.
- Focused `atr_amplitude`/shared-engine regression set (`test_engine`,
  `test_strategy`, `test_backtest`, `test_safety`, `test_config`): **43 passed**.
- `src/kasegu/strategies/atr_amplitude.py` is unchanged from `1026d8f`.
- The documented historical archive replay (5 trades, −0.9469 USDT, 2,808 entry
  orders) could not be independently rerun because its aggTrade archives are not
  present in this worktree. This is a **test limitation**, not a confirmed pass.

The fixes corrected R01 through R04 and R06. R05, R07, and R08 continue to pass.
R09 remains reproducible: `_sync()` cancels every non-matching stop before calling
`_protect()`. In the journal-lags-exchange fixture, the operation trace is
`[cancel old 40% stop, place new 60% stop]`. The exchange finishes with the correct
stop when both calls succeed, but there is a zero-coverage interval and a failed next
call would occur after protection was already removed.

The first revalidation run also showed expected event-taxonomy changes:
`order_unknown` was split into `stop_unknown`, `entry_cancel_unconfirmed`, and
`sync_unconfirmed`, and `sync()` now contains REST failures. Exact reason-string and
exception assertions were generalized to assert the original safety property
(blocking remains and the old stop precedes any emergency close). No exchange-state,
coverage, ordering, exposure, or entry-blocking assertion was weakened.

## Verdict at baseline `1026d8f`

| Level | Verdict | Reason |
|---|---|---|
| Safe for merge? | **No** | Six independent P0 assertions fail, including zero stop coverage and premature global unblock. |
| Safe for Paper / Shadow? | **Yes, observation only** | Paper does not send orders. Use the workflow above; do not interpret Paper fills as executable fills. |
| Safe for Testnet? | **No** | The P0 recovery and protection invariants must pass first; then Testnet must exercise real REST/user-stream ordering. |
| Safe for small real-money execution? | **No** | Confirmed paths can leave a live position without a stop or allow entries while an old entry remains unresolved. |

## Final verdict at updated commit `b5d0c14`

| Level | Verdict | Reason |
|---|---|---|
| Safe for merge? | **No** | R09 is still a reproducible P0 stop-continuity failure during restart reconciliation. |
| Safe for Paper / Shadow? | **Yes, observation only** | No orders leave the process. The added workflow is suitable for collecting evidence, subject to every declared data gap. |
| Safe for Testnet? | **No** | Fix R09 first, rerun the unchanged test, then validate real conditional-order and user-stream sequencing on Testnet. |
| Safe for small real-money execution? | **No** | A process restart with journal/exchange quantity disagreement can temporarily remove the only exchange-side stop. |

## Final integration revalidation at `c097ac1`

The remote branches were fetched again before integration. The test branch merged
`origin/feat/epr15-integration` at
`c097ac12089ff4029026ca5912675002114659ef` without rebasing or force-pushing. The
merge-under-test was `f4f333a17b6f768000082827d90802561030192a` before this report and
mechanical Ruff formatting were committed. No conflict required changing or
weakening an independent safety assertion.

### R01–R09 result

| ID | Latest result | Evidence |
|---|---|---|
| EPR15-R01 | **PASS** | A stop placement timeout never causes an unconfirmed replacement to displace the last confirmed stop. If partial coverage is insufficient, the reduce-only emergency close is submitted before protection is canceled. |
| EPR15-R02 | **PASS** | Failed entry cancellation/query retains per-symbol and therefore global entry blocking. |
| EPR15-R03 | **PASS** | An incomplete open-order snapshot cannot clear a previously unknown entry because the known client id is rechecked first. |
| EPR15-R04 | **PASS** | A REST failure during reconciliation is contained and leaves `sync_unconfirmed` blocking in place. |
| EPR15-R05 | **PASS** | Out-of-order cumulative cancel and incremental fill events converge to the exchange exposure with one full-sized stop. |
| EPR15-R06 | **PASS** | Flat reconciliation clears the local stop reference; a same-sized new position receives a fresh stop id. |
| EPR15-R07 | **PASS** | A pending entry left by a crash is canceled before entries can resume. |
| EPR15-R08 | **PASS** | A protected position retains its original open time and executes an overdue time exit on the first post-restart mark. |
| EPR15-R09 | **PASS** | With journal exposure at 40% and exchange exposure at 60%, the operation trace places and confirms the 60% stop before canceling the old 40% stop. No zero-coverage interval is introduced by operation order. |

The independent R09 assertion was not changed. Claude's additional R09 tests were
also inspected and executed. They verify multiple correct-side stops plus a
wrong-side stop, replacement placement timeout, remaining quantity after a partial
stop trigger, and restart protection cleanup. The implementation now compares stop
`remaining` quantity to exchange exposure, keeps the largest valid existing stop
until a complete replacement is confirmed, and only then removes redundant
correct-side stops. A wrong-side reduce-only stop is not protection and can be
removed before replacement without reducing coverage.

### R09 before and after

- Before (`b5d0c14`): trace was `cancel old 40% stop` followed by
  `place new 60% stop`; a failed second call could leave zero exchange protection.
- After (`c097ac1`): trace is `place and confirm new 60% stop` followed by
  `cancel old 40% stop`. If placement is rejected or unknown, old correct-side stops
  remain until the position is closed or reconciliation succeeds. A partially
  triggered stop counts only its remaining quantity.

### Test and static-analysis record

- `uv run pytest tests/test_epr15_reliability.py -q`: **9 passed**.
- `uv run pytest tests/test_faults.py -q`: **21 passed**.
- `uv run pytest -q`: **158 passed**.
- `uv run ruff check .`: **passed**.
- Initial `uv run ruff format --check .`: found formatting-only differences in three
  independent Shadow/report files. Ruff mechanically formatted only those files.
- Final `uv run ruff format --check .`: **71 files already formatted**.
- Shadow Order Book and Daily Report tests are included in the 158-test full run.

`atr_amplitude`, `TrailingExit`, `Epr15`, `TimeOnlyExit`, `PaperBroker`, and the
existing `LiveBroker` interface are covered by the passing full suite. The frozen
EPR15 parameters and strategy implementation were not modified by this reliability
branch. The integration merge changes production code only through the reviewed
upstream R09 commits.

The historical SOLUSDT/DOGEUSDT aggTrade archives needed to independently reproduce
the documented 2026-10-01 through 2026-10-07 result are not installed in this
worktree. Therefore the historical 5-trade / −0.9469 USDT / 2,808-entry replay is
**not independently revalidated in this run**. Unit, integration, fault-injection,
Shadow Order Book, and Daily Report tests are complete; historical replay remains a
test limitation.

### Remaining safety and rollout conditions

No unresolved simulated P0 blocker was found after `c097ac1`. The following remain
external validation requirements rather than passing evidence:

- Binance Testnet has not exercised actual conditional-order acknowledgements,
  user-stream duplication/order, partial fills, restart adoption, or stop/time-exit
  races.
- At least two weeks of Paper/Shadow evidence and conservative book-fill comparison
  have not yet been collected.
- Real time-exit slippage still lacks a journaled decision-time book/mark reference.
- Mainnet stop slippage, mark-price trigger lag, exchange outages, and gap risk are
  not established by deterministic simulation.

## Final permission verdict at `c097ac1`

| Level | Verdict | Reason |
|---|---|---|
| Safe to merge into `main`? | **Yes, after normal PR review and CI** | R01–R09 pass unchanged independent assertions, the full suite passes, and no new simulated blocker remains. This is a software-integration verdict, not live-trading approval. |
| Safe for Paper / Shadow? | **Yes** | It submits no exchange orders and the collector/reporting workflow is tested. Continue to treat modeled fills as estimates. |
| Safe to enter Binance Testnet? | **Yes, for validation only** | Deterministic safety gates pass. Testnet acceptance scenarios listed above are still mandatory and must use explicitly authorized test credentials/orders. |
| Safe for Mainnet real money? | **No** | No Testnet campaign or two-week Shadow run has been completed, and real exchange latency/slippage/stream behavior remains unverified. |
