"""Shared-capital replay for frozen candidate trades.

The input trades come from a one-position-per-symbol OHLC simulation.  This
replay imposes one global account, deterministic competition among simultaneous
signals, missed/partial maker fills and adverse exit execution.  Because no
historical order book exists, fill fractions and impact are sensitivity inputs,
not measurements of realizable capacity.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class CapacityScenario:
    name: str
    capital: float = 1_000.0
    order_notional: float = 297.0
    leverage: float = 10.0
    max_margin: float = 30.0
    max_notional: float = 350.0
    max_positions: int = 1
    fill_fraction: float = 1.0
    miss_rate: float = 0.0
    stop_extra_slip_pct: float = 0.0
    round_trip_impact_bps: float = 0.0
    latency_ms: int = 0


def stable_uniform(key: str, seed: int) -> float:
    raw = hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def replay(trades: pd.DataFrame, scenario: CapacityScenario, seed: int = 0):
    required = {"symbol", "trade_id", "t", "exit_t", "reason", "net", "mae"}
    missing = required - set(trades)
    if missing:
        raise ValueError(f"missing trade columns: {sorted(missing)}")
    x = trades.copy()
    x["candidate_key"] = x.apply(lambda r: f"{r.symbol}:{int(r.t)}:{int(r.trade_id)}", axis=1)
    x["priority"] = x.candidate_key.map(lambda key: stable_uniform(key, seed))
    x = x.sort_values(["t", "priority", "symbol", "trade_id"])

    # Heap rows: close_time, serial, notional, pnl, adverse_excursion, candidate_key.
    open_heap: list[tuple[int, int, float, float, float, str]] = []
    serial = 0
    exposure = margin = realized = 0.0
    peak = worst_drawdown = 0.0
    max_margin_used = time_margin = active_time = 0.0
    last_t = int(x.t.min()) if len(x) else 0
    rows = []
    event_pnl: dict[int, float] = {}
    accepted_risk = 0.0

    def close_until(now: int) -> None:
        nonlocal exposure, margin, realized, peak, worst_drawdown
        nonlocal time_margin, active_time, last_t
        while open_heap and open_heap[0][0] <= now:
            close_t, _, notional, pnl, adverse, key = heapq.heappop(open_heap)
            duration = max(close_t - last_t, 0)
            time_margin += margin * duration
            active_time += duration if margin > 0 else 0
            last_t = close_t
            exposure -= notional
            margin -= notional / scenario.leverage
            realized += pnl
            peak = max(peak, realized)
            worst_drawdown = max(worst_drawdown, peak - realized)
        duration = max(now - last_t, 0)
        time_margin += margin * duration
        active_time += duration if margin > 0 else 0
        last_t = now

    for r in x.itertuples(index=False):
        entry_t = int(r.t) + scenario.latency_ms
        close_until(entry_t)
        desired = min(
            scenario.order_notional,
            max(scenario.max_notional - exposure, 0.0),
            max(scenario.max_margin - margin, 0.0) * scenario.leverage,
        )
        reason = "accepted"
        if len(open_heap) >= scenario.max_positions:
            reason, desired = "max_positions", 0.0
        elif desired <= 0:
            reason, desired = "margin_or_notional", 0.0
        elif stable_uniform(r.candidate_key + ":fill", seed) < scenario.miss_rate:
            reason, desired = "maker_miss", 0.0
        filled = desired * scenario.fill_fraction
        if filled <= 0:
            rows.append(
                {
                    "candidate_key": r.candidate_key,
                    "symbol": r.symbol,
                    "t": int(r.t),
                    "accepted": False,
                    "reason": reason,
                    "filled_notional": 0.0,
                    "pnl": 0.0,
                    "event_id": int(r.t) // 300_000,
                }
            )
            continue
        extra = scenario.round_trip_impact_bps / 10_000
        if r.reason == "stop_loss":
            extra += scenario.stop_extra_slip_pct / 100
        pnl = filled * (float(r.net) / 100 - extra)
        adverse = filled * abs(min(float(r.mae), 0.0)) / 100
        exposure += filled
        margin += filled / scenario.leverage
        max_margin_used = max(max_margin_used, margin)
        serial += 1
        heapq.heappush(
            open_heap,
            (int(r.exit_t) + 60_000, serial, filled, pnl, adverse, r.candidate_key),
        )
        event_id = int(r.t) // 300_000
        event_pnl[event_id] = event_pnl.get(event_id, 0.0) + pnl
        accepted_risk += adverse
        # Conservative floating check: every open position is at its recorded MAE together.
        worst_equity = realized - sum(item[4] for item in open_heap)
        worst_drawdown = max(worst_drawdown, peak - worst_equity)
        rows.append(
            {
                "candidate_key": r.candidate_key,
                "symbol": r.symbol,
                "t": int(r.t),
                "accepted": True,
                "reason": "accepted_partial" if scenario.fill_fraction < 1 else "accepted",
                "filled_notional": filled,
                "pnl": pnl,
                "event_id": event_id,
            }
        )
    close_until(2**63 - 1)
    elapsed = max((int(x.exit_t.max()) + 60_000 - int(x.t.min())) if len(x) else 0, 1)
    detail = pd.DataFrame(rows)
    accepted = detail[detail.accepted] if len(detail) else detail
    summary = {
        **asdict(scenario),
        "seed": seed,
        "candidates": len(detail),
        "accepted": len(accepted),
        "accept_rate": len(accepted) / len(detail) if len(detail) else 0.0,
        "filled_notional": float(accepted.filled_notional.sum()) if len(accepted) else 0.0,
        "net_usdt": realized,
        "return_on_initial_capital_pct": realized / scenario.capital * 100,
        "max_floating_drawdown_usdt": worst_drawdown,
        "max_floating_drawdown_pct": worst_drawdown / scenario.capital * 100,
        "max_margin_used": max_margin_used,
        "mean_margin_utilization": time_margin / elapsed / scenario.capital,
        "active_margin_utilization": (
            time_margin / active_time / scenario.capital if active_time else 0.0
        ),
        "return_per_mae_risk": realized / accepted_risk if accepted_risk else None,
        "market_events": len(event_pnl),
        "worst_event_usdt": min(event_pnl.values()) if event_pnl else 0.0,
        "historical_order_book": "unknown",
    }
    return detail, summary


def default_scenarios() -> list[CapacityScenario]:
    return [
        CapacityScenario("ohlc_upper_bound"),
        CapacityScenario("partial_50", fill_fraction=0.5, miss_rate=0.25),
        CapacityScenario(
            "queue_stop_stress",
            fill_fraction=0.25,
            miss_rate=0.50,
            stop_extra_slip_pct=0.30,
            round_trip_impact_bps=5,
            latency_ms=250,
        ),
        CapacityScenario(
            "three_positions_100",
            order_notional=100,
            max_margin=90,
            max_notional=900,
            max_positions=3,
            fill_fraction=0.5,
            miss_rate=0.25,
            stop_extra_slip_pct=0.10,
            round_trip_impact_bps=2,
            latency_ms=100,
        ),
        CapacityScenario(
            "three_positions_1000_requested",
            order_notional=1_000,
            max_margin=90,
            max_notional=900,
            max_positions=3,
            fill_fraction=0.25,
            miss_rate=0.50,
            stop_extra_slip_pct=0.30,
            round_trip_impact_bps=10,
            latency_ms=250,
        ),
    ]
