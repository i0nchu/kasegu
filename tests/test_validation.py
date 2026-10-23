import json
from datetime import date

import pandas as pd
import pytest

from research.validation.analyze_shadow import analyze
from research.validation.forward_holdout import days as forward_days
from research.validation.maker import (
    AggTrade,
    BookGap,
    DepthDelta,
    FillStatus,
    LatencyModel,
    LocalBook,
    ShadowOrder,
)
from research.validation.portfolio import CapacityScenario, replay, stable_uniform


def delta(first, final, previous, bids=(), asks=()):
    return DepthDelta("BTCUSDT", 1_000, first, final, previous, tuple(bids), tuple(asks))


def test_forward_holdout_uses_only_completed_october_days():
    assert forward_days() == [date(2026, 10, day) for day in range(1, 23)]


def test_book_requires_snapshot_bridge_and_contiguous_updates():
    book = LocalBook("BTCUSDT")
    book.install_snapshot(100, [["99", "2"]], [["101", "3"]])
    book.apply(delta(99, 101, 98, bids=((100.0, 5.0),)))
    assert book.synchronized
    assert book.best_bid == 100
    book.apply(delta(102, 102, 101, asks=((101.0, 0.0), (102.0, 4.0))))
    assert book.best_ask == 102
    with pytest.raises(BookGap):
        book.apply(delta(104, 104, 103))
    assert not book.synchronized


def test_unsynchronized_history_is_unknown_not_a_fill():
    order = ShadowOrder.create("o", "BTCUSDT", "buy", 100, 2, 1_000, 5_000, 0)
    assert order.activate(LocalBook("BTCUSDT")) == FillStatus.UNKNOWN
    assert order.on_trade(AggTrade("BTCUSDT", 2_000, 90, 100, True)) == 0


def test_queue_model_has_latency_queue_ahead_and_partial_fill():
    book = LocalBook("BTCUSDT")
    book.install_snapshot(100, [["100", "5"]], [["101", "3"]])
    book.apply(delta(99, 100, 98))
    order = ShadowOrder.create("o", "BTCUSDT", "buy", 100, 3, 1_000, 10_000, 500)
    order.activate(book)
    assert order.on_trade(AggTrade("BTCUSDT", 1_400, 100, 99, True)) == 0
    assert order.on_trade(AggTrade("BTCUSDT", 2_000, 100, 4, True)) == 0
    assert order.on_trade(AggTrade("BTCUSDT", 3_000, 100, 3, True)) == 2
    assert order.status == FillStatus.PARTIAL
    assert order.on_trade(AggTrade("BTCUSDT", 4_000, 100, 2, True)) == 1
    assert order.status == FillStatus.FILLED


def test_same_side_aggressor_does_not_fill_passive_buy():
    book = LocalBook("BTCUSDT")
    book.install_snapshot(100, [["100", "0"]], [["101", "3"]])
    book.apply(delta(99, 100, 98))
    order = ShadowOrder.create("o", "BTCUSDT", "buy", 100, 1, 1_000, 10_000, 0)
    order.activate(book)
    assert order.on_trade(AggTrade("BTCUSDT", 2_000, 101, 10, False)) == 0


def test_stable_priority_is_reproducible_and_seed_sensitive():
    assert stable_uniform("x", 7) == stable_uniform("x", 7)
    assert stable_uniform("x", 7) != stable_uniform("x", 8)


def test_latency_jitter_is_bounded_and_repeatable():
    model = LatencyModel(100, 50, 7)
    assert model.sample("o") == model.sample("o")
    assert 50 <= model.sample("o") <= 150


def test_shadow_log_latency_analysis_end_to_end(tmp_path):
    records = [
        {
            "kind": "snapshot",
            "payload": {
                "symbol": "BTCUSDT",
                "lastUpdateId": 100,
                "bids": [["100", "5"]],
                "asks": [["101", "4"]],
            },
        },
        {
            "kind": "depth",
            "payload": {
                "s": "BTCUSDT",
                "E": 1_000,
                "U": 99,
                "u": 100,
                "pu": 98,
                "b": [],
                "a": [],
            },
        },
        {
            "kind": "candidate",
            "payload": {
                "candidate_id": "o",
                "symbol": "BTCUSDT",
                "side": "buy",
                "price": 100,
                "qty": 2,
                "created_ms": 1_000,
                "cancel_ms": 5_000,
            },
        },
        {
            "kind": "agg_trade",
            "payload": {"s": "BTCUSDT", "T": 2_000, "p": "100", "q": "7", "m": True},
        },
    ]
    source = tmp_path / "shadow.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = analyze(source, tmp_path / "result.csv", [LatencyModel(0)])
    assert result.iloc[0].status == FillStatus.FILLED
    assert result.iloc[0].fill_fraction == 1


def test_portfolio_replay_enforces_global_position_limit():
    trades = pd.DataFrame(
        [
            {
                "symbol": "A",
                "trade_id": 0,
                "t": 0,
                "exit_t": 120_000,
                "reason": "max_hold",
                "net": 1.0,
                "mae": -0.5,
            },
            {
                "symbol": "B",
                "trade_id": 0,
                "t": 0,
                "exit_t": 120_000,
                "reason": "max_hold",
                "net": 2.0,
                "mae": -0.4,
            },
            {
                "symbol": "C",
                "trade_id": 0,
                "t": 60_000,
                "exit_t": 180_000,
                "reason": "stop_loss",
                "net": -3.0,
                "mae": -3.0,
            },
        ]
    )
    detail, summary = replay(
        trades,
        CapacityScenario(
            "one", order_notional=100, max_margin=10, max_notional=100, max_positions=1
        ),
        seed=3,
    )
    assert summary["accepted"] == 1
    assert detail.reason.value_counts()["max_positions"] == 2
    assert summary["max_margin_used"] == 10


def test_capacity_stress_applies_partial_fill_stop_slippage_and_impact():
    trades = pd.DataFrame(
        [
            {
                "symbol": "A",
                "trade_id": 0,
                "t": 0,
                "exit_t": 60_000,
                "reason": "stop_loss",
                "net": 1.0,
                "mae": -2.0,
            }
        ]
    )
    scenario = CapacityScenario(
        "stress",
        order_notional=100,
        max_margin=100,
        max_notional=1_000,
        fill_fraction=0.5,
        stop_extra_slip_pct=0.3,
        round_trip_impact_bps=10,
    )
    detail, summary = replay(trades, scenario)
    assert detail.iloc[0].filled_notional == 50
    assert summary["net_usdt"] == pytest.approx(50 * (0.01 - 0.003 - 0.001))
    assert summary["historical_order_book"] == "unknown"
