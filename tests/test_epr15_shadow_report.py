import csv
import json
from datetime import UTC, date, datetime

import pytest

from kasegu.core.position import Trade
from kasegu.core.types import Fill, Order, OrderStatus, OrderType, Purpose, Side
from kasegu.storage.journal import Journal
from scripts.epr15_shadow_report import build_report, render_markdown


def ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


def test_daily_report_joins_journal_status_and_conservative_estimates(tmp_path):
    db_path = tmp_path / "journal.db"
    journal = Journal(db_path, run_id="paper-test")
    start = ms("2026-10-23T00:00:00")
    journal.event("signal", "BTCUSDT", ts=start + 1, ok=True, reason="ok", price=99.0)
    journal.event("signal", "ETHUSDT", ts=start + 2, ok=False, reason="stale_data")
    journal.event("reconnect", ts=start + 3, reason=["market"])
    journal.event(
        "order_unknown", "BTCUSDT", ts=start + 4, level="critical", message="blocked"
    )

    entry = Order(
        "vtn-report",
        "BTCUSDT",
        Side.BUY,
        OrderType.LIMIT,
        1.0,
        Purpose.ENTRY,
        price=99.0,
        status=OrderStatus.FILLED,
        time=start + 100,
    )
    rejected = Order(
        "vtn-rejected",
        "ETHUSDT",
        Side.BUY,
        OrderType.LIMIT,
        1.0,
        Purpose.ENTRY,
        price=50.0,
        status=OrderStatus.REJECTED,
        time=start + 150,
    )
    journal.order(entry)
    journal.order(rejected)
    journal.fill(Fill(entry.client_id, entry.symbol, Side.BUY, 1.0, 99.01, 0.02, True, start + 350))
    journal.trade(
        Trade(
            "BTCUSDT",
            Side.BUY,
            1.0,
            99.01,
            100.0,
            start + 350,
            start + 15 * 60_000 + 350,
            0.99,
            0.04,
            0.01,
            1.2,
            -0.5,
            "time_exit",
        ),
        ts=start + 15 * 60_000 + 350,
    )
    journal.close()

    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "symbols": {
                    "BTCUSDT": {
                        "side": "BUY",
                        "qty": 2,
                        "entry_price": 100,
                        "mark": 101.5,
                    }
                }
            }
        )
    )
    estimates_path = tmp_path / "estimates.csv"
    with estimates_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["order_id", "status", "filled_qty", "fill_fraction"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "order_id": entry.client_id,
                "status": "partial",
                "filled_qty": 0.25,
                "fill_fraction": 0.25,
            }
        )

    report = build_report(
        db_path, date(2026, 10, 23), status_path=status_path, estimates_path=estimates_path
    )
    assert report["signals"] == {
        "total": 2,
        "accepted": 1,
        "rejected": 1,
        "rejection_reasons": {"stale_data": 1},
    }
    assert report["orders"]["rejected_or_expired"] == 1
    assert report["paper_fills"]["first_fill_latency_ms"]["entry"]["avg"] == 250
    assert report["paper_fills"]["adverse_slippage_bps"]["entry"]["avg"] == pytest.approx(
        1.010101
    )
    assert report["conservative_fill_estimates"]["filled_qty"] == 0.25
    assert report["positions"]["duration_minutes"]["avg"] == 15
    assert report["pnl"] == {"realized": pytest.approx(0.96), "unrealized_snapshot": 3.0}
    assert report["websocket"] == {"reconnect": 1}
    assert [alert["kind"] for alert in report["critical_alerts"]] == ["order_unknown"]
    assert "Critical safety alerts: 1" in render_markdown(report)


def test_report_declares_missing_shadow_and_status_evidence(tmp_path):
    db_path = tmp_path / "empty.db"
    Journal(db_path).close()
    report = build_report(db_path, date(2026, 10, 23))
    assert report["pnl"]["unrealized_snapshot"] is None
    assert len(report["data_gaps"]) == 2
