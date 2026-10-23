"""Generate a deterministic daily EPR15 Paper/Shadow reliability report.

The tool is read-only: it consumes Kasegu's Journal SQLite database, the status
snapshot, and (optionally) conservative public-order-book fill estimates.  It
never imports exchange clients, reads credentials, or submits orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

CRITICAL_KINDS = {
    "disconnected",
    "exit_rejected",
    "loss_limit",
    "order_unknown",
    "stop_lost",
    "stop_rejected",
}


def _window(day: date) -> tuple[int, int]:
    start = datetime.combine(day, time.min, UTC)
    return int(start.timestamp() * 1000), int((start + timedelta(days=1)).timestamp() * 1000)


def _json(raw: str | None) -> dict[str, Any]:
    return json.loads(raw) if raw else {}


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "max": None}
    return {"count": len(values), "avg": fmean(values), "max": max(values)}


def _status_pnl(path: Path | None) -> tuple[float | None, dict[str, Any] | None]:
    if path is None or not path.exists():
        return None, None
    status = json.loads(path.read_text())
    total = 0.0
    for position in status.get("symbols", {}).values():
        qty = float(position.get("qty") or 0)
        entry = position.get("entry_price")
        mark = position.get("mark")
        side = position.get("side")
        if not qty or entry is None or mark is None or side not in {"BUY", "SELL"}:
            continue
        total += (float(mark) - float(entry)) * qty * (1 if side == "BUY" else -1)
    return total, status


def _estimate_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    status = Counter(row.get("status") or "unknown" for row in rows)
    filled_qty = sum(float(row.get("filled_qty") or 0) for row in rows)
    fractions = [float(row["fill_fraction"]) for row in rows if row.get("fill_fraction")]
    return {
        "orders": len(rows),
        "status": dict(sorted(status.items())),
        "filled_qty": filled_qty,
        "average_fill_fraction": fmean(fractions) if fractions else None,
        "source": str(path),
    }


def build_report(
    db_path: Path,
    day: date,
    status_path: Path | None = None,
    estimates_path: Path | None = None,
) -> dict[str, Any]:
    start, end = _window(day)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        events = [
            {"ts": row["ts"], "kind": row["kind"], "symbol": row["symbol"], **_json(row["data"])}
            for row in db.execute(
                "SELECT ts, kind, symbol, data FROM events WHERE ts >= ? AND ts < ? ORDER BY id",
                (start, end),
            )
        ]
        order_rows = list(
            db.execute(
                "SELECT client_id, ts, symbol, purpose, status, data FROM orders "
                "WHERE ts >= ? AND ts < ? ORDER BY ts, client_id",
                (start, end),
            )
        )
        fill_rows = list(
            db.execute(
                "SELECT ts, client_id, symbol, side, qty, price, fee, maker FROM fills "
                "WHERE ts >= ? AND ts < ? ORDER BY id",
                (start, end),
            )
        )
        trade_rows = list(
            db.execute(
                "SELECT ts, symbol, data FROM trades WHERE ts >= ? AND ts < ? ORDER BY id",
                (start, end),
            )
        )
    finally:
        db.close()

    signals = [event for event in events if event["kind"] == "signal"]
    signal_rejections = Counter(
        str(event.get("reason", "unspecified")) for event in signals if not event.get("ok", False)
    )
    orders_by_id: dict[str, dict[str, Any]] = {}
    order_status = Counter()
    order_purpose = Counter()
    for row in order_rows:
        data = _json(row["data"])
        data.update(
            client_id=row["client_id"],
            created_ts=row["ts"],
            symbol=row["symbol"],
            purpose=row["purpose"],
            status=row["status"],
        )
        orders_by_id[row["client_id"]] = data
        order_status[str(row["status"])] += 1
        order_purpose[str(row["purpose"])] += 1

    latency: dict[str, list[float]] = defaultdict(list)
    slippage: dict[str, list[float]] = defaultdict(list)
    fill_qty = Counter()
    first_fill: set[str] = set()
    for fill in fill_rows:
        order = orders_by_id.get(fill["client_id"])
        if order is None:
            continue
        purpose = str(order.get("purpose") or "unknown")
        fill_qty[purpose] += float(fill["qty"])
        if fill["client_id"] not in first_fill:
            latency[purpose].append(float(fill["ts"] - order["created_ts"]))
            first_fill.add(fill["client_id"])
        reference = order.get("price") or order.get("stop_price")
        if reference:
            sign = 1 if fill["side"] == "BUY" else -1
            adverse_bps = (
                (float(fill["price"]) - float(reference))
                / float(reference)
                * sign
                * 10_000
            )
            slippage[purpose].append(adverse_bps)

    trades = [_json(row["data"]) for row in trade_rows]
    realized = sum(
        float(item.get("pnl", 0)) - float(item.get("fees", 0)) + float(item.get("funding", 0))
        for item in trades
    )
    durations = [
        (float(item["close_time"]) - float(item["open_time"])) / 60_000
        for item in trades
        if item.get("close_time") is not None and item.get("open_time") is not None
    ]
    exit_reasons = Counter(str(item.get("reason") or "unknown") for item in trades)
    websocket = Counter(
        event["kind"]
        for event in events
        if event["kind"] in {"backfill", "reconnect", "reconnected", "disconnected"}
    )
    critical = [
        event
        for event in events
        if event.get("level") == "critical" or event["kind"] in CRITICAL_KINDS
    ]
    unrealized, status = _status_pnl(status_path)
    estimates = _estimate_summary(estimates_path)
    gaps = []
    if status is None:
        gaps.append("status snapshot missing: unrealized PnL unavailable")
    if estimates is None:
        gaps.append("conservative order-book estimates missing: Paper fill optimism unquantified")
    if any(row["purpose"] == "exit" for row in order_rows):
        gaps.append("market-exit decision price is not journaled: time-exit slippage unavailable")

    return {
        "date_utc": day.isoformat(),
        "window_ms": [start, end],
        "signals": {
            "total": len(signals),
            "accepted": sum(bool(event.get("ok")) for event in signals),
            "rejected": len(signals) - sum(bool(event.get("ok")) for event in signals),
            "rejection_reasons": dict(sorted(signal_rejections.items())),
        },
        "orders": {
            "total": len(order_rows),
            "by_purpose": dict(sorted(order_purpose.items())),
            "by_status": dict(sorted(order_status.items())),
            "rejected_or_expired": order_status["REJECTED"] + order_status["EXPIRED"],
        },
        "paper_fills": {
            "events": len(fill_rows),
            "qty_by_purpose": dict(sorted(fill_qty.items())),
            "first_fill_latency_ms": {key: _stats(value) for key, value in sorted(latency.items())},
            "adverse_slippage_bps": {key: _stats(value) for key, value in sorted(slippage.items())},
        },
        "conservative_fill_estimates": estimates,
        "positions": {
            "closed": len(trades),
            "duration_minutes": _stats(durations),
            "exit_reasons": dict(sorted(exit_reasons.items())),
        },
        "pnl": {"realized": realized, "unrealized_snapshot": unrealized},
        "websocket": dict(sorted(websocket.items())),
        "critical_alerts": critical,
        "status": status,
        "data_gaps": gaps,
    }


def render_markdown(report: dict[str, Any]) -> str:
    signals, orders = report["signals"], report["orders"]
    fills, positions, pnl = report["paper_fills"], report["positions"], report["pnl"]
    estimates = report["conservative_fill_estimates"]
    lines = [
        f"# EPR15 Paper / Shadow Daily Report — {report['date_utc']} UTC",
        "",
        "## Decision summary",
        "",
        f"- Signals: {signals['total']} "
        f"({signals['accepted']} accepted, {signals['rejected']} rejected)",
        f"- Orders: {orders['total']} ({orders['rejected_or_expired']} rejected or expired)",
        f"- Paper fill events: {fills['events']}",
        f"- Closed positions: {positions['closed']}",
        f"- PnL: realized {pnl['realized']:.6f} USDT; unrealized snapshot "
        + (
            "unavailable"
            if pnl["unrealized_snapshot"] is None
            else f"{pnl['unrealized_snapshot']:.6f} USDT"
        ),
        f"- Critical safety alerts: {len(report['critical_alerts'])}",
        "",
        "## Evidence",
        "",
        f"- Signal rejection reasons: `{json.dumps(signals['rejection_reasons'], sort_keys=True)}`",
        f"- Order statuses: `{json.dumps(orders['by_status'], sort_keys=True)}`",
        f"- Exit reasons: `{json.dumps(positions['exit_reasons'], sort_keys=True)}`",
        f"- WebSocket/recovery events: `{json.dumps(report['websocket'], sort_keys=True)}`",
        "- First-fill latency (ms): "
        f"`{json.dumps(fills['first_fill_latency_ms'], sort_keys=True)}`",
        f"- Adverse slippage (bps): `{json.dumps(fills['adverse_slippage_bps'], sort_keys=True)}`",
    ]
    if estimates is not None:
        lines.append(
            "- Conservative book estimate: "
            f"{estimates['orders']} orders, {estimates['filled_qty']:.8f} filled quantity, "
            f"statuses `{json.dumps(estimates['status'], sort_keys=True)}`"
        )
    lines.extend(["", "## Data gaps / blockers", ""])
    lines.extend(f"- {gap}" for gap in report["data_gaps"])
    if not report["data_gaps"]:
        lines.append("- None detected by the report tool.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Paper Journal SQLite database")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--status", type=Path, help="Latest Kasegu status JSON")
    parser.add_argument("--estimates", type=Path, help="Conservative shadow-fill CSV")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.db, args.date, args.status, args.estimates)
    markdown = render_markdown(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown)
    if not args.json_out and not args.md_out:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
