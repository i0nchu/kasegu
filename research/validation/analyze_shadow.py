"""Replay public order-book data against Paper entry orders conservatively."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from research.validation.orderbook import (
    AggTrade,
    BookGap,
    DepthDelta,
    LatencyModel,
    LocalBook,
    ShadowOrder,
)


def journal_candidates(path: Path, lifetime_ms: int) -> list[dict]:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT client_id, ts, symbol, data FROM orders "
            "WHERE purpose = 'entry' ORDER BY ts, client_id"
        ).fetchall()
    finally:
        db.close()
    candidates = []
    for client_id, created, symbol, raw in rows:
        order = json.loads(raw)
        candidates.append(
            {
                "candidate_id": client_id,
                "symbol": symbol,
                "side": str(order["side"]).lower(),
                "price": float(order["price"]),
                "qty": float(order["qty"]),
                "created_ms": created,
                "cancel_ms": created + lifetime_ms,
            }
        )
    return candidates


def analyze(
    market_path: Path,
    candidates: list[dict],
    models: list[LatencyModel],
) -> list[dict]:
    records = [json.loads(line) for line in market_path.read_text().splitlines() if line.strip()]
    rows = []
    for model in models:
        books: dict[str, LocalBook] = {}
        orders = {
            candidate["candidate_id"]: ShadowOrder.create(
                candidate, model.sample(candidate["candidate_id"])
            )
            for candidate in candidates
        }
        for record in records:
            kind, data = record["kind"], record["payload"]
            symbol = data.get("symbol") or data.get("s")
            if kind == "snapshot":
                books.setdefault(symbol, LocalBook(symbol)).install_snapshot(
                    data["lastUpdateId"], data["bids"], data["asks"]
                )
            elif kind == "depth":
                book = books.setdefault(symbol, LocalBook(symbol))
                delta = DepthDelta(
                    symbol,
                    int(data["E"]),
                    int(data["U"]),
                    int(data["u"]),
                    int(data["pu"]),
                    tuple((float(p), float(q)) for p, q in data["b"]),
                    tuple((float(p), float(q)) for p, q in data["a"]),
                )
                try:
                    book.apply(delta)
                except BookGap:
                    book.synchronized = False
                for order in orders.values():
                    if (
                        order.symbol == symbol
                        and order.queue_ahead is None
                        and data["E"] >= order.activation_ms
                    ):
                        order.activate(book)
            elif kind == "book_unknown":
                books.setdefault(symbol, LocalBook(symbol)).synchronized = False
            elif kind == "agg_trade":
                trade = AggTrade(
                    symbol,
                    int(data["T"]),
                    float(data["p"]),
                    float(data["q"]),
                    bool(data["m"]),
                )
                for order in orders.values():
                    if order.symbol != symbol:
                        continue
                    if order.queue_ahead is None and trade.trade_time >= order.activation_ms:
                        order.activate(books.setdefault(symbol, LocalBook(symbol)))
                    order.on_trade(trade)
        for order in orders.values():
            order.expire()
            rows.append(
                {
                    "base_latency_ms": model.base_ms,
                    "jitter_ms": model.jitter_ms,
                    "seed": model.seed,
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "status": order.status,
                    "filled_qty": order.filled_qty,
                    "fill_fraction": order.filled_qty / order.qty,
                    "queue_remaining": order.queue_ahead,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("market", type=Path)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lifetime-ms", type=int, default=60_000)
    args = parser.parse_args()
    models = [
        LatencyModel(0),
        LatencyModel(50),
        LatencyModel(100, 50, 1),
        LatencyModel(250, 100, 1),
        LatencyModel(500, 250, 1),
    ]
    rows = analyze(
        args.market,
        journal_candidates(args.journal, args.lifetime_ms),
        models,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
