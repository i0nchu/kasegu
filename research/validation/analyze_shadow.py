"""Replay recorded shadow data across fixed and jittered latency assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .maker import AggTrade, BookGap, DepthDelta, LatencyModel, LocalBook, ShadowOrder


def analyze(path: Path, out: Path, models: list[LatencyModel]) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for model in models:
        books: dict[str, LocalBook] = {}
        orders: dict[str, ShadowOrder] = {}
        for record in records:
            kind, d = record["kind"], record["payload"]
            symbol = d.get("symbol") or d.get("s")
            if kind == "snapshot":
                book = books.setdefault(symbol, LocalBook(symbol))
                book.install_snapshot(d["lastUpdateId"], d["bids"], d["asks"], d.get("E"))
            elif kind == "depth":
                book = books.setdefault(symbol, LocalBook(symbol))
                delta = DepthDelta(
                    symbol,
                    int(d["E"]),
                    int(d["U"]),
                    int(d["u"]),
                    int(d["pu"]),
                    tuple((float(p), float(q)) for p, q in d["b"]),
                    tuple((float(p), float(q)) for p, q in d["a"]),
                )
                try:
                    book.apply(delta)
                except BookGap:
                    book.synchronized = False
                for order in orders.values():
                    if (
                        order.symbol == symbol
                        and order.queue_ahead is None
                        and order.status == "waiting"
                        and d["E"] >= order.activation_time
                    ):
                        order.activate(book)
            elif kind == "book_unknown":
                books.setdefault(symbol, LocalBook(symbol)).synchronized = False
            elif kind == "candidate":
                latency = model.sample(d["candidate_id"])
                orders[d["candidate_id"]] = ShadowOrder.create(
                    d["candidate_id"],
                    d["symbol"],
                    d["side"],
                    float(d["price"]),
                    float(d["qty"]),
                    int(d["created_ms"]),
                    int(d["cancel_ms"]),
                    latency,
                )
            elif kind == "agg_trade":
                trade = AggTrade(
                    symbol,
                    int(d["T"]),
                    float(d["p"]),
                    float(d["q"]),
                    bool(d["m"]),
                )
                for order in orders.values():
                    if order.symbol != symbol:
                        continue
                    if order.queue_ahead is None and trade.trade_time >= order.activation_time:
                        order.activate(books.setdefault(symbol, LocalBook(symbol)))
                    order.on_trade(trade)
        for order in orders.values():
            order.expire(order.cancel_time)
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
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    models = [
        LatencyModel(0),
        LatencyModel(50),
        LatencyModel(100, 50, 1),
        LatencyModel(250, 100, 1),
        LatencyModel(500, 250, 1),
    ]
    analyze(args.input, args.out, models)


if __name__ == "__main__":
    main()
