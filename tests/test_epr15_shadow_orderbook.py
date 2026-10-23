import pytest

from research.validation.orderbook import (
    AggTrade,
    BookGap,
    DepthDelta,
    FillStatus,
    LatencyModel,
    LocalBook,
    ShadowOrder,
)


def delta(first, final, previous, bid_qty=2.0):
    return DepthDelta(
        "BTCUSDT",
        1000,
        first,
        final,
        previous,
        ((99.0, bid_qty),),
        ((100.0, 2.0),),
    )


def candidate():
    return {
        "candidate_id": "vtn-shadow",
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 99.0,
        "qty": 1.0,
        "created_ms": 1000,
        "cancel_ms": 2000,
    }


def test_gap_invalidates_book_instead_of_claiming_a_fill():
    book = LocalBook("BTCUSDT")
    book.install_snapshot(10, [["99", "2"]], [["100", "2"]])
    assert book.apply(delta(10, 11, 9))
    with pytest.raises(BookGap, match="depth gap"):
        book.apply(delta(12, 13, 99))
    order = ShadowOrder.create(candidate(), latency_ms=0)
    order.activate(book)
    assert order.status is FillStatus.UNKNOWN


def test_queue_model_requires_trade_through_displayed_quantity():
    book = LocalBook("BTCUSDT")
    book.install_snapshot(10, [["99", "2"]], [["100", "2"]])
    book.apply(delta(10, 11, 9))
    order = ShadowOrder.create(candidate(), latency_ms=50)
    order.activate(book)

    order.on_trade(AggTrade("BTCUSDT", 1049, 98.9, 10, buyer_is_maker=True))
    assert order.filled_qty == 0  # still in the deterministic latency window
    order.on_trade(AggTrade("BTCUSDT", 1100, 99.0, 1.5, buyer_is_maker=True))
    assert order.filled_qty == 0 and order.queue_ahead == pytest.approx(0.5)
    order.on_trade(AggTrade("BTCUSDT", 1200, 98.9, 1.0, buyer_is_maker=True))
    assert order.filled_qty == pytest.approx(0.5)
    assert order.status is FillStatus.PARTIAL


def test_latency_jitter_is_reproducible_by_order_id():
    model = LatencyModel(250, 100, seed=7)
    assert model.sample("a") == model.sample("a")
    assert 150 <= model.sample("a") <= 350
