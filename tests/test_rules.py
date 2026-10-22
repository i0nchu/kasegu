import pytest

from kasegu.exchange.rules import parse_rules

INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "0.001",
                    "minQty": "0.001",
                    "maxQty": "1000",
                },
                {"filterType": "MIN_NOTIONAL", "notional": "100"},
            ],
        }
    ]
}


def test_parse_and_round():
    r = parse_rules(INFO)["BTCUSDT"]
    assert r.price_down(100.17) == pytest.approx(100.1)
    assert r.price_up(100.11) == pytest.approx(100.2)
    assert r.price_up(100.1) == pytest.approx(100.1)
    assert r.qty(0.0019) == pytest.approx(0.001)
    assert r.valid(100.0, 0.5) is not None
    assert r.valid(1000.0, 0.5) is None
