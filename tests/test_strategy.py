import numpy as np
import pytest

from kasegu.config import StrategyConfig
from kasegu.core.types import Side
from kasegu.strategies import available, create
from kasegu.strategies import indicators as ind
from tests.helpers import make_state


def test_true_range_uses_prev_close():
    h = np.array([10.0, 12.0, 11.0])
    lo = np.array([9.0, 11.5, 7.0])
    c = np.array([9.5, 11.8, 8.0])
    # bar1: max(0.5, |12-9.5|, |11.5-9.5|) = 2.5; bar2: max(4, |11-11.8|, |7-11.8|) = 4.8
    assert ind.true_range(h, lo, c).tolist() == pytest.approx([2.5, 4.8])


def test_atr_and_amplitude_window():
    c = np.full(100, 100.0)
    h, lo = c + 1, c - 1
    assert ind.atr(h, lo, c, 60) == pytest.approx(2.0)
    assert ind.amplitude_pct(h, lo, c, 60) == pytest.approx(2.0)


def test_ema_matches_pandas_definition():
    c = np.arange(1, 11, dtype=float)
    a = 2 / (3 + 1)
    e = c[0]
    for x in c[1:]:
        e = a * x + (1 - a) * e
    assert ind.ema(c, 3) == pytest.approx(e)


def test_registry():
    assert "atr_amplitude" in available()
    with pytest.raises(ValueError):
        create(StrategyConfig(id="nope"))


def test_no_signal_before_warmup():
    s = create(StrategyConfig())
    assert s.on_bar(make_state(np.linspace(100, 110, 100))) is None


def test_long_in_uptrend():
    s = create(StrategyConfig())
    close = np.linspace(100, 200, 1000)
    st = make_state(close, spread=0.02)
    sig = s.on_bar(st)
    assert sig is not None and sig.side is Side.BUY
    m = sig.metadata
    exp_atr = ind.atr(st.high, st.low, st.close, 60) / st.ask * 100
    exp_amp = ind.amplitude_pct(st.high, st.low, st.close, 60)
    dist = (exp_atr + exp_amp) / 2 * 2.0
    assert m["distance_pct"] == pytest.approx(dist)
    assert sig.limit_price == pytest.approx(st.ask * (1 - dist / 100))
    assert sig.limit_price < st.bid


def test_short_in_downtrend():
    s = create(StrategyConfig(k=1.0))
    st = make_state(np.linspace(200, 100, 1000), spread=0.02)
    sig = s.on_bar(st)
    assert sig.side is Side.SELL
    assert sig.limit_price == pytest.approx(st.ask * (1 + sig.metadata["distance_pct"] / 100))


def test_no_signal_when_close_equals_ema():
    s = create(StrategyConfig())
    assert s.on_bar(make_state(np.full(1000, 100.0))) is None


def test_mid_reference():
    s = create(StrategyConfig(reference="mid"))
    st = make_state(np.linspace(100, 200, 1000), bid=199.0, ask=201.0)
    assert s.on_bar(st).metadata["ref"] == pytest.approx(200.0)
