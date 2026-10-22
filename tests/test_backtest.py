import math
from datetime import date, timedelta

import pytest

from kasegu.backtest.data import days, months
from kasegu.backtest.metrics import summarize
from kasegu.backtest.replay import Funding, Replay, Tick, ms, trade_bars
from kasegu.config import AppConfig
from kasegu.core.position import Trade
from kasegu.core.types import Kline, Side
from kasegu.exchange.rules import SymbolRules
from kasegu.market.data import MINUTE

S = "BTCUSDT"
DAY = date(2026, 1, 2)


def test_ranges():
    assert days(date(2026, 1, 30), date(2026, 2, 1)) == [
        date(2026, 1, 30),
        date(2026, 1, 31),
        date(2026, 2, 1),
    ]
    assert months(date(2026, 1, 30), date(2026, 3, 1)) == ["2026-01", "2026-02", "2026-03"]


def test_trade_bars_fill_gaps_and_seed():
    t0 = 10 * MINUTE
    trades = [(t0 + 1000, 101.0, 1), (t0 + 2000, 99.0, 2), (t0 + 3 * MINUTE + 5, 102.0, 1)]
    out = list(trade_bars(S, trades, start=t0 - MINUTE, seed=100.0))
    bars = [x for x in out if isinstance(x, Kline)]
    ticks = [x for x in out if isinstance(x, Tick)]
    assert [b.open_time for b in bars] == [t0 - MINUTE + i * MINUTE for i in range(5)]
    assert (bars[0].open, bars[0].volume) == (100.0, 0.0)  # seeded empty minute
    assert (bars[1].open, bars[1].high, bars[1].low, bars[1].close) == (101.0, 101.0, 99.0, 99.0)
    assert bars[2].close == 99.0 and bars[2].volume == 0  # empty minute repeats close
    assert bars[-1].close == 102.0
    assert len(ticks) == 3
    times = [x.time if isinstance(x, Tick) else x.close_time for x in out]
    assert times == sorted(times)


def test_funding_lookup():
    f = Funding([(100, 0.0001), (200, -0.0002)])
    assert f.at(50) == (100, 0.0001)
    assert f.at(100) == (200, -0.0002)
    assert f.at(300) == (0, 0.0)


def trade(net, side=Side.BUY, sym=S, reason="trail_t1"):
    return Trade(sym, side, 1, 100, 100 + net, 0, MINUTE, net, 0, 0, 1.0, -0.5, reason)


def test_metrics():
    r = summarize([trade(2), trade(-1), trade(-2), trade(3, Side.SELL)], 8, 4, 0.0)
    assert r["net"] == pytest.approx(2)
    assert r["max_drawdown"] == pytest.approx(3)
    assert r["profit_factor"] == pytest.approx(5 / 3)
    assert r["win_rate"] == 0.5 and r["fill_rate"] == 0.5
    assert r["payoff"] == pytest.approx(2.5 / 1.5)
    assert set(r["by_group"]) == {"BTCUSDT BUY", "BTCUSDT SELL"}


class FakeArchive:
    """Uptrend with a sine wave, enough to fill dips and trail out."""

    def price(self, t):
        x = (t - ms(DAY)) / MINUTE
        return 100 * (1 + 0.00002 * x) * (1 + 0.006 * math.sin(x / 7))

    def klines(self, symbol, day):
        base = ms(day)
        out = []
        for i in range(1440):
            t = base + i * MINUTE
            p = self.price(t)
            out.append(Kline(symbol, t, t + MINUTE - 1, p, p * 1.0005, p * 0.9995, p, 1))
        return out

    def trades(self, symbol, day):
        base = ms(day)
        for i in range(0, 1440 * 60, 5):
            t = base + i * 1000
            yield t, round(self.price(t), 2), 0.1

    funding_calls = 0

    def funding(self, symbol, start, end):
        self.funding_calls += 1
        return []  # month not published yet, forces the REST fallback


async def test_replay_end_to_end():
    cfg = AppConfig(
        symbols=[S],
        strategy={"k": 0.5},
        fee={"slippage_pct": 0.0},
        risk={"max_spread_pct": 1.0, "max_notional": 1e9},
    )
    rules = {S: SymbolRules(S, 0.01, 0.001, 0.001, 1e6, 5)}
    calls = []

    async def api(sym, lo, hi):
        calls.append((sym, lo, hi))
        return [(ms(DAY) + h * 3_600_000, 0.0001) for h in range(0, 40, 8)]

    arc = FakeArchive()
    rp = Replay(cfg, rules, arc, DAY, DAY, funding_api=api)
    r = await rp.run()
    assert arc.funding_calls == 1
    assert calls and calls[0][1] == ms(DAY)
    assert r["funding"] != 0
    assert r["open_positions"] == []
    assert r["entries"] > 0 and r["trades"] > 0
    assert 0 < r["fill_rate"] <= 1
    assert r["fees"] > 0
    assert sum(r["exit_reasons"].values()) == r["trades"]
    # bars continue from warmup with no gap
    assert rp.market.last_open(S) == ms(DAY + timedelta(1)) - MINUTE
