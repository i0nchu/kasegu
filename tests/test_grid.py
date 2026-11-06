import json
import math

import numpy as np
import pandas as pd
import pytest

from kasegu.__main__ import parse_args
from kasegu.core.types import Kline
from kasegu.grid import signals
from kasegu.grid.config import GridConfig, load_grid_config
from kasegu.grid.episode import BAR_MS, GridEpisode, GridParams
from kasegu.grid.paper import DAY_MS, GridPaper

FREE = GridParams(maker=0, taker=0, slip_in=0, slip_out=0, pen=0)


def run(path, side=1, step=0.01, params=FREE, funding=None, hl=0.0):
    e = GridEpisode("X", side, step, 0, params)
    prev = path[0]
    for i, c in enumerate(path):
        o = prev
        f = funding.get(i, 0.0) if funding else 0.0
        e.on_bar(i * BAR_MS, o, max(o, c) * (1 + hl), min(o, c) * (1 - hl), c, f)
        prev = c
        if not e.open:
            break
    return e


def test_weights_sum_and_first_order_is_min_notional():
    w = GridParams().weights()
    assert sum(w) == pytest.approx(1)
    assert w[0] * 104 == pytest.approx(5.0, abs=0.01)
    assert w[5] / w[4] == pytest.approx(1.5)


def test_long_take_profit_on_base_order():
    e = run([100, 100.5, 101.2])
    assert e.reason == "tp" and e.filled == 1
    assert e.pnl == pytest.approx(FREE.weights()[0] * 104 * 0.01)


def test_long_averages_down_then_takes_profit_at_avg_plus_step():
    e = run([100, 99.5, 98.9, 97.9, 99, 101])
    assert e.reason == "tp" and e.filled == 3
    assert e.exit_price == pytest.approx(e.avg() * 1.01)


def test_long_stop_one_step_below_last_level():
    e = run([100] + list(np.linspace(99, 90, 30)))
    assert e.reason == "stop" and e.filled == 6
    assert e.exit_price == pytest.approx(94.0)
    assert e.pnl < 0


def test_short_is_mirror_of_long():
    e = run([100, 99.5, 98.8], side=-1)
    assert e.reason == "tp"
    assert e.pnl == pytest.approx(FREE.weights()[0] * 104 * 0.01)


def test_funding_long_pays_short_receives():
    flat = [100] * 5
    assert run(flat, 1, funding={2: 0.001}).funding < 0
    assert run(flat, -1, funding={2: 0.001}).funding > 0


def test_time_exit_after_hold_bars():
    p = GridParams(maker=0, taker=0, slip_in=0, slip_out=0, pen=0, hold_bars=10)
    e = run([100] * 20, params=p)
    assert e.reason == "time" and e.exit_time == 9 * BAR_MS


def test_no_take_profit_in_a_bar_that_filled_a_new_level():
    e = GridEpisode("X", 1, 0.01, 0, FREE)
    e.on_bar(0, 100, 100, 100, 100)
    # one bar reaches both the next level and the old take profit: adverse move assumed last
    e.on_bar(BAR_MS, 100, 101.5, 98.9, 100)
    assert e.open and e.filled == 2


def test_liquidation_caps_loss_at_margin():
    p = GridParams(maker=0, taker=0, slip_in=0, slip_out=0, pen=0, margin=5.0)
    # the stop at 70 would lose more than the 5 USDT margin
    e = run([100, 95, 90, 85, 80, 75, 50], step=0.05, params=p)
    assert e.reason == "liquidation"
    assert e.pnl == pytest.approx(-5.0 - 0.104)
    # a stop that loses less than the margin stays a stop
    assert run([100, 99, 98, 97, 96, 95, 70], step=0.01, params=p).reason == "stop"


def test_episode_roundtrip():
    e = run([100, 99.5, 98.9])
    e2 = GridEpisode.from_dict(json.loads(json.dumps(e.to_dict())))
    assert e2 == e


def test_vols_match_pandas():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    r = pd.Series(np.log(c)).diff()
    assert signals.rv_3d(c) == pytest.approx(r.iloc[-72:].std() * math.sqrt(24))
    assert signals.sigma_7d(c) == pytest.approx(r.iloc[-168:].std() * math.sqrt(24))
    assert signals.rv_3d(c[:50]) is None


def test_direction():
    up = list(range(1, 120))
    down = up[::-1]
    assert signals.direction(up, up) == 1
    assert signals.direction(down, down) == -1
    assert signals.direction(up, down) == 0
    assert signals.direction(up[:10], up) == 0


def test_universe_ranks_by_30d_volume_and_age():
    now = 200 * DAY_MS
    qv = {"A": [1.0] * 40, "B": [3.0] * 40, "C": [2.0] * 40, "NEW": [9.0] * 40}
    onboard = {"A": 0, "B": 0, "C": 0, "NEW": now - 10 * DAY_MS}
    assert signals.universe(qv, onboard, now, top=2) == ["B", "C"]


# paper runner with a fake market

DAY = 1000 * DAY_MS  # a UTC midnight


def bars(interval_ms, end, n, closes, vol=1.0, sym="X"):
    out = []
    for i in range(n):
        t = end - (n - i) * interval_ms
        c = closes[i]
        out.append(Kline(sym, t, t + interval_ms - 1, c, c * 1.001, c * 0.999, c, vol))
    return out


class FakeMarket:
    def __init__(self, now):
        self.now = now
        rng = np.random.default_rng(1)
        self.hourly = {
            "HOT": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 170))),  # rv3 ~ 0.15
            "WARM": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 170))),
            "CALM": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 170))),
            "DOWN": 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 170))),
            "ETHUSDT": 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 170))),
        }
        self.trend = {"HOT": 1, "WARM": 1, "CALM": 1, "DOWN": -1, "ETHUSDT": 1, "BTCUSDT": 1}
        self.five = {}

    async def perpetuals(self):
        return {s: 0 for s in self.trend}

    async def klines(self, symbol, interval, limit, start=None):
        if interval == "1d":
            closes = (
                np.linspace(50, 150, limit)
                if self.trend[symbol] == 1
                else np.linspace(150, 50, limit)
            )
            ks = bars(DAY_MS, DAY + DAY_MS, limit, closes, vol=10.0, sym=symbol)
        elif interval == "1h":
            ks = bars(3_600_000, DAY + 3_600_000, limit, self.hourly[symbol][-limit:], sym=symbol)
        else:
            ks = [k for k in self.five.get(symbol, []) if k.open_time >= start]
        return [k for k in ks if k.close_time < self.now()]

    async def funding(self, symbol, start, end):
        return []


@pytest.fixture
def clock():
    t = {"now": DAY + 30_000}
    return t


def make(tmp_path, clock, **kw):
    cfg = GridConfig(out_dir=tmp_path, slots=2, **kw)
    m = FakeMarket(lambda: clock["now"])
    return GridPaper(cfg, m, now=lambda: clock["now"]), m


async def test_decide_picks_highest_rv3_with_trend_and_skips_excluded(tmp_path, clock):
    gp, _ = make(tmp_path, clock)
    await gp.step()
    picked = {e.symbol: e.side for e in gp.active}
    # ETH excluded, CALM below the vol cut, DOWN: coin below SMA50 while BTC above -> no trade
    assert picked == {"HOT": 1, "WARM": 1}
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert rec["chosen"] == ["HOT", "WARM"]
    assert all(e.step <= 0.10 for e in gp.active)
    # same day again: no new decision
    await gp.step()
    assert len((tmp_path / "decisions.jsonl").read_text().splitlines()) == 1


async def test_grid_runs_on_5m_bars_and_survives_restart(tmp_path, clock):
    gp, m = make(tmp_path, clock)
    await gp.step()
    hot = next(e for e in gp.active if e.symbol == "HOT")
    p = 100.0
    up = p * (1 + hot.step * 1.2)
    m.five["HOT"] = bars(BAR_MS, DAY + 2 * BAR_MS, 2, [p, p], sym="HOT")
    m.five["WARM"] = bars(BAR_MS, DAY + 2 * BAR_MS, 2, [p, p], sym="WARM")
    clock["now"] = DAY + 2 * BAR_MS + 1000
    await gp.step()
    assert hot.p0 == pytest.approx(p) and hot.open

    gp2, _ = make(tmp_path, clock)  # restart from state.json
    gp2.market = m
    assert {e.symbol for e in gp2.active} == {"HOT", "WARM"}
    m.five["HOT"].append(Kline("HOT", DAY + 2 * BAR_MS, DAY + 3 * BAR_MS - 1, p, up, p, up, 1))
    clock["now"] = DAY + 3 * BAR_MS + 1000
    await gp2.step()
    assert [e.symbol for e in gp2.active] == ["WARM"]
    assert gp2.closed == 1 and gp2.realized > 0
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("HOT,1,")


async def test_account_loss_limit_stops_new_grids(tmp_path, clock):
    gp, _ = make(tmp_path, clock, max_account_loss=1.0)
    gp.realized = -2.0
    await gp.step()
    assert gp.halted and gp.active == []


def test_config_example_loads_and_cli_mode():
    cfg = load_grid_config("configs/mgrid.example.yaml")
    assert cfg == GridConfig()  # the example is the frozen candidate
    assert parse_args(["--mode", "grid-paper"]).mode == "grid-paper"
