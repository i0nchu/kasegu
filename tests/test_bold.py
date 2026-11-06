import json

import pytest

from kasegu.__main__ import parse_args
from kasegu.bold.config import BoldConfig, load_bold_config
from kasegu.bold.paper import DAY_MS, BoldPaper
from kasegu.bold.shot import BAR_MS, Shot, ShotParams
from kasegu.core.types import Kline

FREE = ShotParams(leverage=10, target=2.0, bust=0.1, taker=0, slip=0)


def feed(shot, bars):
    for i, (o, h, low, c, *f) in enumerate(bars):
        shot.on_bar(shot.start + i * BAR_MS, o, h, low, c, f[0] if f else 0.0)
        if not shot.open:
            break
    return shot


def test_long_target_at_ten_percent():
    s = feed(Shot("BTCUSDT", 1, 0, FREE), [(100, 100, 100, 100), (100, 110.5, 100, 110)])
    assert s.reason == "target" and s.equity == pytest.approx(2.0)
    assert s.exit_price == pytest.approx(110.0)


def test_long_bust_at_nine_percent_checked_before_target():
    s = feed(Shot("BTCUSDT", 1, 0, FREE), [(100, 111, 90.5, 100)])
    assert s.reason == "bust" and s.equity == pytest.approx(0.1)
    assert s.exit_price == pytest.approx(91.0)


def test_gap_through_stop_fills_at_open():
    s = feed(Shot("BTCUSDT", 1, 0, FREE), [(100, 100, 100, 100), (85, 86, 84, 85)])
    assert s.reason == "bust" and s.equity == 0.0  # -15% at 10x is more than the stake


def test_short_mirror():
    s = feed(Shot("BTCUSDT", -1, 0, FREE), [(100, 100, 100, 100), (100, 100, 89.5, 90)])
    assert s.reason == "target" and s.equity == pytest.approx(2.0)


def test_costs_and_funding_move_the_levels():
    p = ShotParams(leverage=10, taker=0.0005, slip=0.0005)
    s = Shot("BTCUSDT", 1, 0, p)
    feed(s, [(100, 100, 100, 100, 0.001)])
    assert s.funding == pytest.approx(0.01)  # 0.1% funding x 10x
    assert s.up_move() == pytest.approx((1 + 0.02 + 0.01) / 10)
    assert s.down_move() == pytest.approx((0.9 - 0.02 - 0.01) / 10)


def test_time_exit():
    p = ShotParams(leverage=10, taker=0, slip=0, max_bars=3)
    s = feed(Shot("BTCUSDT", 1, 0, p), [(100, 101, 99, 100)] * 5)
    assert s.reason == "time" and s.exit_time == 2 * BAR_MS and s.equity == pytest.approx(1.0)


def test_roundtrip():
    s = feed(Shot("BTCUSDT", 1, 0, FREE), [(100, 101, 99, 100)])
    assert Shot.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def test_config_rejects_stop_beyond_liquidation():
    BoldConfig(leverage=20)
    with pytest.raises(ValueError):
        BoldConfig(leverage=50)


def test_example_config_and_cli():
    assert load_bold_config("configs/bold.example.yaml") == BoldConfig()
    assert parse_args(["--mode", "bold-paper"]).mode == "bold-paper"


DAY = 1000 * DAY_MS


class Fake:
    def __init__(self, clock):
        self.clock, self.bars = clock, []

    async def klines(self, symbol, interval, limit, start=None):
        return [k for k in self.bars if k.open_time >= start and k.close_time < self.clock["now"]]

    async def funding(self, symbol, start, end):
        return []


def k(t, o, h, low, c):
    return Kline("BTCUSDT", t, t + BAR_MS - 1, o, h, low, c, 0)


async def test_paper_tries_one_at_a_time_and_restarts(tmp_path):
    clock = {"now": DAY + 30_000}
    m = Fake(clock)
    cfg = BoldConfig(out_dir=tmp_path, taker=0, slip=0)
    bp = BoldPaper(cfg, m, now=lambda: clock["now"])
    await bp.step()
    assert bp.shot is not None and bp.shot.start == DAY
    m.bars = [k(DAY, 100, 100, 100, 100), k(DAY + BAR_MS, 100, 111, 100, 110)]
    clock["now"] = DAY + 2 * BAR_MS + 1000
    bp2 = BoldPaper(cfg, m, now=lambda: clock["now"])  # restart from state.json
    await bp2.step()
    assert bp2.shot is None and bp2.counts["target"] == 1 and bp2.pnl == pytest.approx(100)
    # same day: no new try; next day: a new one
    await bp2.step()
    assert bp2.shot is None
    clock["now"] = DAY + DAY_MS + 30_000
    await bp2.step()
    assert bp2.shot is not None and bp2.shot.start == DAY + DAY_MS
    rows = (tmp_path / "tries.csv").read_text().splitlines()
    assert len(rows) == 2 and ",target," in rows[1]
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["tries"] == 1 and st["hit_rate"] == 1.0
