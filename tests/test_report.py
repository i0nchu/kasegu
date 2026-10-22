import json
from datetime import datetime

from kasegu.__main__ import main as cli
from kasegu.backtest import report
from kasegu.backtest.metrics import summarize
from kasegu.core.position import Trade
from kasegu.core.types import Side
from kasegu.market.data import MINUTE


def trade(net, sym="SOLUSDT", reason="max_hold"):
    return Trade(sym, Side.BUY, 1, 100, 100 + net, 0, MINUTE, net, 0, 0, 1.0, -0.5, reason)


def rep(sym, net, label=""):
    return {
        "name": f"{label}_{sym}",
        "label": label,
        "strategy": {"id": "atr_amplitude", "k": 3.0},
        "symbols": [sym],
        "start": "2026-10-01",
        "end": "2026-10-07",
        "result": summarize([trade(net, sym)], 4, 1, 0.02),
    }


def test_run_name():
    now = datetime(2026, 10, 22, 17, 10)
    n = report.run_name(["SOLUSDT"], "2026-10-01", "2026-10-07", "atr_amplitude", 3.0, "exp 1", now)
    assert n == "exp_1_SOL_20261001-20261007_atr_amplitude-k3_20261022-171000"
    many = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "WIFUSDT", "SUIUSDT"]
    n = report.run_name(many, "2026-10-01", "2026-10-07", "atr_only", 2.5, now=now)
    assert n.startswith("BTC+ETH+SOL+2more_20261001-20261007_atr_only-k2.5_")


def test_summary_and_table():
    s = report.summary(rep("SOLUSDT", 1.5, "exp1"))
    assert "exp1 | SOLUSDT | 2026-10-01~2026-10-07 | atr_amplitude k=3" in s
    assert "exits: max_hold 1" in s
    t = report.table([report.row(rep("SOLUSDT", 1.5)), report.row(rep("WIFUSDT", -2))])
    lines = t.splitlines()
    assert lines[0].startswith("symbols") and len(lines) == 4
    assert "100.0%" in t  # time_exit


def test_report_cli(tmp_path, capsys):
    for sym, net in [("SOLUSDT", -1), ("WIFUSDT", 2)]:
        (tmp_path / f"{sym}.json").write_text(json.dumps(rep(sym, net)))
    (tmp_path / "other.json").write_text("{}")
    out = tmp_path / "cmp.csv"
    assert report.main([str(tmp_path), "--csv", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.index("WIFUSDT") < text.index("SOLUSDT")
    assert out.read_text().count("\n") == 3
    assert report.main([str(tmp_path / "none")]) == 1


def test_backtest_each(tmp_path, monkeypatch, capsys):
    import kasegu.app

    seen = []

    async def fake(cfg, label=""):
        seen.append((cfg.symbols, label))
        return rep(cfg.symbols[0], len(seen), label)

    monkeypatch.setattr(kasegu.app, "run_backtest", fake)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(open("configs/config.example.yaml").read())
    argv = ["--config", str(cfg), "--mode", "backtest", "--symbols", "SOLUSDT,WIFUSDT"]
    assert cli(argv + ["--each", "--label", "x"]) == 0
    assert seen == [(["SOLUSDT"], "x"), (["WIFUSDT"], "x")]
    out = capsys.readouterr().out
    assert out.count("x | ") == 2 and "symbols" in out


def test_params_in_name_and_row():
    now = datetime(2026, 10, 22)
    n = report.run_name(
        ["SOLUSDT"], "2026-10-01", "2026-10-07", "quantile", 1.0, now=now, params={"q": 0.9}
    )
    assert "_quantile-k1-q0.9_" in n
    r = rep("SOLUSDT", 1)
    r["strategy"] = {"id": "quantile", "k": 1.0, "params": {"q": 0.9, "lookback": 720}}
    assert report.row(r)["strategy"] == "quantile k=1 lookback=720 q=0.9"
