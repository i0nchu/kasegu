from kasegu.__main__ import main, parse_args


def test_default_mode_is_paper():
    assert parse_args([]).mode == "paper"


def test_live_requires_confirm():
    assert main(["--mode", "live"]) == 2


def test_missing_config(tmp_path):
    assert main(["--config", str(tmp_path / "none.yaml")]) == 2


def test_symbols_override():
    assert parse_args(["--symbols", "solusdt,dogeusdt"]).symbols == "solusdt,dogeusdt"


def test_strategy_override(monkeypatch):
    import kasegu.app

    seen = []

    async def fake(cfg, label=""):
        seen.append(cfg.strategy)
        raise SystemExit(0)

    monkeypatch.setattr(kasegu.app, "run_backtest", fake)
    base = ["--config", "configs/config.example.yaml", "--mode", "backtest"]
    assert main(base + ["--strategy", "nope"]) == 2
    assert main(base + ["--k", "0"]) == 2
    try:
        main(base + ["--strategy", "quantile", "--k", "1.2"])
    except SystemExit:
        pass
    assert seen[0].id == "quantile" and seen[0].k == 1.2 and seen[0].params == {}
