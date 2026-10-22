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


def test_live_needs_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        open(__file__.replace("tests/test_cli.py", "configs/config.example.yaml")).read()
    )
    assert main(["--mode", "live", "--confirm-live", "--config", str(cfg)]) == 2


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("KASEGU_MODE", "live")
    monkeypatch.setenv("KASEGU_CONFIRM_LIVE", "1")
    monkeypatch.setenv("KASEGU_CONFIG", "/app/configs/config.yaml")
    a = parse_args([])
    assert a.mode == "live" and a.confirm_live and a.config == "/app/configs/config.yaml"
    monkeypatch.setenv("KASEGU_CONFIRM_LIVE", "0")
    assert not parse_args([]).confirm_live
