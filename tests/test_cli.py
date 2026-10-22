from kasegu.__main__ import main, parse_args


def test_default_mode_is_paper():
    assert parse_args([]).mode == "paper"


def test_live_requires_confirm():
    assert main(["--mode", "live"]) == 2


def test_missing_config(tmp_path):
    assert main(["--config", str(tmp_path / "none.yaml")]) == 2


def test_symbols_override():
    assert parse_args(["--symbols", "solusdt,dogeusdt"]).symbols == "solusdt,dogeusdt"
