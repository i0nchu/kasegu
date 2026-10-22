from kasegu.__main__ import main, parse_args


def test_default_mode_is_paper():
    assert parse_args([]).mode == "paper"


def test_live_requires_confirm():
    assert main(["--mode", "live"]) == 2
