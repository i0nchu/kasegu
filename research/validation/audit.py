# ruff: noqa: E501
"""Evidence-backed methodology audit of sim.py and E1-E5."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _salted_hashes() -> list[int]:
    values = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.check_output(
            [sys.executable, "-c", "print(hash('BTCUSDT') % 10000)"], env=env, text=True
        )
        values.append(int(out))
    return values


def audit(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sim = (root / "research/sim.py").read_text()
    runs = (root / "research/runs.py").read_text()
    e4 = (root / "research/e4_engine.py").read_text()
    salted = _salted_hashes()
    e1 = pd.read_csv(root / "research/results/E1_decompose/table.csv").set_index("group")
    e3 = pd.read_csv(root / "research/results/E3_robust/table.csv").set_index("group")
    e5 = pd.read_csv(root / "research/results/E5_capitulation/table.csv").set_index("group")
    findings = [
        {
            "id": "A01",
            "severity": "high",
            "topic": "random_control_reproducibility",
            "status": "fail",
            "evidence": f"runs.py uses hash(symbol); PYTHONHASHSEED 1/2 produced {salted}",
            "impact": "Published random controls are not bit-reproducible across Python processes.",
        },
        {
            "id": "A02",
            "severity": "high",
            "topic": "random_control_fairness",
            "status": "fail",
            "evidence": (
                "Random entries are forced at the next bar open via entries=... while simulate() "
                "charges fee_in=maker and applies no entry slippage; E1 baseline/control counts are "
                f"{int(e1.loc['G0_baseline', 'trades'])}/{int(e1.loc['C0_random_baseline_exit', 'trades'])}."
            ),
            "impact": "The control is useful descriptively but is not an execution-matched causal control.",
        },
        {
            "id": "A03",
            "severity": "high",
            "topic": "synthetic_maker_fill",
            "status": "fail",
            "evidence": (
                "sim.py and PaperBroker treat last trade +/- half tick as quotes and fill after a "
                "trade-through; no historical depth, queue position, cancellations or partial fills exist."
            ),
            "impact": "E4 is aggTrade replay, not real maker-execution evidence.",
        },
        {
            "id": "A04",
            "severity": "high",
            "topic": "parameter_search_multiplicity",
            "status": "fail",
            "evidence": (
                "At least 311 explicit configurations/diagnostics appear in drift (20), grid (144), "
                "breakout (96), extreme (18), E3 (25) and E5 (8), before undocumented human choices."
            ),
            "impact": "Nominal holdout language overstates independence; no family-wise selection correction exists.",
        },
        {
            "id": "A05",
            "severity": "medium",
            "topic": "lookahead",
            "status": "pass_with_limits",
            "evidence": (
                "Signals use closed-bar values and orders are evaluated from the following minute. "
                "The rolling quantile includes the just-completed bar outcome, which is known at decision time."
            ),
            "impact": "No direct bar-level future read was found, but research-selection lookahead remains.",
        },
        {
            "id": "A06",
            "severity": "medium",
            "topic": "missing_minutes",
            "status": "inconclusive",
            "evidence": "load_klines forward-fills absent minutes with a flat synthetic OHLC bar.",
            "impact": "Exchange/data outages can be converted into tradable-looking continuity unless separately audited.",
        },
        {
            "id": "A07",
            "severity": "high",
            "topic": "stop_slippage",
            "status": "fail",
            "evidence": (
                f"E3 lead expectancy {e3.loc['L0_lead', 'avg_net%']:.3f}% becomes "
                f"{e3.loc['S2_slip0.3', 'avg_net%']:.3f}% with 0.3% adverse stop/trail slippage."
            ),
            "impact": "The claimed edge is not robust to a plausible but unmeasured crash-execution stress.",
        },
        {
            "id": "A08",
            "severity": "high",
            "topic": "portfolio_capacity",
            "status": "fail",
            "evidence": (
                "runs.py executes one independent account per symbol and sums net_usdt; the production "
                "default has max_positions=1 and max_notional=350."
            ),
            "impact": "Published total USDT, drawdown and capacity are not portfolio-realizable metrics.",
        },
        {
            "id": "A09",
            "severity": "medium",
            "topic": "engine_sample_selection",
            "status": "inconclusive",
            "evidence": (
                "E4 samples only coin-days where the OHLC lead already traded, excludes BTC/ETH, "
                "and replays each coin-day separately."
            ),
            "impact": "It measures conditional simulator disagreement, not population fill rate or shared capacity.",
        },
        {
            "id": "A10",
            "severity": "medium",
            "topic": "frozen_candidate",
            "status": "pass_with_limits",
            "evidence": (
                f"VDA is frozen at commit 36552a3: {int(e5.loc['VDA', 'trades'])} trades, "
                f"published mean {e5.loc['VDA', 'avg_net%']:.3f}%."
            ),
            "impact": "Independent validation can stress execution but must not tune alpha thresholds.",
        },
    ]
    assertions = {
        "sim_uses_next_minute_for_fills": "s_prev, lim_prev" in sim,
        "sim_has_no_order_book": "queue" not in sim.lower() and "depth" not in sim.lower(),
        "control_uses_salted_hash": "hash(symbol)" in runs,
        "e4_conditioned_on_lead_trades": 'RUNS / "E3_robust" / "L0_lead"' in e4,
        "e4_excludes_major_symbols": 'SKIP = {"BTCUSDT", "ETHUSDT"}' in e4,
    }
    result = {"salted_hash_demo": salted, "assertions": assertions, "findings": findings}
    (out / "audit_findings.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(findings).to_csv(out / "audit_findings.csv", index=False)
    return result
