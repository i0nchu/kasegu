"""Readable backtest reports and multi-run comparison.

Usage: uv run python -m kasegu.backtest.report [files or dirs ...] [--sort net] [--csv out.csv]
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def run_name(
    symbols: list[str],
    start: str,
    end: str,
    strategy: str,
    k: float,
    label: str = "",
    now: datetime | None = None,
) -> str:
    """File stem that says what was tested: label, symbols, period, model, time."""
    syms = "+".join(s.removesuffix("USDT") for s in symbols[:3])
    if len(symbols) > 3:
        syms += f"+{len(symbols) - 3}more"
    period = f"{start.replace('-', '')}-{end.replace('-', '')}"
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    parts = [label, syms, period, f"{strategy}-k{k:g}", stamp]
    return re.sub(r"[^\w.+-]", "_", "_".join(p for p in parts if p))


def _pct(v, digits=1):
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _num(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def row(rep: dict) -> dict:
    """One flat row per report for tables and CSV."""
    r, st = rep["result"], rep["strategy"]
    return {
        "name": rep.get("name", ""),
        "label": rep.get("label", ""),
        "symbols": ",".join(rep["symbols"]),
        "period": f"{rep['start']}~{rep['end']}",
        "strategy": f"{st['id']} k={st['k']:g}",
        "trades": r["trades"],
        "fill_rate": r["fill_rate"],
        "net": r["net"],
        "gross": r["gross"],
        "fees": r["fees"],
        "slippage": r["slippage"],
        "funding": r["funding"],
        "max_dd": r["max_drawdown"],
        "pf": r["profit_factor"],
        "win_rate": r["win_rate"],
        "payoff": r["payoff"],
        "mfe": r["avg_mfe_pct"],
        "mae": r["avg_mae_pct"],
        "hold_min": r["avg_hold_min"],
        "time_exit": r["exit_reasons"].get("max_hold", 0) / r["trades"] if r["trades"] else None,
    }


COLUMNS = [
    # header, key, formatter
    ("symbols", "symbols", str),
    ("period", "period", str),
    ("strategy", "strategy", str),
    ("trades", "trades", str),
    ("fill", "fill_rate", lambda v: _pct(v, 2)),
    ("net", "net", _num),
    ("gross", "gross", _num),
    ("fees", "fees", _num),
    ("maxDD", "max_dd", _num),
    ("PF", "pf", _num),
    ("win", "win_rate", _pct),
    ("payoff", "payoff", _num),
    ("MFE%", "mfe", _num),
    ("MAE%", "mae", _num),
    ("hold", "hold_min", lambda v: _num(v, 1)),
    ("time_exit", "time_exit", _pct),
]


def table(rows: list[dict], cols=COLUMNS) -> str:
    cells = [[h for h, _, _ in cols]] + [[f(r[k]) for _, k, f in cols] for r in rows]
    width = [max(len(c[i]) for c in cells) for i in range(len(cols))]
    lines = []
    for n, c in enumerate(cells):
        lines.append(
            "  ".join(
                v.ljust(w) if i < 3 else v.rjust(w)
                for i, (v, w) in enumerate(zip(c, width, strict=True))
            )
        )
        if n == 0:
            lines.append("  ".join("-" * w for w in width))
    return "\n".join(lines)


def summary(rep: dict) -> str:
    """Human readable summary of a single run."""
    r = rep["result"]
    x = row(rep)
    parts = [x["label"], x["symbols"], x["period"], x["strategy"]]
    head = " | ".join(p for p in parts if p)
    reasons = ", ".join(
        f"{k} {v}" for k, v in sorted(r["exit_reasons"].items(), key=lambda kv: -kv[1])
    )
    lines = [
        "=" * len(head),
        head,
        "=" * len(head),
        f"trades {x['trades']}  (entries {r['entries']}, fill {_pct(x['fill_rate'], 2)})",
        f"net {_num(x['net'])}  = gross {_num(x['gross'])} - fees {_num(x['fees'])}"
        f" + funding {_num(x['funding'], 4)}   (slippage in gross {_num(x['slippage'])})",
        f"max drawdown {_num(x['max_dd'])}  PF {_num(x['pf'])}  win {_pct(x['win_rate'])}"
        f"  payoff {_num(x['payoff'])}",
        f"MFE {_num(x['mfe'])}%  MAE {_num(x['mae'])}%  hold {_num(x['hold_min'], 1)} min",
        f"exits: {reasons or '-'}",
    ]
    if r["by_group"]:
        lines.append("")
        g_cols = [
            ("group", "g", str),
            ("trades", "trades", str),
            ("net", "net", _num),
            ("win", "win_rate", _pct),
            ("MFE%", "avg_mfe_pct", _num),
            ("MAE%", "avg_mae_pct", _num),
        ]
        g_rows = [{"g": g, **v} for g, v in r["by_group"].items()]
        lines.append(table(g_rows, g_cols))
    if rep.get("file"):
        lines += ["", f"saved: {rep['file']}"]
    return "\n".join(lines)


def load(paths: list[str]) -> list[dict]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob("*.json"))
        elif p.is_file():
            files.append(p)
    out = []
    for f in files:
        rep = json.loads(f.read_text())
        if "result" in rep:
            rep.setdefault("name", f.stem)
            out.append(rep)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kasegu.backtest.report")
    p.add_argument("paths", nargs="*", default=["data/backtests"])
    p.add_argument("--sort", default="net", help="column key to sort by (desc)")
    p.add_argument("--csv", help="also write the table to this CSV file")
    args = p.parse_args(argv)

    rows = [row(r) for r in load(args.paths)]
    if not rows:
        print("no reports found")
        return 1
    vals = [r.get(args.sort) for r in rows if r.get(args.sort) is not None]
    if vals and all(isinstance(v, int | float) for v in vals):
        rows.sort(key=lambda r: (r.get(args.sort) is None, -(r.get(args.sort) or 0)))
    else:
        rows.sort(key=lambda r: str(r.get(args.sort, "")))
    print(table(rows))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\ncsv: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
