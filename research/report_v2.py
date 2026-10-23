"""Build the round 2 report (HTML) from research/results and the full logs in data/research/runs.

Usage: uv run python research/report_v2.py out.html
"""

import html
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sim import Cfg

RES = Path("research/results")
RUNS = Path("data/research/runs")
COMMITS = {
    "E1_decompose": "8206de4",
    "E2_holdout": "e83d379",
    "E3_robust": "eacb86f",
    "E4_engine": "ea3f6fb",
    "E5_capitulation": "36552a3",
}
DEFAULT = asdict(Cfg())
LABELS = {
    "k": "k",
    "trend": "趨勢",
    "sides": "方向",
    "sl": "停損%",
    "max_hold": "最長持倉(分)",
    "exit": "出場",
    "tp": "停利%",
    "maker": "掛單費%",
    "taker": "吃單費%",
    "slip": "滑價%",
    "level_slip": "停損額外滑價%",
    "penetrate": "需穿過%",
    "ema": "EMA",
    "vol": "波動視窗",
}


def esc(x) -> str:
    return html.escape(str(x))


def pct(v, d=3) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:+.{d}f}%"


def cls(v) -> str:
    return "" if v is None or pd.isna(v) else ("gain" if v > 0 else "loss")


def snapshot(cfg: dict) -> str:
    """Settings that differ from the engine baseline, plus filters and period."""
    parts = [
        f"{LABELS.get(k, k)}={v}" for k, v in cfg.items() if k in LABELS and v != DEFAULT.get(k)
    ]
    ex = cfg.get("extra") or {}
    for k, v in ex.items():
        if k == "months":
            parts.append(f"期間={v[0]}～{v[-1]}")
        else:
            parts.append(f"{k}={v}")
    return "、".join(parts) or "引擎現行設定"


def group_rows(exp: str) -> str:
    rows = []
    for g in sorted((RES / exp).iterdir()):
        if not g.is_dir():
            continue
        c = json.loads((g / "config.json").read_text())
        s = json.loads((g / "summary.json").read_text())
        e = s["expectancy"]
        cand = s.get("candidates") or {}
        if c.get("control_random_entry"):
            cand_txt = f"隨機進場 {cand.get('random_entries', 0):,} 次（無掛單）"
        elif cand:
            cand_txt = "<br>".join(f"{esc(k)} {v:,}" for k, v in sorted(cand.items()))
        else:
            cand_txt = "只記錄成交（掃描組）"
        reasons = "、".join(f"{k} {v}" for k, v in s["exit_reasons"].items())
        rows.append(
            f"<tr><td><b>{esc(g.name)}</b><div class='sub'>{esc(c.get('note', ''))}</div>"
            f"<div class='snap'>{esc(snapshot(c['cfg']))}</div></td>"
            f"<td class='r'>{e['trades']:,}</td>"
            f"<td class='r {cls(s['avg_net_pct'])}'>{pct(s['avg_net_pct'])}</td>"
            f"<td class='r'>{e['win_rate'] * 100:.1f}%</td>"
            f"<td class='r'>{(e['payoff'] or 0):.2f}</td>"
            f"<td class='r'>{(e['breakeven_win_rate'] or 0) * 100:.1f}%</td>"
            f"<td class='r {cls(s['net_usdt'])}'>{s['net_usdt']:+,.0f}</td>"
            f"<td class='r'>{s['pos_months']}/{s['months']}</td>"
            f"<td class='small'>{cand_txt}</td><td class='small'>{esc(reasons)}</td></tr>"
        )
    head = (
        "<tr><th>組別與設定快照</th><th class='r'>筆數</th><th class='r'>每筆淨</th>"
        "<th class='r'>勝率</th><th class='r'>盈虧比</th><th class='r'>打平勝率</th>"
        "<th class='r'>合計 USDT</th><th class='r'>正月</th>"
        "<th>候選訂單結果</th><th>出場原因</th></tr>"
    )
    body = "".join(rows)
    return (
        f"<div class='tbl'><table class='groups'><thead>{head}</thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def trades_json() -> str:
    cols = [
        "symbol",
        "t",
        "side",
        "entry",
        "exit",
        "reason",
        "gross",
        "net",
        "mfe",
        "mae",
        "hold",
        "vol_ratio",
        "ret60",
        "atr_pct",
    ]
    out = {}
    for exp, g in (("E5_capitulation", "VDA"), ("E5_capitulation", "VDA_ctrl")):
        tr = pd.read_csv(RUNS / exp / g / "trades.csv")[cols]
        tr["t"] = pd.to_datetime(tr.t, unit="ms").dt.strftime("%Y-%m-%d %H:%M")
        out[f"{exp}/{g}"] = tr.round(5).values.tolist()
    m = pd.read_csv(RES / "E4_engine" / "matched.csv")
    m["m"] = pd.to_datetime(m.m, unit="ms").dt.strftime("%Y-%m-%d %H:%M")
    m["_merge"] = m["_merge"].map(
        {"both": "兩邊都成交", "left_only": "只有模擬成交", "right_only": "只有引擎成交"}
    )
    out["E4"] = m.round(4).astype(object).where(m.notna(), None).values.tolist()
    return json.dumps(out, ensure_ascii=False)


def bars(items, width=640, unit="%", fmt="{:+.3f}", height=26) -> str:
    """Horizontal bars around zero: items = [(label, value, highlight)]."""
    vmax = max(abs(v) for _, v, _ in items) or 1
    lab_w, mid = 230, 230 + (width - 230 - 70) / 2
    half = (width - 230 - 70) / 2
    h = height * len(items) + 24
    out = [
        f"<svg viewBox='0 0 {width} {h}' width='100%' role='img'>",
        f"<line x1='{mid}' y1='4' x2='{mid}' y2='{h - 16}' stroke='var(--ink)'/>",
    ]
    for i, (lab, v, hi) in enumerate(items):
        y = 8 + i * height
        w = abs(v) / vmax * half
        x = mid if v >= 0 else mid - w
        col = "var(--gain)" if v > 0 else "var(--loss)"
        op = "1" if hi else "0.55"
        out.append(f"<text x='{lab_w - 8}' y='{y + 13}' text-anchor='end'>{esc(lab)}</text>")
        out.append(
            f"<rect x='{x:.1f}' y='{y}' width='{max(w, 1):.1f}' height='{height - 9}' "
            f"fill='{col}' opacity='{op}'/>"
        )
        tx = x + w + 4 if v >= 0 else x - 4
        anchor = "start" if v >= 0 else "end"
        out.append(
            f"<text class='val' x='{tx:.1f}' y='{y + 13}' text-anchor='{anchor}'>"
            f"{fmt.format(v)}{unit}</text>"
        )
    out.append("</svg>")
    return "".join(out)


def main(path: str):
    e1 = pd.read_csv(RES / "E1_decompose" / "table.csv").set_index("group")
    e3 = pd.read_csv(RES / "E3_robust" / "table.csv").set_index("group")
    e5 = pd.read_csv(RES / "E5_capitulation" / "table.csv").set_index("group")
    e4 = json.loads((RES / "E4_engine" / "summary.json").read_text())
    tpl = Path("research/report_v2.html").read_text()
    chart_e1 = bars(
        [
            ("原策略進場，成本 0", e1.loc["G4_zero_cost", "avg_net%"], True),
            ("隨機進場，成本 0", e1.loc["C4_random_zero_cost", "avg_net%"], False),
            ("原策略進場 + 現行出場", e1.loc["G0_baseline", "avg_net%"], True),
            ("原策略進場 + 只留停損與時間", e1.loc["G1_time_exit", "avg_net%"], True),
            ("原策略進場 + 固定停利停損", e1.loc["G2_bracket", "avg_net%"], True),
            ("原策略進場 + 抱 30 分不停損", e1.loc["G3_hold_only", "avg_net%"], True),
            ("隨機進場 + 現行出場", e1.loc["C0_random_baseline_exit", "avg_net%"], False),
            ("隨機進場 + 抱 30 分", e1.loc["C3_random_hold_only", "avg_net%"], False),
        ]
    )
    chart_e3 = bars(
        [(f"{r.note}", r["avg_net%"], r.kind != "exit_ctrl") for _, r in e3.iterrows()], height=24
    )
    chart_e5 = bars([(r.note, r["y2_avg%"], g != "VDA_ctrl") for g, r in e5.iterrows()])
    sections = {
        exp: group_rows(exp)
        for exp in ("E1_decompose", "E2_holdout", "E3_robust", "E5_capitulation")
    }
    page = (
        tpl.replace("{{CHART_E1}}", chart_e1)
        .replace("{{CHART_E3}}", chart_e3)
        .replace("{{CHART_E5}}", chart_e5)
        .replace("{{TRADES}}", trades_json())
        .replace("{{E4}}", json.dumps(e4, ensure_ascii=False))
    )
    for exp, rows in sections.items():
        page = page.replace("{{GROUPS_" + exp + "}}", rows)
    for exp, h in COMMITS.items():
        page = page.replace("{{C_" + exp + "}}", h)
    Path(path).write_text(page)
    print("wrote", path, len(page) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1])
