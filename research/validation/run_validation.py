# ruff: noqa: E501
"""Run the frozen-candidate audit, reproduction, reconciliation and capacity replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit import audit
from .portfolio import default_scenarios, replay
from .reconcile import reconcile
from .reproduce import reproduce

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/validation"
OUT = ROOT / "research/validation/results"


def manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA.glob("monthly/**/*.zip")):
        rows.append(
            {
                "path": str(path.relative_to(DATA)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for path in sorted((DATA / "runs").glob("**/*.csv")):
        rows.append(
            {
                "path": str(path.relative_to(DATA)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for path in (DATA / "ticks.json",):
        rows.append(
            {
                "path": str(path.relative_to(DATA)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for path in sorted((DATA / "forward").glob("**/*")):
        if path.is_file() and path.suffix != ".part":
            rows.append(
                {
                    "path": str(path.relative_to(DATA)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "data_manifest.csv", index=False)
    return frame


def capacity() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(DATA / "runs/E5_capitulation/VDA.csv")
    clusters = (
        trades.assign(event_id=trades.t // 300_000)
        .groupby("event_id")
        .agg(
            event_time=("t", "min"),
            candidates=("symbol", "size"),
            symbols=("symbol", "nunique"),
            independent_net_usdt=("net_usdt", "sum"),
            simultaneous_mae_usdt=("mae", lambda x: float((-x.clip(upper=0) * 2.97).sum())),
        )
        .reset_index()
    )
    clusters.to_csv(OUT / "portfolio_event_clusters.csv", index=False)
    all_rows = []
    for scenario in default_scenarios():
        for seed in range(50):
            detail, row = replay(trades, scenario, seed)
            all_rows.append(row)
            if seed == 0:
                detail.to_csv(OUT / f"capacity_{scenario.name}_trades.csv", index=False)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT / "capacity_seed_results.csv", index=False)
    summary = (
        raw.groupby("name")
        .agg(
            seeds=("seed", "size"),
            accepted_mean=("accepted", "mean"),
            accept_rate_mean=("accept_rate", "mean"),
            net_mean_usdt=("net_usdt", "mean"),
            net_p05_usdt=("net_usdt", lambda x: x.quantile(0.05)),
            net_p95_usdt=("net_usdt", lambda x: x.quantile(0.95)),
            return_mean_pct=("return_on_initial_capital_pct", "mean"),
            max_floating_dd_mean_pct=("max_floating_drawdown_pct", "mean"),
            max_margin_used=("max_margin_used", "max"),
            mean_margin_utilization=("mean_margin_utilization", "mean"),
            active_margin_utilization=("active_margin_utilization", "mean"),
            return_per_mae_risk=("return_per_mae_risk", "mean"),
            worst_event_usdt=("worst_event_usdt", "min"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "capacity_summary.csv", index=False)
    return raw, summary


def _fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def report(
    audit_result: dict,
    rec: dict,
    rep: dict,
    cap: pd.DataFrame,
    forward: pd.DataFrame,
    files: int,
) -> None:
    stress = pd.DataFrame(rep["scenarios"]).set_index("scenario")
    forward = forward.set_index("scenario")
    exact = rep["exact"]
    cap_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(r["name"]),
                _fmt(r["accepted_mean"], 1),
                _fmt(r["net_mean_usdt"], 2),
                f"[{_fmt(r['net_p05_usdt'], 2)}, {_fmt(r['net_p95_usdt'], 2)}]",
                _fmt(r["max_floating_dd_mean_pct"], 2) + "%",
                _fmt(r["max_margin_used"], 2),
                _fmt(r["mean_margin_utilization"] * 100, 4) + "%",
                _fmt(r["active_margin_utilization"] * 100, 2) + "%",
                _fmt(r["return_per_mae_risk"], 3),
                _fmt(r["worst_event_usdt"], 2),
            ]
        )
        + " |"
        for _, r in cap.iterrows()
    )
    audit_rows = "\n".join(
        f"| {f['id']} | {f['severity']} | {f['status']} | {f['topic']} | {f['evidence']} |"
        for f in audit_result["findings"]
    )
    adverse = stress.loc["penetration_0.2_stop_slip_0.3"]
    fwd = forward.loc["frozen"]
    fwd_adverse = forward.loc["penetration_0.2_stop_slip_0.3"]
    verdict = "FAIL" if adverse.avg_net_pct <= 0 else "INCONCLUSIVE"
    text = f"""# Kasegu Independent Validation & Capacity Report

驗證分支：`research/codex-validation`，基底 `8a30b681e97b48ba05af24eb0f3439bc18ff702e`。
驗證日期：2026-10-23。全程未啟動 live/paper broker，未讀取 API key，也未送出訂單。

## Executive verdict

| 對象 | 結論 | 原因 |
| --- | --- | --- |
| 現行 baseline | **FAIL** | E1 每筆淨值 −0.115%，零成本價格期望仍為 −0.009%，沒有扣成本正期望。 |
| VDA 訊號的一分鐘 OHLC 重現 | **PASS（限研究重現）** | {exact["matched_keys"]}/{exact["published_trades"]} 筆 key 對上；最大 net 差 {_fmt(exact["max_abs_net_diff_pct"], 12)}%。 |
| 2026-10 未使用短期資料 | **PASS WITH LIMITS** | 22 個完整日、{int(fwd.trades)} 筆，凍結設定平均 {_fmt(fwd.avg_net_pct)}%；期間短且仍是 OHLC。 |
| aggTrades 引擎一致性 | **PASS WITH LARGE ERROR** | 216 筆匹配，但引擎平均僅為分鐘模擬的 {rec["engine_to_sim_expectancy_ratio"]:.1%}；另有 {rec["sim_only"]} / {rec["engine_only"]} 筆單邊成交。 |
| 真實 maker execution | **INCONCLUSIVE** | E4 沒有真實委託簿、排隊順位、撤單或部分成交；synthetic quote 不能證明 maker 成交。 |
| Portfolio capacity | **INCONCLUSIVE** | 已施加共用資本限制，但歷史 book depth 不存在，規模、漏成交與衝擊只能做敏感度，不能估計真實容量。 |
| 凍結候選 `VDA-capitulation-v1` | **{verdict} / DO NOT PROMOTE** | 多重搜尋、對照組不公平、maker 成交未知；保守執行壓力下每筆 {_fmt(adverse.avg_net_pct)}%，不足以完成可執行正期望的證明。 |

最重要的結論是：目前證據支持「極端急跌後有條件反彈」作為研究現象，但尚未支持
「能在 Binance 真實 maker 排隊、共用帳戶限制與危機滑價下實現正收益」。報告中的
`inconclusive` 不是中性推薦；在驗收標準下它等同不得進入實盤。

## 1. Research reliability audit

| ID | 嚴重度 | 判定 | 主題 | 證據 |
| --- | --- | --- | --- | --- |
{audit_rows}

### Look-ahead 與選模

未發現 `sim.py` 在單次訊號計算中直接讀取未來 K 線：訊號在已收盤分鐘形成，成交從下一分鐘判定。
`quantile` 的本根低／高價在本根收盤時已知。但研究流程有嚴重的 selection bias：至少 311 個明列組合與診斷，
再選出 `k=12/sl=3/long/against`，而報告沒有 family-wise correction。E2 的 2024–25 年資料在發現候選時確實未使用，
但它是向歷史更早期間回看，後續 E3/E5 又重複使用全部 24 個月，因此不能繼續稱為未使用資料。

### 隨機對照

`seed + hash(symbol)` 受 Python hash salt 影響，本次以兩個 `PYTHONHASHSEED` 得到
`{audit_result["salted_hash_demo"]}`。更關鍵的是 random control 在下一根 open 強制成交，卻套 maker 費且無入場滑價；
策略組則必須穿過限價才成交。E1 想配相同筆數，最終仍是 46,052 vs 42,094。它不是 execution-matched control，
因此「隨機組虧損」不能單獨證明策略 timing 的因果優勢。

### E1–E5 逐實驗判讀

| 實驗 | 可支持的結論 | 不能支持的結論／主要缺口 |
| --- | --- | --- |
| E1 baseline | 原始 baseline 在既有 OHLC 模型中穩健偏弱，連未扣成本價格期望也略負。 | 隨機對照不可重現且執行條件不同，不能作為 timing 因果證據。 |
| E2 跨期搜尋 | `k=12` 在較早年度與新幣集合仍呈正值，值得形成候選假說。 | 是回看歷史、沒有 multiple-testing correction，且沒有 maker book 證據。 |
| E3 鄰域／成本 | 相鄰參數仍有正值，並揭露結果對成交與停損滑價的敏感度。 | 穿價只是 proxy；L0 加 0.3% 停損／trail 滑價後由 +0.178% 變 −0.018%。 |
| E4 aggTrades engine | 能量化分鐘模擬與逐筆重播的交易、價格及收益差異。 | 樣本由原模擬已成交 coin-days 篩出、排除 BTC/ETH、使用合成報價且逐幣獨立，不能推估全體 fill/capacity。 |
| E5 capitulation filters | VDA 在既有資料可逐筆精確重現，條件後的歷史期望較高。 | 門檻由 Y1 選出並重用資料；E4 驗證的是 L0，不是 VDA，因此原報告的「real execution 仍獲利」表述過強。 |

## 2. Simulator levels and execution error

1. `research/sim.py`：一分鐘 OHLC，對棒內路徑作固定 O-L-H-C / O-H-L-C 假設。
2. E4 engine replay：逐筆 aggTrade，但 bid/ask 是 last trade ± half tick 合成；沒有真實 book。
3. Shadow order-book observation：只有未來啟動 collector 後才存在。過去狀態明確標為 `unknown_book`。

逐筆對帳在 [trade_reconciliation.csv](validation/results/trade_reconciliation.csv)，誤差統計在
[execution_error.csv](validation/results/execution_error.csv)。匹配交易的 simulator 平均
{rec["matched_sim_avg_pct"]:.3f}%，engine 平均 {rec["matched_engine_avg_pct"]:.3f}%，平均誤差
{rec["matched_mean_error_pct"]:.3f}%，平均絕對誤差 {rec["matched_abs_error_pct"]:.3f}%。
E4 還條件式抽取「原模擬已成交」的 coin-days、排除 BTC/ETH、逐幣獨立執行，不能推廣成總體 fill rate 或容量估計。

## 3. Frozen alpha reproduction and adverse execution

| 情境 | 交易數 | 每筆淨值 | 獨立帳戶合計 USDT | 正幣種 | 正月份 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | {int(stress.loc["frozen", "trades"])} | {_fmt(stress.loc["frozen", "avg_net_pct"])}% | {_fmt(stress.loc["frozen", "net_usdt_independent_accounts"], 2)} | {int(stress.loc["frozen", "positive_symbols"])} | {int(stress.loc["frozen", "positive_months"])} |
| 穿價 0.2% | {int(stress.loc["penetration_0.2pct", "trades"])} | {_fmt(stress.loc["penetration_0.2pct", "avg_net_pct"])}% | {_fmt(stress.loc["penetration_0.2pct", "net_usdt_independent_accounts"], 2)} | {int(stress.loc["penetration_0.2pct", "positive_symbols"])} | {int(stress.loc["penetration_0.2pct", "positive_months"])} |
| 停損額外滑 0.3% | {int(stress.loc["stop_slip_0.3pct", "trades"])} | {_fmt(stress.loc["stop_slip_0.3pct", "avg_net_pct"])}% | {_fmt(stress.loc["stop_slip_0.3pct", "net_usdt_independent_accounts"], 2)} | {int(stress.loc["stop_slip_0.3pct", "positive_symbols"])} | {int(stress.loc["stop_slip_0.3pct", "positive_months"])} |
| 兩者同時 | {int(adverse.trades)} | {_fmt(adverse.avg_net_pct)}% | {_fmt(adverse.net_usdt_independent_accounts, 2)} | {int(adverse.positive_symbols)} | {int(adverse.positive_months)} |

這些 adverse variants 只改 execution，沒有調整凍結 alpha 門檻。即使它們仍為正，也不能補上 queue 的未知資料；
「價格穿過 0.2%」只是保守 proxy，不是排隊模型的校準值。

### 2026-10 forward holdout

先前研究資料止於 2026-09。本次另取當時未參與選模的 2026-10-01 至 2026-10-22 完整公開日檔，
保留 2026-09 只作 rolling warm-up，並以同一份凍結設定、同一個模擬器執行。

| 情境 | 完整日 | 交易數 | 每筆淨值 | 獨立帳戶合計 USDT | 正幣種 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | {int(fwd.days)} | {int(fwd.trades)} | {_fmt(fwd.avg_net_pct)}% | {_fmt(fwd.net_usdt_independent_accounts, 2)} | {int(fwd.positive_symbols)} |
| 穿價 0.2% + 停損滑價 0.3% | {int(fwd_adverse.days)} | {int(fwd_adverse.trades)} | {_fmt(fwd_adverse.avg_net_pct)}% | {_fmt(fwd_adverse.net_usdt_independent_accounts, 2)} | {int(fwd_adverse.positive_symbols)} |

這段資料對「急跌後條件反彈」提供新的支持證據，但只有 22 天、單一市場 regime，且仍缺少歷史 order book。
它能降低純粹重用舊資料的疑慮，不能單獨證明 maker execution、容量或可持續性；最終判定維持 **INCONCLUSIVE / DO NOT PROMOTE**。

## 4. Shared portfolio and capacity sensitivity

Portfolio replay 使用同一個 1,000 USDT 初始資本、全域 margin/notional/position 限制，並把同一 5 分鐘的信號視為共同市場事件。
同時訊號以預先固定 SHA-256 priority、50 個 seed 測試，不用事後收益選單。`max_floating_drawdown` 將所有同時持倉視為一起到達
各自記錄的 MAE，是偏保守的 event stress。歷史委託簿仍未知，因此以下是 sensitivity envelope。

| 情境 | 平均接受筆數 | 平均淨 USDT | 5–95% seed | 平均最大浮動 DD | 最大 margin | 全期 margin 使用率 | 持倉期間使用率 | 淨利 / MAE 風險 | 最差共同事件 USDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{cap_rows}

逐情境逐筆結果與 seed 分布位於 `research/validation/results/capacity_*`。
未套全域限制前的五分鐘共同事件分組在 [portfolio_event_clusters.csv](validation/results/portfolio_event_clusters.csv)，
包含每個事件的候選幣數、獨立帳戶合計收益及同時 MAE；套限制後可由各情境逐筆檔的 `event_id` 對帳。
`ohlc_upper_bound` 保留原 OHLC 全成交，應視為上界；其餘 partial/miss/impact 數字是壓力假設，
不是從歷史深度估出的成交率。不同名義規模的結論仍是 **capacity unknown**。

## 5. Shadow Market Data Collector

`research/validation/shadow_collector.py` 只使用 public REST depth 與 public WebSocket，沒有任何下單端點。
它先緩衝 diff depth、取得 snapshot，驗證 `U/u/pu` 連續性，記錄 snapshot、增量、BBO、spread、depth、aggTrades、
book gap 與候選單生命週期到 append-only JSONL。`maker.py` 只在 book 已同步後啟用候選單；排隊前方數量不因取消而樂觀減少，
只由對向主動成交消耗，支援部分成交、固定延遲與延遲敏感度。未觀測狀態回傳 `unknown_book`。

collector 尚未在本次驗證中啟動，所以沒有把未來觀測冒充歷史證據。正式評估至少需涵蓋多次急跌事件，並事先固定 latency、
queue multiplier、cancel timing 與 fill rule；在那之前不得寫成「真實 maker 成交」。

## 6. Reproducibility

本次資料快照共有 {files} 個檔案，SHA-256 與大小見 [data_manifest.csv](validation/results/data_manifest.csv)。
快照位於 gitignored `data/validation/`，報告中的 manifest 能偵測變更但不能代替原始 Binance archive。

```bash
PYTHONPATH=research .venv/bin/python -m validation.forward_holdout
PYTHONPATH=research .venv/bin/python -m validation.run_validation --reproduce
.venv/bin/python -m pytest tests/test_validation.py -q
.venv/bin/python -m ruff check research/validation tests/test_validation.py
```

只重建報告與 capacity（沿用已產生 reproduction）可省略 `--reproduce`。

官方同步規則參考：
[Binance local order book](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly)。
官方要求先緩衝 stream、取 snapshot、首筆覆蓋 `lastUpdateId`，其後每筆 `pu` 必須等於前一筆 `u`；本 collector 依此實作。
"""
    (ROOT / "research/validation_report.md").write_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reproduce", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_result = audit(ROOT, OUT)
    rec = reconcile(
        DATA / "runs/E4_engine/matched.csv",
        DATA / "runs/E4_engine/engine_trades.csv",
        OUT,
    )
    reproduction_file = OUT / "reproduction_bundle.json"
    if args.reproduce:
        rep = reproduce(
            DATA,
            DATA / "runs/E5_capitulation/VDA.csv",
            OUT,
        )
        reproduction_file.write_text(json.dumps(rep, indent=2) + "\n")
    elif reproduction_file.exists():
        rep = json.loads(reproduction_file.read_text())
    else:
        raise SystemExit("Run once with --reproduce to create frozen-candidate evidence")
    _, cap = capacity()
    forward_file = OUT / "forward_holdout_summary.csv"
    if not forward_file.exists():
        raise SystemExit("Run validation.forward_holdout before building the report")
    forward = pd.read_csv(forward_file)
    files = len(manifest())
    report(audit_result, rec, rep, cap, forward, files)


if __name__ == "__main__":
    main()
