# Kasegu Independent Validation & Capacity Report

驗證分支：`research/codex-validation`，基底 `8a30b681e97b48ba05af24eb0f3439bc18ff702e`。
驗證日期：2026-10-23。全程未啟動 live/paper broker，未讀取 API key，也未送出訂單。

## Executive verdict

| 對象 | 結論 | 原因 |
| --- | --- | --- |
| 現行 baseline | **FAIL** | E1 每筆淨值 −0.115%，零成本價格期望仍為 −0.009%，沒有扣成本正期望。 |
| VDA 訊號的一分鐘 OHLC 重現 | **PASS（限研究重現）** | 710/710 筆 key 對上；最大 net 差 0.000000000000%。 |
| 2026-10 未使用短期資料 | **PASS WITH LIMITS** | 22 個完整日、45 筆，凍結設定平均 0.671%；期間短且仍是 OHLC。 |
| aggTrades 引擎一致性 | **PASS WITH LARGE ERROR** | 216 筆匹配，但引擎平均僅為分鐘模擬的 42.1%；另有 9 / 8 筆單邊成交。 |
| 真實 maker execution | **INCONCLUSIVE** | E4 沒有真實委託簿、排隊順位、撤單或部分成交；synthetic quote 不能證明 maker 成交。 |
| Portfolio capacity | **INCONCLUSIVE** | 已施加共用資本限制，但歷史 book depth 不存在，規模、漏成交與衝擊只能做敏感度，不能估計真實容量。 |
| 凍結候選 `VDA-capitulation-v1` | **INCONCLUSIVE / DO NOT PROMOTE** | 多重搜尋、對照組不公平、maker 成交未知；保守執行壓力下每筆 0.047%，不足以完成可執行正期望的證明。 |

最重要的結論是：目前證據支持「極端急跌後有條件反彈」作為研究現象，但尚未支持
「能在 Binance 真實 maker 排隊、共用帳戶限制與危機滑價下實現正收益」。報告中的
`inconclusive` 不是中性推薦；在驗收標準下它等同不得進入實盤。

## 1. Research reliability audit

| ID | 嚴重度 | 判定 | 主題 | 證據 |
| --- | --- | --- | --- | --- |
| A01 | high | fail | random_control_reproducibility | runs.py uses hash(symbol); PYTHONHASHSEED 1/2 produced [2158, 5712] |
| A02 | high | fail | random_control_fairness | Random entries are forced at the next bar open via entries=... while simulate() charges fee_in=maker and applies no entry slippage; E1 baseline/control counts are 46052/42094. |
| A03 | high | fail | synthetic_maker_fill | sim.py and PaperBroker treat last trade +/- half tick as quotes and fill after a trade-through; no historical depth, queue position, cancellations or partial fills exist. |
| A04 | high | fail | parameter_search_multiplicity | At least 311 explicit configurations/diagnostics appear in drift (20), grid (144), breakout (96), extreme (18), E3 (25) and E5 (8), before undocumented human choices. |
| A05 | medium | pass_with_limits | lookahead | Signals use closed-bar values and orders are evaluated from the following minute. The rolling quantile includes the just-completed bar outcome, which is known at decision time. |
| A06 | medium | inconclusive | missing_minutes | load_klines forward-fills absent minutes with a flat synthetic OHLC bar. |
| A07 | high | fail | stop_slippage | E3 lead expectancy 0.178% becomes -0.018% with 0.3% adverse stop/trail slippage. |
| A08 | high | fail | portfolio_capacity | runs.py executes one independent account per symbol and sums net_usdt; the production default has max_positions=1 and max_notional=350. |
| A09 | medium | inconclusive | engine_sample_selection | E4 samples only coin-days where the OHLC lead already traded, excludes BTC/ETH, and replays each coin-day separately. |
| A10 | medium | pass_with_limits | frozen_candidate | VDA is frozen at commit 36552a3: 710 trades, published mean 0.414%. |

### Look-ahead 與選模

未發現 `sim.py` 在單次訊號計算中直接讀取未來 K 線：訊號在已收盤分鐘形成，成交從下一分鐘判定。
`quantile` 的本根低／高價在本根收盤時已知。但研究流程有嚴重的 selection bias：至少 311 個明列組合與診斷，
再選出 `k=12/sl=3/long/against`，而報告沒有 family-wise correction。E2 的 2024–25 年資料在發現候選時確實未使用，
但它是向歷史更早期間回看，後續 E3/E5 又重複使用全部 24 個月，因此不能繼續稱為未使用資料。

### 隨機對照

`seed + hash(symbol)` 受 Python hash salt 影響，本次以兩個 `PYTHONHASHSEED` 得到
`[2158, 5712]`。更關鍵的是 random control 在下一根 open 強制成交，卻套 maker 費且無入場滑價；
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
0.361%，engine 平均 0.152%，平均誤差
-0.209%，平均絕對誤差 0.793%。
E4 還條件式抽取「原模擬已成交」的 coin-days、排除 BTC/ETH、逐幣獨立執行，不能推廣成總體 fill rate 或容量估計。

## 3. Frozen alpha reproduction and adverse execution

| 情境 | 交易數 | 每筆淨值 | 獨立帳戶合計 USDT | 正幣種 | 正月份 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 710 | 0.414% | 873.97 | 24 | 21 |
| 穿價 0.2% | 528 | 0.305% | 478.50 | 20 | 20 |
| 停損額外滑 0.3% | 710 | 0.158% | 332.27 | 18 | 19 |
| 兩者同時 | 528 | 0.047% | 73.83 | 13 | 17 |

這些 adverse variants 只改 execution，沒有調整凍結 alpha 門檻。即使它們仍為正，也不能補上 queue 的未知資料；
「價格穿過 0.2%」只是保守 proxy，不是排隊模型的校準值。

### 2026-10 forward holdout

先前研究資料止於 2026-09。本次另取當時未參與選模的 2026-10-01 至 2026-10-22 完整公開日檔，
保留 2026-09 只作 rolling warm-up，並以同一份凍結設定、同一個模擬器執行。

| 情境 | 完整日 | 交易數 | 每筆淨值 | 獨立帳戶合計 USDT | 正幣種 |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen | 22 | 45 | 0.671% | 89.61 | 16 |
| 穿價 0.2% + 停損滑價 0.3% | 22 | 35 | 0.215% | 22.34 | 14 |

這段資料對「急跌後條件反彈」提供新的支持證據，但只有 22 天、單一市場 regime，且仍缺少歷史 order book。
它能降低純粹重用舊資料的疑慮，不能單獨證明 maker execution、容量或可持續性；最終判定維持 **INCONCLUSIVE / DO NOT PROMOTE**。

## 4. Shared portfolio and capacity sensitivity

Portfolio replay 使用同一個 1,000 USDT 初始資本、全域 margin/notional/position 限制，並把同一 5 分鐘的信號視為共同市場事件。
同時訊號以預先固定 SHA-256 priority、50 個 seed 測試，不用事後收益選單。`max_floating_drawdown` 將所有同時持倉視為一起到達
各自記錄的 MAE，是偏保守的 event stress。歷史委託簿仍未知，因此以下是 sensitivity envelope。

| 情境 | 平均接受筆數 | 平均淨 USDT | 5–95% seed | 平均最大浮動 DD | 最大 margin | 全期 margin 使用率 | 持倉期間使用率 | 淨利 / MAE 風險 | 最差共同事件 USDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ohlc_upper_bound | 272.2 | 366.44 | [308.29, 423.02] | 2.05% | 29.70 | 0.0063% | 2.97% | 1.715 | -18.44 |
| partial_50 | 220.9 | 154.02 | [123.68, 185.43] | 1.20% | 14.85 | 0.0025% | 1.49% | 1.752 | -9.27 |
| queue_stop_stress | 165.6 | 52.71 | [33.99, 69.81] | 0.74% | 7.42 | 0.0009% | 0.74% | 1.627 | -7.75 |
| three_positions_100 | 325.4 | 87.59 | [68.24, 105.33] | 0.85% | 15.00 | 0.0010% | 0.59% | 1.921 | -12.97 |
| three_positions_1000_requested | 243.8 | 206.34 | [138.35, 271.70] | 3.67% | 52.03 | 0.0031% | 2.48% | 1.472 | -43.59 |

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

本次資料快照共有 2571 個檔案，SHA-256 與大小見 [data_manifest.csv](validation/results/data_manifest.csv)。
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
