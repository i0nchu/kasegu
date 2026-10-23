# Alpha 研究（research/claude-alpha）

目標：找出有可解釋市場機制、扣除成本後仍可能獲利的現象，而不是一組回測賺錢的參數。

## 資料與時間戳

| 資料 | 來源 | 粒度 | 何時可用 |
|---|---|---|---|
| K 線（含主動買量 taker_buy_volume、成交筆數） | data.binance.vision monthly klines | 1m | 該根 K 線收盤時 |
| 未平倉量 OI | data.binance.vision daily metrics | 5m | 標記時間 + 5 分鐘（保守延遲一期） |
| 逐筆成交 aggTrades | data.binance.vision daily aggTrades | 逐筆 | 成交當下（只用來驗證成交，不作特徵） |
| 資金費率 | data.binance.vision monthly fundingRate | 8h | 結算時 |

- 範圍：40 個幣（前兩輪 28 個 + 新增 12 個），2024-10-01 到 2026-09-30。第一年（Y1）到 2025-09，第二年（Y2）從 2025-10。
- 所有特徵都在「下單那根 K 線收盤時」計算（`features.py`），成交那根 K 線或之後的資訊不得當作篩選條件。
- 「市場」是 40 個幣 1m 對數報酬的等權平均（`common.market()`）。
- 市場事件分類（`common.market_days()`）：當天等權市場 60 分鐘內最大跌幅 > 5% 為 crash（49 天）、2–5% 為 stress（342 天）、其餘 calm（339 天）。
- 信賴區間一律以「日」為單位做 bootstrap（同一天所有幣的事件一起抽），避免崩盤日被重複計算。

## 程式

| 檔案 | 用途 |
|---|---|
| `download_metrics.py` / `download_trades.py` | 下載 OI 與指定日的逐筆成交 |
| `data.py` | 對齊到同一分鐘格線的面板快取（`data.py klines`、`data.py oi`） |
| `features.py` | 事前特徵：報酬、主動買賣失衡、成交量倍數、ATR、波動擴張、OI 變化、清算代理 |
| `passive.py` | 被動限價單事件研究：每張掛單、是否成交、成交後 markout / MFE / MAE |
| `stats.py` | 以日為群的 bootstrap、分組統計、尾部貢獻 |
| `a1_passive_reversal.py` | 第一階段：極端被動反轉，含深度曲線、鏡像、固定深度等對照 |
| `a2_fill_realism.py` | 第一階段：用逐筆成交檢查掛單是否真的成交 |
| `a3_candidate.py` | 第一階段：候選規則的逐筆交易模擬（一幣一倉、完整掛單紀錄） |
| `b1_shock_reversal.py` | 第二階段：流動性衝擊後吃單反向 |
| `b2_flow_momentum.py` | 第二階段：訂單流動能（時間序列與橫截面） |
| `b3_relative_value.py` | 第二階段：橫截面相對價值（殘差反轉） |

重跑（在專案根目錄）：

```bash
uv run python research/alpha/download_metrics.py
uv run python research/alpha/data.py klines && uv run python research/alpha/data.py oi
uv run python research/alpha/a1_passive_reversal.py
```

完整事件與交易紀錄寫在 `data/research/runs/<實驗>/`（太大不進 git），可審閱的摘要表複製到 `research/results/<實驗>/`。

## 文件

- `docs/H1_extreme_passive_reversal.md`：第一階段，唯一通過的候選
- `docs/H2_liquidity_shock_reversal.md`、`docs/H3_order_flow_momentum.md`、`docs/H4_cross_sectional_relative_value.md`
- `docs/LOG.md`：所有嘗試過的假設與失敗結果
- `docs/HANDOFF_codex.md`：凍結版本與交給 Codex 的驗證說明
