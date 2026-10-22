# kasegu

Binance USDⓈ-M Futures 波動率交易系統（Strategy-Based Event-Driven Architecture）。

## 環境需求

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.12（uv 會依 `.python-version` 自動安裝）

## 快速開始

```bash
uv sync
cp configs/config.example.yaml configs/config.yaml   # 本機設定，不進 git
uv run python -m kasegu --mode paper      # 預設模式
uv run python -m kasegu --mode backtest
uv run python -m kasegu --mode live --confirm-live
uv run pytest
uv run ruff check .
```

## 回測

以 [data.binance.vision](https://data.binance.vision) 的 aggTrades 逐筆重播，1m K 線只用來預熱指標。
資料會快取在 `data/history/`，報告輸出到 `data/backtests/`。

```bash
uv run python -m kasegu --mode backtest --start 2026-10-01 --end 2026-10-07
uv run python -m kasegu --mode backtest --symbols SOLUSDT   # 覆寫設定檔的 symbols
uv run python -m kasegu --mode backtest --symbols SOLUSDT,WIFUSDT,SUIUSDT --each --label exp1
```

- `--each`：每個幣種各自跑一次，最後印出比較表（依 net 排序）。
- `--label`：實驗名稱，會放在檔名與摘要最前面。
- `--strategy`、`--k`：不改設定檔直接換模型或倍率（換模型時不沿用設定檔的 `params`）。
- 報告檔名格式：`{label}_{幣種}_{起訖日}_{策略}-k{k}_{執行時間}.json`，
  例如 `exp1_SOL_20261001-20261007_atr_amplitude-k3_20261022-171000.json`。

比較已存在的報告（檔案或資料夾皆可，可輸出 CSV）：

```bash
uv run python -m kasegu.backtest.report data/backtests
uv run python -m kasegu.backtest.report data/backtests/exp1_* --sort pf --csv exp1.csv
```

### 進場模型

三個模型都用 EMA(240) 決定方向，只差在掛單距離 `Distance%`：

| id | Distance% | k 的意義 |
|---|---|---|
| `atr_amplitude` | (ATR% + 平均振幅%) / 2 × k | 波動倍率（baseline） |
| `atr_only` | ATR% × k | 波動倍率 |
| `quantile` | 歷史「下一根 K 線逆向偏離」的 q 分位數 × k | 分位數的縮放，k=1 即原始分位數 |

`quantile` 多單看下一根最低價跌破本根收盤的幅度、空單看下一根最高價突破的幅度，分開估計，
只用決策當下已收盤的 K 線。參數放在 `strategy.params`：`q`（預設 0.95）、`lookback`（預設 720 根）。
三者的 k 意義不同，比較時不要只看 k 數值相同。

```bash
uv run python -m kasegu --mode backtest --symbols WIFUSDT --strategy atr_only --k 3 --label m1
uv run python -m kasegu --mode backtest --symbols WIFUSDT --strategy quantile --k 1 --label m1
uv run python -m kasegu.backtest.report data/backtests/m1_*
```

回測、paper、live 共用同一套 Engine、Risk Manager 與 Exit Manager。

`live` 不會是預設模式，必須同時指定 `--mode live --confirm-live`。

## Live（Testnet）

```bash
# 1. 在 https://testnet.binancefuture.com 建立 API key，設定環境變數（見下方）
# 2. 檢查下單、停損、user data stream 與自動撤單倒數
uv run python -m kasegu.exchange.check --symbol BTCUSDT
# 3. 啟動
uv run python -m kasegu --mode live --confirm-live
```

- `exchange.testnet: true` 時下單與行情都走 Testnet（策略價格必須來自同一本委託簿）。
  改成 `false` 才會連到正式環境，啟動時會再警告一次。
- 帳戶必須是單向持倉（One-way Mode），啟動時會依 `risk.leverage` 設定各幣種槓桿。
- 進場：Post-Only 限價單；出場：reduce-only 市價單；停損：交易所端 `STOP_MARKET`
  條件單（Binance algo order，以標記價格觸發）。
- 成交以 user data stream 為主，REST 回應與重連後的對帳只補上尚未記錄的成交量，不會重複計算。
- 重啟時以交易所為準：撤掉舊進場單，沿用數量相符的停損單，多餘的停損單撤銷；
  倉位與本地紀錄不符時停止該幣種的新進場。
- 狀態存在 `data/journal-testnet.db` / `data/journal-mainnet.db`，與 paper 分開。
- `live.auto_cancel_s`：交易所端倒數撤單，程式失聯時自動撤掉進場單。
  Testnet 實測它也會撤掉停損條件單，所以只在該幣種沒有倉位時啟用，
  建立停損前先關閉倒數；啟動時也會先清掉上次留下的倒數。check 的第 2 段會驗證這個流程，
  通過後再打開（預設關閉）。

## API Key

透過環境變數載入，不寫入 Git（可複製 `.env.example` 為 `.env`）：

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
```

## 專案結構

```
configs/config.example.yaml  # 範例配置，複製成 configs/config.yaml 使用
src/kasegu/
├── __main__.py              # CLI 入口
├── config.py                # Pydantic 配置
├── exchange/                # Binance REST / WebSocket、live broker、testnet check
├── market/                  # K 線、報價、標記價格
├── core/                    # engine, execution, position, risk, types
├── strategies/              # 可替換的進場距離模型
├── exits/                   # 共用追蹤離場
├── backtest/                # 歷史重播
└── storage/                 # SQLite journal
tests/
```
