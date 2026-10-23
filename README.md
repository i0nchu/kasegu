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

前三個模型都用 EMA(240) 決定方向（收盤在 EMA 上只做多、下只做空），只差在掛單距離 `Distance%`：

| id | Distance% | k 的意義 |
|---|---|---|
| `atr_amplitude` | (ATR% + 平均振幅%) / 2 × k | 波動倍率（baseline） |
| `atr_only` | ATR% × k | 波動倍率 |
| `quantile` | 歷史「下一根 K 線逆向偏離」的 q 分位數 × k | 分位數的縮放，k=1 即原始分位數 |
| `epr15` | (ATR/收盤% + 平均振幅%) / 2 × 12，只在收盤 < EMA240 時掛多單 | 凍結為 12，見 [docs/epr15.md](docs/epr15.md) |

`epr15` 的參數與出場已凍結（git tag `alpha-epr15-v1`），設定不符會在啟動時拒絕執行，
小額帳戶 / Shadow 範例在 `configs/epr15.example.yaml`，研究重現設定（40 幣、每筆 297 USDT，非實盤建議）在 `configs/epr15.research.yaml`。

`quantile` 多單看下一根最低價跌破本根收盤的幅度、空單看下一根最高價突破的幅度，分開估計，
只用決策當下已收盤的 K 線。參數放在 `strategy.params`：`q`（預設 0.95）、`lookback`（預設 720 根）。
三者的 k 意義不同，比較時不要只看 k 數值相同。

```bash
uv run python -m kasegu --mode backtest --symbols WIFUSDT --strategy atr_only --k 3 --label m1
uv run python -m kasegu --mode backtest --symbols WIFUSDT --strategy quantile --k 1 --label m1
uv run python -m kasegu.backtest.report data/backtests/m1_*
```

回測、paper、live 共用同一套 Engine、Risk Manager 與出場政策。

### 出場政策

出場與進場分開設定，`exit.policy` 選擇：

| policy | 規則 |
|---|---|
| `trailing`（預設） | 固定停損 + 三級追蹤止盈 + `max_holding_minutes` |
| `time_only` | 固定停損 + 成交後 `max_holding_minutes` 分鐘市價出場，不啟動任何追蹤級距 |

兩者都會在成交後掛交易所端 reduce-only 停損單（`stop_loss_pct`）。

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

### 保護機制

- 交易所停損單被拒時，立刻以 reduce-only 市價單平倉。
- 啟動或重連時，若交易所有本地沒有紀錄的倉位，會接手管理：掛停損並套用出場規則。
- `risk.max_daily_loss`（USDT／UTC 日）與 `risk.max_consecutive_losses` 觸發後暫停新進場，
  既有倉位照常保護與出場；重啟後會從 journal 讀回當日已實現損益。
- 委託狀態未知、撤單未確認（含交易所回傳舊狀態）、停損未確認或被交易所端取消時，
  發出 critical 告警並暫停所有幣種的新進場，每 30 秒重新對帳；所有委託、倉位與停損都確認後才恢復。
  細節見 [docs/epr15.md](docs/epr15.md) 第 6 節。
- 撤單結果未確認時不會掛新的進場單，避免兩張進場單同時在簿上。
- 倉位數量改變時先掛新停損，交易所確認後才撤舊停損；新停損未確認且舊停損不足時市價平倉。
- `risk.max_loss_per_trade`：以固定停損價估算的單筆虧損上限（USDT），提高槓桿不會繞過；
  不含跳空與滑價，實際損失可能更大。

## Docker 部署

```bash
cp configs/config.example.yaml configs/config.yaml   # 策略、幣種、風控、告警
cp .env.example .env                                 # 模式、API key、Telegram
docker compose up -d --build
docker compose logs -f
docker compose stop        # SIGTERM：撤掉進場單，保留倉位與停損
```

- 模式由 `.env` 決定：`KASEGU_MODE=paper|live`，live 還需要 `KASEGU_CONFIRM_LIVE=1`。
- `./data` 掛載進容器，存放 journal、`status.json`、`events.jsonl` 與歷史資料快取。
- `status.json` 每 5 秒更新一次，健康檢查超過 60 秒沒更新就標記為 unhealthy；
  容器設定 `restart: unless-stopped`。
- 回測：`docker compose run --rm kasegu --mode backtest --symbols WIFUSDT --each`

## 告警與事件（可選）

引擎把重要事件發佈到事件匯流排，各個輸出互不影響，也不會卡住交易：

| 輸出 | 設定 | 用途 |
|---|---|---|
| `data/events.jsonl` | `alerts.events_path` | 所有事件，一行一個 JSON |
| Telegram | `alerts.telegram: true` + `.env` 的 `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID` | 手機通知，`alerts.min_level` 過濾等級 |
| Webhook | `alerts.webhook_url` | 每個事件 POST 一個 JSON，可接未來的 WebUI |
| `data/status.json` | `status_path` | 目前倉位、掛單、當日損益、暫停狀態的快照 |

事件格式：`{"source", "tag", "kind", "level", "message", "symbol", "ts", "data"}`。
主要事件有 `started`、`stopped`、`entry_filled`、`trade_closed`、`tier`（只記 journal）、
`stop_rejected`、`stop_lost`、`stop_unknown`、`order_unknown`、`entry_cancel_unconfirmed`、
`stop_cancel_unconfirmed`、`sync_unconfirmed`、`untradable`、`reconciled`、`exit_rejected`、
`position_adopted`、`position_resized`、`position_gone`、`loss_limit`、`disconnected`、`reconnected`。

## API Key

透過環境變數載入，不寫入 Git（可複製 `.env.example` 為 `.env`）：

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
```

## 專案結構

```
configs/config.example.yaml  # 範例配置，複製成 configs/config.yaml 使用
configs/epr15.example.yaml   # EPR15 小額帳戶 / Shadow 範例
configs/epr15.research.yaml  # EPR15 研究重現（非實盤建議）
docs/epr15.md                # EPR15 規格、研究版差異、Shadow 操作與上線檢查
scripts/epr15_compare.py     # 研究模擬器與正式引擎逐筆比較
src/kasegu/
├── __main__.py              # CLI 入口
├── config.py                # Pydantic 配置
├── exchange/                # Binance REST / WebSocket、live broker、testnet check
├── market/                  # K 線、報價、標記價格
├── core/                    # engine, execution, position, risk, types
├── strategies/              # 可替換的進場模型（含 epr15）
├── exits/                   # 出場政策：trailing、time_only
├── events.py / notify.py    # 事件匯流排與告警輸出
├── health.py                # 容器健康檢查
├── backtest/                # 歷史重播
└── storage/                 # SQLite journal
tests/
```
