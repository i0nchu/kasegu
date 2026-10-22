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
- 報告檔名格式：`{label}_{幣種}_{起訖日}_{策略}-k{k}_{執行時間}.json`，
  例如 `exp1_SOL_20261001-20261007_atr_amplitude-k3_20261022-171000.json`。

比較已存在的報告（檔案或資料夾皆可，可輸出 CSV）：

```bash
uv run python -m kasegu.backtest.report data/backtests
uv run python -m kasegu.backtest.report data/backtests/exp1_* --sort pf --csv exp1.csv
```

回測、paper、live 共用同一套 Engine、Risk Manager 與 Exit Manager。

`live` 不會是預設模式，必須同時指定 `--mode live --confirm-live`。

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
├── exchange/                # Binance REST / WebSocket
├── market/                  # K 線、報價、標記價格
├── core/                    # engine, execution, position, risk, types
├── strategies/              # 可替換的進場距離模型
├── exits/                   # 共用追蹤離場
├── backtest/                # 歷史重播
└── storage/                 # SQLite journal
tests/
```
