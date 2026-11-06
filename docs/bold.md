# 翻倍押注（paper）

研究：`research/alpha/mgrid-bold/REPORT.md`。這不是有優勢的策略，期望值約為 0 減掉成本，是「用最高機率把小資金翻倍」的大膽下注。**只有 paper 模式，不下單。**

## 規則（預設）

- 每天 00:00 UTC，如果沒有進行中的押注，就以 BTCUSDT 5m K 線的開盤價做多 10 倍，本金 100 USD。
- 權益到 2 倍（價格約 +10.2%）止盈；權益剩 0.1 倍（價格約 −8.8%）出場，在強平之前；30 天到期市價平倉。
- 手續費和滑價（taker 0.05% + 0.05%，進出各一次）先扣，funding 依結算時間計入；同一根 5m K 線先檢查不利方向；跳空越過停損時以開盤價計（可能虧光）。
- 每一次嘗試都從 100 USD 重新開始，用來統計前瞻的翻倍率。
- 設定檔會檢查停損是否在強平之前（`maintenance`），槓桿太高會拒絕啟動。

## 執行

```bash
cp configs/bold.example.yaml configs/bold.yaml
uv run kasegu --mode bold-paper --config configs/bold.yaml
# 或 docker
docker compose --profile bold up -d --build kasegu-bold
```

輸出在 `data/bold/`：`tries.csv`（每次嘗試一列：開始、結束、結果、權益倍數、損益、funding、天數）、`status.json`（目前的押注、止盈/停損價、累計翻倍率）、`state.json`（重啟續跑）。

## 參考值（2020-07~2026-09 回測，每天進場各算一次）

BTC 10 倍：翻倍 50.6%，達標中位數 11 天，失敗中位數 9.5 天。這段期間 BTC 長期上漲，所以未來應以約 50% 減去成本來預期。
