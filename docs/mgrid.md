# 順勢馬丁格爾網格（paper）

研究來源：`research/alpha/mgrid-regime`（因子研究，REJECT）與 `research/alpha/mgrid-trend`（本策略，候選 `MGT-BOTH-k1.0-RV3-K3`，val 後判定 INCONCLUSIVE）。**只有 paper 模式，不會對交易所下任何單。**

## 規則

每天 00:00 UTC（等最後一根 5m K 線收盤，預設延遲 20 秒）：

1. 幣池：USDT 永續中，近 30 個已收盤日的平均成交額前 40 名，上市滿 60 天；排除 BTC、ETH 與股票/貴金屬/指數代幣（`grid/signals.py` 的 `EXCLUDE`）。
2. 方向：BTC 收盤 > 100 日均線**且**該幣收盤 > 50 日均線 → 做多網格；兩者都在線下 → 做空網格；不一致 → 不開。
3. 波動：近 72 根 1h 報酬的日波動 RV3 ≥ 0.0696，由高到低補滿空位（最多 3 個幣，同一幣同時只有一個網格）。
4. 格距 = 近 168 根 1h 報酬的日波動，夾在 [0.5%, 10%]。

每個網格：開盤價市價開第 1 筆（5 USDT），往不利方向每格一張限價，6 筆名目 5 / 7.5 / 11.3 / 16.9 / 25.3 / 38 USDT（合計 104）。止盈 = 均價 ± 1 格（限價全平）；停損 = 最後一層再 1 格（市價）；72 小時到期市價平倉；逐倉保證金 = 帳戶 / 3，虧損超過保證金視為強平。結束後隔天 00:00 才能再用這個空位。

成交規則與研究完全相同：限價要穿價 0.05% 才算成交；同一根 5m 若有新加碼就不在該根止盈；maker 0.02%、taker 0.05%，市價另加 0.02%（進）/ 0.05%（出）滑價；funding 依結算時間計入。

帳戶累計虧損（已實現 + 未實現）達 `max_account_loss`（預設 40 USD）後不再開新網格，已開的照規則跑完。要重新開始，刪除 `data/grid/state.json` 中的 `"halted": true`（或整個 state.json）。

## 執行

```bash
cp configs/mgrid.example.yaml configs/mgrid.yaml
uv run kasegu --mode grid-paper --config configs/mgrid.yaml
# 或 docker
docker compose --profile grid up -d --build kasegu-grid
```

只用公開行情 API，不需要 API key。可以和現有的 EPR15 paper 同時跑（輸出在不同目錄）。

## 輸出（`data/grid/`）

| 檔案 | 內容 |
|---|---|
| `state.json` | 未結束的網格、已實現損益、上次決策日；重啟會從這裡接續 |
| `trades.csv` | 每個結束的網格一列：方向、開始/結束時間、出場原因、損益、手續費、funding、成交層數、格距、RV3 |
| `decisions.jsonl` | 每天的選幣紀錄：幣池、所有候選的 RV3/波動/方向、選中哪些 |
| `daily.csv` | 每天 00:00 的已實現、未實現、權益 |
| `status.json` | 目前狀態（每 5 分鐘更新，docker 健康檢查用） |

## 驗證

- `research/alpha/mgrid-trend/code/parity.py`：3000 條隨機路徑，`kasegu.grid.episode` 和研究回測的出場時間、原因、損益逐筆一致。
- `research/alpha/mgrid-trend/code/replay_paper.py`：用歷史資料重播整個 paper 執行器（選幣 + 撮合），和研究 val 帳本比對。
- `tests/test_grid.py`：網格狀態機、訊號、選幣、重啟、帳戶虧損上限。

## 和回測可能的差異

- 上市天數用交易所 `onboardDate`，研究用資料天數。
- 某幣 K 線抓取失敗時，該網格當作仍佔用空位（不會多開）。
- 下市中的幣：交易所不再列為 TRADING 就不會被選；已開的網格照 K 線跑到結束。

## 前瞻評估怎麼看

val 的參考值：每月約 37 個網格、勝率 70%、年化 +61~70 USD、最大回撤 37~44 USD（100 USD 本金）。前瞻 paper 至少要累積約 3 個月（~110 個網格）才看得出平均值，判斷重點：每月網格數、勝率、平均損益、出場原因比例、成交層數分布、funding 是否和回測一致。
