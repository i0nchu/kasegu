# Alpha monitoring plan — bold-entry

## 前瞻觀測欄位

- 每筆：signal UTC / price & input snapshot / symbol / side / decision / intended entry / actual entry / order status / maker-taker fees / funding / slippage / exit / risk R / mark-to-market / reject or missed fill reason。
- 每日：PnL/equity、gross/net exposure、risk concentration、baseline PnL、fill rate、data staleness、backend incident、cost drift。
- 定期：估計 net expectancy + CI、baseline incremental PnL、信號與資料分布 drift、最大回撤、主要市場 regime、capacity。

## 停損與治理（研究前填值，不能事後追績效）

- 訊號/資料失真立即暫停條件：
- 系統/下單風險 kill switch：
- 單筆/單日/整體最大曝險：
- 統計退化告警、有效樣本數：
- 成本退化停用門檻：
- 恢復條件與審批：
- `KEEP / REDUCE / PAUSE / RETIRE` 決策與紀錄位置：
