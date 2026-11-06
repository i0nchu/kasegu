# Experiment ledger — mgrid-base

| ID | UTC | Split | 內容 | 結果 | 判定 |
|---|---|---|---|---|---|
| B000 | 2026-11-06 16:05 | — | 預先登記（commit c87d0fd） | — | — |
| B001 | 2026-11-06 16:15 | dev | 3 分類器 × 狀態，之後 30 天報酬 / 回撤（results/S2_states.csv） | UP 之後 30 天 +11~18%，RANGE −1~+5%，DOWN −0~+7%。DOWN 減其他的月區塊 bootstrap 95% CI 全部跨 0；DOWN 與 RANGE 的回撤差不多（−11~−15%） | **S2 未過**：DOWN 和 RANGE 分不開，只有 UP 分得出來 |
| B002 | 2026-11-06 16:25 | dev | （S2 未過後的探索）B0/B1/G1/G2 × 3 分類器（results/S3_dev.csv） | B0 Calmar 1.05、maxDD −77%；MA50_200 G2 Calmar 2.02、maxDD −52%、2022 −11%（B0 −46%）；但 MA20_100 的網格比同分類器 B1 差（1.15 vs 1.81） | 網格增量不一致 |
| B003 | 2026-11-06 16:30 | — | **偏離登記**：S2 未過照規則應停。使用者要求要能分辨機會、要具體策略，所以依登記的 S3 規則（dev Calmar 最高）凍結 MGB-G2-MA50_200，val、hold 各跑一次，結果只當描述，不算通過任何關卡 | — | 凍結 |
| B004 | 2026-11-06 16:35 | val、hold（各一次） | 凍結的 MGB-G2-MA50_200 與同期 B0、B1、其他組（results/S3_val.csv、S3_hold.csv） | G2：val 年化 +33%（B0 +80%），hold −13%（B0 −9%）。B1 MA50_200：val +61%、maxDD −33%；hold +14%、maxDD −42%（B0 −60%） | **凍結的網格候選 FAIL**：兩段都輸給買進持有。B1 趨勢持有三段 Calmar：1.49 / 1.88 / 0.33，對 B0：1.05 / 2.30 / −0.15 |
