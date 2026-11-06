# Experiment ledger — mgrid-roll

| ID | UTC | Split | 變體 | 結果 | 判定 |
|---|---|---|---|---|---|
| R000 | 2026-11-06 11:15 | — | 預先登記 HYPOTHESIS.md（commit 84a956d） | — | — |
| R001 | 2026-11-06 11:35 | dev | Q1 挑幣 11 個特徵 IC + partial IC（results/S2_selection_dev.csv） | 通過：RV14（partial t 11.9）、AGE（越新越好，−7.3）、FUND3a（我方付費越多越差，−4.7）、MOM30a（4.6）、BETA30（4.1）、DIST50a（3.9）、DD90a（−3.3），加上基準 RV3。未過：MOM7a、QVR、LQV | 7 個新特徵過關 |
| R002 | 2026-11-06 11:35 | dev | Q2 擇時 5 個市場特徵三分位（results/S2_timing_dev.csv） | 通過：BTCTR t 3.2（2022 無高分位日）、BTCRV7 t 2.8（三期皆正）、MRV t 2.8（三期皆正）；BREADTH t 2.5、MFUND 子期間不足未過 | 3 個過關 |
| R003 | 2026-11-06 11:40 | — | 程式修正：partial IC 門檻原寫成 p_t > 2，負向特徵（AGE、DD90a、FUND3a）被誤判；改為 \|p_t\| > 2 且與 IC 同號（登記本意）。**多個成分過關時的組裝規則（S3 前決定，登記未寫明）**：挑幣 = 所有過關特徵（含 RV3）依方向的每日百分位 rank 等權平均 = SCORE；擇時 = 過關且三個子期間都有值、同號者中 t 最高 = BTCRV7（過去 7 天 BTC 日波動 ≤ 0.0291 當天不開新網格）。BTCTR 因 2022 沒有資料且與方向規則重疊不選。 | — | S3 組合：MGT / SCORE / SCORE_T |
| R004 | 2026-11-06 11:55 | dev | S3 複利帳戶：4 個變體（LONG 基準 / MGT / SCORE / SCORE_T）× 9 種 sizing × 成本基準/壓測（results/S3_dev_summary.csv） | SCORE 對 MGT 配對差 t −0.3 ~ −0.98（全部 sizing 為負）；SCORE_T 對 SCORE t −0.2 ~ +0.7；擇時大幅降回撤（K1 L1.04：−25%→−12%）但日均差不顯著。LONG 在 L≥3.12 於 2022 爆倉 | **挑幣組合與擇時都未過 t>2 增量門檻 → 依登記移除**，候選 = MGT（BOTH + RV3 排序） |
| R005 | 2026-11-06 12:00 | dev | sizing 選擇（壓測、bootstrap P(1 年<30 USD) ≤ 10% 中 1 年中位數最高） | MGT K=1 L=3.12：中位數 346 USD、P(<30) 4.2%、翻倍 72%、歷史 maxDD −68%、2023H1 −47%。K1 L4.16 P(<30) 22% 不合格 | 凍結 MGR-K1-L3（FROZEN.json sha256 前 16 碼 e392ac6db187ba7c）；事前宣告次要對照 MGR-K3-L3（K3 L3.12，壓測 maxDD −32%、P(<30) 0%） |

## Split read/peek log

| UTC | Split | 操作 | 已看結果？ |
|---|---|---|---|
| 2026-11-06 11:15 | val | R006 一次性重算（MGR-K1-L3、MGR-K3-L3 + MGT/LONG 對照，基準與壓測成本） | 是（val 先前已被 mgrid-trend 消耗；本次為非獨立重算） |
