# Experiment ledger — mgrid-roll

| ID | UTC | Split | 變體 | 結果 | 判定 |
|---|---|---|---|---|---|
| R000 | 2026-11-06 11:15 | — | 預先登記 HYPOTHESIS.md（commit 84a956d） | — | — |
| R001 | 2026-11-06 11:35 | dev | Q1 挑幣 11 個特徵 IC + partial IC（results/S2_selection_dev.csv） | 通過：RV14（partial t 11.9）、AGE（越新越好，−7.3）、FUND3a（我方付費越多越差，−4.7）、MOM30a（4.6）、BETA30（4.1）、DIST50a（3.9）、DD90a（−3.3），加上基準 RV3。未過：MOM7a、QVR、LQV | 7 個新特徵過關 |
| R002 | 2026-11-06 11:35 | dev | Q2 擇時 5 個市場特徵三分位（results/S2_timing_dev.csv） | 通過：BTCTR t 3.2（2022 無高分位日）、BTCRV7 t 2.8（三期皆正）、MRV t 2.8（三期皆正）；BREADTH t 2.5、MFUND 子期間不足未過 | 3 個過關 |
| R003 | 2026-11-06 11:40 | — | 程式修正：partial IC 門檻原寫成 p_t > 2，負向特徵（AGE、DD90a、FUND3a）被誤判；改為 \|p_t\| > 2 且與 IC 同號（登記本意）。**多個成分過關時的組裝規則（S3 前決定，登記未寫明）**：挑幣 = 所有過關特徵（含 RV3）依方向的每日百分位 rank 等權平均 = SCORE；擇時 = 過關且三個子期間都有值、同號者中 t 最高 = BTCRV7（過去 7 天 BTC 日波動 ≤ 0.0291 當天不開新網格）。BTCTR 因 2022 沒有資料且與方向規則重疊不選。 | — | S3 組合：MGT / SCORE / SCORE_T |

## Split read/peek log

| UTC | Split | 操作 | 已看結果？ |
|---|---|---|---|
