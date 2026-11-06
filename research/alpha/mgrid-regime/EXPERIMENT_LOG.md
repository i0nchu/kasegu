# Experiment ledger — mgrid-regime

此檔紀錄**所有**試驗。思考過程見 [LOG.md](LOG.md)，結論見 [REPORT.md](REPORT.md)。
資料版本：babypig `data/research/fast/5m/*.npz`（G1 產生），轉 float32 後 385 檔 558 MB（`data/research/mgrid/stage/`）。程式：`code/`。

| ID | UTC | Split | 變體 / 參數 | Code commit | N / clusters | 結果（淨） | 對基準 | 判定 |
|---|---|---|---|---|---|---|---|---|
| E000 | 2026-11-06 10:40 | — | 預先登記 HYPOTHESIS.md | b1ed4e7 | — | — | — | S0 通過 |
| E001 | 2026-11-06 11:05 | dev | 4 個網格（L/S × k 0.5/1.0）逐 episode 描述統計 | 本 commit | 40,300 幣日 / 147 幣 | 每 episode 平均 −0.01% ~ +0.02%，勝率 73–91% | 無濾網即基準 | 典型馬丁：高勝率、期望值約 0 |
| E002 | 2026-11-06 11:10 | dev | 16 因子 × {T2 網格損益, T1 未來 ER} 每日 IC，Holm 32 檢定 | 本 commit | 1,092 日 | T1：無一因子 Holm 後有意義（最大 AC13 IC 0.025）。T2：RV3 IC 0.30、五分位差 0.33%/ep，三子期間一致 | RV 本身就是事前列的基準因子 | 只有 RV3 過 S2 門檻（RV14 差 0.28 < 0.30） |
| E003 | 2026-11-06 11:15 | dev | RV3 拆解：side、step 單位、停損率；其他 12 因子控制 RV3 後的 partial IC（12 檢定，探索性） | 本 commit | 1,092 日 | RV3 高五分位 +0.039 step/ep、低 −0.094；ER/ADX/CHOP 的 T2 效果大多被 RV3 吸收，方向與「震盪好」相反 | — | RV3 帶入 S3；經典震盪指標放棄 |
| E004 | 2026-11-06 11:25 | dev | 帳戶回測 12 組（3 方向 × 2 k × {無, RV3}），100 USD、K=3（neutral K=2） | 本 commit | 1,092 日 | 最佳 long k1.0 RV3：+100 USD/年、t 3.3、maxDD −71；long k0.5 RV3：+128/年、t 3.0、maxDD −71 | 配對差 t 2.3 / 3.1 | maxDD 超過 40 USD 門檻 |
| E005 | 2026-11-06 11:30 | dev | 壓測：手續費與滑價加倍 + 限價穿價 0.2% | 本 commit | 同上 | long k1.0 RV3 +86/年 t 2.8；long k0.5 RV3 +94/年，但 2022、2023H1 皆負 | — | — |
| E006 | 2026-11-06 11:30 | dev | 同 bar 先止盈的有利順序 | 本 commit | 同上 | 與 E004 幾乎相同（差 < 1 USD） | — | 路徑假設不影響結論 |
| E007 | 2026-11-06 11:40 | dev | beta 拆解（episode 回歸 fRET，日群聚）+ 選幣隨機 seed 噪音（10 seeds）+ 2022-01~2023-06 子期間 | 本 commit | 13,377 ep | RV3 多單網格 alpha 0.06%/ep（t 1.0）、k1.0 0.16%（t 1.9）；無濾網帳戶年化在 seeds 間 sd 26 USD；排除 2021 牛市後 long k1.0 RV3 −8 USD/年（壓測 −19），maxDD −60 | — | 獲利集中在 2020H2–21 山寨牛市 |

## Split read/peek log

| UTC | Split | 操作 | 已看結果？ | 受污染假設/候選 | Commit |
|---|---|---|---|---|---|
| 2026-11-06 | dev | E001–E007 全部 | 是 | 全部 | 本 commit |
| — | val | 未讀取（程式只對 dev 期間跑模擬） | 否 | — | — |
| — | hold | 未讀取 | 否 | — | — |

## 結案/階段變化

| UTC | 前階段 → 後階段 | 事前門檻 | 證據摘要 | 判定 | Commit |
|---|---|---|---|---|---|
| 2026-11-06 10:40 | S0 → S1 | 假設可反駁、切分已登記 | HYPOTHESIS.md | APPROVE | b1ed4e7 |
| 2026-11-06 11:15 | S2 → S3 | Holm 顯著 + 五分位差 ≥ 0.3% + 三期一致 | 僅 RV3 過；經典震盪指標對未來震盪無預測力 | APPROVE（只帶 RV3） | 本 commit |
| 2026-11-06 11:45 | S3 → 結案 | t>2、配對 t>2、≥2/3 期正、壓測 >0、maxDD<40 USD | 12 組全數 maxDD 超過 40 USD；最佳組的獲利 >100% 來自 2020H2–21；beta 中性版本（neutral）壓測後 t 0.7 | **REJECT**，不使用 val/hold | 本 commit |
