# Experiment ledger — bold-entry

| ID | UTC | Split | 內容 | 結果 | 判定 |
|---|---|---|---|---|---|
| E000 | 2026-11-06 16:50 | — | 預先登記 8 個因子、關卡、切分（HYPOTHESIS.md） | — | — |
| E001 | 2026-11-06 17:00 | dev | numba 模擬 vs `kasegu.bold.shot.Shot` 隨機 40 天 × 2 槓桿（code/parity.py） | 80/80 完全一致 | 模擬可信 |
| E002 | 2026-11-06 17:05 | dev | 8 因子 × 2 設定（L10、L20），三分位 HIT、NW t、Holm、ETH 與子期間一致性（results/S2_dev.csv） | 沒有任何因子通過。最大的是 L10 RET30 高三分位 +7.0pp（t 1.48），但 2022 年 −26.5pp；L20 RV7 低三分位 +8.3pp（t 1.97、Holm p 0.39），2023H1 反向。L10 Holm 後最小 p = 1.0 | **S2 未過 → REJECT**，依規則不跑 val/hold |
| E003 | 2026-11-06 17:10 | dev | 執行面消融（不是因子）：限價進場 + 限價止盈 vs 兩邊市價（results/exec_ablation_dev.csv） | BTC 10x HIT 43.9% → 45.2%（+1.3pp），平均權益 1.006 → 1.026；BTC 20x +0.8pp；ETH +0.3~0.9pp | 機械性的小改善，與理論一致（成本少約 1.5% 權益） |
