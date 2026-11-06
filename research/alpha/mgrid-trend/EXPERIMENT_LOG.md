# Experiment ledger — mgrid-trend

| ID | UTC | Split | 變體 | 結果（淨，100 USD 帳戶） | 判定 |
|---|---|---|---|---|---|
| T000 | 2026-11-06 11:00 | — | 預先登記（commit e4cafaa） | — | — |
| T001 | 2026-11-06 11:10 | dev | 12 組 + LONG/SHORT/RANDOM 基準，K=3（results/dev_K3_summary.csv） | 最佳 BOTH k1.0 RV3：+111 USD/年、t 4.3、maxDD −33、三期 +291/+41/+0.5 | 對永遠做多配對差 +11/年、t 0.35，**未過 t>2 門檻** |
| T002 | 2026-11-06 11:15 | dev | 同上壓測（成本×2、穿價 0.2%） | BOTH k1.0 RV3：+97/年、t 3.8、maxDD −33；C_2023H1 −8 | 其他門檻皆過 |
| T003 | 2026-11-06 11:20 | dev | BOTH k1.0 RV3 的 K=1/2/3 | K=3 壓測 maxDD −33 < 40 → K=3 | — |
| T004 | 2026-11-06 11:25 | — | **偏離預先登記**：依登記規則應 REJECT 不跑 val（配對差門檻未過）。使用者明確要求不能只交「找不到」，因此凍結唯一候選 MGT-BOTH-k1.0-RV3-K3（FROZEN.json sha256 前 16 碼 c22e7ac20bef2626），在 val 上只跑一次。val 判定沿用 HYPOTHESIS.md；就算 val 通過，因 dev 增量門檻沒過，最多記為 INCONCLUSIVE→需 forward paper。 | — | 凍結 |

## Split read/peek log

| UTC | Split | 操作 | 已看結果？ |
|---|---|---|---|
| 2026-11-06 | dev | T001–T003 | 是 |
