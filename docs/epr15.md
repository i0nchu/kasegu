# EPR15 正式整合

來源：git tag `alpha-epr15-v1`（`research/claude-alpha`），規格 `research/alpha/docs/HANDOFF_codex.md`，
研究實作 `research/alpha/a3_candidate.py` + `research/sim.py`。本文件只描述整合，不改研究結論與參數。

## 1. 模組

| 部分 | 位置 | 說明 |
|---|---|---|
| 進場 | `src/kasegu/strategies/epr15.py` | `Epr15(AtrAmplitude)`，重用 `indicators.ema / atr / amplitude_pct` 與 `DistanceStrategy` 的掛單流程 |
| 出場 | `src/kasegu/exits/time_only.py` | `TimeOnlyExit`：固定停損 + 成交後 N 分鐘市價出場，不啟動追蹤級距 |
| 出場介面 | `src/kasegu/exits/base.py` | `ExitPolicy` 與 `create_exit`；原本的三級追蹤改名 `TrailingExit`（`ExitManager` 仍可匯入） |
| 設定 | `configs/epr15.example.yaml` | Shadow / Testnet / 小額帳戶起點：6 幣、每筆 19.8 USDT、保守風控 |
| 研究重現設定 | `configs/epr15.research.yaml` | 研究的 40 幣、每筆 297 USDT、無全域上限；只用於回測、比較與 Shadow 對照，**不是實盤建議** |
| 比較 | `scripts/epr15_compare.py` | 研究交易與正式引擎 tick 回放逐筆比對 |

Engine、Broker、RiskManager、PositionManager 沒有複製；backtest、paper、live 用同一個 `Epr15` 與 `TimeOnlyExit`。
原本的 `atr_amplitude`、`atr_only`、`quantile` 與 `trailing` 出場不變，回歸測試照常通過；
`configs/config.example.yaml`（atr_amplitude）在 SOLUSDT、DOGEUSDT 2026-10-06~07 回測，main 與本分支結果完全相同（5 筆、淨 −0.9469 USDT、2,808 張進場單）。

## 2. 凍結參數

| 項目 | 值 | 設定 |
|---|---|---|
| 方向 | 只做多 | 程式固定 |
| 趨勢 | 下單那根 1m 收盤 < EMA(240)，`adjust=False` | `ema_period: 240` |
| 波動單位 vol% | (mean(TR, 60) / 收盤 × 100 + mean((H − L) / C × 100, 60)) / 2 | `vol_period: 60` |
| 掛單價 | Ask × (1 − 12 × vol% / 100)，向下取整到 tick | `k: 12`、`reference: ask` |
| 委託 | Post-Only（GTX）限價買單，每根 1m 收盤撤單重掛 | 引擎既有流程 |
| 預熱 | 720 根 | `ema_period × 3` |
| 停損 | 成交均價 −3%，交易所端 reduce-only `STOP_MARKET`（標記價格觸發） | `exit.stop_loss_pct: 3.0` |
| 時間出場 | 成交後 15 分鐘 reduce-only 市價賣出 | `exit.policy: time_only`、`max_holding_minutes: 15` |
| 追蹤止盈 | 不啟動 | `time_only` 忽略 tier 設定 |

`Epr15` 在建構時檢查 `k / ema_period / vol_period / reference`，`params` 必須為空；
`Engine` 與 CLI 啟動時呼叫 `check_exit`，出場不是 `time_only / 3% / 15 分鐘` 就拒絕執行。
`--k`、`--strategy` 覆寫若破壞凍結設定會直接以 exit code 2 結束。

## 3. 研究版與正式引擎的語意差異

不是完全等價。研究模擬器以 1m K 線近似，正式引擎以事件驅動（live / paper 為即時行情，回測為 aggTrades 逐筆回放）。
下表列出每一項差異、方向與在哪個測試或比較中看得到。

| # | 類別 | 研究模擬器（`sim.py`） | 正式引擎 | 影響 |
|---|---|---|---|---|
| 1 | 時間語意 | 訊號在第 i 根收盤決定，掛單只在第 i+1 根有效 | 同樣在收盤決定；掛單從送出到下一根收盤撤單為止。Live 有撤單→重掛的網路延遲，期間沒有掛單或舊單多掛幾百毫秒 | 研究無延遲；需 Testnet 量測 |
| 2 | 時間語意 | 15 根 K 線：成交那根算第 1 根，第 15 根**收盤**市價出場，實際持有 14~15 分鐘（成交在那根內的位置未知） | 成交時間戳起算滿 15 分鐘，下一個標記價格事件出場（出場原因 `time_exit`，研究叫 `max_hold`）；live 標記價格每秒更新，回放在無成交時段會晚到下一筆成交 | 引擎平均晚 36 秒出場（比較：1.1~60 秒） |
| 3 | 時間語意 | 出場那根收盤即可掛下一張單（出場與下一張單同一時刻） | 出場在第 j+15 根中段，下一張單要等那根收盤，最早第 j+16 根成交 | 引擎可能少一筆緊接的交易（比較中 1 筆 `time:in_position`） |
| 4 | 成交模型 | 下一根最低價 + tick/2 < 掛單價 | 回放：成交價 + tick/2（Ask）< 掛單價，同條件；paper：bookTicker Ask < 掛單價；live：真實排隊 | 回放與研究一致（551/556 進場價相同）；live 成交率會較低 |
| 5 | 成交模型 | 掛單參考價 = 收盤 + tick/2 | 參考價 = 當下 Ask（回放中等於最後成交 + tick/2） | 5 筆差 1 tick（浮點取整邊界） |
| 6 | 分鐘內順序 | 假設路徑：陽線 O→L→H→C、陰線 O→H→L→C；成交後從掛單價接著走 | 真實逐筆順序 | 研究可能把停損算在反彈之前或之後 |
| 7 | 出場模型 | 3% 停損以**最後成交價**的 1m 低點判斷；觸價成交在 −3% − 0.03% 再扣滑價；跳空則用開盤價 | 交易所 `STOP_MARKET` 以**標記價格**觸發（回放以每秒最後成交近似），觸發後市價以 Bid − 滑價成交；另有軟體端 −3% 市價出場，兩者都是 reduce-only | 亞秒級插針研究會停損、引擎不會（11 筆）；真正停損時引擎成交更差（61 筆平均 −3.52% vs −3.06%） |
| 8 | 出場模型 | 時間出場價 = 第 15 根收盤 − tick/2 − 滑價 | 出場當下 Bid − 滑價 | 與 #2 合併，平均差 +0.003% |
| 9 | 手續費 | Maker 0.02% 進 + Taker 0.05% × 出場價/進場價 | 每筆成交按成交價乘費率（或帳戶實際費率 `use_account_rate`） | 公式相同；只隨出場價不同而變 |
| 10 | 滑價 | 出場 0.02%，停損再加 0.03% | `fee.slippage_pct` 0.02%，停損沒有額外 0.03% | 研究停損假設較保守，但實測引擎停損滑價更大（#7） |
| 11 | 資金費率 | [成交, 出場+1 分) 期間的費率加總 × 方向 | 結算時刻持倉時，以當下標記價格 × 數量收付 | 4 筆差異，量很小 |
| 12 | EMA 種子 | 從 2024-10 起算 | 最近 1500 根（live `history_bars` ≤ 1500）；種子權重 (1−2/241)^1500 ≈ 4e−6 | 可忽略 |
| 13 | 倉位與風控 | 每幣一倉、沒有全域上限、名義固定 297 USDT（以 % 計） | `max_positions`、`max_notional`（含掛單中的名義）、`max_loss_per_trade`、每日虧損與連敗暫停；數量依 lot size 向下取整 | 正式環境交易數會比研究少；比較時關閉全域上限 |
| 14 | 資料保護 | 無 | 報價逾時（`stale_after_s`）、價差、委託簿數量、未知委託狀態都會拒絕進場 | 回放中安靜時段沒有成交就沒有報價更新（1 筆 `stale_data`），live 的 bookTicker 不會有這個問題 |

對應的自動化測試在 `tests/test_epr15.py`：EMA 方向與只做多、掛單距離與 tick 取整、720 根預熱、
凍結參數拒絕、time_only 不啟動 tier、時間出場的 15 分鐘邊界、交易所停損先成交後不重複平倉、
時間出場與停損競爭（reduce-only 拒單、不會開出反向倉位）、部分成交與撤單競爭、撤單結果未知、
停損狀態未知、停損被交易所取消、重啟後沿用成交時間、報價逾時、全域倉位與曝險上限、連敗暫停、
最小名義與 lot size、提高槓桿無法繞過單筆虧損上限。

## 4. 逐筆比較結果

`uv run python scripts/epr15_compare.py --jobs 8`（約 1 分鐘），完整數字在 [epr15_compare.json](epr15_compare.json)。

範圍：凍結研究結果 `research/results/A3_candidate/EPR15/trades.csv` 中，本機有 aggTrades 快取的
215 天、40 幣、509 個（幣，日）組合，共 560 筆研究交易（全部 1,918 筆的 29%）。這些日子是 A2 為了檢查成交真實性而抽樣的交易日，
不是隨機抽樣，所以只用來比對語意，不拿來估計績效。引擎每次回放一個幣、一個 UTC 日，使用 `configs/epr15.research.yaml` 並關閉全域風控上限，成本與研究相同。
審查修正（委託確認、對帳）後重跑，結果與修正前完全相同。

| 項目 | 結果 |
|---|---|
| 研究交易 / 引擎交易 / 配對成功 | 560 / 556 / 556（99.3%） |
| 進場價完全相同 | 551 / 556，其餘 5 筆差 1 tick |
| 研究有、引擎沒有 | 4 筆：2 筆掛單在回放日前一分鐘（邊界）、1 筆 `stale_data`（回放無成交 > 120 秒）、1 筆引擎仍持有上一筆 |
| 引擎有、研究沒有 | 0 |
| 跨日被回放截斷（排除） | 14 筆 |

已配對且未跨日的 542 筆，依主要差異分類（`d_net` = 引擎淨報酬 − 研究淨報酬，%／筆）：

| 類別 | 筆數 | 平均 d_net | 說明 |
|---|---|---|---|
| time（兩邊都是時間出場） | 470 | +0.003% | 中位數絕對差 0.05%；引擎平均晚 36 秒出場，持有 15.00~15.24 分 |
| exit:stop_path（兩邊都停損） | 61 | −0.46% | 引擎停損成交平均 −3.52%（中位數 −3.24%，最差 −6.36%），研究 −3.06% |
| exit:stop_trigger（研究停損、引擎時間出場） | 11 | +4.94% | 研究在 70 毫秒級的插針停損；引擎每秒取樣標記價格沒有觸發 |

平均淨報酬：研究 +0.256%，引擎 +0.307%。差距幾乎全部來自出場模型（#7）：
引擎對插針較寬容、但真正跌破時停損更差，兩者方向相反。時間語意（#2、#3）與成交模型（#4、#5）的影響很小。

需要注意：
- `stop_trigger` 的有利結果依賴「交易所以標記價格觸發停損」。回放以每秒最後成交近似標記價格，
  真實標記價格的平滑程度不同，這部分要用 Testnet 與交易所實際紀錄確認，不能當作額外的 alpha。
- 停損滑價比研究假設大，研究 A3 的 slip 敏感度（slip0.2 淨值歸零）應以這個實測值重新看待。
- 這只驗證「同一套規則在兩個模擬器中產生同樣的交易」，不驗證真實成交率；live 的排隊位置會讓成交數變少，且偏向更差的成交。

## 5. Shadow Trading 操作

Shadow 指「正式行情 + 模擬成交」：`--mode paper`，行情來自 Binance 正式環境（`exchange.data_testnet: false`），
委託只進本地 `PaperBroker`，不會送出任何訂單，也不需要 API key。

```bash
cp configs/epr15.example.yaml configs/epr15.yaml     # 依帳戶調整 symbols 與 risk
# 想看研究同規模的 Shadow 對照，另開一份 configs/epr15.research.yaml（不同 journal）
uv run python -m kasegu --mode paper --config configs/epr15.yaml --log-file data/logs/epr15.log
```

Docker：在 `.env` 設 `KASEGU_MODE=paper`、`KASEGU_CONFIG=/app/configs/epr15.yaml`，再 `docker compose up -d --build`。

產出（與其他策略分開）：

| 檔案 | 內容 |
|---|---|
| `data/journal-epr15.db` | orders、fills、trades、events（含每根收盤的 `signal`） |
| `data/events-epr15.jsonl` | 告警事件 |
| `data/status-epr15.json` | 目前掛單、倉位、暫停狀態，每 5 秒更新 |

每日檢查：

```bash
sqlite3 data/journal-epr15.db "select symbol, json_extract(data,'$.reason') r, count(*),
  round(avg(json_extract(data,'$.pnl') - json_extract(data,'$.fees') + json_extract(data,'$.funding')),4)
  from trades group by 1, 2"
sqlite3 data/journal-epr15.db "select json_extract(data,'$.reason'), count(*) from events
  where kind = 'signal' group by 1 order by 2 desc"
```

看的重點：

1. 每幣每月約 2 筆成交（研究 1,918 筆 / 40 幣 / 24 月）；明顯偏多代表成交判定或波動計算有誤。
2. `time_exit` 與 `stop_loss / exchange_stop` 比例，研究約 87% / 13%。
3. 時間出場的實際持有時間應落在 15:00 ~ 15:02（標記價格每秒一次）。
4. `signal` 被拒原因：`would_take`（Post-Only 會吃單）、`stale_data`、`max_positions`、`max_notional`。
5. 任何 critical 事件（`order_unknown`、`stop_lost`、`stop_rejected`、`loss_limit`、`disconnected`）。

Paper 的限制：成交只看 bookTicker 的 Ask 低於掛單價，不模擬排隊位置，成交率偏樂觀；
重啟後 paper 倉位不保留（`sync` 以空的 PaperBroker 為準）。成交品質要到 Testnet / 小額實盤才看得到。

之後的 Testnet 驗證：`exchange.testnet: true`，先跑 `uv run python -m kasegu.exchange.check --symbol BTCUSDT`，
再 `--mode live --confirm-live`。Testnet 行情與正式環境不同，只驗證下單流程，不驗證策略績效。

## 6. 交易安全與異常處理

**阻擋進場的條件**（`Engine.blocked`，任何一個幣種被阻擋，所有幣種都不掛新進場單；既有倉位照常保護與出場）：

| 事件 | 觸發 | 引擎動作 |
|---|---|---|
| `order_unknown` | 下單重試 3 次都失敗、查詢也失敗 | 該單列為未確認，持續追蹤 |
| `entry_cancel_unconfirmed` | 撤單失敗或交易所回傳仍為 NEW / PARTIALLY_FILLED（含部分成交後撤剩餘量） | 不當作已撤，不掛新單，之後的成交照常套用並調整停損 |
| `stop_unknown` | 新停損送單結果未知 | 保留舊停損（不撤）；舊停損不足以覆蓋整個倉位時，以 reduce-only 市價平倉 |
| `stop_rejected` | 新停損被拒 | 同上，保留舊停損或平倉 |
| `stop_cancel_unconfirmed` | 換新停損或平倉後，舊停損撤不掉 | reduce-only，不會開倉，但等對帳清掉 |
| `stop_lost` | 交易所端停損被取消或過期（非本程式撤單） | 立即重掛，重掛失敗則依上面兩列處理 |
| `exit_rejected` | 出場單被拒（常見於交易所停損剛好先成交） | 下一個標記價格重試，並對帳 |
| `sync_unconfirmed` | 對帳時任何一項無法確認，或對帳本身失敗 | 維持阻擋 |

**停損替換順序**：倉位變大（部分成交後又成交）時，先送新停損，交易所確認有效且數量覆蓋整個倉位後才撤舊停損。
新停損未確認時舊停損一定保留。

**對帳（`sync`）解除阻擋的條件**，每個幣種都要同時成立，否則保持阻擋並每 30 秒重試：

1. 先前未確認的委託都已查到結果（查不到 = 交易所沒有這張單）。
2. 交易所上本程式的進場單、出場單全部撤銷成功（含上次程式異常終止時留下的）。
3. 本地倉位改為交易所數字（不符時發 `position_resized` / `position_gone` / `position_adopted`）。
4. 有倉位時：存在一張已確認、方向正確、剩餘數量覆蓋整個倉位的停損；多餘的停損都撤銷成功。
   重啟時舊停損數量不符（例如停機前又有成交）：先送新的全量停損，交易所確認後才撤舊停損，
   確認前最大的一張舊停損一直保留；方向錯誤的停損（不具保護作用）直接撤銷。
   新停損未確認或被拒時舊停損全部保留、阻擋進場，並以 reduce-only 市價單平倉。
5. 空倉時：所有停損撤銷成功，本地 `entry / stop / close` 參照全部清空，下一筆成交一定重新掛停損。

**程式異常終止**（kill -9、當機、斷電）：交易所上的進場單會留著，最多再活到下次啟動。
重新啟動時的 `sync` 會撤掉所有本程式留下的進場與出場單（`vt` 開頭的 client id，其他工具的單不動）；
若這段期間進場單已成交，倉位會被接手（`position_adopted`）並掛上停損。
停機期間的保護只有交易所端停損；停機中成交的進場單在重啟前**沒有**停損。
要縮短這個空窗，可在 Testnet 驗證後開啟 `live.auto_cancel_s`（交易所倒數撤單，只在空倉時啟用）。

**斷線或停機超過 15 分鐘**：

- 重啟：本地倉位從 journal 讀回，成交時間不變，第一個標記價格就以 `time_exit` 平倉。
- 未重啟只斷線：沒有標記價格就不會出場，交易所停損仍在；串流恢復後第一個標記價格平倉。
- 接手的倉位（本地沒有紀錄）從接手時刻起算 15 分鐘，因為無法得知原本的成交時間。

**`max_loss_per_trade` 只是預估**：以 `名義 × stop_loss_pct` 計算，代表價格剛好停在 −3% 時的損失。
不包含跳空、標記價格落後、停損觸發後的市價滑價與手續費。第 4 節的回放中，真正停損的成交平均 −3.52%，最差 −6.36%，
實際單筆損失可能是預估的兩倍以上。最壞情況要以 `max_positions × 名義 × 跳空幅度` 估算，
因為急殺時所有掛著的買單可能同時成交（掛單中的單也計入 `max_positions` 與 `max_notional`）。

**小額帳戶**：啟動時以最新收盤價檢查每個幣種的下單數量，低於 `minQty` 或 `MIN_NOTIONAL` 的會發 `untradable` 警告，
這些幣種的訊號會被 Risk Manager 拒絕，數量永遠向下取整，不會為了達到最小名義而放大。

## 7. Live 上線前尚未完成的安全檢查

這個 PR 不啟用 live。以下項目完成並留下紀錄前，不應以正式資金執行：

- [ ] 獨立 Code Review 通過本 PR（策略、`time_only`、引擎安全修改）。
- [ ] Codex 獨立驗證（`HANDOFF_codex.md` 第 5 節）完成，特別是 aggTrades 實測出場滑價：
      中位數 > 0.2% 時現象不可交易（研究 A3 slip0.2 淨值歸零）。
- [ ] Shadow 至少 2 週，每幣每月成交數、出場原因比例、持有時間與第 5 節預期一致。
- [ ] Testnet：`kasegu.exchange.check` 全部通過；實際跑出 Post-Only 被拒（-5022）、
      撤單重掛、部分成交、停損觸發、時間出場、重啟接手各至少一次，journal 與交易所紀錄一致。
- [ ] Testnet 量測每根 1m 收盤到撤單＋重掛完成的延遲；急殺時 Post-Only 是否被拒、重掛延遲是否 < 數秒。
- [ ] 斷線演練：拔網路、`docker compose stop/start`、kill -9，確認倉位保有交易所停損、
      `order_unknown` 告警與阻擋進場、恢復後 `reconciled`。
- [ ] Telegram 或 webhook 告警實際送達（critical 事件）。
- [ ] 小額帳戶：從 `configs/epr15.example.yaml` 開始，不要用 `epr15.research.yaml`；啟動 log 沒有 `untradable`；
      資金不足時降低 `margin_budget` 或減少幣種，不要提高 `leverage`；
      `max_positions × 名義 × 可能跳空幅度`、`max_daily_loss` 在可承受範圍內。
- [ ] 帳戶為單向持倉、逐幣槓桿已設好，帳戶中沒有其他程式的掛單或倉位（會被接手）。
- [ ] 確認 `live.auto_cancel_s` 的設定（預設 0）；開啟前先在 Testnet 跑 check 第 2 段。
- [ ] 資金費率：持有 15 分鐘通常不跨結算，但跨結算時照實收付，Shadow 報表要能看到。
- [ ] 已知未處理：交易所停損以標記價格觸發，急殺時標記價格落後成交價，實際停損價可能比 −3% 更差；
      研究假設 −3% 再多 0.03%。需要 Testnet / aggTrades 實測。
- [ ] 時間出場與交易所停損同時觸發時，reduce-only 會拒絕第二張單，引擎發 `exit_rejected`（critical 一次）、
      暫停新進場並在 30 秒內對帳。需在 Testnet 實際觸發一次確認流程。
- [ ] 上線當日由人工執行 `--mode live --confirm-live`，`exchange.testnet: false` 只在確認後手動修改。
