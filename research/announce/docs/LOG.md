# ANN: exchange announcement event alpha

Thread 研究交易所公告事件 Alpha, branch claude/project-thread-9zcx4f. Protocol: alpha-research (S0..S7).

## S0 THESIS (2026-11-06)

Exchange announcements change a coin's access to liquidity (new venue, removal, risk tag).
The flow they trigger is forced or mechanical (new buyers on Binance/Upbit KRW, holders
exiting before delisting, risk-averse funds cutting tagged coins), so the price impact may
not be absorbed in the first seconds, or may overshoot and revert.

Differs from NLA: NLA measured the new coin after its first Binance trade. ANN measures
coins that are ALREADY tradeable (mostly on a Binance USDS-M perp) at the moment of the
announcement, so we can act on them.

Event types (t0 = announcement publish time):
- BN_LIST: Binance spot listing of a coin that already has a Binance perp.
- BN_FUT: Binance new perp listing (coin already trading elsewhere; only spot reaction is
  observable on Binance, if spot exists).
- UP_LIST: Upbit KRW market listing; reaction of the coin on Binance perp.
- BN_DELIST: Binance spot delisting announcement; reaction on perp if it still trades.
- BN_MON: monitoring-tag additions.
- UP_WARN: Upbit investment warning (유의종목) designation, if data allows.

Hypotheses (two directions per type, decided only on dev):
- H_drift: after a realistic detection latency (entry >= t0 + 1m), the move continues.
- H_fade: the first-hour move over-shoots and reverts over 1h-24h.

Simple baselines: same coin, same clock time on a random non-event day; BTC and alt
basket over the same window.

Realistic latency for us: poll the announcement API every few seconds -> entry at the
close of the first full 1m bar after t0 at best; also test +2m and +5m.

## Data

- Announcements: Binance CMS article API and Upbit announcements API, fetched on babypig
  (cloud proxy blocks both) with research/announce/fetch_announcements.py.
- Prices: data.binance.vision via S3 path-style endpoint (research/announce/archive.py,
  copied from NLA).
