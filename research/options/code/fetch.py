"""Fetch Deribit/OKX option data (stdlib only). Run where exchanges are reachable."""
import csv, gzip, json, sys, time, urllib.request, urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "research/options/data")
OUT.mkdir(parents=True, exist_ok=True)
DB = "https://www.deribit.com/api/v2/public/"
HIST = "https://history.deribit.com/api/v2/public/"
START = datetime(2021, 4, 1, tzinfo=timezone.utc)
NOW = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def get(url, params=None, tries=6):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kasegu-research"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def ms(dt):
    return int(dt.timestamp() * 1000)


def write_gz(name, header, rows):
    with gzip.open(OUT / name, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(name, len(rows), flush=True)


def dvol(cur):
    rows, t = {}, START
    while t < NOW:
        e = min(t + timedelta(hours=999), NOW)
        d = get(DB + "get_volatility_index_data", dict(currency=cur, start_timestamp=ms(t), end_timestamp=ms(e), resolution=3600))
        for r in d["result"]["data"]:
            rows[r[0]] = r
        t = e
    write_gz(f"dvol_{cur}_1h.csv.gz", ["ts", "o", "h", "l", "c"], [rows[k] for k in sorted(rows)])


def perp(cur):
    rows, t = {}, START
    while t < NOW:
        e = min(t + timedelta(hours=999), NOW)
        d = get(DB + "get_tradingview_chart_data", dict(instrument_name=f"{cur}-PERPETUAL", start_timestamp=ms(t), end_timestamp=ms(e), resolution=60))["result"]
        for i, ts in enumerate(d.get("ticks", [])):
            rows[ts] = [ts, d["open"][i], d["high"][i], d["low"][i], d["close"][i]]
        t = e
    write_gz(f"perp_{cur}_1h.csv.gz", ["ts", "o", "h", "l", "c"], [rows[k] for k in sorted(rows)])


def delivery(cur):
    rows, off = [], 0
    while True:
        d = get(DB + "get_delivery_prices", dict(index_name=f"{cur.lower()}_usd", offset=off, count=1000))["result"]
        rows += [[x["date"], x["delivery_price"]] for x in d["data"]]
        off += 1000
        if off >= d["records_total"] or not d["data"]:
            break
    rows.sort()
    write_gz(f"delivery_{cur}.csv.gz", ["date", "price"], rows)


def expiry(name):
    s = name.split("-")[1]
    return datetime(2000 + int(s[-2:]), MON[s[-5:-2]], int(s[:-5]), 8, tzinfo=timezone.utc)


def day_trades(cur, day):
    t0, t1 = ms(day + timedelta(hours=8)), ms(day + timedelta(hours=9))
    out, seen, start = [], set(), t0
    for _ in range(20):
        d = get(HIST + "get_last_trades_by_currency_and_time", dict(currency=cur, kind="option", start_timestamp=start, end_timestamp=t1, count=1000, sorting="asc"))["result"]
        tr = d["trades"]
        for x in tr:
            if x["trade_id"] in seen:
                continue
            seen.add(x["trade_id"])
            out.append(x)
        if not d.get("has_more") or not tr:
            break
        start = tr[-1]["timestamp"]
    agg = defaultdict(list)
    for x in out:
        try:
            ex = expiry(x["instrument_name"])
        except Exception:
            continue
        if (ex - day).total_seconds() / 86400 <= 4.5:
            agg[x["instrument_name"]].append(x)
    rows = []
    for name, xs in agg.items():
        xs.sort(key=lambda x: x["timestamp"])
        amt = sum(x["amount"] for x in xs)
        vw = sum(x["price"] * x["amount"] for x in xs) / amt
        f = xs[0]
        rows.append([day.date().isoformat(), name, len(xs), round(amt, 3), f["timestamp"], f["price"], f.get("mark_price"), f.get("iv"), f.get("index_price"),
                     round(vw, 6), min(x["price"] for x in xs), max(x["price"] for x in xs),
                     round(sum(x.get("iv", 0) or 0 for x in xs) / len(xs), 2), sum(1 for x in xs if x.get("direction") == "buy")])
    return rows


def trades(cur):
    days = [START + timedelta(days=i) for i in range((NOW - START).days)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda d: _safe(day_trades, cur, d), days))
    rows = [r for rs in res for r in rs]
    write_gz(f"opt_trades_{cur}_0809.csv.gz", ["day", "instrument", "n", "amount", "first_ts", "first_price", "first_mark", "first_iv", "first_index",
                                                "vwap", "min_price", "max_price", "mean_iv", "n_buy"], rows)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print("ERR", a, e, flush=True)
        return []


def snapshot():
    snap = {"ts": NOW.isoformat(), "taken": datetime.now(timezone.utc).isoformat()}
    for cur in ("BTC", "ETH"):
        snap[f"deribit_{cur}_summary"] = _safe(lambda: get(DB + "get_book_summary_by_currency", dict(currency=cur, kind="option"))["result"])
        snap[f"deribit_{cur}_instruments"] = _safe(lambda: get(DB + "get_instruments", dict(currency=cur, kind="option"))["result"])
        fam = f"{cur}-USD"
        okx = "https://www.okx.com/api/v5/"
        snap[f"okx_{cur}_tickers"] = _safe(lambda: get(okx + "market/tickers", dict(instType="OPTION", instFamily=fam)))
        snap[f"okx_{cur}_instruments"] = _safe(lambda: get(okx + "public/instruments", dict(instType="OPTION", instFamily=fam)))
        snap[f"okx_{cur}_summary"] = _safe(lambda: get(okx + "public/opt-summary", dict(instFamily=fam)))
    # order books for nearest 2-4 day expiries, OTM strikes
    books = {}
    for cur in ("BTC", "ETH"):
        summ = snap.get(f"deribit_{cur}_summary") or []
        idx = next((s["underlying_price"] for s in summ if s.get("underlying_price")), None)
        if not idx:
            continue
        for s in summ:
            n = s["instrument_name"]
            dte = (expiry(n) - datetime.now(timezone.utc)).total_seconds() / 86400
            k = float(n.split("-")[2])
            otm = (k / idx - 1) if n.endswith("C") else (1 - k / idx)
            if 1.5 <= dte <= 3.6 and 0.0 <= otm <= 0.12:
                books["deribit:" + n] = _safe(lambda: get(DB + "get_order_book", dict(instrument_name=n, depth=10))["result"])
        tick = snap.get(f"okx_{cur}_tickers") or {}
        for t in tick.get("data", []):
            n = t["instId"]  # BTC-USD-261108-70000-C
            p = n.split("-")
            ex = datetime.strptime(p[2], "%y%m%d").replace(hour=8, tzinfo=timezone.utc)
            dte = (ex - datetime.now(timezone.utc)).total_seconds() / 86400
            k = float(p[3])
            otm = (k / idx - 1) if p[4] == "C" else (1 - k / idx)
            if 1.5 <= dte <= 3.6 and 0.0 <= otm <= 0.12:
                books["okx:" + n] = _safe(lambda: get("https://www.okx.com/api/v5/market/books", dict(instId=n, sz=10)))
    snap["books"] = books
    with gzip.open(OUT / "snapshot.json.gz", "wt") as f:
        json.dump(snap, f)
    print("snapshot", len(books), flush=True)


if __name__ == "__main__":
    what = sys.argv[2].split(",") if len(sys.argv) > 2 else ["snap", "dvol", "perp", "delivery", "trades"]
    for cur in ("BTC", "ETH"):
        if "dvol" in what: _safe(dvol, cur)
        if "perp" in what: _safe(perp, cur)
        if "delivery" in what: _safe(delivery, cur)
    if "snap" in what: snapshot()
    if "trades" in what:
        for cur in ("BTC", "ETH"):
            _safe(trades, cur)
