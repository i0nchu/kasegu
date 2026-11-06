import gzip, json
import pandas as pd
s = json.load(gzip.open("../data/snapshot.json.gz"))
print(s["taken"])
for k, v in s.items():
    if k != "books":
        print(k, type(v).__name__, (len(v) if v else v) if not isinstance(v, dict) else list(v)[:5])
rows = []
for k, b in s["books"].items():
    if not b:
        continue
    ex, name = k.split(":", 1)
    if ex == "deribit":
        bids, asks = b.get("bids", []), b.get("asks", [])
        idx = b.get("underlying_price") or b.get("index_price")
        mark = b.get("mark_price")
        unit = 1.0  # contracts in underlying units
    else:
        d = b["data"][0]
        bids = [[float(x[0]), float(x[1])] for x in d["bids"]]
        asks = [[float(x[0]), float(x[1])] for x in d["asks"]]
        idx, mark, unit = None, None, 0.01
    bb = bids[0][0] if bids else None
    ba = asks[0][0] if asks else None
    p = name.replace("BTC-USD-", "BTC-").replace("ETH-USD-", "ETH-")
    rows.append(dict(ex=ex, inst=name, bid=bb, ask=ba, bid_sz=bids[0][1] * unit if bids else 0, ask_sz=asks[0][1] * unit if asks else 0,
                     ask_depth=sum(a[1] for a in asks[:5]) * unit, mark=mark))
df = pd.DataFrame(rows)
df["spread_pct"] = (df.ask - df.bid) / ((df.ask + df.bid) / 2) * 100
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df.sort_values(["ex", "inst"]).to_string())
