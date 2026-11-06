"""Fetch historical exchange announcements (Binance CMS, Upbit notices).

Run on a machine that can reach the exchanges (stdlib only):
    python research/announce/fetch_announcements.py --out research/announce/data
Writes JSONL (raw records) plus a fetch_log.txt. Safe to rerun; resumes details.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (research script)", "Accept": "application/json"}
BN = "https://www.binance.com/bapi/composite/v1/public/cms"
# 48 new listing, 161 delisting, 49 latest news, 50 new fiat, 93 activities,
# 157 airdrop/hodler (ids not all guaranteed; unknown ones are logged and skipped)
BN_CATALOGS = [48, 161, 49, 50, 93, 157, 51, 128]
DETAIL_CATALOGS = {48, 161}
DETAIL_KEYWORDS = ("monitoring tag", "seed tag", "delist", "cease", "will list",
                   "futures will launch", "perpetual contract", "hodler", "alpha")


def get(url, log, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            log(f"ERR {url} try{i}: {e!r}")
            time.sleep(2 ** i)
    return None


def binance_list(out, log):
    path = out / "binance_list.jsonl"
    seen = set()
    with path.open("w", encoding="utf-8") as f:
        for cid in BN_CATALOGS:
            page, n = 1, 0
            while True:
                url = f"{BN}/article/list/query?type=1&catalogId={cid}&pageNo={page}&pageSize=50"
                js = get(url, log)
                cats = ((js or {}).get("data") or {}).get("catalogs") or []
                arts = []
                for c in cats:
                    for a in c.get("articles") or []:
                        a["catalogId"] = c.get("catalogId", cid)
                        a["catalogName"] = c.get("catalogName")
                        arts.append(a)
                if not arts:
                    break
                for a in arts:
                    key = (a.get("code"), a["catalogId"])
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
                    n += 1
                page += 1
                time.sleep(0.4)
                if page > 400:
                    break
            log(f"binance catalog {cid}: {n} articles, {page - 1} pages")
    return path


def binance_details(out, log):
    src = out / "binance_list.jsonl"
    dst = out / "binance_detail.jsonl"
    done = set()
    if dst.exists():
        for line in dst.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["code"])
            except Exception:  # noqa: BLE001
                pass
    todo = []
    for line in src.open(encoding="utf-8"):
        a = json.loads(line)
        t = (a.get("title") or "").lower()
        if a["catalogId"] in DETAIL_CATALOGS or any(k in t for k in DETAIL_KEYWORDS):
            if a.get("code") and a["code"] not in done:
                todo.append(a["code"])
    todo = list(dict.fromkeys(todo))
    log(f"binance details todo {len(todo)} (done {len(done)})")
    with dst.open("a", encoding="utf-8") as f:
        for i, code in enumerate(todo):
            js = get(f"{BN}/article/detail/query?articleCode={code}", log)
            d = (js or {}).get("data")
            if d:
                f.write(json.dumps({"code": code, "data": d}, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                log(f"  details {i}/{len(todo)}")
            time.sleep(0.3)


def upbit(out, log):
    path = out / "upbit_list.jsonl"
    n = 0
    with path.open("w", encoding="utf-8") as f:
        page = 1
        while True:
            q = urllib.parse.urlencode({"os": "web", "page": page, "per_page": 20,
                                        "category": "all"})
            js = get(f"https://api-manager.upbit.com/api/v1/announcements?{q}", log)
            d = (js or {}).get("data") or {}
            items = d.get("notices") or d.get("list") or []
            if not items:
                break
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n += 1
            if page == 1:
                log(f"upbit total_pages={d.get('total_pages')} total={d.get('total_count')}")
            if d.get("total_pages") and page >= d["total_pages"]:
                break
            page += 1
            time.sleep(0.4)
    log(f"upbit announcements: {n}")
    # legacy notice API (older history), best effort
    path2 = out / "upbit_legacy.jsonl"
    n2 = 0
    with path2.open("w", encoding="utf-8") as f:
        page = 1
        while page < 2000:
            q = urllib.parse.urlencode({"page": page, "per_page": 20, "thread_name": "general"})
            js = get(f"https://api-manager.upbit.com/api/v1/notices?{q}", log, tries=2)
            d = (js or {}).get("data") or {}
            items = d.get("list") or []
            if not items:
                break
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n2 += 1
            if d.get("total_pages") and page >= d["total_pages"]:
                break
            page += 1
            time.sleep(0.4)
    log(f"upbit legacy notices: {n2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/announce/data")
    ap.add_argument("--skip", default="", help="comma list: bnlist,bndetail,upbit")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = set(a.skip.split(","))
    logf = (out / "fetch_log.txt").open("a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if "bnlist" not in skip:
        binance_list(out, log)
    if "bndetail" not in skip:
        binance_details(out, log)
    if "upbit" not in skip:
        upbit(out, log)
    log("done")


if __name__ == "__main__":
    main()
