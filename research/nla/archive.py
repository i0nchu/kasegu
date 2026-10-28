"""Shared helpers for the data.binance.vision archive, read via the S3 path-style endpoint."""

import http.client
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = Path("data/history/vision")
KCOLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


def get(url: str) -> bytes:
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if i == 5:
                raise
        except (OSError, http.client.HTTPException):
            if i == 5:
                raise
        time.sleep(2**i)


def listing(prefix: str, dirs: bool) -> list[str]:
    out, marker = [], ""
    while True:
        q = urllib.parse.quote
        url = f"{S3}?prefix={q(prefix)}&marker={q(marker)}" + ("&delimiter=/" if dirs else "")
        x = get(url).decode()
        tag = "Prefix" if dirs else "Key"
        found = [f for f in re.findall(rf"<{tag}>([^<]+)</{tag}>", x) if f != prefix]
        out += found
        if "<IsTruncated>true</IsTruncated>" not in x or not found:
            return out
        marker = found[-1]


def fetch(key: str) -> Path | None:
    """Download one archive file into the cache; None when it does not exist."""
    path = CACHE / key.removeprefix("data/")
    miss = path.with_suffix(".404")
    if path.exists():
        return path
    if miss.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = get(f"{S3}/{urllib.parse.quote(key)}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            miss.touch()
            return None
        raise
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


def ms(t: np.ndarray) -> np.ndarray:
    """Spot files from 2025 on use microseconds; bring every timestamp to milliseconds."""
    t = t.astype(np.int64)
    return np.where(t > 10**14, t // 1000, t)


def read_klines(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(df.iat[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = KCOLS
    df = df.drop(columns="ignore").astype(float)
    df["open_time"] = ms(df["open_time"].to_numpy())
    df["close_time"] = ms(df["close_time"].to_numpy())
    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
