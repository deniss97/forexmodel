"""Свежие минутные свечи акций с Мосбиржи (ISS, бесплатно) — для работы стратегии в реальном времени.

Берёт историю data/raw/<TICKER>_1m_*.csv, докачивает минуты после её последней свечи с
iss.moex.com (режим TQBR, тот же формат колонок: open, close, high, low, value, volume, begin, end)
и пишет историю + свежие минуты в data/raw/live/<TICKER>_1m_live.csv. Исходные файлы не меняются;
при повторном запуске докачивается только новое.

    python scripts/fetch_moex_iss.py                 # SBER GAZP LKOH
    python scripts/fetch_moex_iss.py SBER MOEX
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LIVE = RAW / "live"
SOURCES = {"SBER": "SBER_1m_15-26.csv", "GAZP": "GAZP_1m_15-26_upd.csv", "LKOH": "LKOH_1m_15-26.csv",
           "MOEX": "MOEX_1m_15-26.csv", "MTSS": "MTSS_1m_15-26.csv"}
URL = ("https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{t}/candles.json"
       "?interval=1&from={d}&iss.meta=off&start={s}")
COLS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def fetch(ticker: str, since: pd.Timestamp) -> pd.DataFrame:
    rows, start = [], 0
    while True:
        with urllib.request.urlopen(URL.format(t=ticker, d=since.strftime("%Y-%m-%d"), s=start), timeout=30) as r:
            data = json.load(r)["candles"]["data"]
        if not data:
            break
        rows += data
        start += len(data)
        time.sleep(0.2)                       # бережно к бесплатному API
    return pd.DataFrame(rows, columns=COLS)


def update(ticker: str) -> Path:
    LIVE.mkdir(parents=True, exist_ok=True)
    out = LIVE / f"{ticker}_1m_live.csv"
    hist = pd.read_csv(out if out.exists() else RAW / SOURCES[ticker])[COLS]   # у GAZP в истории лишняя колонка-индекс
    last = pd.Timestamp(hist["begin"].iloc[-1])
    new = fetch(ticker, last.normalize())
    new = new[pd.to_datetime(new["begin"]) > last]
    if len(new):
        hist = pd.concat([hist, new], ignore_index=True)
        hist.to_csv(out, index=False)
    elif not out.exists():
        hist.to_csv(out, index=False)
    print(f"[{ticker}] +{len(new)} минут, данные до {hist['begin'].iloc[-1]} → {out.relative_to(ROOT)}", flush=True)
    return out


def main(argv=None) -> int:
    for t in (argv or ["SBER", "GAZP", "LKOH"]):
        update(t.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
