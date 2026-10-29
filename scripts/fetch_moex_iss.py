"""Свежие минутные свечи акций с Мосбиржи (ISS, бесплатно) — для работы стратегии в реальном времени.

Берёт историю data/raw/<TICKER>_1m_*.csv, докачивает минуты после её последней свечи с
iss.moex.com (режим TQBR, тот же формат колонок: open, close, high, low, value, volume, begin, end)
и пишет историю + свежие минуты в data/raw/live/<TICKER>_1m_live.csv. Исходные файлы не меняются;
при повторном запуске докачивается только новое и дописывается в конец файла (29.10: раньше файл ~180 МБ переписывался
целиком каждый час, когда контур переключался на ISS).

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
from moex_data import MIN_FREE_APPEND_GB, check_disk, last_begin

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
    if not out.exists():
        pd.read_csv(RAW / SOURCES[ticker])[COLS].to_csv(out, index=False)    # у GAZP в истории лишняя колонка-индекс
    last = last_begin(out)                                                   # по хвосту файла
    new = fetch(ticker, last.normalize())
    new = new[pd.to_datetime(new["begin"]) > last]
    if len(new):
        check_disk(out.parent, MIN_FREE_APPEND_GB)
        new[COLS].to_csv(out, mode="a", header=False, index=False)
    end = new["begin"].iloc[-1] if len(new) else last
    print(f"[{ticker}] +{len(new)} минут, данные до {end} → {out.relative_to(ROOT)}", flush=True)
    return out


def main(argv=None) -> int:
    for t in (argv or ["SBER", "GAZP", "LKOH"]):
        update(t.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
