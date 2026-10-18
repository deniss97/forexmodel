"""Свежие минутные свечи акций из Т-Инвестиций — вместо ISS Мосбиржи (задача T30, созвон 18.10).

Тот же результат, что scripts/fetch_moex_iss.py: история data/raw/<TICKER>_1m_*.csv (или уже собранный
data/raw/live/<TICKER>_1m_live.csv) + свежие минуты → data/raw/live/<TICKER>_1m_live.csv в формате минуток
Мосбиржи (open, close, high, low, value, volume, begin, end; время московское), поэтому scripts/live_signals.py
работает без изменений. В отличие от бесплатного ISS, у клиентов брокера минутки без задержки. Только завершённые
свечи, биржевые (без торгов выходного дня). Токен — TINVEST_TOKEN или ~/.config/forexmodel/tinvest_token.

    python scripts/fetch_tinvest.py                    # SBER GAZP LKOH
    python scripts/fetch_tinvest.py SBER --sandbox     # токен песочницы: рыночные данные там те же
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from forexmodel.broker.tinvest import MSK, TInvest, to_moex_minutes
from fetch_moex_iss import COLS, LIVE, RAW, ROOT, SOURCES


def update(api: TInvest, ticker: str) -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    out = LIVE / f"{ticker}_1m_live.csv"
    hist = pd.read_csv(out if out.exists() else RAW / SOURCES[ticker])[COLS]
    last = pd.Timestamp(hist["begin"].iloc[-1])                        # московское время
    share = api.share(ticker)
    start = last + pd.Timedelta(minutes=1) - MSK                       # → UTC
    c = api.candles_1m(share.uid, start, pd.Timestamp.utcnow().tz_localize(None))
    new = to_moex_minutes(c, share.lot)
    new = new[pd.to_datetime(new["begin"]) > last]
    if len(new):
        pd.concat([hist, new[COLS]], ignore_index=True).to_csv(out, index=False)
    elif not out.exists():
        hist.to_csv(out, index=False)
    end = new["begin"].iloc[-1] if len(new) else hist["begin"].iloc[-1]
    print(f"[{ticker}] +{len(new)} минут из Т-Инвестиций, данные до {end} → {out.relative_to(ROOT)}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tickers", nargs="*", default=["SBER", "GAZP", "LKOH"])
    ap.add_argument("--sandbox", action="store_true", help="токен песочницы (рыночные данные те же)")
    a = ap.parse_args(argv)
    api = TInvest(sandbox=a.sandbox)
    for t in a.tickers:
        update(api, t.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
