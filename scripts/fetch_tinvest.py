"""Свежие минутные свечи акций из Т-Инвестиций — вместо ISS Мосбиржи (задача T30, созвон 18.10).

Тот же результат, что scripts/fetch_moex_iss.py: история data/raw/<TICKER>_1m_*.csv (или уже собранный
data/raw/live/<TICKER>_1m_live.csv) + свежие минуты → data/raw/live/<TICKER>_1m_live.csv в формате минуток
Мосбиржи (open, close, high, low, value, volume, begin, end; время московское), поэтому scripts/live_signals.py
работает без изменений. В отличие от бесплатного ISS, у клиентов брокера минутки без задержки. Только завершённые
свечи, биржевые (без торгов выходного дня). Токен — в configs/tinvest_token (или TINVEST_TOKEN).

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
from moex_data import MIN_FREE_APPEND_GB, check_disk, last_begin


def update(api: TInvest, ticker: str, out=None) -> None:
    """Свежие минуты дописываются в конец файла: история не перезаписывается (прежде файл ~170 МБ переписывался
    каждый час — перезагрузка посреди записи оставила бы его обрезанным), последняя минута — по хвосту файла.
    out — готовый файл минуток (новые акции H66 дописываются прямо в data/raw/moex, без копии истории в live)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    if out is None:
        out = LIVE / f"{ticker}_1m_live.csv"
        if not out.exists():
            pd.read_csv(RAW / SOURCES[ticker])[COLS].to_csv(out, index=False)  # у GAZP в истории лишняя колонка-индекс
    last = last_begin(out)                                             # московское время
    share = api.share(ticker)
    start = last + pd.Timedelta(minutes=1) - MSK                       # → UTC
    c = api.candles_1m(share.uid, start, pd.Timestamp.now("UTC").tz_localize(None))
    new = to_moex_minutes(c, share.lot) if len(c) else pd.DataFrame(columns=COLS)   # ночью и в закрытую биржу свечей нет
    new = new[pd.to_datetime(new["begin"]) > last]
    if len(new):
        check_disk(out.parent, MIN_FREE_APPEND_GB)             # дописка — сотни килобайт, порог ниже, чем у выгрузки истории
        new[COLS].to_csv(out, mode="a", header=False, index=False)
    end = new["begin"].iloc[-1] if len(new) else last
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
