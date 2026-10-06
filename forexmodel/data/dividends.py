"""Дивиденды акций MOEX: даты отсечки и обратная корректировка цен.

Минутные цены акций в data/raw не скорректированы: в первый день без дивиденда акция открывается
ниже примерно на его размер. Для стратегии это два искажения:

* механический гэп отсечки даёт ложный «импульс» вниз (сигнал на продажу там, где цена просто
  отдала дивиденд);
* позиция через отсечку в расчёте теряет (лонг) или получает (шорт) гэп, хотя в жизни лонг получает
  дивиденд, а шорт его платит (у CFD — дивидендная корректировка счёта).

Обе проблемы снимает обратная корректировка: все бары до отсечки умножаются на (C − D) / C, где C —
последняя цена закрытия перед отсечкой, D — дивиденд на акцию. Доходность по скорректированным ценам
равна полной доходности (с дивидендом), гэп отсечки исчезает, ATR и сигналы считаются без него.
Стоп по скорректированным ценам соответствует стопу, переставленному на дивиденд перед отсечкой
(даты отсечек известны заранее).

Таблица дивидендов — data/dividends/moex_dividends.csv (scripts/fetch_dividends.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


def load_dividends(path: str | Path, ticker: str) -> pd.DataFrame:
    """Дивиденды тикера: ex_time (начало первой сессии без дивиденда) и amount (на акцию), по времени."""
    d = pd.read_csv(path, parse_dates=["ex_time"])
    d = d[d["ticker"].str.upper() == ticker.upper()]
    return d[["ex_time", "amount"]].sort_values("ex_time").reset_index(drop=True)


def adjustment_factors(minute: pd.DataFrame, dividends: pd.DataFrame, time_col: str = "time") -> np.ndarray:
    """Множитель цены для каждого минутного бара (1 после последней отсечки)."""
    t = minute[time_col].to_numpy()
    close = minute["close"].to_numpy(dtype=float)
    factor = np.ones(len(minute))
    for ex, amount in zip(dividends["ex_time"], dividends["amount"]):
        i = int(np.searchsorted(t, np.datetime64(pd.Timestamp(ex)), side="left"))
        if i <= 0 or i >= len(t) or not amount > 0:
            continue                                   # отсечка вне данных
        c = close[i - 1]
        if not np.isfinite(c) or c <= amount:
            continue
        factor[:i] *= (c - amount) / c
    return factor


def adjust_for_dividends(minute: pd.DataFrame, dividends: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Копия минуток с обратно скорректированными OHLC (объём не меняется)."""
    factor = adjustment_factors(minute, dividends, time_col)
    out = minute.copy()
    for col in OHLC:
        if col in out.columns:
            out[col] = (out[col].to_numpy(dtype=float) * factor).astype(out[col].dtype)
    return out
