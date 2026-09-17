"""Общая механика first-touch разметки по минутным барам.

Исходная реализация обходила минутные бары через `for ts, row in
future_fine.iterrows()` — самое медленное, что можно сделать в pandas (на
10 годах минуток это часы). Здесь тот же алгоритм на numpy-срезах: порядок
касаний определяется индексом первого True в булевом массиве.

Семантика касания сохранена один в один, включая разрешение ничьей: если оба
уровня задеты одним и тем же минутным баром, приоритет отдаётся нижнему
(в старом коде `first_up_time < first_down_time` в этом случае давало False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["MinuteBook", "first_index", "BarrierHit", "scan_barriers"]


@dataclass
class MinuteBook:
    """Минутные бары в виде numpy-массивов (готовые к searchsorted-срезам)."""

    times: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @classmethod
    def from_frame(cls, df_fine: pd.DataFrame, time_col: str = "time") -> "MinuteBook":
        fine = df_fine.copy()
        fine[time_col] = pd.to_datetime(fine[time_col])
        fine = fine.sort_values(time_col)
        return cls(
            times=fine[time_col].values.astype("datetime64[ns]"),
            high=fine["high"].to_numpy(dtype=float),
            low=fine["low"].to_numpy(dtype=float),
            close=fine["close"].to_numpy(dtype=float),
        )

    def window(self, start: pd.Timestamp, end: pd.Timestamp) -> slice:
        a = int(np.searchsorted(self.times, pd.Timestamp(start).to_datetime64(), side="left"))
        b = int(np.searchsorted(self.times, pd.Timestamp(end).to_datetime64(), side="right"))
        return slice(a, b)


def first_index(mask: np.ndarray) -> float:
    """Индекс первого True или inf, если его нет."""
    idx = np.flatnonzero(mask)
    return float(idx[0]) if idx.size else np.inf


@dataclass
class BarrierHit:
    up: float   # индекс первого касания верхнего уровня (inf — не было)
    down: float


def scan_barriers(
    book: MinuteBook,
    win: slice,
    up_level: float,
    down_level: float,
    require_close_beyond: bool = False,
) -> Optional[BarrierHit]:
    """Первое касание верхнего/нижнего уровня внутри окна минутных баров."""
    if win.stop <= win.start:
        return None

    if require_close_beyond:
        series_up = book.close[win]
        series_down = book.close[win]
    else:
        series_up = book.high[win]
        series_down = book.low[win]

    return BarrierHit(up=first_index(series_up >= up_level), down=first_index(series_down <= down_level))
