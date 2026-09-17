"""Базовые индикаторы.

ADX/DI реализованы здесь по Уайлдеру, а не берутся из pandas_ta: библиотека
регулярно ломается на новых версиях numpy/pandas (`from numpy import NaN`),
а это единственное, ради чего она была нужна. Никаких других зависимостей
от pandas_ta в проекте нет.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "wilder_smooth",
    "true_range",
    "atr",
    "rsi",
    "macd",
    "bollinger_bands",
    "adx",
    "linreg_slope",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Сглаживание Уайлдера (RMA): alpha = 1/n."""
    return series.ewm(alpha=1.0 / window, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14, method: str = "sma") -> pd.Series:
    tr = true_range(df)
    if method == "wilder":
        return wilder_smooth(tr, n)
    return tr.rolling(n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def bollinger_bands(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    return ma + k * std, ma - k * std, ma


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Возвращает DataFrame с колонками adx / dmp / dmn (аналог ta.adx)."""
    high, low = df["high"], df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_n = wilder_smooth(true_range(df), length)
    plus_di = 100 * wilder_smooth(pd.Series(plus_dm, index=df.index), length) / tr_n.replace(0, np.nan)
    minus_di = 100 * wilder_smooth(pd.Series(minus_dm, index=df.index), length) / tr_n.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder_smooth(dx, length), "dmp": plus_di, "dmn": minus_di}, index=df.index)


def linreg_slope(series: pd.Series, window: int) -> pd.Series:
    """Наклон линейной регрессии за окно (через ковариацию — без polyfit в цикле)."""
    x = pd.Series(np.arange(len(series), dtype=float), index=series.index)
    mean_x = x.rolling(window).mean()
    mean_y = series.rolling(window).mean()
    cov = (x * series).rolling(window).mean() - mean_x * mean_y
    var = (x * x).rolling(window).mean() - mean_x * mean_x
    return cov / var.replace(0, np.nan)
