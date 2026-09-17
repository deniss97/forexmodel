"""Технические признаки рабочего ТФ.

Отличия от ноутбука:

  * `rsi_z` считается rolling-окном, а не по среднему/std всей выборки
    (раньше это был look-ahead на train и другой масштаб на sim);
  * для каждого «абсолютного» индикатора добавлена безразмерная версия
    (ratio / деление на ATR), а сами абсолютные значения помечаются как
    служебные и по умолчанию не попадают в обучение (см. selection.py);
  * тренд-фильтр отсюда убран — он живёт в features/trend.py и применяется
    на этапе сборки признаков, а не внутри расчёта индикаторов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from . import indicators as ind

__all__ = ["add_candle_features", "add_technical_indicators"]

# Колонки в абсолютных ценовых единицах (body, sma_*, ema_*, bb_*, macd*, atr_*)
# считаются, потому что нужны для расчёта производных и для симуляции, но в
# обучение по умолчанию не идут: список таких колонок собирает
# features.selection.absolute_price_columns() по тому же конфигу.


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Геометрия свечи. Наружу отдаются только безразмерные версии."""
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)

    body = df["close"] - df["open"]
    df["body"] = body
    df["body_abs"] = body.abs()
    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]

    df["body_to_range"] = df["body_abs"] / rng
    df["body_dir"] = np.sign(body)
    df["upper_wick_ratio"] = df["upper_wick"] / rng
    df["lower_wick_ratio"] = df["lower_wick"] / rng
    df["close_pos_in_range"] = (df["close"] - df["low"]) / rng
    return df


def add_technical_indicators(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Индикаторы + их безразмерные производные."""
    df = df.copy()
    close = df["close"]

    atr_col = f"atr_{cfg.atr_period}"
    df[atr_col] = ind.atr(df, cfg.atr_period)
    atr_safe = df[atr_col].replace(0, np.nan)

    df["atr_pct"] = df[atr_col] / close
    df["atr_ratio"] = df[atr_col] / df[atr_col].rolling(50).mean().replace(0, np.nan)
    df["atr_ema_ratio"] = df[atr_col] / ind.ema(df[atr_col], 10).replace(0, np.nan)

    # --- SMA / EMA: в обучение идут только отношения ---
    for p in cfg.sma_periods:
        col = f"sma_{p}"
        df[col] = ind.sma(close, p)
        df[f"close_sma_{p}_ratio"] = close / df[col].replace(0, np.nan) - 1
        df[f"close_sma_{p}_atr"] = (close - df[col]) / atr_safe

    for p in cfg.ema_periods:
        col = f"ema_{p}"
        df[col] = ind.ema(close, p)
        df[f"close_ema_{p}_ratio"] = close / df[col].replace(0, np.nan) - 1
        df[f"close_ema_{p}_atr"] = (close - df[col]) / atr_safe

    if len(cfg.sma_periods) >= 2:
        fast, slow = f"sma_{cfg.sma_periods[0]}", f"sma_{cfg.sma_periods[-1]}"
        df["sma_spread_atr"] = (df[fast] - df[slow]) / atr_safe
    if len(cfg.ema_periods) >= 2:
        fast, slow = f"ema_{cfg.ema_periods[0]}", f"ema_{cfg.ema_periods[-1]}"
        df["ema_spread_atr"] = (df[fast] - df[slow]) / atr_safe

    # --- RSI ---
    rsi_col = f"rsi_{cfg.rsi_period}"
    df[rsi_col] = ind.rsi(close, cfg.rsi_period)
    roll = df[rsi_col].rolling(cfg.rsi_z_window, min_periods=cfg.rsi_z_window // 4)
    df["rsi_z"] = (df[rsi_col] - roll.mean()) / roll.std().replace(0, np.nan)

    # --- ADX / DI ---
    adx_df = ind.adx(df, cfg.adx_period)
    adx_col = f"adx_{cfg.adx_period}"
    df[adx_col] = adx_df["adx"]
    df[f"dmp_{cfg.adx_period}"] = adx_df["dmp"]
    df[f"dmn_{cfg.adx_period}"] = adx_df["dmn"]
    df["di_spread"] = adx_df["dmp"] - adx_df["dmn"]
    df["adx_slope"] = adx_df["adx"].diff()
    df["rsi_adx_product"] = df[rsi_col] * df[adx_col] / 100.0

    # --- MACD: только нормированный на ATR ---
    macd_line, macd_sig, macd_hist = ind.macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd_line, macd_sig, macd_hist
    df["macd_atr"] = macd_line / atr_safe
    df["macd_hist_atr"] = macd_hist / atr_safe
    df["macd_diff_atr"] = (macd_line - macd_sig) / atr_safe

    # --- Bollinger ---
    bb_up, bb_low, bb_mid = ind.bollinger_bands(close, cfg.bb_period, cfg.bb_std)
    df["bb_up"], df["bb_low"], df["bb_mid"] = bb_up, bb_low, bb_mid
    width = (bb_up - bb_low)
    df["bb_width"] = width / bb_mid.replace(0, np.nan)
    df["bb_pos"] = (close - bb_low) / width.replace(0, np.nan)
    df["bb_width_atr"] = width / atr_safe

    # --- Динамика цены ---
    df["log_return"] = np.log(close / close.shift(1))
    df["candle_range_atr"] = (df["high"] - df["low"]) / atr_safe

    for lag in cfg.lags:
        df[f"ret_lag_{lag}"] = close / close.shift(lag) - 1
        df[f"ret_lag_{lag}_atr"] = (close - close.shift(lag)) / atr_safe
        df[f"rsi_lag_{lag}"] = df[rsi_col].shift(lag)
        df[f"macd_hist_atr_lag_{lag}"] = df["macd_hist_atr"].shift(lag)

    # --- Календарь (безразмерно, циклически) ---
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow"] = df["time"].dt.dayofweek.astype(float)

    return df
