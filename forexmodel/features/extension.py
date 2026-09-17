"""Признаки «где мы находимся внутри движения».

Ключевая гипотеза разбора: входы систематически случаются на излёте движения
(winrate 25% при симметричных барьерах — это не отсутствие сигнала, а
антисигнал). Чтобы модель могла отличить старт движения от его конца, ей
нужны признаки растяжения — их в исходном feature_cols не было вовсе
(dist_to_high_20 / dist_to_low_20 были закомментированы).

Всё нормировано на ATR или безразмерно, поэтому не зависит от уровня цены.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig

__all__ = ["add_extension_features", "extension_columns"]


def add_extension_features(df: pd.DataFrame, cfg: FeatureConfig, atr_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    atr_col = atr_col or f"atr_{cfg.atr_period}"
    if atr_col not in df.columns:
        raise KeyError(f"Нет колонки {atr_col!r} — сначала вызовите add_technical_indicators")

    atr = df[atr_col].replace(0, np.nan)
    close = df["close"]

    for w in cfg.extension_windows:
        # сколько ATR пройдено от экстремума окна — прямая мера «поздности» входа
        df[f"ext_from_low_{w}"] = (close - df["low"].rolling(w).min()) / atr
        df[f"ext_from_high_{w}"] = (df["high"].rolling(w).max() - close) / atr
        # чистое смещение за окно
        df[f"runup_{w}"] = (close - close.shift(w)) / atr
        # efficiency ratio Кауфмана: 1 = прямая линия (тренд), ~0 = пила
        path = close.diff().abs().rolling(w).sum().replace(0, np.nan)
        df[f"er_{w}"] = (close - close.shift(w)).abs() / path
        # положение цены в окне
        m, s = close.rolling(w).mean(), close.rolling(w).std()
        df[f"z_close_{w}"] = (close - m) / s.replace(0, np.nan)

    # серия одноцветных свечей (знак * длина)
    sign = np.sign(close - df["open"])
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    df["candle_streak"] = streak * sign

    # «возраст» последнего обновления 24-барного экстремума
    hi24 = df["high"].rolling(24).max()
    lo24 = df["low"].rolling(24).min()
    new_high = (df["high"] >= hi24).astype(int)
    new_low = (df["low"] <= lo24).astype(int)
    df["bars_since_new_high"] = new_high.groupby(new_high.cumsum()).cumcount()
    df["bars_since_new_low"] = new_low.groupby(new_low.cumsum()).cumcount()

    # ранг сжатия волатильности: низкий bb_width относительно истории = перед пробоем
    if "bb_width" in df.columns:
        df["bb_width_pct"] = df["bb_width"].rolling(100).rank(pct=True)

    if cfg.use_volume_features and "volume" in df.columns:
        vol = df["volume"]
        df["rel_vol_20"] = vol / vol.rolling(20).mean().replace(0, np.nan)
        df["rel_vol_50"] = vol / vol.rolling(50).mean().replace(0, np.nan)
        df["vol_x_body"] = df["rel_vol_20"] * df.get("body_to_range", 0)
        df["vol_trend"] = vol.rolling(5).mean() / vol.rolling(50).mean().replace(0, np.nan)

    return df


def extension_columns(cfg: FeatureConfig) -> list[str]:
    """Имена признаков растяжения — их же стоит добавить в мета-модель:
    именно она отвечает на вопрос «сигнал есть, но не поздно ли»."""
    cols: list[str] = []
    for w in cfg.extension_windows:
        cols += [f"ext_from_low_{w}", f"ext_from_high_{w}", f"runup_{w}", f"er_{w}", f"z_close_{w}"]
    cols += ["candle_streak", "bars_since_new_high", "bars_since_new_low", "bb_width_pct"]
    if cfg.use_volume_features:
        cols += ["rel_vol_20", "rel_vol_50", "vol_x_body", "vol_trend"]
    return cols
