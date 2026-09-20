"""Сборка всех признаков в одном месте.

Важно: признаки считаются НА НЕПРЕРЫВНОМ ряде и только потом режутся на
train/test/sim. В ноутбуке каждая выборка обрабатывалась отдельно, поэтому
первые ~50-100 баров sim уходили в модель с NaN/warm-up значениями
(rolling(50), rolling(100), EMA-разогрев) — на коротком sim-периоде это
заметная доля данных.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .extension import add_extension_features
from .technical import add_candle_features, add_technical_indicators
from .trend import add_trend_filter

log = get_logger(__name__)

__all__ = ["build_features"]


def build_features(df_tf: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Полный набор признаков для рабочего ТФ (непрерывный ряд целиком)."""
    df = df_tf.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df = add_candle_features(df)
    df = add_technical_indicators(df, cfg.features)

    if cfg.features.use_extension_features:
        df = add_extension_features(df, cfg.features, atr_col=cfg.atr_col)

    if cfg.features.use_orderflow_features:
        if not cfg.data.orderflow_path:
            raise ValueError("features.use_orderflow_features=true, но data.orderflow_path не задан")
        from .orderflow import add_orderflow_features, aggregate_orderflow, load_orderflow

        of = load_orderflow(cfg.data.orderflow_path, cfg.data.orderflow_tz_shift_hours)
        of_bars = aggregate_orderflow(of, cfg.features, cfg.data.base_timeframe)
        del of  # ~700 МБ секундных строк, дальше не нужны
        df = add_orderflow_features(df, of_bars, cfg.features, atr_col=cfg.atr_col)

    df = add_trend_filter(df, cfg.trend)

    log.info("Признаки построены: %d баров, %d колонок", len(df), df.shape[1])
    return df
