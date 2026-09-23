"""Отбор колонок для обучения.

В ноутбуке feature_cols собирался как «все числовые колонки, кроме label/time»,
и в модель попадали абсолютные уровни цены: close_lag_*, sma_*, ema_*,
bb_up/bb_low, atr_14, macd*, body, вики. За период обучения инструмент прошёл
путь в разы по цене, так что уровень однозначно кодирует эпоху — дерево её
запоминает, а на новом периоде уровни выходят за диапазон обучения.

Здесь список исключений собирается детерминированно из того же конфига, по
которому считались признаки, поэтому он проверяем тестом и не разъезжается
с technical.py.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd

from ..config import FeatureConfig
from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["absolute_price_columns", "SERVICE_COLUMNS", "select_feature_columns"]

#: Никогда не признаки: сырые цены, метки, время и служебные поля пайплайна.
SERVICE_COLUMNS: set[str] = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
    "label_enc",
    "ds",
    "minmax_diff",
    "sample_weight",
    "label_t1",
    "y_pred_cb",
    "y_pred_nn",
    "y_pred_cb_raw",
    "confidence_cb",
    "confidence_nn",
    "final_class",
    "final_signal",
    "meta_proba",
    "position_size",
    "oof_pred",
    "oof_confidence",
    "oof_proba_0",
    "oof_proba_1",
    "oof_proba_2",
    "expected_value",
    "trend_gate",  # гейт входа в симуляции, не признак (см. TrendGateConfig)
}


def absolute_price_columns(cfg: FeatureConfig) -> set[str]:
    """Колонки в абсолютных ценовых единицах (рубли), которые нельзя учить."""
    cols = {
        "body",
        "body_abs",
        "upper_wick",
        "lower_wick",
        "macd",
        "macd_sig",
        "macd_hist",
        "bb_up",
        "bb_low",
        "bb_mid",
        "ema_4h",
        f"atr_{cfg.atr_period}",
    }
    cols |= {f"sma_{p}" for p in cfg.sma_periods}
    cols |= {f"ema_{p}" for p in cfg.ema_periods}
    cols |= {f"close_lag_{lag}" for lag in cfg.lags}
    cols |= {f"macd_lag_{lag}" for lag in cfg.lags}
    return cols


def select_feature_columns(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    extra_exclude: Iterable[str] = (),
) -> List[str]:
    exclude = set(SERVICE_COLUMNS) | set(cfg.extra_exclude) | set(extra_exclude)
    if cfg.dimensionless_only:
        exclude |= absolute_price_columns(cfg)

    features = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    dropped_all_nan = [c for c in features if df[c].isna().all()]
    if dropped_all_nan:
        log.warning("Колонки из одних NaN исключены: %s", dropped_all_nan)
        features = [c for c in features if c not in dropped_all_nan]

    constant = [c for c in features if df[c].nunique(dropna=True) <= 1]
    if constant:
        log.warning("Константные колонки исключены: %s", constant)
        features = [c for c in features if c not in constant]

    log.info("Отобрано признаков: %d", len(features))
    return features


def assert_features_present(df: pd.DataFrame, features: Sequence[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            f"В данных нет признаков, на которых обучалась модель: {missing}. "
            "Скорее всего, изменился конфиг features — переобучите модель."
        )
