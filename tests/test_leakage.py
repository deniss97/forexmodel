"""Тесты на отсутствие заглядывания в будущее.

Главный приём: признак каузален тогда и только тогда, когда его значение на
баре i не меняется от того, есть ли в данных бары после i. Поэтому считаем
признаки на полном ряде и на его префиксе и сравниваем.

Именно этот тест ловит две ошибки исходного пайплайна:
  * `rsi_z` нормировался по mean/std ВСЕЙ выборки;
  * `add_trend_filter_enhanced` мёрджил 4H/daily по времени начала бина.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.features.builder import build_features
from forexmodel.features.technical import add_candle_features, add_technical_indicators
from forexmodel.features.trend import add_trend_filter_confirm, add_trend_filter_early

PREFIX = 400


def _causal_check(full: pd.DataFrame, prefix: pd.DataFrame, skip: set[str] | None = None) -> list[str]:
    skip = skip or set()
    bad = []
    for col in prefix.columns:
        if col in skip or col == "time" or not pd.api.types.is_numeric_dtype(prefix[col]):
            continue
        a = full[col].iloc[: len(prefix)].to_numpy(dtype=float)
        b = prefix[col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True):
            bad.append(col)
    return bad


def test_technical_indicators_are_causal(hourly_df, cfg):
    full = add_technical_indicators(add_candle_features(hourly_df), cfg.features)
    prefix = add_technical_indicators(add_candle_features(hourly_df.iloc[:PREFIX].copy()), cfg.features)

    bad = _causal_check(full, prefix)
    assert not bad, f"Признаки зависят от будущих баров: {bad}"


def test_rsi_z_uses_rolling_window_not_full_sample(hourly_df, cfg):
    """Прямая проверка конкретной утечки из ноутбука."""
    featured = add_technical_indicators(add_candle_features(hourly_df), cfg.features)
    rsi = featured[f"rsi_{cfg.features.rsi_period}"]
    naive = (rsi - rsi.mean()) / rsi.std()

    # rolling-версия не должна совпадать с нормировкой по всей выборке
    assert not np.allclose(featured["rsi_z"].dropna(), naive.loc[featured["rsi_z"].dropna().index], atol=1e-6)


@pytest.mark.parametrize("builder", [add_trend_filter_early, add_trend_filter_confirm])
def test_trend_filters_are_causal(hourly_df, cfg, builder):
    df = add_technical_indicators(add_candle_features(hourly_df), cfg.features)
    full = builder(df, cfg.trend)
    prefix = builder(df.iloc[:PREFIX].copy(), cfg.trend)

    bad = _causal_check(full, prefix)
    assert not bad, f"Тренд-фильтр {builder.__name__} заглядывает в будущее: {bad}"


def test_trend_value_changes_only_at_bin_boundaries(hourly_df, cfg):
    """Значение 4H-тренда приходит из уже ЗАКРЫТОГО бина, поэтому меняться оно
    может только на границе 4H-бина (00:00, 04:00, 08:00, ...)."""
    df = add_technical_indicators(add_candle_features(hourly_df), cfg.features)
    out = add_trend_filter_early(df, cfg.trend)

    changed = out["trend_4h"].ne(out["trend_4h"].shift())
    changed.iloc[0] = False
    offending = out.loc[changed & (out["time"].dt.hour % 4 != 0), "time"]
    assert offending.empty, f"Тренд меняется внутри 4H-бина: {list(offending[:5])}"


def test_full_feature_pipeline_is_causal(hourly_df, cfg):
    full = build_features(hourly_df, cfg)
    prefix = build_features(hourly_df.iloc[:PREFIX].copy(), cfg)

    bad = _causal_check(full, prefix)
    assert not bad, f"Сборка признаков заглядывает в будущее: {bad}"
