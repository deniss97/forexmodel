"""Тесты разметки на вручную сконструированных сериях."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import LabelingConfig
from forexmodel.labeling.atr_barriers import generate_labels_atr_asym
from forexmodel.labeling.first_touch import generate_labels_first_touch
from forexmodel.labeling.uniqueness import average_uniqueness


def _frames(minute_path: list[float], horizon: int = 2):
    """Часовые бары из заданной минутной траектории (60 минут на бар)."""
    n_min = len(minute_path)
    minutes = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n_min, freq="1min"),
            "open": minute_path,
            "high": minute_path,
            "low": minute_path,
            "close": minute_path,
        }
    )
    hours = []
    for start in range(0, n_min, 60):
        chunk = minute_path[start : start + 60]
        hours.append(
            {
                "time": minutes["time"].iloc[start],
                "open": chunk[0],
                "high": max(chunk),
                "low": min(chunk),
                "close": chunk[-1],
            }
        )
    return pd.DataFrame(hours), minutes


def test_first_touch_up_before_down():
    path = [100.0] * 60 + [100.0] * 30 + [101.0] * 30 + [99.0] * 60 + [100.0] * 120
    hours, minutes = _frames(path)
    cfg = LabelingConfig(mode="first_touch", horizon=2, threshold=0.005, use_next_open=True)

    out = generate_labels_first_touch(hours, minutes, cfg)
    assert out["label"].iloc[0] == 2  # +1% достигнут раньше, чем -1%


def test_first_touch_down_before_up():
    path = [100.0] * 60 + [100.0] * 10 + [99.0] * 20 + [101.0] * 30 + [100.0] * 180
    hours, minutes = _frames(path)
    cfg = LabelingConfig(mode="first_touch", horizon=2, threshold=0.005)

    out = generate_labels_first_touch(hours, minutes, cfg)
    assert out["label"].iloc[0] == 0


def test_first_touch_no_touch_is_class_one():
    path = [100.0] * 60 + [100.1] * 180
    hours, minutes = _frames(path)
    cfg = LabelingConfig(mode="first_touch", horizon=2, threshold=0.01)

    out = generate_labels_first_touch(hours, minutes, cfg)
    assert out["label"].iloc[0] == 1


def test_asymmetric_barriers_punish_late_entries():
    """Движение вверх на 1 ATR, затем откат на 1 ATR.

    При симметричных барьерах такой бар получил бы класс 2 (цель близко).
    При tp=1.5 ATR / sl=0.75 ATR цель не достигается, стоп срабатывает —
    это и есть механизм, заставляющий модель отличать излёт от старта.
    """
    atr = 1.0
    path = [100.0] * 60 + [100.9] * 60 + [99.0] * 120
    hours, minutes = _frames(path)
    hours["atr_14"] = atr

    cfg = LabelingConfig(mode="atr_asym", horizon=2, tp_atr=1.5, sl_atr=0.75, atr_col="atr_14")
    out = generate_labels_atr_asym(hours, minutes, cfg)
    assert out["label"].iloc[0] == 0  # вниз-цель достигнута, вверх-цель нет


def test_asymmetric_barriers_reward_early_entries():
    atr = 1.0
    path = [100.0] * 60 + [100.5] * 30 + [102.0] * 30 + [102.0] * 120
    hours, minutes = _frames(path)
    hours["atr_14"] = atr

    cfg = LabelingConfig(mode="atr_asym", horizon=2, tp_atr=1.5, sl_atr=0.75, atr_col="atr_14")
    out = generate_labels_atr_asym(hours, minutes, cfg)
    assert out["label"].iloc[0] == 2


def test_labels_are_nan_without_full_horizon():
    path = [100.0] * 240
    hours, minutes = _frames(path)
    cfg = LabelingConfig(mode="first_touch", horizon=2, threshold=0.005)

    out = generate_labels_first_touch(hours, minutes, cfg)
    assert np.isnan(out["label"].iloc[-1])
    assert np.isnan(out["label"].iloc[-2])


def test_uniqueness_equal_for_non_overlapping_labels():
    # метка 0 занимает бар 1, метка 1 — бар 2: перекрытия нет
    weights = average_uniqueness(np.array([1, 2]), n_bars=2)
    assert np.allclose(weights, [1.0, 1.0])


def test_uniqueness_penalises_overlap():
    """Метка 0 занимает бары 1-2, метка 1 — только бар 2 (общий).

    Уникальность: 0 -> (1 + 1/2)/2 = 0.75, 1 -> 1/2 = 0.5, то есть
    перекрывающаяся метка весит меньше ровно в 1.5 раза.
    """
    weights = average_uniqueness(np.array([2, 2]), n_bars=2)
    assert weights[0] / weights[1] == pytest.approx(1.5)


def test_uniqueness_ignores_unlabeled_rows():
    weights = average_uniqueness(np.array([1, -1, 3]), n_bars=3)
    assert weights[1] == 0.0
    assert np.all(weights[[0, 2]] > 0)
