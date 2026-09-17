"""Метрики классификации.

Ключевая метрика для этой задачи — «полярная точность»: доля верных среди
предсказаний классов 0/2, то есть среди тех баров, где модель реально
предлагает сделку. Обычная accuracy здесь бессмысленна: класс 1 («никуда не
дошли») можно предсказывать всегда и иметь хорошую метрику при нулевой
торговой ценности.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["polar_precision", "classification_summary", "confidence_table", "baseline_polar_rate"]

POLAR_CLASSES = (0, 2)


def polar_precision(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int] = POLAR_CLASSES) -> float:
    mask = np.isin(y_pred, classes)
    if mask.sum() == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def baseline_polar_rate(y_true: np.ndarray, classes: Sequence[int] = POLAR_CLASSES) -> float:
    """Доля полярных классов в самих данных — база для сравнения.

    Если полярная точность модели не выше этой величины, edge отсутствует:
    именно так и было на holdout исходного пайплайна (0.487 при базе ~0.5).
    """
    mask = np.isin(y_true, classes)
    if mask.sum() == 0:
        return float("nan")
    return float(mask.mean())


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    classes: Sequence[int] = POLAR_CLASSES,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)
    polar = polar_precision(y_true, y_pred, classes)
    base = _naive_polar_baseline(y_true, y_pred, classes)

    log.info("=== %s ===\n%s\nConfusion matrix:\n%s", title, report, matrix)
    log.info(
        "Полярная точность (classes %s): %.4f | база «угадай по частоте»: %.4f | edge: %+.4f",
        list(classes),
        polar,
        base,
        polar - base,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "polar_precision": polar,
        "polar_baseline": base,
        "polar_edge": polar - base,
        "n_polar_predictions": int(np.isin(y_pred, classes).sum()),
    }


def _naive_polar_baseline(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> float:
    """Сколько бы дал случайный выбор среди тех же полярных предсказаний.

    Для каждого предсказанного полярного класса считаем его долю в истинных
    метках на том же подмножестве — это и есть «попадание по частоте».
    """
    mask = np.isin(y_pred, classes)
    if mask.sum() == 0:
        return float("nan")
    truth = y_true[mask]
    shares = [(truth == c).mean() for c in classes]
    return float(np.mean(shares))


def confidence_table(
    df_preds: pd.DataFrame,
    bins: Optional[Iterable[float]] = None,
    classes: Sequence[int] = POLAR_CLASSES,
) -> pd.DataFrame:
    """Точность по корзинам уверенности — только для полярных предсказаний."""
    bins = list(bins or [0.0, 0.5, 0.6, 0.7, 0.9, 1.0])
    sub = df_preds[df_preds["y_pred"].isin(classes)].copy()
    if sub.empty:
        log.warning("Нет предсказаний по полярным классам")
        return pd.DataFrame(columns=["samples", "accuracy"])

    sub["correct"] = (sub["y_true"] == sub["y_pred"]).astype(int)
    sub["conf_bin"] = pd.cut(sub["confidence"], bins=bins)
    table = sub.groupby("conf_bin", observed=True)["correct"].agg(["count", "mean"])
    return table.rename(columns={"count": "samples", "mean": "accuracy"})
