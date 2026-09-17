"""Веса наблюдений по уникальности (де Прадо, AFML гл. 4).

Метка бара i «занимает» бары [i+1, i+1+horizon]. При horizon=10 на часовых
барах каждый бар входит в 10 меток подряд, то есть выборка сильно
автокоррелирована: модель видит одно и то же событие десять раз и уверенно
переобучается. Вес = средняя уникальность = среднее 1/concurrency по всем
барам, которые метка занимает.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["average_uniqueness"]


def average_uniqueness(t1_idx: pd.Series | np.ndarray, n_bars: int | None = None) -> np.ndarray:
    """Средняя уникальность для каждой метки.

    t1_idx: индекс последнего бара, который занимает метка (-1 = метки нет).
    Возвращает массив весов той же длины; для строк без метки — 0.
    """
    t1 = np.asarray(t1_idx, dtype=float)
    n = len(t1)
    n_bars = n_bars or n

    valid = np.isfinite(t1) & (t1 >= 0)
    starts = np.arange(n)
    ends = np.where(valid, t1, starts).astype(int)

    # concurrency через разностный массив: +1 на старте, -1 после конца
    diff = np.zeros(n_bars + 2, dtype=float)
    for i in np.flatnonzero(valid):
        a = min(starts[i] + 1, n_bars)
        b = min(ends[i] + 1, n_bars + 1)
        if b <= a:
            continue
        diff[a] += 1
        diff[b] -= 1

    concurrency = np.cumsum(diff)[: n_bars + 1]
    inv = np.divide(1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
    cum_inv = np.concatenate([[0.0], np.cumsum(inv)])

    weights = np.zeros(n, dtype=float)
    for i in np.flatnonzero(valid):
        a = min(starts[i] + 1, n_bars)
        b = min(ends[i] + 1, n_bars + 1)
        span = b - a
        if span <= 0:
            continue
        weights[i] = (cum_inv[b] - cum_inv[a]) / span

    if weights.max() > 0:
        weights = weights / weights[weights > 0].mean()  # средний вес ~1, чтобы не менять масштаб лосса
    log.info("Веса уникальности: mean=%.3f min=%.3f max=%.3f", weights.mean(), weights.min(), weights.max())
    return weights
