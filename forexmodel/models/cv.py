"""Хронологические разбиения с embargo.

Единственный способ резать эту выборку: метка бара i смотрит на horizon баров
вперёд, поэтому между train и val/holdout всегда должен быть зазор минимум в
horizon баров. В ноутбуке embargo был только в OOF-функции, а обычный holdout
(`train_and_evaluate`) резался встык — последние метки train смотрели прямо в
holdout.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["purged_walk_forward_splits", "chronological_split"]


def purged_walk_forward_splits(
    n_samples: int, n_splits: int = 5, embargo: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(train_idx, val_idx) для walk-forward CV с вырезанием embargo-баров."""
    fold_size = n_samples // (n_splits + 1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(1, n_splits + 1):
        train_end = k * fold_size
        val_start = train_end + embargo
        val_end = min(val_start + fold_size, n_samples)
        if val_start >= val_end:
            continue

        train_idx = np.arange(0, max(0, train_end - embargo))
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


def chronological_split(
    n_samples: int, test_size: float, val_size: float = 0.0, embargo: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """train / val / holdout подряд по времени, между блоками — embargo.

    val нужен для early stopping. Холдаут в подборе итераций НЕ участвует —
    иначе его метрика завышена (в ноутбуке мета-модель обучалась с
    eval_set=(X_hold, y_hold) при use_best_model=True по умолчанию, и на этом
    же holdout печатался AUC).
    """
    n_hold = int(n_samples * test_size)
    n_val = int(n_samples * val_size)

    hold_start = n_samples - n_hold
    val_end = hold_start - embargo
    val_start = val_end - n_val
    train_end = val_start - embargo if n_val > 0 else val_end

    if train_end <= 0:
        raise ValueError(
            f"Недостаточно данных: n={n_samples}, test_size={test_size}, val_size={val_size}, embargo={embargo}"
        )

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(max(val_start, 0), max(val_end, 0)) if n_val > 0 else np.array([], dtype=int)
    hold_idx = np.arange(hold_start, n_samples) if n_hold > 0 else np.array([], dtype=int)
    return train_idx, val_idx, hold_idx
