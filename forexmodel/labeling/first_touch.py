"""Классическая first-touch разметка с симметричными барьерами ±threshold.

Класс 2 — верхний уровень достигнут раньше нижнего.
Класс 0 — нижний раньше верхнего.
Класс 1 — ни один уровень не достигнут за горизонт.
NaN    — недостаточно будущих данных.

Оставлена для сравнения/воспроизводимости старых результатов. Рабочий режим —
`labeling.mode: atr_asym` (см. atr_barriers.py): симметричные барьеры не
позволяют модели отличить старт движения от излёта, потому что до ±0.5% цена
добегает и там, и там.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LabelingConfig
from ..logging_utils import get_logger
from .core import MinuteBook, scan_barriers

log = get_logger(__name__)

__all__ = ["generate_labels_first_touch"]


def generate_labels_first_touch(
    df_main: pd.DataFrame,
    df_fine: pd.DataFrame,
    cfg: LabelingConfig,
    return_debug: bool = False,
) -> pd.DataFrame:
    """Возвращает DataFrame с колонками label / label_t1 (+ debug-поля)."""
    main = df_main.sort_values("time").reset_index(drop=True)
    book = MinuteBook.from_frame(df_fine)

    n = len(main)
    labels = np.full(n, np.nan)
    t1_idx = np.full(n, -1, dtype=int)
    rows = []

    times = pd.to_datetime(main["time"])
    opens = main["open"].to_numpy(dtype=float)
    closes = main["close"].to_numpy(dtype=float)
    delay = pd.Timedelta(minutes=cfg.min_minutes_after)

    for i in range(n):
        j_end = i + 1 + cfg.horizon
        if j_end >= n:
            continue

        base = opens[i + 1] if cfg.use_next_open else closes[i]
        up_level = base * (1 + cfg.threshold)
        down_level = base * (1 - cfg.threshold)

        win = book.window(times.iloc[i + 1] + delay, times.iloc[j_end])
        hit = scan_barriers(book, win, up_level, down_level, cfg.require_close_beyond)
        if hit is None:
            continue

        if np.isinf(hit.up) and np.isinf(hit.down):
            label = 1
        elif hit.up < hit.down:
            label = 2
        else:
            label = 0

        labels[i] = label
        t1_idx[i] = j_end

        if return_debug:
            rows.append(
                {
                    "index": i,
                    "signal_time": times.iloc[i],
                    "entry_time": times.iloc[i + 1],
                    "entry_price": base,
                    "up_level": up_level,
                    "down_level": down_level,
                    "label": label,
                    "up_hit_min": hit.up,
                    "down_hit_min": hit.down,
                }
            )

    out = pd.DataFrame({"label": labels, "label_t1": t1_idx}, index=main.index)
    _log_distribution(out["label"])

    if return_debug:
        return out, pd.DataFrame(rows)
    return out


def _log_distribution(labels: pd.Series) -> None:
    dist = labels.value_counts(dropna=True, normalize=True).sort_index()
    log.info("Распределение меток: %s", {int(k): round(float(v), 3) for k, v in dist.items()})
