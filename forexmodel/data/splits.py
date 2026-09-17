"""Разбиение на train / test / sim с проверкой непересечения и embargo.

В ноутбуке было:

    df_min_train = df_min[df_min.time < '2025-12-30']
    df_min_test  = df_min[(df_min.time >= '2024-12-30') & (df_min.time < '2025-12-30')]

то есть test целиком лежал ВНУТРИ train. Любая оценка на нём была in-sample.
Здесь такое падает при загрузке конфига.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .loader import slice_time

log = get_logger(__name__)


@dataclass
class Split:
    name: str
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]

    def apply(self, df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        return slice_time(df, self.start, self.end, time_col)


def build_splits(cfg: Config) -> Dict[str, Split]:
    sc = cfg.data.split_config()

    ts = lambda v: pd.Timestamp(v) if v else None  # noqa: E731
    splits = {
        "train": Split("train", None, ts(sc.train_end)),
        "test": Split("test", ts(sc.test_start), ts(sc.test_end)),
        "sim": Split("sim", ts(sc.sim_start), None),
    }
    _assert_no_overlap(splits)
    return splits


def _assert_no_overlap(splits: Dict[str, Split]) -> None:
    order = ["train", "test", "sim"]
    for a, b in zip(order, order[1:]):
        left, right = splits[a], splits[b]
        if left.end is None or right.start is None:
            raise ValueError(f"Границы выборок {a}/{b} заданы не полностью — не могу проверить непересечение")
        if right.start < left.end:
            raise ValueError(
                f"Выборки пересекаются: {b} начинается {right.start}, а {a} заканчивается {left.end}. "
                "Проверьте data.splits в конфиге."
            )


def apply_embargo(df: pd.DataFrame, embargo_bars: int, time_col: str = "time") -> pd.DataFrame:
    """Убирает последние `embargo_bars` баров.

    Метки этих баров смотрят вперёд, за границу выборки, поэтому в обучении
    они создают оверлап со следующим периодом.
    """
    if embargo_bars <= 0 or df.empty:
        return df.reset_index(drop=True)
    if embargo_bars >= len(df):
        raise ValueError(f"embargo_bars={embargo_bars} >= размера выборки ({len(df)})")
    out = df.iloc[:-embargo_bars].reset_index(drop=True)
    log.info("Embargo: отброшено %d последних баров (до %s)", embargo_bars, out[time_col].iloc[-1])
    return out
