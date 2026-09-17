"""Разметка: единая точка входа `attach_labels`."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .atr_barriers import generate_labels_atr_asym
from .first_touch import generate_labels_first_touch
from .uniqueness import average_uniqueness

log = get_logger(__name__)

__all__ = [
    "attach_labels",
    "generate_labels_first_touch",
    "generate_labels_atr_asym",
    "average_uniqueness",
]


def attach_labels(df_main: pd.DataFrame, df_fine: pd.DataFrame, cfg: Config, dropna: bool = True) -> pd.DataFrame:
    """Считает метки выбранным способом и добавляет label / label_t1 / sample_weight."""
    if cfg.labeling.mode == "atr_asym":
        labels = generate_labels_atr_asym(df_main, df_fine, cfg.labeling)
    else:
        labels = generate_labels_first_touch(df_main, df_fine, cfg.labeling)

    out = df_main.copy()
    out["label"] = labels["label"].values
    out["label_t1"] = labels["label_t1"].values

    if cfg.labeling.use_uniqueness_weights:
        out["sample_weight"] = average_uniqueness(out["label_t1"], n_bars=len(out))
    else:
        out["sample_weight"] = 1.0

    if dropna:
        before = len(out)
        out = out.dropna(subset=["label"]).reset_index(drop=True)
        out["label"] = out["label"].astype(int)
        log.info("Размечено %d баров из %d (остальные — без полного горизонта)", len(out), before)

    return out
