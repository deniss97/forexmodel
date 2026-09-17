"""Сборка датасета: котировки -> признаки -> метки -> выборки.

Порядок операций здесь принципиален:

  1. признаки считаются на НЕПРЕРЫВНОМ часовом ряде (иначе каждая выборка
     получает свой warm-up: rolling(50), rolling(100), разогрев EMA — на
     коротком sim-периоде это заметная доля баров и другой масштаб признаков);
  2. метки считаются там же, по всему минутному ряду;
  3. и только потом ряд режется на train/test/sim, причём из train вырезается
     embargo: метки последних баров смотрят вперёд, за границу выборки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..config import Config
from ..data.loader import load_minute_csv, minute_window_for, resample_ohlcv
from ..data.splits import apply_embargo, build_splits
from ..evaluation.diagnostics import check_split_sanity
from ..features.builder import build_features
from ..features.selection import select_feature_columns
from ..labeling import attach_labels
from ..logging_utils import get_logger, section

log = get_logger(__name__)

__all__ = ["Dataset", "build_dataset"]


@dataclass
class Dataset:
    minute: pd.DataFrame
    full: pd.DataFrame                 # непрерывный ряд рабочего ТФ с признаками и метками
    splits: Dict[str, pd.DataFrame]
    features: List[str]
    cfg: Config

    @property
    def train(self) -> pd.DataFrame:
        return self.splits["train"]

    @property
    def test(self) -> pd.DataFrame:
        return self.splits["test"]

    @property
    def sim(self) -> pd.DataFrame:
        return self.splits["sim"]

    def minute_slice(self, name: str) -> pd.DataFrame:
        """Минутные бары под выборку + запас на горизонт сделки."""
        return minute_window_for(self.splits[name], self.minute, self.cfg.horizon_minutes)


def build_dataset(cfg: Config, with_labels: bool = True) -> Dataset:
    section(log, "Шаг 1/4 · чтение минутного CSV")
    minute = load_minute_csv(cfg.data.csv_path, cfg.data.time_col, cfg.data.volume_candidates)
    hourly = resample_ohlcv(minute, cfg.data.base_timeframe)
    log.info("Рабочий ТФ %s: %d баров", cfg.data.base_timeframe, len(hourly))

    section(log, "Шаг 2/4 · признаки на непрерывном ряде")
    full = build_features(hourly, cfg)

    warmup = max(100, cfg.features.rsi_z_window)
    if warmup < len(full):
        log.info("Отброшен warm-up признаков: первые %d баров", warmup)
        full = full.iloc[warmup:].reset_index(drop=True)

    if with_labels:
        section(log, "Шаг 3/4 · triple-barrier разметка по минутным барам")
        full = attach_labels(full, minute, cfg, dropna=False)

    section(log, "Шаг 4/4 · нарезка train/test/sim и отбор признаков")
    split_defs = build_splits(cfg)
    splits: Dict[str, pd.DataFrame] = {}
    for name, split in split_defs.items():
        part = split.apply(full)
        if with_labels and name in ("train", "test"):
            # для обучения нужны только размеченные бары + embargo на стыке выборок
            part = part.dropna(subset=["label"]).reset_index(drop=True)
            part["label"] = part["label"].astype(int)
            part = apply_embargo(part, cfg.embargo_bars)
        splits[name] = part

    check_split_sanity(splits)

    features = select_feature_columns(splits["train"] if not splits["train"].empty else full, cfg.features)
    log.info("Датасет готов: %d признаков, выборки %s", len(features),
             {name: len(df) for name, df in splits.items()})
    return Dataset(minute=minute, full=full, splits=splits, features=features, cfg=cfg)
