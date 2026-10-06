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
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..config import Config
from ..data.loader import load_minute_compact, load_minute_csv, minute_window_for, resample_ohlcv
from ..data.splits import apply_embargo, build_splits
from ..evaluation.diagnostics import check_split_sanity
from ..features.builder import build_features
from ..features.selection import select_feature_columns
from ..labeling import attach_labels
from ..logging_utils import get_logger, section

log = get_logger(__name__)

__all__ = ["Dataset", "build_dataset", "resplit_dataset"]


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
    if cfg.data.minute_loader == "compact":
        cache_dir = Path(cfg.paths.reports_dir) / "_cache"
        minute = load_minute_compact(cfg.data.csv_path, cfg.data.time_col, cache_dir, cfg.data.volume_candidates)
    else:
        minute = load_minute_csv(cfg.data.csv_path, cfg.data.time_col, cfg.data.volume_candidates)
    if cfg.data.dividends_path:
        from ..data.dividends import adjust_for_dividends, load_dividends

        divs = load_dividends(cfg.data.dividends_path, cfg.data.dividend_ticker or Path(cfg.data.csv_path).stem.split("_")[0])
        minute = adjust_for_dividends(minute, divs)
        log.info("Цены скорректированы на дивиденды: %d отсечек", len(divs))
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
    splits, features = _split_full(full, cfg, with_labels)
    return Dataset(minute=minute, full=full, splits=splits, features=features, cfg=cfg)


def resplit_dataset(ds: Dataset, cfg: Config) -> Dataset:
    """Та же сборка признаков и разметки, но с другими границами выборок.

    Признаки и метки от границ не зависят (считаются на непрерывном ряде), поэтому
    walk-forward может собрать датасет один раз и только перерезать его на каждом
    окне — вместо того чтобы на каждом окне заново читать минутки и секундную ленту.
    Конфиг `cfg` должен отличаться от `ds.cfg` только секцией data.splits / train_query.
    """
    splits, features = _split_full(ds.full, cfg, with_labels="label" in ds.full.columns)
    return Dataset(minute=ds.minute, full=ds.full, splits=splits, features=features, cfg=cfg)


def _split_full(full: pd.DataFrame, cfg: Config, with_labels: bool):
    split_defs = build_splits(cfg)
    splits: Dict[str, pd.DataFrame] = {}
    for name, split in split_defs.items():
        part = split.apply(full)
        if with_labels and name in ("train", "test"):
            # для обучения нужны только размеченные бары + embargo на стыке выборок
            part = part.dropna(subset=["label"]).reset_index(drop=True)
            part["label"] = part["label"].astype(int)
            part = apply_embargo(part, cfg.embargo_bars)
            if name == "train" and cfg.data.train_query:
                before = len(part)
                part = part.query(cfg.data.train_query).reset_index(drop=True)
                log.info("train_query %r: осталось %d баров из %d", cfg.data.train_query, len(part), before)
        splits[name] = part

    check_split_sanity(splits)

    from ..features.extension import regime_columns

    # режимные признаки, посчитанные только для мета-модели, в primary не идут
    meta_only = [] if cfg.features.use_regime_features else regime_columns(cfg.features, force=True)
    features = select_feature_columns(splits["train"] if not splits["train"].empty else full, cfg.features, extra_exclude=meta_only)
    log.info("Датасет готов: %d признаков, выборки %s", len(features),
             {name: len(df) for name, df in splits.items()})
    return splits, features
