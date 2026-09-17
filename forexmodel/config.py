"""Конфигурация пайплайна.

Весь пайплайн управляется одним YAML-файлом (см. `configs/default.yaml`).
Здесь он разбирается в типизированные dataclass-ы, чтобы:

  * опечатка в имени параметра падала сразу при загрузке, а не через 40 минут
    обучения;
  * любой модуль мог принимать только «свою» часть конфига
    (`cfg.labeling`, `cfg.simulation`, ...) и не тащить глобальные переменные,
    как это было в ноутбуке.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = [
    "Config",
    "DataConfig",
    "FeatureConfig",
    "LabelingConfig",
    "TrendConfig",
    "CatBoostConfig",
    "NNConfig",
    "MetaConfig",
    "SimulationConfig",
    "PathsConfig",
    "load_config",
]


# ----------------------------------------------------------------------
# Вспомогательный конструктор dataclass из dict (со строгой проверкой ключей)
# ----------------------------------------------------------------------
def _build(cls, data: Optional[Dict[str, Any]], path: str = ""):
    """Собирает dataclass из словаря, ругаясь на неизвестные ключи."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__}: ожидался словарь, получено {type(data).__name__}")

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{path or cls.__name__}: неизвестные параметры {sorted(unknown)}. "
            f"Допустимые: {sorted(known)}"
        )
    return cls(**data)


@dataclass
class SplitConfig:
    """Границы выборок. Задаются строками-датами (включительно/исключительно по смыслу ниже).

    train: time <  train_end
    test:  test_start <= time < test_end
    sim:   time >= sim_start

    Дополнительно из train вырезается `embargo_bars` последних баров основного ТФ:
    их метки смотрят вперёд на `labeling.horizon` баров, то есть в тестовый период.
    """

    train_end: str = "2024-12-30 21:00:00"
    test_start: str = "2024-12-30 21:00:00"
    test_end: str = "2025-12-30 21:00:00"
    sim_start: str = "2025-12-30 21:00:00"
    embargo_bars: Optional[int] = None  # None -> labeling.horizon + 1


@dataclass
class DataConfig:
    csv_path: str = "data/raw/LKOH_1m_15-26.csv"
    time_col: str = "begin"
    volume_candidates: List[str] = field(default_factory=lambda: ["value", "volume"])
    base_timeframe: str = "1h"          # рабочий ТФ сигналов
    signal_tf_minutes: int = 60         # длина бара рабочего ТФ в минутах
    splits: Dict[str, Any] = field(default_factory=dict)

    def split_config(self) -> SplitConfig:
        return _build(SplitConfig, self.splits, "data.splits")


@dataclass
class FeatureConfig:
    """Параметры индикаторов.

    `dimensionless_only=True` выкидывает из обучения все признаки в абсолютных
    ценовых единицах (close_lag_*, sma_*, ema_*, bb_up/bb_low, atr_14, macd, ...).
    Инструмент за период обучения прошёл путь в разы по цене, поэтому абсолютный
    уровень однозначно кодирует эпоху: дерево его запоминает, а на новых данных
    уровни выходят за диапазон обучения.
    """

    sma_periods: List[int] = field(default_factory=lambda: [3, 7])
    ema_periods: List[int] = field(default_factory=lambda: [3, 7])
    rsi_period: int = 4
    adx_period: int = 4
    macd_fast: int = 3
    macd_slow: int = 8
    macd_signal: int = 3
    atr_period: int = 14
    bb_period: int = 5
    bb_std: float = 2.0
    lags: List[int] = field(default_factory=lambda: [1, 2, 5, 8, 13])
    rsi_z_window: int = 200             # rolling-окно для z-score RSI (было: по всей выборке)
    extension_windows: List[int] = field(default_factory=lambda: [6, 12, 24])
    use_extension_features: bool = True
    use_volume_features: bool = True
    dimensionless_only: bool = True
    extra_exclude: List[str] = field(default_factory=list)


@dataclass
class LabelingConfig:
    """Разметка.

    mode:
      first_touch — симметричные барьеры ±threshold (исходный вариант);
      atr_asym    — triple-barrier с асимметричными барьерами в ATR.
                    При tp_atr > sl_atr бар «на излёте» движения почти никогда не
                    успевает дойти до цели раньше стопа, поэтому получает класс 1
                    или противоположный. Это единственный способ дать модели
                    отличить старт движения от его конца.
    """

    mode: str = "atr_asym"
    horizon: int = 10                   # баров основного ТФ
    threshold: float = 0.0075           # для first_touch
    tp_atr: float = 1.5                 # для atr_asym
    sl_atr: float = 0.75                # для atr_asym
    atr_col: str = "atr_14"
    min_minutes_after: int = 1
    use_next_open: bool = True
    require_close_beyond: bool = False
    use_uniqueness_weights: bool = True  # веса де Прадо для пересекающихся меток


@dataclass
class TrendConfig:
    """Тренд-фильтр старшего ТФ. Оба варианта мёрджатся каузально (со сдвигом на
    длину бина), заглядывания в будущее нет ни в одном."""

    enabled: bool = True
    mode: str = "early"                 # early | confirm
    timeframe: str = "4h"
    # early
    slope_period: int = 6
    adx_period: int = 9
    adx_min: float = 15.0
    adx_max: float = 40.0               # перегретый тренд = поздняя фаза, вход запрещён
    require_adx_rising: bool = True
    # confirm
    ema_period: int = 7
    adx_threshold: float = 15.0
    max_ffill_bars: int = 5
    require_slope_agreement: bool = True
    pullback_atr_threshold: float = 0.4


@dataclass
class CatBoostConfig:
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 6.0
    random_seed: int = 42
    verbose: int = 200
    early_stopping_rounds: int = 100
    test_size: float = 0.2              # доля холдаута (хронологически последняя)
    val_size: float = 0.15              # доля train под early stopping
    use_class_weights: bool = True


@dataclass
class NNConfig:
    enabled: bool = True
    seq_len: int = 20
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    conv_channels: int = 32
    dropout: float = 0.3
    test_size: float = 0.2
    early_stopping_patience: int = 5
    random_seed: int = 42
    device: str = "auto"                # auto | cpu | cuda


@dataclass
class MetaConfig:
    enabled: bool = True
    n_splits: int = 5
    embargo_bars: Optional[int] = None   # None -> 2 * labeling.horizon
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 4
    l2_leaf_reg: float = 6.0
    early_stopping_rounds: int = 50
    test_size: float = 0.2
    val_size: float = 0.15
    threshold: float = 0.55
    size_by_proba: bool = False
    random_seed: int = 42


@dataclass
class SimulationConfig:
    """Симуляция.

    ВАЖНО (исправление относительно ноутбука): комиссия больше НЕ зашита в
    цену открытия. Барьеры TP/SL считаются от сырой цены входа — ровно так же,
    как они считались в разметке, иначе модель учится на одних уровнях, а
    торгует на других. Комиссия вычитается из PnL отдельно, на обе стороны.
    """

    exit_mode: str = "trailing"          # fixed_pct | atr | trailing
    commission_pct: float = 0.15         # за круг (вход+выход), % от цены
    open_delay_minutes: int = 61         # должно совпадать с use_next_open в разметке
    horizon_minutes: Optional[int] = None  # None -> labeling.horizon * signal_tf_minutes
    # fixed_pct
    stop_loss_pct: float = 0.75
    take_profit_pct: float = 0.75
    # atr / trailing
    sl_atr: float = 1.0
    tp_atr: float = 2.0
    trail_atr: float = 1.5
    activate_atr: float = 1.0
    atr_col: str = "atr_14"
    # вход
    signal_source: str = "ensemble"      # ensemble | cb | nn | meta
    ensemble_rule: str = "agreement"     # agreement | cb_priority | nn_priority | confidence
    min_conf_cb: Optional[float] = None
    min_conf_nn: Optional[float] = None
    use_expected_value_filter: bool = True
    min_expected_value_pct: float = 0.0  # p*TP - (1-p)*SL - cost > порог
    max_extension_atr: Optional[float] = None  # грубый отсев поздних входов до переобучения
    extension_col: str = "ext_from_low_24"     # пара к ext_from_high_24 берётся автоматически
    # тренд-фильтр в симуляции
    use_trend_filter: bool = True
    trend_col: str = "trend_4h"
    close_on_trend_flip: bool = False
    trend_flip_mode: str = "opposite"    # opposite | not_aligned
    allow_overlapping_positions: bool = False


@dataclass
class PathsConfig:
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    run_name: str = "lkoh_1h"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # ---------------- производные значения ----------------
    @property
    def embargo_bars(self) -> int:
        """Сколько баров основного ТФ вырезать между выборками.

        Метка бара i смотрит на horizon баров вперёд, поэтому последние
        horizon+1 баров train «знают» о начале следующего периода.
        """
        cfg_value = self.data.split_config().embargo_bars
        return int(cfg_value) if cfg_value is not None else self.labeling.horizon + 1

    @property
    def meta_embargo_bars(self) -> int:
        if self.meta.embargo_bars is not None:
            return int(self.meta.embargo_bars)
        return 2 * self.labeling.horizon

    @property
    def horizon_minutes(self) -> int:
        if self.simulation.horizon_minutes is not None:
            return int(self.simulation.horizon_minutes)
        return self.labeling.horizon * self.data.signal_tf_minutes

    @property
    def atr_col(self) -> str:
        return f"atr_{self.features.atr_period}"

    def artifacts_path(self) -> Path:
        return Path(self.paths.artifacts_dir) / self.paths.run_name

    def reports_path(self) -> Path:
        return Path(self.paths.reports_dir) / self.paths.run_name

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)


_SECTIONS = {
    "data": DataConfig,
    "features": FeatureConfig,
    "labeling": LabelingConfig,
    "trend": TrendConfig,
    "catboost": CatBoostConfig,
    "nn": NNConfig,
    "meta": MetaConfig,
    "simulation": SimulationConfig,
    "paths": PathsConfig,
}


def config_from_dict(raw: Optional[Dict[str, Any]]) -> Config:
    raw = dict(raw or {})
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Неизвестные секции конфига: {sorted(unknown)}")

    kwargs = {name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()}
    cfg = Config(**kwargs)

    # валидация секции splits (она хранится как dict, но проверить нужно сразу)
    cfg.data.split_config()
    _validate(cfg)
    _warn_barrier_mismatch(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.labeling.mode not in {"first_touch", "atr_asym"}:
        raise ValueError(f"labeling.mode: ожидалось first_touch|atr_asym, получено {cfg.labeling.mode!r}")
    if cfg.trend.mode not in {"early", "confirm"}:
        raise ValueError(f"trend.mode: ожидалось early|confirm, получено {cfg.trend.mode!r}")
    if cfg.simulation.exit_mode not in {"fixed_pct", "atr", "trailing"}:
        raise ValueError(
            f"simulation.exit_mode: ожидалось fixed_pct|atr|trailing, получено {cfg.simulation.exit_mode!r}"
        )
    if cfg.simulation.signal_source not in {"ensemble", "cb", "nn", "meta"}:
        raise ValueError(
            f"simulation.signal_source: ожидалось ensemble|cb|nn|meta, получено {cfg.simulation.signal_source!r}"
        )
    if cfg.simulation.signal_source == "meta" and not cfg.meta.enabled:
        raise ValueError("simulation.signal_source='meta', но meta.enabled=False")
    if cfg.labeling.atr_col != f"atr_{cfg.features.atr_period}":
        raise ValueError(
            f"labeling.atr_col={cfg.labeling.atr_col!r} не совпадает с features.atr_period="
            f"{cfg.features.atr_period} (ожидалось atr_{cfg.features.atr_period})"
        )


def _warn_barrier_mismatch(cfg: Config) -> None:
    """Барьеры симуляции должны совпадать с барьерами разметки.

    Иначе модель учится на одних уровнях, а торгует на других — ровно та
    ошибка, из-за которой в ноутбуке «правильная метка» и «сработавший TP»
    были разными событиями.
    """
    from .logging_utils import get_logger

    log = get_logger(__name__)
    sim, lab = cfg.simulation, cfg.labeling

    if sim.exit_mode == "atr" and lab.mode == "atr_asym":
        if (sim.tp_atr, sim.sl_atr) != (lab.tp_atr, lab.sl_atr):
            log.warning(
                "Барьеры симуляции (tp=%.2f, sl=%.2f ATR) не совпадают с разметкой (tp=%.2f, sl=%.2f ATR)",
                sim.tp_atr,
                sim.sl_atr,
                lab.tp_atr,
                lab.sl_atr,
            )
    elif sim.exit_mode == "fixed_pct" and lab.mode == "atr_asym":
        log.warning(
            "Разметка в ATR, а выход фиксированный в процентах — модель учится не на тех уровнях, "
            "на которых торгует. Либо exit_mode: atr/trailing, либо labeling.mode: first_touch."
        )
    elif sim.exit_mode == "trailing" and lab.mode == "atr_asym" and abs(sim.sl_atr - lab.sl_atr) > 1e-9:
        log.info(
            "Трейлинг: стоп %.2f ATR против %.2f ATR в разметке — допустимо, но помните, что "
            "метки описывают более узкий стоп.",
            sim.sl_atr,
            lab.sl_atr,
        )


def load_config(path: Path | str) -> Config:
    """Читает YAML и возвращает валидированный Config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)
