"""Primary-модель: CatBoost на трёхклассовой разметке.

Отличия от `train_and_evaluate` из ноутбука:

  * holdout отрезается с embargo (раньше метки последних баров train смотрели
    прямо в holdout);
  * есть отдельный val-блок и early stopping — вместо фиксированных 500
    итераций при depth=8 без eval_set;
  * добавлен l2_leaf_reg;
  * поддержаны веса уникальности (пересекающиеся метки);
  * предсказание отдаёт ПОЛНЫЕ вероятности по классам, а не только max:
    без них невозможен ни фильтр по ожидаемой ценности, ни нормальная
    мета-модель.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import CatBoostConfig
from ..evaluation.metrics import classification_summary
from ..logging_utils import get_logger, section
from ..features.selection import assert_features_present
from ..progress import CatBoostProgress, fmt_duration
from .cv import chronological_split

log = get_logger(__name__)

__all__ = ["PrimaryModel", "train_primary", "predict_primary"]


@dataclass
class PrimaryModel:
    """Модель + всё, без чего её нельзя применить (порядок фич, маппинг классов)."""

    model: Any
    features: List[str]
    class_map: Dict[int, int]        # исходная метка -> индекс класса CatBoost
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def inv_class_map(self) -> Dict[int, int]:
        return {v: k for k, v in self.class_map.items()}


def _class_weights(y: pd.Series, classes: List[int]) -> List[float]:
    counts = Counter(y)
    total = sum(counts.values())
    return [total / (len(classes) * counts[c]) for c in classes]


def train_primary(
    df: pd.DataFrame,
    features: List[str],
    cfg: CatBoostConfig,
    embargo: int = 0,
    target_col: str = "label",
    weight_col: Optional[str] = "sample_weight",
) -> PrimaryModel:
    from catboost import CatBoostClassifier, Pool

    assert_features_present(df, features)
    df = df.reset_index(drop=True)

    section(log, "CatBoost primary: подготовка")
    train_idx, val_idx, hold_idx = chronological_split(len(df), cfg.test_size, cfg.val_size, embargo)
    log.info(
        "Блоки (хронологически): train=%d | val=%d (early stopping) | holdout=%d (только отчёт) | embargo=%d баров",
        len(train_idx),
        len(val_idx),
        len(hold_idx),
        embargo,
    )

    y_all = df[target_col].astype(int)
    classes = sorted(y_all.iloc[train_idx].unique())
    class_map = {int(c): i for i, c in enumerate(classes)}
    shares = y_all.iloc[train_idx].value_counts(normalize=True).sort_index()
    log.info("Признаков: %d | классы в train: %s", len(features), {int(k): round(float(v), 3) for k, v in shares.items()})

    def pool(idx: np.ndarray):
        if len(idx) == 0:
            return None
        weights = df.loc[idx, weight_col].to_numpy() if weight_col and weight_col in df.columns else None
        return Pool(
            df.loc[idx, features],
            y_all.iloc[idx].map(class_map).to_numpy(),
            weight=weights,
        )

    train_pool, val_pool, hold_pool = pool(train_idx), pool(val_idx), pool(hold_idx)

    params = dict(
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        loss_function="MultiClass",
        random_seed=cfg.random_seed,
        verbose=False,  # прогресс идёт через callback в лог (с ETA), а не в stdout
    )
    if cfg.use_class_weights:
        params["class_weights"] = _class_weights(y_all.iloc[train_idx], classes)

    section(log, f"CatBoost primary: обучение, до {cfg.iterations} деревьев")
    log.info(
        "depth=%d lr=%.3g l2=%.3g | early stopping: %s",
        cfg.depth,
        cfg.learning_rate,
        cfg.l2_leaf_reg,
        f"{cfg.early_stopping_rounds} раундов без улучшения val" if val_pool is not None else "выключено (нет val)",
    )

    every_n = cfg.verbose if isinstance(cfg.verbose, int) and cfg.verbose > 0 else None
    tracker = CatBoostProgress(cfg.iterations, label="CatBoost primary", logger=log, every_n=every_n)
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=cfg.early_stopping_rounds if val_pool is not None else None,
        use_best_model=val_pool is not None,
        callbacks=[tracker],
    )
    tracker.progress.finish(
        f"в модели осталось {model.tree_count_} деревьев"
        + (f" (лучшая итерация по val — {model.get_best_iteration()})" if val_pool is not None else "")
    )
    log.info("Среднее время на дерево: %s", fmt_duration(tracker.progress.elapsed / max(model.tree_count_, 1)))

    metrics: Dict[str, float] = {}
    if hold_pool is not None:
        proba = model.predict_proba(df.loc[hold_idx, features])
        inv = {v: k for k, v in class_map.items()}
        y_pred = np.array([inv[i] for i in proba.argmax(axis=1)])
        metrics = classification_summary(
            y_true=y_all.iloc[hold_idx].to_numpy(),
            y_pred=y_pred,
            title="CatBoost holdout",
        )

    best_iterations = model.tree_count_
    if cfg.refit_on_full_train and len(val_idx) + len(hold_idx) > 0:
        model = _refit_on_full(df, features, y_all, class_map, params, best_iterations, weight_col)
        metrics["refit_on_full_train"] = 1.0
        metrics["refit_iterations"] = float(best_iterations)

    return PrimaryModel(model=model, features=list(features), class_map=class_map, metrics=metrics)


def _refit_on_full(
    df: pd.DataFrame,
    features: List[str],
    y_all: pd.Series,
    class_map: Dict[int, int],
    params: Dict[str, Any],
    iterations: int,
    weight_col: Optional[str],
) -> Any:
    """Переобучение на ВСЕЙ обучающей выборке с уже найденным числом деревьев.

    Зачем: блоки val и holdout — это хронологически последние 35% истории, то
    есть годы, ближайшие к периоду торговли. Модель, обученная только на первых
    65%, к моменту бэктеста отстаёт от рынка на пару лет. Холдаут нужен, чтобы
    ЧЕСТНО измерить качество и выбрать число итераций; после того как измерение
    сделано, выбрасывать эти данные из обучения незачем.

    Число итераций берётся от первой модели (его выбрал val, который в этой
    оценке не участвовал), поэтому eval_set здесь не нужен и early stopping
    не запускается — иначе число деревьев подбиралось бы по данным, на которых
    модель учится.
    """
    from catboost import CatBoostClassifier, Pool

    section(log, f"CatBoost primary: рефит на всей выборке ({len(df)} баров, {iterations} деревьев)")
    weights = df[weight_col].to_numpy() if weight_col and weight_col in df.columns else None
    full_pool = Pool(df[features], y_all.map(class_map).to_numpy(), weight=weights)

    full_params = dict(params)
    full_params["iterations"] = iterations
    tracker = CatBoostProgress(iterations, label="CatBoost primary (рефит)", logger=log)

    model = CatBoostClassifier(**full_params)
    model.fit(full_pool, callbacks=[tracker])
    tracker.progress.finish(f"модель видит историю целиком: {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")
    return model


def predict_primary(bundle: PrimaryModel, df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Предсказания на готовых признаках.

    В ноутбуке `evaluate_catboost` пересчитывал признаки внутри себя по
    `params` — из-за этого sim-выборка получала свой warm-up и свой масштаб
    `rsi_z`. Здесь признаки приходят снаружи, посчитанные на непрерывном ряде.
    """
    assert_features_present(df, bundle.features)

    X = df[bundle.features].replace([np.inf, -np.inf], np.nan)
    proba = bundle.model.predict_proba(X)
    inv = bundle.inv_class_map

    out = pd.DataFrame({time_col: df[time_col].values})
    for idx, label in inv.items():
        out[f"proba_{label}"] = proba[:, idx]

    out["y_pred"] = [inv[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1)
    return out
