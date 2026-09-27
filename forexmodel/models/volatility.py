"""Прогноз волатильности и размер позиции по нему.

Зачем. На серебре преимущество модели в процентах растёт с волатильностью на входе
(docs/results/model_changelog.md, 09-28): при низкой волатильности сделка после
комиссии −0.03%, при высокой +0.06%. Комиссия фиксирована в процентах, а ход — в
единицах ATR, поэтому в тихие периоды её не перекрыть. Отсюда: предсказать
волатильность на горизонт сделки и не торговать (или торговать меньше), когда она
низкая, и больше — когда высокая.

Цель — средний true range следующих `horizon` баров в % цены:

    target[i] = mean(TR[i+1 .. i+horizon]) / close[i] * 100

Она смотрит только вперёд; строки, у которых горизонт не помещается в данные,
получают NaN. На стыке train/test используется тот же embargo, что и для меток
(labeling.horizon + 1 >= horizon), поэтому цель последних баров train не видит test.

Модель — CatBoostRegressor на log(target) по тем же признакам, что primary. Наивный
прогноз для сравнения — текущий ATR в % цены (`atr_pct` × 100).

Опорный уровень для размера позиции — медиана цели по последним `ref_bars` барам
ОБУЧАЮЩЕЙ выборки: известен в момент торговли, не зависит от будущего.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import VolatilityConfig
from ..logging_utils import get_logger, section
from .cv import chronological_split

log = get_logger(__name__)

__all__ = ["VolModel", "volatility_target", "train_volatility_model", "predict_volatility", "position_sizes"]


@dataclass
class VolModel:
    model: Any                       # CatBoostRegressor или None для source=atr
    features: List[str]
    reference: float                 # опорная волатильность, % цены (медиана цели на хвосте train)
    source: str = "model"            # model | atr
    metrics: Dict[str, float] = field(default_factory=dict)


def volatility_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Средний true range следующих `horizon` баров в % от close текущего бара."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    # mean(TR[i+1..i+h]) — rolling по окну, заканчивающемуся на i+h, сдвинутый на −h
    fwd = tr.rolling(horizon).mean().shift(-horizon)
    return fwd / close * 100.0


def _naive(df: pd.DataFrame, atr_col: str) -> np.ndarray:
    return (df[atr_col] / df["close"] * 100.0).to_numpy(dtype=float)


def train_volatility_model(df_train: pd.DataFrame, features: List[str], cfg: VolatilityConfig,
                           horizon: int, atr_col: str, embargo: int = 0) -> VolModel:
    """Обучает прогноз волатильности на обучающей выборке (или только опорный уровень для source=atr)."""
    df = df_train.reset_index(drop=True).copy()
    if "_vol_target" not in df.columns:
        # вызывающий обязан считать цель на НЕПРЕРЫВНОМ ряде (train после train_query — с разрывами);
        # здесь — запасной путь для непрерывной выборки
        df["_vol_target"] = volatility_target(df, horizon)
    tail = df["_vol_target"].dropna().iloc[-cfg.ref_bars:]
    reference = float(tail.median()) if len(tail) else float("nan")

    if cfg.source == "atr":
        naive_tail = pd.Series(_naive(df, atr_col)).dropna().iloc[-cfg.ref_bars:]
        return VolModel(model=None, features=[], reference=float(naive_tail.median()), source="atr",
                        metrics={"reference": float(naive_tail.median())})

    from catboost import CatBoostRegressor, Pool

    d = df.dropna(subset=["_vol_target"]).reset_index(drop=True)
    y = np.log(d["_vol_target"].clip(lower=1e-6))
    tr_idx, va_idx, ho_idx = chronological_split(len(d), cfg.test_size, cfg.val_size, embargo)
    section(log, f"Прогноз волатильности: {len(d)} баров, горизонт {horizon}")
    params = dict(iterations=cfg.iterations, learning_rate=cfg.learning_rate, depth=cfg.depth,
                  l2_leaf_reg=cfg.l2_leaf_reg, loss_function="RMSE", random_seed=cfg.random_seed, verbose=False)
    model = CatBoostRegressor(**params)
    model.fit(Pool(d.loc[tr_idx, features], y.iloc[tr_idx]),
              eval_set=Pool(d.loc[va_idx, features], y.iloc[va_idx]) if len(va_idx) else None,
              early_stopping_rounds=cfg.early_stopping_rounds if len(va_idx) else None,
              use_best_model=bool(len(va_idx)))
    metrics: Dict[str, float] = {"trees": float(model.tree_count_), "reference": reference}
    if len(ho_idx):
        pred = np.exp(model.predict(d.loc[ho_idx, features]))
        true = d.loc[ho_idx, "_vol_target"].to_numpy()
        naive = _naive(d.loc[ho_idx], atr_col)
        for name, p in (("model", pred), ("naive_atr", naive)):
            ok = np.isfinite(p) & np.isfinite(true)
            metrics[f"corr_{name}"] = float(np.corrcoef(np.log(p[ok]), np.log(true[ok]))[0, 1])
            metrics[f"mape_{name}"] = float(np.mean(np.abs(p[ok] - true[ok]) / true[ok]))
        log.info("Волатильность, holdout: corr(log) модель %.3f / наивный ATR %.3f | MAPE %.3f / %.3f",
                 metrics["corr_model"], metrics["corr_naive_atr"], metrics["mape_model"], metrics["mape_naive_atr"])
    if cfg.refit_on_full_train:
        n = model.tree_count_
        model = CatBoostRegressor(**dict(params, iterations=n))
        model.fit(Pool(d[features], y))
        metrics["refit_iterations"] = float(n)
    return VolModel(model=model, features=list(features), reference=reference, source="model", metrics=metrics)


def predict_volatility(vm: VolModel, df: pd.DataFrame, atr_col: str) -> np.ndarray:
    """Прогноз волатильности на горизонт в % цены."""
    if vm.source == "atr":
        return _naive(df, atr_col)
    X = df[vm.features].replace([np.inf, -np.inf], np.nan)
    return np.exp(vm.model.predict(X))


def position_sizes(pred_vol: np.ndarray, reference: float, min_ratio: float, power: float, cap: float) -> np.ndarray:
    """Размер позиции по прогнозу волатильности.

    ratio = pred / reference; size = 0 при ratio < min_ratio, иначе min(ratio^power, cap).
    power = 0 — только фильтр (размер 1 выше порога).
    """
    ratio = np.asarray(pred_vol, dtype=float) / reference
    size = np.where(ratio >= min_ratio, np.minimum(np.power(np.clip(ratio, 0, None), power), cap), 0.0)
    return np.where(np.isfinite(size), size, 0.0)
