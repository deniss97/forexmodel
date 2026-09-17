"""Мета-модель (meta-labeling, де Прадо, AFML гл. 3).

Схема:
  1. OOF-предсказания primary-модели на train через purged/embargoed
     walk-forward CV — чтобы мета-модель училась на реалистичных, а не на
     переобученных in-sample сигналах.
  2. Каждый OOF-сигнал прогоняется через РЕАЛЬНУЮ симуляцию (те же TP/SL/
     комиссия, что в бою) -> бинарная метка «сделка была прибыльной».
  3. Бинарный CatBoost предсказывает P(profit | сигнал + рыночный контекст).
  4. На инференсе торгуем только там, где P(profit) >= threshold.

По сути мета-модель здесь — это ДЕТЕКТОР ПОЗДНИХ ВХОДОВ: primary отвечает на
вопрос «куда», мета — на вопрос «не поздно ли». Поэтому в её признаки
обязательно идут признаки растяжения (ext_from_*, runup_*, er_*, trend_age_4h).

Исправления относительно ноутбука:
  * `eval_set=(X_hold, y_hold)` при use_best_model=True означало, что holdout
    выбирал число итераций, и на нём же печаталась AUC — метрика была завышена.
    Теперь early stopping идёт по отдельному val-блоку;
  * при сборе мета-меток симуляция запускается с allow_overlapping_positions=True:
    нужен исход КАЖДОГО сигнала независимо, иначе часть их просто выпадает из
    обучающей выборки по причине «позиция уже открыта»;
  * вероятности по классам primary-модели теперь доступны и на инференсе,
    поэтому их можно безопасно использовать как мета-признаки.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger, section
from ..progress import CatBoostProgress, Progress, fmt_duration
from ..simulation.simulator import simulate_trades
from .cv import chronological_split, purged_walk_forward_splits

log = get_logger(__name__)

__all__ = [
    "MetaModel",
    "get_oof_primary_predictions",
    "build_meta_labels",
    "build_meta_features",
    "train_meta_model",
    "apply_meta_filter",
]

LONG, SHORT = 2, 0


@dataclass
class MetaModel:
    model: Any
    features: List[str]
    threshold: float
    metrics: Dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# 1. OOF-предсказания primary-модели
# ----------------------------------------------------------------------
def get_oof_primary_predictions(
    df_train: pd.DataFrame,
    features: Sequence[str],
    cfg: Config,
    target_col: str = "label",
) -> pd.DataFrame:
    from catboost import CatBoostClassifier

    df = df_train.reset_index(drop=True).copy()
    n = len(df)
    embargo = cfg.meta_embargo_bars

    splits = purged_walk_forward_splits(n, n_splits=cfg.meta.n_splits, embargo=embargo)
    if not splits:
        raise ValueError("Недостаточно данных для walk-forward CV с заданными n_splits/embargo")

    oof_pred = np.full(n, np.nan)
    proba_cols = {c: np.full(n, np.nan) for c in (0, 1, 2)}

    cb = cfg.catboost
    section(log, f"OOF-предсказания primary: {len(splits)} фолдов purged walk-forward")
    log.info(
        "На каждом фолде обучается отдельный CatBoost (%d деревьев, depth=%d) — это самый долгий шаг обучения",
        cb.iterations,
        cb.depth,
    )
    folds_bar = Progress(len(splits), label="OOF walk-forward", unit="фолд", logger=log, min_interval=0.0)

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        y_tr = df.loc[train_idx, target_col].astype(int)
        classes = sorted(y_tr.unique())
        class_map = {int(c): i for i, c in enumerate(classes)}
        counts = Counter(y_tr)
        total = sum(counts.values())

        model = CatBoostClassifier(
            iterations=cb.iterations,
            learning_rate=cb.learning_rate,
            depth=cb.depth,
            l2_leaf_reg=cb.l2_leaf_reg,
            loss_function="MultiClass",
            random_seed=cb.random_seed,
            verbose=False,
            class_weights=[total / (len(classes) * counts[c]) for c in classes],
        )
        weights = df.loc[train_idx, "sample_weight"] if "sample_weight" in df.columns else None
        log.info("Фолд %d/%d: обучение на %d барах, предсказание на %d", fold, len(splits), len(train_idx), len(val_idx))
        tracker = CatBoostProgress(cb.iterations, label=f"  фолд {fold}/{len(splits)}", logger=log, min_interval=20.0)
        model.fit(
            df.loc[train_idx, list(features)],
            y_tr.map(class_map),
            sample_weight=weights,
            callbacks=[tracker],
        )

        proba = model.predict_proba(df.loc[val_idx, list(features)])
        inv = {i: c for c, i in class_map.items()}
        oof_pred[val_idx] = [inv[i] for i in proba.argmax(axis=1)]
        for c, i in class_map.items():
            proba_cols[c][val_idx] = proba[:, i]

        folds_bar.set(fold, f"обучено {fold * cb.iterations} деревьев всего")

    folds_bar.finish(f"OOF-предсказаний: {int(np.isfinite(oof_pred).sum())} из {n} баров")

    df["oof_pred"] = oof_pred
    for c in (0, 1, 2):
        df[f"oof_proba_{c}"] = proba_cols[c]
    df["oof_confidence"] = df[[f"oof_proba_{c}" for c in (0, 1, 2)]].max(axis=1)

    return df.dropna(subset=["oof_pred"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Мета-метки через реальную симуляцию
# ----------------------------------------------------------------------
def build_meta_labels(df_oof: pd.DataFrame, df_prices_1m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    section(log, "Мета-метки: прогон OOF-сигналов через минутную симуляцию")
    started = time.perf_counter()
    sig = df_oof[df_oof["oof_pred"].isin([LONG, SHORT])].copy()
    if sig.empty:
        raise ValueError("Primary-модель не дала ни одного полярного OOF-сигнала")

    log.info(
        "Полярных OOF-сигналов: %d из %d баров (%.1f%%) — у каждого нужно узнать исход сделки",
        len(sig),
        len(df_oof),
        100 * len(sig) / max(len(df_oof), 1),
    )
    sig["final_class"] = sig["oof_pred"].astype(int)

    sim_cfg = dataclasses.replace(
        cfg.simulation,
        signal_source="cb",
        min_conf_cb=None,
        min_conf_nn=None,
        use_expected_value_filter=False,   # нужен исход КАЖДОГО сигнала
        max_extension_atr=None,
        use_trend_filter=False,            # тренд — это признак мета-модели, а не фильтр на этом шаге
        close_on_trend_flip=False,
        allow_overlapping_positions=True,  # иначе часть сигналов выпадет из обучения
    )
    trades_df, report = simulate_trades(sig, df_prices_1m, dataclasses.replace(cfg, simulation=sim_cfg))

    if trades_df.empty:
        raise ValueError("Симуляция OOF-сигналов не дала ни одной сделки — проверьте данные/пороги")

    log.info(
        "Симуляция OOF-сигналов заняла %s: сделок %d | winrate %.2f%% | сумма PnL %+.2f%%",
        fmt_duration(time.perf_counter() - started),
        report["total_trades"],
        report["winrate"],
        report["total_pnl_pct"],
    )

    trades_df = trades_df.copy()
    trades_df["meta_label"] = (trades_df["profit_pct"] > 0).astype(int)

    meta_df = sig.merge(
        trades_df[["signal_dt", "meta_label", "profit_pct", "exit_reason"]],
        left_on="time",
        right_on="signal_dt",
        how="inner",
    )
    log.info("Мета-выборка: %d строк, доля прибыльных %.3f", len(meta_df), meta_df["meta_label"].mean())
    return meta_df


# ----------------------------------------------------------------------
# 3. Признаки и обучение мета-модели
# ----------------------------------------------------------------------
def build_meta_features(
    meta_df: pd.DataFrame,
    market_features: Sequence[str],
    include_class_probas: bool = True,
) -> List[str]:
    cols = list(market_features) + ["oof_pred", "oof_confidence"]
    if include_class_probas:
        cols += ["oof_proba_0", "oof_proba_1", "oof_proba_2"]
    return [c for c in dict.fromkeys(cols) if c in meta_df.columns]


def train_meta_model(meta_df: pd.DataFrame, meta_features: Sequence[str], cfg: Config) -> MetaModel:
    from catboost import CatBoostClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    mc = cfg.meta
    meta_df = meta_df.sort_values("signal_dt").reset_index(drop=True)
    train_idx, val_idx, hold_idx = chronological_split(
        len(meta_df), mc.test_size, mc.val_size, cfg.meta_embargo_bars
    )
    section(log, f"Мета-модель: обучение, до {mc.iterations} деревьев")
    log.info(
        "Блоки: train=%d | val=%d (early stopping) | holdout=%d (только отчёт) | мета-признаков %d",
        len(train_idx),
        len(val_idx),
        len(hold_idx),
        len(list(meta_features)),
    )

    X = meta_df[list(meta_features)]
    y = meta_df["meta_label"]

    counts = Counter(y.iloc[train_idx])
    total = sum(counts.values())
    class_weights = [total / (2 * counts.get(c, 1)) for c in (0, 1)]

    model = CatBoostClassifier(
        iterations=mc.iterations,
        learning_rate=mc.learning_rate,
        depth=mc.depth,
        l2_leaf_reg=mc.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=mc.random_seed,
        verbose=False,  # прогресс — через callback, чтобы попал в файл лога
        class_weights=class_weights,
    )
    eval_set = (X.iloc[val_idx], y.iloc[val_idx]) if len(val_idx) else None
    tracker = CatBoostProgress(mc.iterations, label="Мета-модель", logger=log, min_interval=10.0)
    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=eval_set,
        early_stopping_rounds=mc.early_stopping_rounds if eval_set else None,
        use_best_model=eval_set is not None,
        callbacks=[tracker],
    )
    tracker.progress.finish(f"в модели осталось {model.tree_count_} деревьев (после early stopping по val)")

    metrics: Dict[str, float] = {}
    if len(hold_idx):
        proba = model.predict_proba(X.iloc[hold_idx])[:, 1]
        pred = (proba >= mc.threshold).astype(int)
        y_hold = y.iloc[hold_idx]
        log.info(
            "=== Meta-model holdout (в подборе итераций НЕ участвовал) ===\n%s\n%s",
            classification_report(y_hold, pred, digits=4, zero_division=0),
            confusion_matrix(y_hold, pred),
        )
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_hold, proba))
            log.info("ROC-AUC (holdout): %.4f", metrics["roc_auc"])
        except ValueError:
            pass
        metrics["precision_at_threshold"] = float((y_hold[pred == 1] == 1).mean()) if (pred == 1).any() else float("nan")
        metrics["kept_share"] = float((pred == 1).mean())

    return MetaModel(model=model, features=list(meta_features), threshold=mc.threshold, metrics=metrics)


# ----------------------------------------------------------------------
# 4. Применение на новых сигналах
# ----------------------------------------------------------------------
def apply_meta_filter(
    df_signals: pd.DataFrame,
    meta_model: MetaModel,
    primary_pred_col: str = "y_pred_cb",
    primary_conf_col: str = "confidence_cb",
    threshold: Optional[float] = None,
    size_by_proba: bool = False,
) -> pd.DataFrame:
    """Добавляет meta_proba / final_signal / position_size.

    Чтобы этот фильтр действительно влиял на сделки, симуляция должна
    запускаться с simulation.signal_source='meta' — в ноутбуке она читала
    y_pred_cb/y_pred_nn, и final_signal не использовался вообще.
    """
    df = df_signals.copy()
    threshold = meta_model.threshold if threshold is None else threshold

    df["oof_pred"] = pd.to_numeric(df[primary_pred_col], errors="coerce")
    df["oof_confidence"] = pd.to_numeric(df[primary_conf_col], errors="coerce")
    for c in (0, 1, 2):
        src = f"proba_{c}_cb"
        if src in df.columns:
            df[f"oof_proba_{c}"] = df[src]

    missing = [c for c in meta_model.features if c not in df.columns]
    if missing:
        raise KeyError(
            f"Для мета-модели не хватает колонок: {missing}. "
            "Признаки инференса должны совпадать с признаками обучения."
        )

    mask = df["oof_pred"].isin([LONG, SHORT])
    df["meta_proba"] = np.nan
    if mask.any():
        df.loc[mask, "meta_proba"] = meta_model.model.predict_proba(df.loc[mask, meta_model.features])[:, 1]

    df["final_signal"] = np.where(mask & (df["meta_proba"] >= threshold), df["oof_pred"], np.nan)

    if size_by_proba:
        df["position_size"] = np.where(
            df["final_signal"].notna(),
            ((df["meta_proba"] - threshold) / max(1 - threshold, 1e-9)).clip(0, 1),
            0.0,
        )

    kept = int(df["final_signal"].notna().sum())
    log.info("Мета-фильтр (порог %.2f): оставлено %d из %d сигналов", threshold, kept, int(mask.sum()))
    return df
