"""Полный цикл обучения: данные -> primary -> NN -> мета-модель -> артефакты."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Config
from ..features.extension import extension_columns
from ..logging_utils import get_logger
from ..models.catboost_primary import PrimaryModel, train_primary
from ..models.meta import MetaModel, build_meta_features, build_meta_labels, get_oof_primary_predictions, train_meta_model
from ..models.nn_cnn_bilstm import NNModel, train_nn
from ..models.persistence import TrainingArtifacts, save_bundle, save_json
from ..progress import StageTracker
from .dataset import Dataset, build_dataset

log = get_logger(__name__)

__all__ = ["TrainResult", "run_training", "format_training_summary"]


@dataclass
class TrainResult:
    dataset: Dataset
    primary: PrimaryModel
    nn: Optional[NNModel]
    meta: Optional[MetaModel]
    features: List[str]
    metrics: Dict[str, Any]


def _stage_names(cfg: Config, dataset: Optional[Dataset]) -> List[str]:
    names = [] if dataset is not None else ["Данные и признаки"]
    names.append("Primary (CatBoost)")
    if cfg.nn.enabled:
        names.append("CNN-BiLSTM")
    if cfg.meta.enabled:
        names += ["OOF-предсказания", "Мета-метки (симуляция)", "Мета-модель"]
    names.append("Сохранение артефактов")
    return names


def run_training(cfg: Config, dataset: Optional[Dataset] = None, save: bool = True) -> TrainResult:
    stages = StageTracker(
        title=f"ОБУЧЕНИЕ · run_name={cfg.paths.run_name}",
        stages=_stage_names(cfg, dataset),
        logger=log,
        timings_path=cfg.artifacts_path() / "timings_train.json" if save else None,
    )

    if dataset is None:
        with stages.stage("Данные и признаки"):
            ds = build_dataset(cfg)
    else:
        ds = dataset
    features = ds.features
    embargo = cfg.embargo_bars

    with stages.stage("Primary (CatBoost)"):
        primary = train_primary(ds.train, features, cfg.catboost, embargo=embargo)

    nn_model: Optional[NNModel] = None
    if cfg.nn.enabled:
        with stages.stage("CNN-BiLSTM"):
            nn_model = train_nn(ds.train, features, cfg.nn, embargo=embargo)

    meta_model: Optional[MetaModel] = None
    if cfg.meta.enabled:
        with stages.stage("OOF-предсказания"):
            df_oof = get_oof_primary_predictions(ds.train, features, cfg)
        with stages.stage("Мета-метки (симуляция)"):
            meta_df = build_meta_labels(df_oof, ds.minute_slice("train"), cfg)
        with stages.stage("Мета-модель"):
            # признаки растяжения обязаны быть в мета-модели: именно она отвечает
            # на вопрос «сигнал есть, но не поздно ли»
            market_features = list(
                dict.fromkeys(features + [c for c in extension_columns(cfg.features) if c in meta_df.columns])
            )
            meta_features = build_meta_features(meta_df, market_features)
            meta_model = train_meta_model(meta_df, meta_features, cfg)

    metrics = {
        "primary": primary.metrics,
        "nn": nn_model.metrics if nn_model else None,
        "meta": meta_model.metrics if meta_model else None,
        "n_train": len(ds.train),
        "n_features": len(features),
    }

    with stages.stage("Сохранение артефактов"):
        if save:
            artifacts = TrainingArtifacts(cfg.artifacts_path()).ensure()
            save_bundle(primary, artifacts.primary, cfg)
            if nn_model is not None:
                save_bundle(nn_model, artifacts.nn, cfg)
            if meta_model is not None:
                save_bundle(meta_model, artifacts.meta, cfg)
            save_json({"features": features}, artifacts.features)
            save_json(metrics, artifacts.metrics)
            cfg.dump(artifacts.config)
            log.info("Модели и метрики записаны в %s", cfg.artifacts_path())
        else:
            log.info("save=False — модели не сохраняются (режим перебора параметров)")

    summary = format_training_summary(cfg, metrics, ds)
    log.info(summary)
    timings = stages.finish(save=save)

    if save:
        path = cfg.artifacts_path() / "train_summary.txt"
        path.write_text(summary + "\n" + timings + "\n", encoding="utf-8")
        log.info("Сводка обучения: %s", path)

    return TrainResult(dataset=ds, primary=primary, nn=nn_model, meta=meta_model, features=features, metrics=metrics)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "н/д" if value != value else f"{value:.{digits}f}"
    return str(value)


def format_training_summary(cfg: Config, metrics: Dict[str, Any], ds: Dataset) -> str:
    """Итог обучения одной таблицей: что обучили и есть ли у модели преимущество."""
    lines = [
        "",
        "=" * 78,
        f"  ИТОГ ОБУЧЕНИЯ · {cfg.paths.run_name}",
        "=" * 78,
        f"  Обучающая выборка : {metrics['n_train']} баров, признаков {metrics['n_features']}",
        f"  Период обучения   : {ds.train['time'].iloc[0]} .. {ds.train['time'].iloc[-1]}",
        "  Финальные модели  : "
        + (
            "переобучены на ВСЕЙ обучающей выборке (refit_on_full_train)"
            if cfg.catboost.refit_on_full_train
            else f"обучены на первых {(1 - cfg.catboost.test_size - cfg.catboost.val_size) * 100:.0f}% выборки "
            "(refit_on_full_train: false — val и holdout в обучении не участвуют)"
        ),
        f"  Разметка          : {cfg.labeling.mode}, horizon={cfg.labeling.horizon}, "
        f"TP={cfg.labeling.tp_atr} ATR / SL={cfg.labeling.sl_atr} ATR",
        "",
        "  Primary (CatBoost), holdout — в подборе итераций НЕ участвовал:",
    ]
    p = metrics.get("primary") or {}
    lines += [
        f"    полярная точность   : {_fmt(p.get('polar_precision'))}   (доля верных там, где модель предлагает сделку)",
        f"    база «по частоте»   : {_fmt(p.get('polar_baseline'))}",
        f"    EDGE                : {_fmt(p.get('polar_edge'))}   <- главная цифра: >0 значит преимущество есть",
        f"    accuracy / f1       : {_fmt(p.get('accuracy'))} / {_fmt(p.get('f1_weighted'))}",
        f"    полярных предсказаний: {_fmt(p.get('n_polar_predictions'))}",
    ]

    nn = metrics.get("nn")
    lines.append("")
    if nn:
        lines += [
            "  CNN-BiLSTM, val:",
            f"    полярная точность / edge: {_fmt(nn.get('polar_precision'))} / {_fmt(nn.get('polar_edge'))}",
        ]
    else:
        lines.append("  CNN-BiLSTM: выключена (nn.enabled=false)")

    meta = metrics.get("meta")
    lines.append("")
    if meta:
        lines += [
            f"  Мета-модель (порог {cfg.meta.threshold}), holdout:",
            f"    ROC-AUC             : {_fmt(meta.get('roc_auc'))}   (0.5 = бесполезна)",
            f"    точность на порогe  : {_fmt(meta.get('precision_at_threshold'))}   "
            "(доля прибыльных среди пропущенных сделок)",
            f"    пропущено сигналов  : {_fmt(meta.get('kept_share'))}   (какую долю сигналов мета-модель оставляет)",
        ]
    else:
        lines.append("  Мета-модель: выключена (meta.enabled=false)")

    lines += ["=" * 78]
    return "\n".join(lines)
