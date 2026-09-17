"""Бэктест: предсказания моделей -> сигнал -> минутная симуляция -> отчёт."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import Config
from ..evaluation.diagnostics import exit_reason_breakdown, late_entry_diagnostic
from ..evaluation.metrics import classification_summary, confidence_table
from ..logging_utils import get_logger, section
from ..models.catboost_primary import PrimaryModel, predict_primary
from ..models.meta import MetaModel, apply_meta_filter
from ..models.nn_cnn_bilstm import NNModel, predict_nn
from ..models.persistence import TrainingArtifacts, load_bundle
from ..progress import StageTracker
from ..simulation.report import format_report
from ..simulation.signals import attach_model_predictions, build_signal_column
from ..simulation.simulator import simulate_trades
from .dataset import Dataset, build_dataset

log = get_logger(__name__)

__all__ = ["BacktestResult", "run_backtest", "load_models", "format_signal_funnel"]


@dataclass
class BacktestResult:
    signals: pd.DataFrame
    trades: pd.DataFrame
    report: Dict[str, Any]
    diagnostics: Dict[str, pd.DataFrame]


def load_models(cfg: Config) -> tuple[PrimaryModel, Optional[NNModel], Optional[MetaModel]]:
    artifacts = TrainingArtifacts(cfg.artifacts_path())
    primary = load_bundle(artifacts.primary)
    nn_model = load_bundle(artifacts.nn) if cfg.nn.enabled and artifacts.nn.exists() else None
    meta_model = load_bundle(artifacts.meta) if cfg.meta.enabled and artifacts.meta.exists() else None
    return primary, nn_model, meta_model


def run_backtest(
    cfg: Config,
    split: str = "sim",
    dataset: Optional[Dataset] = None,
    primary: Optional[PrimaryModel] = None,
    nn_model: Optional[NNModel] = None,
    meta_model: Optional[MetaModel] = None,
    save: bool = True,
) -> BacktestResult:
    stage_names = ["Данные"] if dataset is None else []
    stage_names += ["Предсказания моделей", "Сигналы", "Минутная симуляция", "Диагностика"]
    if save:
        stage_names += ["Сохранение и графики"]

    stages = StageTracker(
        title=f"БЭКТЕСТ · выборка {split} · run_name={cfg.paths.run_name}",
        stages=stage_names,
        logger=log,
        timings_path=cfg.artifacts_path() / f"timings_backtest_{split}.json" if save else None,
    )

    if dataset is None:
        with stages.stage("Данные"):
            ds = build_dataset(cfg)
    else:
        ds = dataset

    if primary is None:
        primary, nn_model, meta_model = load_models(cfg)
        log.info("Модели загружены из %s", cfg.artifacts_path())

    df = ds.splits[split]
    if df.empty:
        raise ValueError(f"Выборка {split!r} пуста — проверьте data.splits")

    with stages.stage("Предсказания моделей"):
        log.info(
            "Выборка %s: %s .. %s (%d баров)", split, df["time"].iloc[0], df["time"].iloc[-1], len(df)
        )
        cb_preds = predict_primary(primary, df)
        nn_preds = predict_nn(nn_model, df) if nn_model is not None else None
        _log_model_quality(df, cb_preds, nn_preds)

    with stages.stage("Сигналы"):
        signals = attach_model_predictions(df, cb_preds, nn_preds)

        if cfg.simulation.signal_source == "meta":
            if meta_model is None:
                raise ValueError("signal_source='meta', но мета-модель не обучена/не найдена")
            # порог берём из АКТУАЛЬНОГО конфига, а не из бандла: в бандле он
            # заморожен на момент обучения, и meta.threshold в YAML/--set
            # молча не действовал — самый важный регулятор был недоступен
            if cfg.meta.threshold != meta_model.threshold:
                log.info(
                    "Порог мета-модели: %.2f из конфига (в бандле при обучении было %.2f)",
                    cfg.meta.threshold,
                    meta_model.threshold,
                )
            signals = apply_meta_filter(
                signals,
                meta_model,
                threshold=cfg.meta.threshold,
                size_by_proba=cfg.meta.size_by_proba,
            )

        signals = build_signal_column(signals, cfg)

    with stages.stage("Минутная симуляция"):
        trades, report = simulate_trades(signals, ds.minute_slice(split), cfg)
        funnel = format_signal_funnel(cfg, signals, trades, report)
        log.info(funnel)
        log.info(format_report(report, title=f"РЕЗУЛЬТАТ СИМУЛЯЦИИ · {split}"))

    diagnostics: Dict[str, pd.DataFrame] = {}
    with stages.stage("Диагностика"):
        if not trades.empty:
            section(log, "Диагностика: не входим ли мы на излёте движения")
            trades_ext, ext_summary = late_entry_diagnostic(trades, df, atr_col=cfg.atr_col)
            diagnostics["late_entry"] = ext_summary
            diagnostics["exit_reasons"] = exit_reason_breakdown(trades)
            trades = trades_ext
        else:
            log.warning("Сделок нет — диагностика пропущена")

    if save:
        with stages.stage("Сохранение и графики"):
            _save_outputs(cfg, split, signals, trades, report, diagnostics)

    stages.finish(save=save)
    return BacktestResult(signals=signals, trades=trades, report=report, diagnostics=diagnostics)


def format_signal_funnel(cfg: Config, signals: pd.DataFrame, trades: pd.DataFrame, report: Dict[str, Any]) -> str:
    """Воронка «бары -> сигналы -> сделки» одним блоком.

    Эти числа и раньше были в логе, но по одному в пяти разных модулях, и
    понять, какой именно фильтр съел сигналы, можно было только собрав их
    вручную. Обычно виноват не тот фильтр, на который думаешь.
    """
    def polar(col: str) -> Optional[int]:
        if col not in signals.columns:
            return None
        return int(pd.to_numeric(signals[col], errors="coerce").isin([0, 2]).sum())

    rows: List[tuple[str, Optional[int], str]] = [
        ("баров в выборке", len(signals), ""),
        ("полярных предсказаний CatBoost", polar("y_pred_cb"), "модель предлагает сделку (класс 0 или 2)"),
        ("полярных предсказаний NN", polar("y_pred_nn"), "nn.enabled"),
    ]
    if "final_signal" in signals.columns:
        rows.append((
            f"прошло мета-фильтр (порог {cfg.meta.threshold})",
            polar("final_signal"),
            "meta.threshold",
        ))
    rows.append((
        "итоговых сигналов",
        polar("final_class"),
        "после фильтра ожидаемой ценности" if cfg.simulation.use_expected_value_filter else "",
    ))
    rows.append(("сделок открыто", len(trades), ""))

    width = max(len(name) for name, _, _ in rows)
    lines = ["", "=" * 78, f"  ВОРОНКА СИГНАЛОВ · {cfg.simulation.signal_source}", "=" * 78]
    base = len(signals) or 1
    for name, value, note in rows:
        if value is None:
            lines.append(f"  {name:<{width}} : —      (выключено)")
            continue
        suffix = f"  <- {note}" if note else ""
        lines.append(f"  {name:<{width}} : {value:<6} ({value / base * 100:5.2f}% баров){suffix}")

    dropped = [
        ("уже открыта позиция", report.get("skipped_overlap")),
        ("против тренд-фильтра", report.get("skipped_trend")),
        ("не нашлось минутных цен", report.get("skipped_no_prices")),
        ("нет ATR на баре", report.get("skipped_no_atr")),
    ]
    dropped = [(name, int(count)) for name, count in dropped if count]
    if dropped:
        lines.append("")
        lines.append("  Сигналы, не ставшие сделками:")
        for name, count in dropped:
            lines.append(f"    {name:<28} {count}")

    lines.append("=" * 78)
    return "\n".join(lines)


def _log_model_quality(df: pd.DataFrame, cb_preds: pd.DataFrame, nn_preds: Optional[pd.DataFrame]) -> None:
    """Качество моделей на бэктест-периоде (там, где метки есть)."""
    if "label" not in df.columns:
        log.warning("В выборке нет меток — качество моделей не считается, только сделки")
        return

    truth = df[["time", "label"]].dropna(subset=["label"])
    if truth.empty:
        return

    for name, preds in (("CatBoost", cb_preds), ("CNN-BiLSTM", nn_preds)):
        if preds is None:
            continue
        merged = preds.merge(truth, on="time", how="inner")
        if merged.empty:
            continue
        classification_summary(merged["label"].astype(int).to_numpy(), merged["y_pred"].to_numpy(), title=f"{name} ({len(merged)} баров)")
        table = confidence_table(merged.rename(columns={"label": "y_true"}))
        if not table.empty:
            log.info("Точность по корзинам уверенности (%s):\n%s", name, table)


def _save_outputs(
    cfg: Config,
    split: str,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    report: Dict[str, Any],
    diagnostics: Dict[str, pd.DataFrame],
) -> None:
    out_dir: Path = cfg.reports_path()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for path, frame in (
        (out_dir / f"trades_{split}.csv", trades),
        (out_dir / f"report_{split}.csv", pd.DataFrame([report])),
        (out_dir / f"signals_{split}.csv", signals),
    ):
        frame.to_csv(path, index=False)
        written.append(path)

    for name, table in diagnostics.items():
        if not table.empty:
            path = out_dir / f"diag_{name}_{split}.csv"
            table.to_csv(path)
            written.append(path)

    summary_path = out_dir / f"summary_{split}.txt"
    summary_path.write_text(_build_summary_text(cfg, split, signals, trades, report, diagnostics), encoding="utf-8")
    written.append(summary_path)

    written += _build_charts(cfg, split, signals, trades)

    log.info("Записано %d файлов в %s:", len(written), out_dir)
    for path in written:
        log.info("    %s", path)


def _build_charts(cfg: Config, split: str, signals: pd.DataFrame, trades: pd.DataFrame) -> list[Path]:
    """Графики строятся сразу после бэктеста: отдельная команда `chart` остаётся,
    но лазить за ней после каждого прогона незачем."""
    from ..viz.charts import create_equity_chart, create_trading_chart

    out_dir = cfg.reports_path()
    paths: list[Path] = []

    trades_chart = out_dir / f"chart_{split}.html"
    try:
        create_trading_chart(
            signals,
            trades,
            output_file=trades_chart,
            title=f"{cfg.paths.run_name} — {split}: свечи, тренд и сделки",
            trend_col=cfg.simulation.trend_col,
        )
        paths.append(trades_chart)
    except Exception as exc:  # график — не причина терять результаты прогона
        log.warning("График сделок не построен: %s", exc)

    if not trades.empty:
        equity_chart = out_dir / f"equity_{split}.html"
        try:
            create_equity_chart(trades, output_file=equity_chart, title=f"{cfg.paths.run_name} — {split}")
            paths.append(equity_chart)
        except Exception as exc:
            log.warning("График капитала не построен: %s", exc)

    return paths


def _build_summary_text(
    cfg: Config,
    split: str,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    report: Dict[str, Any],
    diagnostics: Dict[str, pd.DataFrame],
) -> str:
    """Один текстовый файл, который можно открыть и всё понять без логов."""
    parts = [
        f"Прогон        : {cfg.paths.run_name}",
        f"Выборка       : {split}  ({signals['time'].iloc[0]} .. {signals['time'].iloc[-1]}, {len(signals)} баров)",
        f"Источник сигнала: {cfg.simulation.signal_source} | выходы: {cfg.simulation.exit_mode} | "
        f"комиссия {cfg.simulation.commission_pct}% за круг",
        format_signal_funnel(cfg, signals, trades, report),
        format_report(report, title=f"РЕЗУЛЬТАТ СИМУЛЯЦИИ · {split}"),
    ]

    for name, title in (("late_entry", "PnL по растяжению на входе (ATR)"), ("exit_reasons", "Причины выхода")):
        table = diagnostics.get(name)
        if table is not None and not table.empty:
            parts += ["", title, "-" * len(title), table.to_string()]

    if not trades.empty:
        cols = ["signal_dt", "side", "open_price", "exit_price", "profit_pct", "exit_reason"]
        if len(trades) <= 10:
            title, sample = f"Все сделки ({len(trades)})", trades
        else:
            title, sample = "Первые и последние 5 сделок", pd.concat([trades.head(5), trades.tail(5)])
        parts += ["", title, "-" * len(title), sample[cols].to_string(index=False)]

    return "\n".join(parts) + "\n"
