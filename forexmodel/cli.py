"""Командный интерфейс пайплайна.

    python -m forexmodel.cli data      -c configs/default.yaml
    python -m forexmodel.cli train     -c configs/default.yaml
    python -m forexmodel.cli backtest  -c configs/default.yaml --split sim
    python -m forexmodel.cli run       -c configs/default.yaml        # train + backtest
    python -m forexmodel.cli sweep     -c configs/default.yaml -p labeling.horizon=8,10,15 -p simulation.sl_atr=0.75,1.0
    python -m forexmodel.cli chart     -c configs/default.yaml --split sim

Любой параметр конфига можно переопределить из командной строки:
    python -m forexmodel.cli train -c configs/default.yaml --set catboost.depth=4 --set nn.enabled=false
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .config import Config, load_config
from .logging_utils import add_log_file, get_logger, log_files, setup_logging
from .progress import fmt_duration

log = get_logger(__name__)


# ----------------------------------------------------------------------
# разбор --set / -p
# ----------------------------------------------------------------------
def _parse_value(raw: str) -> Any:
    """YAML-разбор значения: 10 -> int, 0.75 -> float, false -> bool, [1,2] -> list."""
    return yaml.safe_load(raw)


def _apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    from .pipelines.sweep import set_by_path

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Ожидался формат path=value, получено: {item!r}")
        path, raw = item.split("=", 1)
        value = _parse_value(raw)
        set_by_path(cfg, path.strip(), value)
        log.info("Переопределено: %s = %r", path.strip(), value)
    return cfg


def _parse_grid(params: List[str]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    for item in params:
        if "=" not in item:
            raise ValueError(f"Ожидался формат path=v1,v2,v3, получено: {item!r}")
        path, raw = item.split("=", 1)
        grid[path.strip()] = [_parse_value(v.strip()) for v in raw.split(",")]
    return grid


# ----------------------------------------------------------------------
# команды
# ----------------------------------------------------------------------
def cmd_data(cfg: Config, args: argparse.Namespace) -> None:
    from .pipelines.dataset import build_dataset

    ds = build_dataset(cfg)
    log.info("Признаков: %d", len(ds.features))
    log.info("Первые 30 признаков: %s", ds.features[:30])
    for name, df in ds.splits.items():
        labeled = int(df["label"].notna().sum()) if "label" in df.columns else 0
        log.info("%-6s: %d баров, размечено %d", name, len(df), labeled)


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from .pipelines.train_pipeline import run_training

    run_training(cfg)


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> None:
    from .pipelines.backtest_pipeline import run_backtest

    run_backtest(cfg, split=args.split)


def cmd_run(cfg: Config, args: argparse.Namespace) -> None:
    from .pipelines.backtest_pipeline import run_backtest
    from .pipelines.dataset import build_dataset
    from .pipelines.train_pipeline import run_training

    ds = build_dataset(cfg)
    trained = run_training(cfg, dataset=ds)
    run_backtest(
        cfg,
        split=args.split,
        dataset=ds,
        primary=trained.primary,
        nn_model=trained.nn,
        meta_model=trained.meta,
    )


def cmd_sweep(cfg: Config, args: argparse.Namespace) -> None:
    from .pipelines.sweep import run_sweep

    grid = _parse_grid(args.param)
    if not grid:
        raise ValueError("Не задана сетка: используйте -p labeling.horizon=8,10,15")
    out = cfg.reports_path() / "sweep_results.csv"
    df = run_sweep(cfg, grid, split=args.split, output_csv=out)
    log.info("Топ результатов:\n%s", df.head(20).to_string())


def cmd_chart(cfg: Config, args: argparse.Namespace) -> None:
    import pandas as pd

    from .viz.charts import create_equity_chart, create_trade_detail_chart, create_trading_chart

    reports = cfg.reports_path()
    signals_path = reports / f"signals_{args.split}.csv"
    if not signals_path.exists():
        raise FileNotFoundError(f"Нет {signals_path} — сначала прогоните backtest на выборке {args.split}")

    signals = pd.read_csv(signals_path, parse_dates=["time"])
    trades_path = reports / f"trades_{args.split}.csv"
    trades = pd.read_csv(trades_path, parse_dates=["signal_dt", "open_dt", "close_dt"]) if trades_path.exists() else pd.DataFrame()

    create_trading_chart(
        signals,
        trades,
        output_file=reports / f"chart_{args.split}.html",
        title=f"{cfg.paths.run_name} — {args.split}: свечи, тренд и сделки",
        trend_col=cfg.simulation.trend_col,
    )
    if trades.empty:
        return

    create_equity_chart(trades, output_file=reports / f"equity_{args.split}.html",
                        title=f"{cfg.paths.run_name} — {args.split}")

    # для разбора сделок нужны минутные цены — их приходится собрать заново,
    # в CSV отчётов они не лежат (это сотни мегабайт)
    from .pipelines.dataset import build_dataset

    log.info("Собираю минутные цены для разбора сделок (это ~30 с)")
    ds = build_dataset(cfg)
    sim = cfg.simulation
    for select, label in (("worst", "худшие"), ("best", "лучшие")):
        create_trade_detail_chart(
            trades,
            ds.minute_slice(args.split),
            output_file=reports / f"trades_{select}_{args.split}.html",
            title=f"{cfg.paths.run_name} — {args.split}: {label} сделки",
            sl_atr=sim.sl_atr,
            trail_atr=sim.trail_atr,
            activate_atr=sim.activate_atr,
            select=select,
        )


COMMANDS = {
    "data": cmd_data,
    "train": cmd_train,
    "backtest": cmd_backtest,
    "run": cmd_run,
    "sweep": cmd_sweep,
    "chart": cmd_chart,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forexmodel", description="ML-пайплайн для торговли по часовым барам")
    parser.add_argument("command", choices=sorted(COMMANDS), help="что делать")
    parser.add_argument("-c", "--config", default="configs/default.yaml", help="путь к YAML-конфигу")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="переопределение: path=value")
    parser.add_argument("-p", "--param", action="append", default=[], help="сетка для sweep: path=v1,v2")
    parser.add_argument("--split", default="sim", choices=["train", "test", "sim"], help="на какой выборке считать")
    parser.add_argument("--log-level", default="INFO", help="INFO (по умолчанию) | DEBUG | WARNING")
    parser.add_argument(
        "--log-file",
        default=None,
        help="куда писать лог; по умолчанию reports/<run_name>/logs/<команда>_<дата>.log",
    )
    parser.add_argument("--no-log-file", action="store_true", help="не писать лог в файл, только в консоль")
    return parser


def _default_log_file(cfg: Config, command: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return cfg.reports_path() / "logs" / f"{command}_{stamp}.log"


def _log_artifact_locations(cfg: Config, args: argparse.Namespace, elapsed: float) -> None:
    """Финальный блок: где лежат логи, отчёты и графики. Иначе после долгого
    прогона приходится вспоминать структуру папок."""
    reports = cfg.reports_path()
    artifacts = cfg.artifacts_path()
    split = args.split

    log.info("")
    log.info("=" * 78)
    log.info("  ГОТОВО · команда «%s» · заняло %s", args.command, fmt_duration(elapsed))
    log.info("=" * 78)

    if args.command in ("train", "run"):
        log.info("  Модели и метрики : %s", artifacts)
        log.info("      primary.pkl / nn.pkl / meta.pkl, metrics.json, train_summary.txt, config.yaml")

    if args.command in ("backtest", "run", "chart"):
        log.info("  Графики (HTML, открыть в браузере):")
        for name, what in (
            (f"chart_{split}.html", "свечи, фон тренда и все сделки"),
            (f"equity_{split}.html", "кривая капитала, просадка, распределение PnL"),
            (f"trades_worst_{split}.html", "12 худших сделок: минутный путь цены и уровни"),
            (f"trades_best_{split}.html", "12 лучших сделок"),
        ):
            path = reports / name
            mark = "" if path.exists() else "  (не создан)"
            log.info("      %s%s  <- %s", path, mark, what)
        log.info("  Отчёты и таблицы : %s", reports)
        log.info("      summary_%s.txt — вся сводка текстом", split)
        log.info("      trades_%s.csv — все сделки, report_%s.csv — метрики одной строкой", split, split)
        log.info("      signals_%s.csv — бары с предсказаниями и сигналами", split)
        log.info("      diag_late_entry_%s.csv — PnL в разрезе растяжения на входе", split)

    for path in log_files():
        log.info("  Лог этого прогона: %s", path)
    log.info("=" * 78)


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level, args.log_file)

    cfg = load_config(Path(args.config))
    cfg = _apply_overrides(cfg, args.overrides)

    if args.log_file is None and not args.no_log_file:
        path = add_log_file(_default_log_file(cfg, args.command))
        log.info("Лог прогона пишется в %s", path)

    started = time.perf_counter()
    try:
        COMMANDS[args.command](cfg, args)
    except Exception:
        log.exception("Команда «%s» упала через %s", args.command, fmt_duration(time.perf_counter() - started))
        raise
    _log_artifact_locations(cfg, args, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
