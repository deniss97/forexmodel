"""forexmodel — ML-пайплайн для внутридневной торговли по часовым барам.

Быстрый старт из ноутбука/REPL:

    from forexmodel import load_config, build_dataset, run_training, run_backtest

    cfg = load_config("configs/default.yaml")
    ds = build_dataset(cfg)
    trained = run_training(cfg, dataset=ds)
    result = run_backtest(cfg, dataset=ds, primary=trained.primary,
                          nn_model=trained.nn, meta_model=trained.meta)

Тяжёлые зависимости (catboost, torch, plotly) импортируются лениво — внутри
функций, которым они нужны, поэтому `import forexmodel` работает и там, где
установлены только pandas/numpy.
"""

from .config import Config, load_config
from .logging_utils import get_logger, setup_logging

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "get_logger",
    "build_dataset",
    "run_training",
    "run_backtest",
    "run_sweep",
    "__version__",
]


def __getattr__(name: str):
    """Ленивая выдача пайплайнов (чтобы не тянуть pandas при импорте пакета)."""
    if name in {"build_dataset", "run_training", "run_backtest", "run_sweep"}:
        from . import pipelines

        return getattr(pipelines, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
