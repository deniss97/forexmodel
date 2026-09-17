"""Перебор параметров (замена вложенных циклов HORIZON_VALUES x THRESHOLD_VALUES
из ноутбука).

Отличия: сетка задаётся точечными путями внутри конфига, каждый прогон пишет
собственную строку результата, а падение одной комбинации не роняет перебор и
не теряется в потоке print-ов — в лог уходит traceback с указанием комбинации.
"""

from __future__ import annotations

import copy
import itertools
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .backtest_pipeline import run_backtest
from .dataset import build_dataset
from .train_pipeline import run_training

log = get_logger(__name__)

__all__ = ["run_sweep", "set_by_path"]


def set_by_path(cfg: Config, path: str, value: Any) -> None:
    """set_by_path(cfg, 'labeling.horizon', 12); работает и для dict-секций
    вроде 'data.splits.train_end'."""
    obj: Any = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)

    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
        return
    if not hasattr(obj, last):
        raise AttributeError(f"В конфиге нет параметра {path!r}")
    setattr(obj, last, value)


def run_sweep(
    base_cfg: Config,
    grid: Dict[str, Sequence[Any]],
    split: str = "sim",
    output_csv: Path | str | None = None,
) -> pd.DataFrame:
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    log.info("Перебор: %d комбинаций по параметрам %s", len(combos), keys)

    rows: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        params = dict(zip(keys, combo))
        for path, value in params.items():
            set_by_path(cfg, path, value)
        cfg.paths.run_name = f"{base_cfg.paths.run_name}_sweep{i:03d}"

        log.info("[%d/%d] %s", i, len(combos), params)
        try:
            dataset = build_dataset(cfg)
            trained = run_training(cfg, dataset=dataset, save=False)
            result = run_backtest(
                cfg,
                split=split,
                dataset=dataset,
                primary=trained.primary,
                nn_model=trained.nn,
                meta_model=trained.meta,
                save=False,
            )
            rows.append({**params, **result.report, "polar_edge": trained.primary.metrics.get("polar_edge")})
        except Exception:
            log.error("Комбинация %s упала:\n%s", params, traceback.format_exc())
            rows.append({**params, "error": 1})

    df = pd.DataFrame(rows)
    if not df.empty and "total_pnl_pct" in df.columns:
        df = df.sort_values(["total_pnl_pct", "profit_factor"], ascending=False).reset_index(drop=True)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info("Результаты перебора: %s", output_csv)

    return df
