"""Перебор правил выхода на УЖЕ обученных моделях (без переобучения).

Идея, которую скрипт проверяет: дать цене «погулять». Текущий трейлинг ставит
стоп в 1 ATR и включает сопровождение после хода в 1 ATR — на инструменте с
ATR больше 1% цены это означает, что почти любой нормальный откат выбивает
позицию до того, как движение состоится. Стоп шире и трейлинг позже должны
уменьшить долю выходов «по стопу» и удлинить сделки.

Важно: правила ВЫХОДА к обучению отношения не имеют, поэтому перебирать их
можно на готовых моделях. А вот барьеры РАЗМЕТКИ (`labeling.tp_atr`/`sl_atr`)
менять так нельзя — на них учится модель, и для них нужен полный `sweep`.

    python scripts/compare_exits.py -c configs/silver.yaml --set simulation.signal_source=cb
    python scripts/compare_exits.py -c configs/default.yaml --set meta.threshold=0.40
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (путь до пакета)
import pandas as pd
import yaml

from forexmodel.config import load_config
from forexmodel.logging_utils import get_logger, setup_logging
from forexmodel.pipelines.backtest_pipeline import load_models, run_backtest
from forexmodel.pipelines.dataset import build_dataset
from forexmodel.pipelines.sweep import set_by_path

log = get_logger(__name__)

#: Набор правил выхода. Первый — то, что стоит в конфиге, остальные «отпускают»
#: цену всё сильнее: шире стоп, позже включается трейлинг, шире сам трейл,
#: длиннее горизонт удержания.
EXIT_VARIANTS: dict[str, dict] = {
    "как в конфиге": {},

    "стоп 1.5 ATR": {"simulation.sl_atr": 1.5},
    "стоп 2.0 ATR": {"simulation.sl_atr": 2.0},

    "трейл включается после 1.5 ATR": {"simulation.activate_atr": 1.5},
    "трейл включается после 2.0 ATR": {"simulation.activate_atr": 2.0},

    "трейл 2.5 ATR": {"simulation.trail_atr": 2.5},

    "стоп 1.5 + трейл после 2.0": {"simulation.sl_atr": 1.5, "simulation.activate_atr": 2.0},
    "стоп 2.0 + трейл 2.5 после 2.0": {
        "simulation.sl_atr": 2.0,
        "simulation.trail_atr": 2.5,
        "simulation.activate_atr": 2.0,
    },

    "фикс. барьеры 1.5/0.75 ATR (как разметка)": {
        "simulation.exit_mode": "atr",
        "simulation.tp_atr": 1.5,
        "simulation.sl_atr": 0.75,
    },
    "фикс. барьеры 3.0/1.5 ATR": {
        "simulation.exit_mode": "atr",
        "simulation.tp_atr": 3.0,
        "simulation.sl_atr": 1.5,
    },

    "горизонт 20 баров": {"simulation.horizon_minutes": 1200},
    "горизонт 40 баров + стоп 1.5": {"simulation.horizon_minutes": 2400, "simulation.sl_atr": 1.5},
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Перебор правил выхода на обученных моделях")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=["test", "sim"])
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="общие для всех вариантов переопределения (напр. simulation.signal_source=cb)",
    )
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "--keep-ev",
        action="store_true",
        help="не выключать фильтр ожидаемой ценности (он зависит от sl_atr и исказит сравнение)",
    )
    parser.add_argument(
        "--sl-sweep",
        nargs="+",
        type=float,
        default=None,
        metavar="ATR",
        help="вместо набора правил — чистая кривая по ширине стопа, напр. --sl-sweep 0.75 1 1.5 2 3",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.WARNING))
    base = load_config(Path(args.config))
    for item in args.overrides:
        path, raw = item.split("=", 1)
        set_by_path(base, path.strip(), yaml.safe_load(raw))

    if base.simulation.use_expected_value_filter and not args.keep_ev:
        # Фильтр ожидаемой ценности считает порог входа как
        # p > (sl_atr + комиссия) / (tp_atr + sl_atr), то есть использует ТЕ ЖЕ
        # sl_atr/tp_atr, что и правило выхода. Расширяя стоп, мы автоматически
        # поднимаем требуемую вероятность: при sl_atr=2.0 на серебре нужно
        # p > 0.509 при максимуме модели 0.637 — и все варианты с широким стопом
        # дают ноль сделок. Сравнивать правила выхода так нельзя: меняется ещё и
        # набор сигналов. Поэтому на время перебора EV-фильтр выключается, и все
        # варианты получают ОДИН И ТОТ ЖЕ набор входов.
        log.warning("EV-фильтр выключен на время перебора: он зависит от sl_atr/tp_atr "
                    "и менял бы набор сигналов вместе с правилом выхода (--keep-ev отключает)")
        set_by_path(base, "simulation.use_expected_value_filter", False)

    dataset = build_dataset(base)
    primary, nn_model, meta_model = load_models(base)

    variants = (
        {f"стоп {sl:g} ATR": {"simulation.sl_atr": sl} for sl in args.sl_sweep}
        if args.sl_sweep
        else EXIT_VARIANTS
    )

    rows = []
    for name, overrides in variants.items():
        cfg = copy.deepcopy(base)
        for path, value in overrides.items():
            set_by_path(cfg, path, value)

        for split in args.splits:
            result = run_backtest(
                cfg, split=split, dataset=dataset, primary=primary,
                nn_model=nn_model, meta_model=meta_model, save=False,
            )
            report, trades = result.report, result.trades
            reasons = trades["exit_reason"].value_counts(normalize=True) if len(trades) else {}
            rows.append({
                "правило выхода": name,
                "выборка": split,
                "сделок": report["total_trades"],
                "winrate": report["winrate"],
                "PnL": report["total_pnl_pct"],
                "PF": report["profit_factor"],
                "капитал": report["equity_multiple"],
                "просадка": report["max_drawdown_pct"],
                "минут": report["avg_minutes"],
                # доля выходов по стопу — прямая мера «не дали цене погулять»
                "%стоп": round(float(reasons.get("stop_loss", 0)) * 100, 1),
                "%трейл": round(float(reasons.get("trailing_stop", 0)) * 100, 1),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    output = Path(args.output) if args.output else base.reports_path() / "exit_variants.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nТаблица сохранена: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
