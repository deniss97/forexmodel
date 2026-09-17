"""Сравнение правил входа на УЖЕ обученных моделях (без переобучения).

Зачем отдельный скрипт: `sweep` переобучает модели на каждой комбинации, а
пороги входа (`meta.threshold`, EV-фильтр, тренд-фильтр) к обучению отношения
не имеют — их можно перебирать на готовых моделях за секунды. Именно такой
перебор отвечает на вопрос «почему сделок так мало и что будет, если ослабить
фильтры».

Ключевая колонка — `до_комиссии`: средний результат сделки ДО вычета комиссии.
Если он меньше комиссии, стратегия убыточна по построению, и добавлять сделки
бессмысленно — каждая будет стоить разницу.

    python scripts/compare_filters.py -c configs/silver.yaml
    python scripts/compare_filters.py -c configs/default.yaml --splits test sim
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (путь до пакета)
import pandas as pd

from forexmodel.config import load_config
from forexmodel.logging_utils import get_logger, setup_logging
from forexmodel.pipelines.backtest_pipeline import load_models, run_backtest
from forexmodel.pipelines.dataset import build_dataset
from forexmodel.pipelines.sweep import set_by_path

log = get_logger(__name__)

def build_variants(thresholds: list[float]) -> dict[str, dict]:
    """Имя варианта -> переопределения конфига.

    Порядок — от строгого набора фильтров к мягкому, чтобы в таблице была видна
    зависимость «больше сделок -> хуже результат» (или её отсутствие). Варианты
    с `signal_source: cb` обходят мета-модель: если она выродилась в константу
    (а это видно по ROC-AUC около 0.5), её порог ничего не отбирает, и сравнивать
    надо именно с прямым сигналом primary-модели.
    """
    variants: dict[str, dict] = {}
    for th in thresholds:
        variants[f"meta {th:.2f} + EV + тренд"] = {"meta.threshold": th}
        variants[f"meta {th:.2f}, без EV"] = {
            "meta.threshold": th,
            "simulation.use_expected_value_filter": False,
        }
    variants["без меты: cb + EV + тренд"] = {"simulation.signal_source": "cb"}
    variants["без меты: cb + EV, без тренда"] = {
        "simulation.signal_source": "cb",
        "simulation.use_trend_filter": False,
    }
    variants["без меты: cb, без EV и тренда"] = {
        "simulation.signal_source": "cb",
        "simulation.use_expected_value_filter": False,
        "simulation.use_trend_filter": False,
    }
    return variants


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Перебор порогов входа на обученных моделях")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=["test", "sim"])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.55, 0.50, 0.45, 0.40],
        help="пороги мета-модели; подбирать по фактическому разбросу meta_proba в signals_*.csv",
    )
    parser.add_argument("-o", "--output", default=None, help="куда сохранить CSV (по умолчанию reports/<run>/filter_variants.csv)")
    parser.add_argument("--log-level", default="WARNING", help="WARNING скрывает логи вложенных бэктестов")
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.WARNING))
    cfg = load_config(Path(args.config))

    # датасет и модели читаются один раз на все варианты — это 95% времени
    dataset = build_dataset(cfg)
    primary, nn_model, meta_model = load_models(cfg)

    rows = []
    for name, overrides in build_variants(args.thresholds).items():
        variant_cfg = copy.deepcopy(cfg)
        for path, value in overrides.items():
            set_by_path(variant_cfg, path, value)

        for split in args.splits:
            result = run_backtest(
                variant_cfg,
                split=split,
                dataset=dataset,
                primary=primary,
                nn_model=nn_model,
                meta_model=meta_model,
                save=False,  # варианты не должны перетирать отчёты основного прогона
            )
            report, trades = result.report, result.trades
            rows.append(
                {
                    "вариант": name,
                    "выборка": split,
                    "сделок": report["total_trades"],
                    "winrate": report["winrate"],
                    "PnL": report["total_pnl_pct"],
                    "PF": report["profit_factor"],
                    "на_сделку": round(report["avg_profit_pct"], 3),
                    "до_комиссии": round(float(trades["gross_pct"].mean()), 3) if len(trades) else None,
                }
            )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    gross = df["до_комиссии"].dropna()
    if len(gross):
        print(
            f"\nСредний результат сделки до комиссии по вариантам: {gross.mean():+.3f}%"
            f" | комиссия в конфиге: {cfg.simulation.commission_pct}% за круг"
        )

    output = Path(args.output) if args.output else cfg.reports_path() / "filter_variants.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Таблица сохранена: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
