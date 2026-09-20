"""Проверка гипотезы «признак -> направление» как правила входа, без модели.

Если IC у признака есть (см. feature_ic.py), следующий вопрос — торгуется ли он:
вход LONG при feature >= порога, SHORT при <= -порога, те же выходы и комиссия,
что у основного пайплайна (signal_source=rule). Перебираются пороги; на каждом —
число сделок, winrate, PnL и результат до комиссии.

Фильтр ожидаемой ценности выключается: он берёт вероятности CatBoost, а здесь
проверяется признак сам по себе. Тренд-фильтр по умолчанию тоже выключен —
включается --trend, чтобы увидеть, помогает ли он именно этому правилу.

    python scripts/compare_rules.py -c configs/lkoh_2024_orderflow.yaml \
        --feature of_large_delta_z_6 --thresholds 0.5 1 1.5 2
    python scripts/compare_rules.py -c configs/silver.yaml \
        --feature ret_lag_1 --thresholds 0.005 0.01 0.015 0.02   # «бар >= 1% -> продолжение»
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Признак как правило входа: перебор порогов")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--feature", required=True, help="колонка признака, напр. of_large_delta_z_6 или ret_lag_1")
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "sim"])
    parser.add_argument("--invert", action="store_true", help="выше порога -> SHORT (контртренд)")
    parser.add_argument("--trend", action="store_true", help="оставить тренд-фильтр симуляции")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="path=value")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.WARNING))
    base = load_config(Path(args.config))
    for item in args.overrides:
        path, raw = item.split("=", 1)
        import yaml

        set_by_path(base, path.strip(), yaml.safe_load(raw))

    set_by_path(base, "simulation.signal_source", "rule")
    set_by_path(base, "simulation.rule_feature", args.feature)
    set_by_path(base, "simulation.rule_invert", bool(args.invert))
    set_by_path(base, "simulation.use_expected_value_filter", False)
    set_by_path(base, "simulation.use_trend_filter", bool(args.trend))

    dataset = build_dataset(base)
    if args.feature not in dataset.full.columns:
        raise SystemExit(f"Признака {args.feature!r} нет. Похожие: "
                         f"{[c for c in dataset.full.columns if args.feature.split('_')[0] in c][:15]}")
    primary, nn_model, meta_model = load_models(base)

    rows = []
    for th in args.thresholds:
        cfg = copy.deepcopy(base)
        set_by_path(cfg, "simulation.rule_threshold", float(th))
        for split in args.splits:
            n_bars = len(dataset.splits[split])
            result = run_backtest(cfg, split=split, dataset=dataset, primary=primary,
                                  nn_model=nn_model, meta_model=meta_model, save=False)
            rep, tr = result.report, result.trades
            rows.append({
                "порог": th,
                "выборка": split,
                "баров": n_bars,
                "сигналов": rep.get("n_signals_in", 0),
                "сделок": rep["total_trades"],
                "лонгов%": rep.get("long_share"),
                "winrate": rep["winrate"],
                "PnL": rep["total_pnl_pct"],
                "PF": rep["profit_factor"],
                "на_сделку": round(rep["avg_profit_pct"], 3),
                "до_комиссии": round(float(tr["gross_pct"].mean()), 3) if len(tr) else None,
                "просадка": rep["max_drawdown_pct"],
            })

    df = pd.DataFrame(rows)
    mode = "контртренд (инверсия)" if args.invert else "по направлению признака"
    print(f"\nПравило: {args.feature}, {mode}, тренд-фильтр {'вкл' if args.trend else 'выкл'}, "
          f"комиссия {base.simulation.commission_pct}%, выход {base.simulation.exit_mode}")
    print(df.to_string(index=False))

    output = Path(args.output) if args.output else base.reports_path() / f"rule_{args.feature}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nТаблица сохранена: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
