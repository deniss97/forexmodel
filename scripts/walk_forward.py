"""Walk-forward: обучение до начала года, торговля весь год, по каждому году.

Ответ на рекомендацию 4 из RESULTS.md (часть V): одна пара test/sim не отличает
преимущество модели от режима рынка — серебро 2025 убыточно при любых настройках,
2026 прибыльно при любых. Здесь для каждого года Y модель обучается на данных
до 1 января Y (со своим holdout внутри train, как обычно) и торгует год Y.
Ни один бар года Y в обучение не попадает.

Сравнивать конфиги надо по числу прибыльных лет и по результату на сделку до
комиссии, а не по сумме: сумму делает один год.

    python scripts/walk_forward.py -c configs/silver_direction.yaml --years 2020 2026
    python scripts/walk_forward.py -c configs/silver.yaml --years 2020 2026 \\
        --set nn.enabled=false --set simulation.signal_source=cb --name silver_base

Итог — таблица в консоли и reports/walk_forward/<name>.csv; сделки каждого года —
reports/walk_forward/<name>_trades.csv.

`--variants` прогоняет несколько правил входа на ОДНОЙ обученной модели года —
так эффект мета-модели отделяется от разброса обучения:

    python scripts/walk_forward.py -c configs/silver.yaml --years 2019 2026 \
        --set meta.enabled=true --set meta.context_features=true \
        --variants cb meta:0.50 meta:0.52 meta:0.55

Вариант `cb` — сигнал primary без меты, `meta:<порог>` — мета-фильтр с порогом.
Файлы: <name>__<вариант>.csv на каждый вариант.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from forexmodel.config import load_config
from forexmodel.logging_utils import setup_logging
from forexmodel.pipelines.backtest_pipeline import run_backtest
from forexmodel.pipelines.dataset import build_dataset
from forexmodel.pipelines.sweep import set_by_path
from forexmodel.pipelines.train_pipeline import run_training

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("FROM", "TO"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="path=value, как в CLI")
    ap.add_argument("--name", default=None, help="имя для файлов результата (по умолчанию run_name конфига)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="правила входа на одной модели: cb, meta:<порог>; по умолчанию — как в конфиге")
    args = ap.parse_args(argv)
    setup_logging(logging.WARNING)
    pd.set_option("display.width", 250)

    base = load_config(ROOT / args.config)
    for item in args.overrides:
        path, raw = item.split("=", 1)
        set_by_path(base, path.strip(), yaml.safe_load(raw))
    name = args.name or base.paths.run_name
    out_dir = ROOT / "reports" / "walk_forward"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variants or [None]
    rows = {v: [] for v in variants}
    trades_by = {v: [] for v in variants}
    for year in range(args.years[0], args.years[1] + 1):
        cfg = copy.deepcopy(base)
        cfg.paths.run_name = f"wf_{name}_{year}"
        cfg.data.splits.update({
            "train_end": f"{year}-01-01 00:00:00",
            "test_start": f"{year}-01-01 00:00:00",
            "test_end": f"{year + 1}-01-01 00:00:00",
            "sim_start": f"{year + 1}-01-01 00:00:00",
        })
        print(f"[{name}] {year}: обучение до {year}-01-01, торговля {year}...", flush=True)
        ds = build_dataset(cfg)
        if ds.test.empty:
            print(f"  нет данных за {year}, пропуск")
            continue
        trained = run_training(cfg, dataset=ds, save=False)
        meta_m = trained.metrics.get("meta") or {}
        for v in variants:
            vcfg = copy.deepcopy(cfg)
            if v == "cb":
                vcfg.simulation.signal_source = "cb"
            elif v and v.startswith("meta:"):
                vcfg.simulation.signal_source = "meta"
                vcfg.meta.threshold = float(v.split(":", 1)[1])
            res = run_backtest(vcfg, split="test", dataset=ds, primary=trained.primary, nn_model=trained.nn,
                               meta_model=trained.meta, save=False)
            t = res.trades
            pnl = t["profit_pct"] if len(t) else pd.Series(dtype=float)
            loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
            ex2 = pnl.drop(pnl.nlargest(2).index) if len(pnl) > 2 else pnl
            row = {
                "год": year,
                "деревьев": trained.metrics.get("primary", {}).get("refit_iterations"),
                "edge_holdout": round(trained.metrics.get("primary", {}).get("polar_edge", float("nan")), 4),
                "meta_auc": round(meta_m.get("roc_auc", float("nan")), 3),
                "meta_деревьев": meta_m.get("trees"),
                "meta_std": round(meta_m.get("proba_std", float("nan")), 4),
                "сделок": len(t),
                "итог_%": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
                "без_2_лучших_%": round(float(ex2.sum()), 2) if len(pnl) else 0.0,
                "PF": round(float(pnl[pnl > 0].sum() / loss), 2) if loss > 0 else float("inf"),
                "до_комиссии": round(float(t["gross_pct"].mean()), 3) if len(t) else float("nan"),
                "winrate": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else float("nan"),
            }
            rows[v].append(row)
            if len(t):
                trades_by[v].append(t.assign(год=year))
            print(f"  {v or 'конфиг'}: {row}", flush=True)

    for v in variants:
        table = pd.DataFrame(rows[v])
        suffix = "" if v is None else "__" + v.replace(":", "_")
        print(f"\nWALK-FORWARD · {name}{suffix}")
        print(table.to_string(index=False))
        if len(table):
            pos = int((table["итог_%"] > 0).sum())
            print(f"прибыльных лет: {pos} из {len(table)} | сумма {table['итог_%'].sum():+.2f}% | "
                  f"медиана года {table['итог_%'].median():+.2f}% | сделка до комиссии в среднем по годам "
                  f"{table['до_комиссии'].mean():+.3f}%")
        table.to_csv(out_dir / f"{name}{suffix}.csv", index=False)
        if trades_by[v]:
            all_t = pd.concat(trades_by[v])
            all_t.to_csv(out_dir / f"{name}{suffix}_trades.csv", index=False)
            # главная мера: средний результат сделки и его t-статистика по ВСЕМ сделкам.
            # Годовые итоги шумят на ±5 п.п. между прогонами из-за последовательности
            # сделок (запрет перекрытия), а по сделкам различия видны честно
            for label, col in (("до комиссии", "gross_pct"), ("после комиссии", "profit_pct")):
                x = all_t[col]
                t_stat = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else float("nan")
                print(f"сделка {label}: {x.mean():+.3f}% ± {x.std() / np.sqrt(len(x)):.3f} (t = {t_stat:+.2f}, n = {len(x)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
