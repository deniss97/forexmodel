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
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
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

    rows, all_trades = [], []
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
        res = run_backtest(cfg, split="test", dataset=ds, primary=trained.primary, nn_model=trained.nn,
                           meta_model=trained.meta, save=False)
        t = res.trades
        pnl = t["profit_pct"] if len(t) else pd.Series(dtype=float)
        loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
        ex2 = pnl.drop(pnl.nlargest(2).index) if len(pnl) > 2 else pnl
        rows.append({
            "год": year,
            "деревьев": trained.metrics.get("primary", {}).get("refit_iterations"),
            "edge_holdout": round(trained.metrics.get("primary", {}).get("polar_edge", float("nan")), 4),
            "сделок": len(t),
            "итог_%": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
            "без_2_лучших_%": round(float(ex2.sum()), 2) if len(pnl) else 0.0,
            "PF": round(float(pnl[pnl > 0].sum() / loss), 2) if loss > 0 else float("inf"),
            "до_комиссии": round(float(t["gross_pct"].mean()), 3) if len(t) else float("nan"),
            "winrate": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else float("nan"),
        })
        if len(t):
            all_trades.append(t.assign(год=year))
        print(f"  {rows[-1]}", flush=True)

    table = pd.DataFrame(rows)
    print(f"\nWALK-FORWARD · {name}")
    print(table.to_string(index=False))
    if len(table):
        pos = int((table["итог_%"] > 0).sum())
        print(f"прибыльных лет: {pos} из {len(table)} | сумма {table['итог_%'].sum():+.2f}% | "
              f"медиана года {table['итог_%'].median():+.2f}% | сделка до комиссии в среднем по годам "
              f"{table['до_комиссии'].mean():+.3f}%")
    table.to_csv(out_dir / f"{name}.csv", index=False)
    if all_trades:
        pd.concat(all_trades).to_csv(out_dir / f"{name}_trades.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
