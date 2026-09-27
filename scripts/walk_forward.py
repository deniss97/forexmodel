"""Walk-forward: обучение до начала окна, торговля всё окно, по каждому окну.

Ответ на рекомендацию 4 из RESULTS.md (часть V): одна пара test/sim не отличает
преимущество модели от режима рынка — серебро 2025 убыточно при любых настройках,
2026 прибыльно при любых. Здесь для каждого окна (год или квартал) модель
обучается на данных до его начала (со своим holdout внутри train, как обычно) и
торгует это окно. Ни один бар окна в обучение не попадает.

Главная мера — средний результат сделки и его t-статистика по ВСЕМ сделкам
(печатается в конце): итоги окон между прогонами расходятся на ±5 п.п. из-за
последовательности сделок, а по сделкам различия видны честно.

    python scripts/walk_forward.py -c configs/silver_direction.yaml --years 2020 2026
    python scripts/walk_forward.py -c configs/silver.yaml --years 2020 2026 \\
        --set nn.enabled=false --set simulation.signal_source=cb --name silver_base

Квартальные окна — для коротких данных (лента сделок LKOH есть только с 2024):

    python scripts/walk_forward.py -c configs/lkoh_2024_orderflow_reg.yaml \\
        --quarters 2024Q3 2026Q1 --set simulation.signal_source=cb --set meta.enabled=false

Итог — таблица в консоли и reports/walk_forward/<name>.csv; сделки —
reports/walk_forward/<name>_trades.csv.

`--variants` прогоняет несколько правил входа на ОДНОЙ обученной модели окна —
так эффект мета-модели отделяется от разброса обучения:

    python scripts/walk_forward.py -c configs/silver.yaml --years 2019 2026 \\
        --set meta.enabled=true --set meta.context_features=true \\
        --variants cb meta:0.50 meta:0.52 meta:0.55

Размер позиции по прогнозу волатильности (models/volatility.py; требует
--set volatility.enabled=true): `vol:<min_ratio>:<power>` — прогноз моделью,
`volatr:<min_ratio>:<power>` — наивный прогноз «текущий ATR». Сигнал — primary (cb).

Варианты: `cb` — сигнал primary без меты; `meta:<порог>` — мета-фильтр с порогом;
`rule:<признак>:<порог>` — правило входа по признаку без модели (EV- и
тренд-фильтр выключены; `rule:<признак>:<порог>:trend` — с тренд-фильтром).
Если все варианты — правила, модель не обучается вовсе.
Файлы: <name>__<вариант>.csv на каждый вариант.

Признаки и разметка от границ окон не зависят, поэтому датасет собирается один
раз и на каждом окне только перерезается (`resplit_dataset`).
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from forexmodel.config import load_config
from forexmodel.logging_utils import setup_logging
from forexmodel.pipelines.backtest_pipeline import run_backtest
from forexmodel.pipelines.dataset import build_dataset, resplit_dataset
from forexmodel.pipelines.sweep import set_by_path
from forexmodel.pipelines.train_pipeline import run_training
from forexmodel.simulation.signals import build_signal_column
from forexmodel.simulation.simulator import simulate_trades

ROOT = Path(__file__).resolve().parents[1]


def windows(years=None, quarters=None) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(метка, начало, конец) окон торговли."""
    if years:
        return [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in range(years[0], years[1] + 1)]
    parse = lambda q: pd.Period(re.sub(r"(\d{4})Q(\d)", r"\1Q\2", q.upper()), freq="Q")  # noqa: E731
    out, p = [], parse(quarters[0])
    while p <= parse(quarters[1]):
        out.append((str(p), p.start_time.normalize(), (p + 1).start_time.normalize()))
        p += 1
    return out


def _stats_row(key: str, label: str, t: pd.DataFrame, metrics: dict) -> dict:
    pnl = t["profit_pct"] if len(t) else pd.Series(dtype=float)
    loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
    ex2 = pnl.drop(pnl.nlargest(2).index) if len(pnl) > 2 else pnl
    meta_m = metrics.get("meta") or {}
    prim = metrics.get("primary") or {}
    vol_m = metrics.get("volatility") or {}
    return {
        key: label,
        "деревьев": prim.get("refit_iterations"),
        "edge_holdout": round(prim.get("polar_edge", float("nan")), 4),
        "meta_auc": round(meta_m.get("roc_auc", float("nan")), 3),
        "meta_деревьев": meta_m.get("trees"),
        "meta_std": round(meta_m.get("proba_std", float("nan")), 4),
        "сделок": len(t),
        "итог_%": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "без_2_лучших_%": round(float(ex2.sum()), 2) if len(pnl) else 0.0,
        "PF": round(float(pnl[pnl > 0].sum() / loss), 2) if loss > 0 else float("inf"),
        "до_комиссии": round(float(t["gross_pct"].mean()), 3) if len(t) else float("nan"),
        "winrate": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else float("nan"),
        "средний_размер": round(float(t["size"].mean()), 3) if len(t) and "size" in t else float("nan"),
        "vol_corr_model": round(vol_m.get("corr_model", float("nan")), 3),
        "vol_corr_atr": round(vol_m.get("corr_naive_atr", float("nan")), 3),
    }


def _rule_trades(ds, cfg, variant: str) -> pd.DataFrame:
    parts = variant.split(":")
    feature, threshold = parts[1], float(parts[2])
    c = copy.deepcopy(cfg)
    c.simulation.signal_source = "rule"
    c.simulation.rule_feature = feature
    c.simulation.rule_threshold = threshold
    c.simulation.use_expected_value_filter = False
    c.simulation.use_trend_filter = len(parts) > 3 and parts[3] == "trend"
    signals = build_signal_column(ds.test.copy(), c)
    trades, _ = simulate_trades(signals, ds.minute_slice("test"), c)
    return trades


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--config", required=True)
    period = ap.add_mutually_exclusive_group(required=True)
    period.add_argument("--years", nargs=2, type=int, metavar=("FROM", "TO"))
    period.add_argument("--quarters", nargs=2, metavar=("FROM", "TO"), help="напр. 2024Q3 2026Q1")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="path=value, как в CLI")
    ap.add_argument("--name", default=None, help="имя для файлов результата (по умолчанию run_name конфига)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="правила входа: cb, meta:<порог>, rule:<признак>:<порог>[:trend]; по умолчанию — как в конфиге")
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

    wins = windows(args.years, args.quarters)
    key = "год" if args.years else "квартал"
    variants = args.variants or [None]
    only_rules = all(v and v.startswith("rule:") for v in variants)

    def fold_cfg(label, start, end):
        cfg = copy.deepcopy(base)
        cfg.paths.run_name = f"wf_{name}_{label}"
        cfg.data.splits.update({
            "train_end": f"{start:%Y-%m-%d %H:%M:%S}",
            "test_start": f"{start:%Y-%m-%d %H:%M:%S}",
            "test_end": f"{end:%Y-%m-%d %H:%M:%S}",
            "sim_start": f"{end:%Y-%m-%d %H:%M:%S}",
            "sim_end": None,
        })
        return cfg

    print(f"[{name}] сборка признаков и разметки (один раз на все окна)...", flush=True)
    ds_all = build_dataset(fold_cfg(*wins[0]))

    rows = {v: [] for v in variants}
    trades_by = {v: [] for v in variants}
    for label, start, end in wins:
        cfg = fold_cfg(label, start, end)
        print(f"[{name}] {label}: обучение до {start:%Y-%m-%d}, торговля до {end:%Y-%m-%d}...", flush=True)
        ds = resplit_dataset(ds_all, cfg)
        if ds.test.empty:
            print(f"  нет данных за {label}, пропуск")
            continue
        trained = None if only_rules else run_training(cfg, dataset=ds, save=False)
        metrics = trained.metrics if trained else {}
        for v in variants:
            if v and v.startswith("rule:"):
                t = _rule_trades(ds, cfg, v)
            else:
                vcfg = copy.deepcopy(cfg)
                vol_model = trained.vol
                vcfg.volatility.enabled = False
                if v == "cb":
                    vcfg.simulation.signal_source = "cb"
                elif v and v.startswith("meta:"):
                    vcfg.simulation.signal_source = "meta"
                    vcfg.meta.threshold = float(v.split(":", 1)[1])
                elif v and v.split(":")[0] in ("vol", "volatr"):
                    kind, mn, pw = v.split(":")[:3]
                    vcfg.simulation.signal_source = "cb"
                    vcfg.volatility.enabled = True
                    vcfg.volatility.min_ratio, vcfg.volatility.power = float(mn), float(pw)
                    if kind == "volatr":
                        from forexmodel.models.volatility import train_volatility_model

                        a = copy.deepcopy(vcfg.volatility)
                        a.source = "atr"
                        vol_model = train_volatility_model(ds.train, [], a, 1, vcfg.atr_col)
                    elif vol_model is None:
                        raise ValueError("вариант vol:* требует --set volatility.enabled=true (модель волатильности)")
                elif v is None:
                    vcfg.volatility.enabled = base.volatility.enabled
                t = run_backtest(vcfg, split="test", dataset=ds, primary=trained.primary, nn_model=trained.nn,
                                 meta_model=trained.meta, save=False, vol_model=vol_model).trades
            row = _stats_row(key, label, t, metrics)
            rows[v].append(row)
            if len(t):
                trades_by[v].append(t.assign(**{key: label}))
            print(f"  {v or 'конфиг'}: {row}", flush=True)

    for v in variants:
        table = pd.DataFrame(rows[v])
        suffix = "" if v is None else "__" + v.replace(":", "_")
        print(f"\nWALK-FORWARD · {name}{suffix}")
        print(table.to_string(index=False))
        if len(table):
            pos = int((table["итог_%"] > 0).sum())
            print(f"прибыльных окон: {pos} из {len(table)} | сумма {table['итог_%'].sum():+.2f}% | "
                  f"медиана окна {table['итог_%'].median():+.2f}% | сделка до комиссии в среднем по окнам "
                  f"{table['до_комиссии'].mean():+.3f}%")
        table.to_csv(out_dir / f"{name}{suffix}.csv", index=False)
        if trades_by[v]:
            all_t = pd.concat(trades_by[v])
            all_t.to_csv(out_dir / f"{name}{suffix}_trades.csv", index=False)
            # главная мера: средний результат сделки и его t-статистика по ВСЕМ сделкам
            for lab, col in (("до комиссии", "gross_pct"), ("после комиссии", "profit_pct")):
                x = all_t[col]
                t_stat = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else float("nan")
                print(f"сделка {lab}: {x.mean():+.3f}% ± {x.std() / np.sqrt(len(x)):.3f} (t = {t_stat:+.2f}, n = {len(x)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
