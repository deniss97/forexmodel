"""Важность признаков модели CatBoost по окнам walk-forward: на обучении и вне выборки.

На каждом окне (год Y) модель учится на данных до Y, как в scripts/walk_forward.py, и для неё
снимаются две важности:
  * «на обучении» — PredictionValuesChange: насколько признак меняет прогноз модели (норма — 100
    в сумме по признакам). Показывает, на что модель опирается;
  * «вне выборки» — LossFunctionChange на барах года Y (тех, где модель торгует: data.train_query
    применяется и к ним): насколько вырастет ошибка модели на НОВЫХ данных, если убрать признак.
    Плюс — признак помогает на следующем году, минус — мешает (модель выучила на нём то, что не
    повторилось).
Итог — средние по окнам и доля окон, где признак помог вне выборки, плюс сводка по группам.

    python scripts/feature_importance.py -c configs/silver.yaml --years 2019 2026 --name silver
    python scripts/feature_importance.py -c configs/silver.yaml --years 2019 2026 --name silver_emazone \\
        --set features.use_ema_zone_features=true
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
from forexmodel.pipelines.dataset import build_dataset, resplit_dataset
from forexmodel.pipelines.sweep import set_by_path
from forexmodel.pipelines.train_pipeline import run_training
from walk_forward import windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "feature_importance"

# группы признаков по имени (первое совпадение)
GROUPS = [
    ("время (час, день недели)", r"^(hour_|dow$)"),
    ("слабая зона EMA5", r"^(ema5_|trend_ema5_)"),
    ("тренд-фильтр 4ч", r"(_4h$)"),
    ("волатильность (ATR, ширина канала, размах свечи)", r"^(atr_|bb_width|candle_range_atr)"),
    ("доходности и их лаги", r"^(log_return|ret_lag_)"),
    ("свеча (тело, тени, серия)", r"^(body_|lower_wick|upper_wick|candle_streak)"),
    ("RSI", r"^rsi"),
    ("MACD", r"^macd"),
    ("скользящие средние (отклонение, спред)", r"^(close_(sma|ema)_|ema_spread|sma_spread)"),
    ("ADX / DI", r"^(adx_4$|adx_slope$|di_spread|dmn_|dmp_)"),
    ("положение в диапазоне, растяжение, z-оценки", r"^(ext_from|close_pos|bb_pos|z_close|runup|bars_since)"),
    ("эффективность движения (Кауфман)", r"^er_"),
]


def group_of(name: str) -> str:
    for g, rx in GROUPS:
        if re.search(rx, name):
            return g
    return "прочее"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    ap.add_argument("--regroup", action="store_true", help="только пересобрать таблицы из сохранённых окон")
    args = ap.parse_args(argv)
    if args.regroup:
        return summarize(pd.read_csv(OUT / f"{args.name}_by_window.csv"), args.name)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    from catboost import Pool

    base = load_config(ROOT / args.config)
    for item in ["nn.enabled=false", "meta.enabled=false", "simulation.signal_source=cb",
                 f"catboost.random_seed={args.seed}", *args.overrides]:
        path, raw = item.split("=", 1)
        set_by_path(base, path.strip(), yaml.safe_load(raw))

    def fold_cfg(label, start, end):
        cfg = copy.deepcopy(base)
        cfg.paths.run_name = f"fi_{args.name}_{label}"
        cfg.data.splits.update({"train_end": f"{start:%Y-%m-%d %H:%M:%S}", "test_start": f"{start:%Y-%m-%d %H:%M:%S}",
                                "test_end": f"{end:%Y-%m-%d %H:%M:%S}", "sim_start": f"{end:%Y-%m-%d %H:%M:%S}",
                                "sim_end": None})
        return cfg

    wins = windows(years=args.years)
    print(f"[{args.name}] сборка датасета...", flush=True)
    ds_all = build_dataset(fold_cfg(*wins[0]))
    rows = []
    for label, start, end in wins:
        cfg = fold_cfg(label, start, end)
        ds = resplit_dataset(ds_all, cfg)
        if ds.test.empty:
            continue
        trained = run_training(cfg, dataset=ds, save=False)
        pm = trained.primary
        feats = pm.features
        pvc = np.asarray(pm.model.get_feature_importance(), dtype=float)
        test = ds.test.query(cfg.data.train_query) if cfg.data.train_query else ds.test
        y = test["label"].map(pm.class_map)
        ok = y.notna()
        lfc = np.asarray(pm.model.get_feature_importance(Pool(test.loc[ok, feats], y[ok].astype(int)),
                                                         type="LossFunctionChange"), dtype=float)
        for f, a, b in zip(feats, pvc, lfc):
            rows.append({"окно": label, "признак": f, "на_обучении": a, "вне_выборки": b})
        print(f"  {label}: деревьев {pm.model.tree_count_}, признаков {len(feats)}, тест {int(ok.sum())} баров",
              flush=True)
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / f"{args.name}_by_window.csv", index=False)
    return summarize(r, args.name)


def summarize(r: pd.DataFrame, name: str) -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    n_win = r["окно"].nunique()
    # вне выборки — в долях от суммы положительных за окно, чтобы окна были сопоставимы
    r["вне_выборки_норм"] = r.groupby("окно")["вне_выборки"].transform(lambda x: 100 * x / x.clip(lower=0).sum())
    agg = r.groupby("признак").agg(на_обучении=("на_обучении", "mean"), вне_выборки=("вне_выборки_норм", "mean"),
                                   окон_помог=("вне_выборки", lambda x: int((x > 0).sum())))
    agg["группа"] = [group_of(f) for f in agg.index]
    agg = agg.sort_values("на_обучении", ascending=False)
    agg.to_csv(OUT / f"{name}.csv")
    grp = agg.groupby("группа").agg(признаков=("на_обучении", "size"), на_обучении=("на_обучении", "sum"),
                                    вне_выборки=("вне_выборки", "sum")).sort_values("на_обучении", ascending=False)
    grp.to_csv(OUT / f"{name}_groups.csv")
    print(f"\nВАЖНОСТЬ ПРИЗНАКОВ · {name} · {n_win} окон (на обучении — % прогноза; вне выборки — % от "
          f"суммы помогающих на следующем году, минус — мешает)")
    print(agg.round(2).head(30).to_string())
    print("\nПО ГРУППАМ")
    print(grp.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
