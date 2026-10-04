"""Walk-forward с отбором признаков по прошлым окнам: модель на 10–20 признаках вместо 81.

Идея (docs/results/trend_filter_ema.md §8): у 43 из 81 признака модели серебра вклад вне выборки
отрицательный, почти вся польза — время суток и волатильность. Если оставить только признаки,
которые помогали на НОВЫХ данных в прошлом, у модели меньше способов подстроиться под шум.

Без заглядывания: на год Y признаки отбираются по важности вне выборки окон < Y
(reports/feature_importance/<history>_by_window.csv: модель окна Y' учится до Y', важность —
LossFunctionChange на году Y'; всё это известно до начала Y). Модель года Y учится на отобранных
признаках (features.extra_exclude), торгует год Y.

Варианты отбора: все признаки; топ-10 и топ-20 по средней важности вне выборки прошлых окон;
«помогали» — средняя важность > 0 и помог хотя бы в половине прошлых окон; для сравнения — топ-10
по важности на обучении (на что модель опирается, а не что помогает).

    python scripts/feature_selection_wf.py -c configs/silver.yaml --years 2019 2026 --history silver_2016
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
from forexmodel.pipelines.backtest_pipeline import run_backtest
from forexmodel.pipelines.dataset import build_dataset, resplit_dataset
from forexmodel.pipelines.sweep import set_by_path
from forexmodel.pipelines.train_pipeline import run_training
from walk_forward import windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "feature_importance"


def select(hist: pd.DataFrame, year: int, rule: str) -> list:
    past = hist[hist["окно"].astype(int) < year]
    if past.empty:
        return []
    past = past.assign(norm=past.groupby("окно")["вне_выборки"].transform(lambda x: 100 * x / x.clip(lower=0).sum()))
    g = past.groupby("признак").agg(oos=("norm", "mean"), helped=("вне_выборки", lambda x: (x > 0).mean()),
                                    train=("на_обучении", "mean"))
    if rule == "топ-10 вне выборки":
        return g.sort_values("oos", ascending=False).head(10).index.tolist()
    if rule == "топ-20 вне выборки":
        return g.sort_values("oos", ascending=False).head(20).index.tolist()
    if rule == "помогали":
        return g[(g["oos"] > 0) & (g["helped"] >= 0.5)].index.tolist()
    if rule == "топ-10 на обучении":
        return g.sort_values("train", ascending=False).head(10).index.tolist()
    raise ValueError(rule)


RULES = ["все признаки", "топ-10 вне выборки", "топ-20 вне выборки", "помогали", "топ-10 на обучении"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True)
    ap.add_argument("--history", required=True, help="имя прогона scripts/feature_importance.py (окна до --years)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1])
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--name", default="silver_fsel")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    hist = pd.read_csv(OUT / f"{args.history}_by_window.csv")
    base = load_config(ROOT / args.config)
    for item in ["nn.enabled=false", "meta.enabled=false", "simulation.signal_source=cb", *args.overrides]:
        path, raw = item.split("=", 1)
        set_by_path(base, path.strip(), yaml.safe_load(raw))

    def fold_cfg(start, end, seed):
        cfg = copy.deepcopy(base)
        cfg.catboost.random_seed = seed
        cfg.data.splits.update({"train_end": f"{start:%Y-%m-%d %H:%M:%S}", "test_start": f"{start:%Y-%m-%d %H:%M:%S}",
                                "test_end": f"{end:%Y-%m-%d %H:%M:%S}", "sim_start": f"{end:%Y-%m-%d %H:%M:%S}",
                                "sim_end": None})
        return cfg

    wins = windows(years=args.years)
    # продолжение после обрыва: сделки и выбор признаков дописываются после каждого окна
    OUT.mkdir(parents=True, exist_ok=True)
    part_t, part_c = OUT / f"{args.name}_trades_part.csv", OUT / f"{args.name}_chosen_part.csv"
    done = set()
    if part_t.exists():
        prev = pd.read_csv(part_t)
        done = set(zip(prev["seed"], prev["год"].astype(str), prev["отбор"]))
        print(f"продолжение: уже посчитано {len(done)} окон", flush=True)
    print("сборка датасета...", flush=True)
    ds_all = build_dataset(fold_cfg(wins[0][1], wins[0][2], args.seeds[0]))
    all_feats = ds_all.features
    for seed in args.seeds:
        for label, start, end in wins:
            for rule in RULES:
                if (seed, str(label), rule) in done:
                    continue
                cfg = fold_cfg(start, end, seed)
                if rule != "все признаки":
                    keep = [f for f in select(hist, int(label), rule) if f in all_feats]
                    cfg.features.extra_exclude = sorted(set(all_feats) - set(keep))
                    row = {"seed": seed, "год": label, "отбор": rule, "признаков": len(keep), "признаки": ", ".join(keep)}
                    pd.DataFrame([row]).to_csv(part_c, mode="a", header=not part_c.exists(), index=False)
                ds = resplit_dataset(ds_all, cfg)
                if ds.test.empty:
                    continue
                trained = run_training(cfg, dataset=ds, save=False)
                t = run_backtest(cfg, split="test", dataset=ds, primary=trained.primary, save=False).trades
                t = t.assign(seed=seed, год=label, отбор=rule, признаков=len(trained.features))
                if not len(t):        # пустое окно тоже отмечаем как посчитанное
                    t = pd.DataFrame([{"seed": seed, "год": label, "отбор": rule, "profit_pct": np.nan}])
                t.to_csv(part_t, mode="a", header=not part_t.exists(), index=False)
                print(f"  seed {seed} {label} {rule}: признаков {len(trained.features)}, сделок {len(t)}, "
                      f"итог {t['profit_pct'].sum() if len(t) else 0:+.2f}%", flush=True)
    t = pd.read_csv(part_t).dropna(subset=["profit_pct"])
    t["год"] = t["год"].astype(str)
    t.to_csv(OUT / f"{args.name}_trades.csv", index=False)
    pd.read_csv(part_c).to_csv(OUT / f"{args.name}_chosen.csv", index=False)
    rows = []
    for (rule, seed), g in t.groupby(["отбор", "seed"], sort=False):
        p = g["profit_pct"]
        y = g.groupby("год")["profit_pct"].sum()
        rows.append({"отбор": rule, "seed": seed, "сделок": len(p), "итог": p.sum(), "на_сделку": p.mean(),
                     "t": p.mean() / (p.std() / np.sqrt(len(p))), "лет_в_плюсе": f"{int((y > 0).sum())}/{len(y)}",
                     "до_комиссии": g["gross_pct"].mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"{args.name}.csv", index=False)
    print(r.round(3).to_string(index=False))
    s = r.groupby("отбор", sort=False).agg(сделок=("сделок", "mean"), итог=("итог", "mean"), на_сделку=("на_сделку", "mean"),
                                           t=("t", "mean"))
    s["t по seed"] = r.groupby("отбор", sort=False)["t"].apply(lambda x: " / ".join(f"{v:+.2f}" for v in x))
    print("\nСРЕДНЕЕ ПО SEED")
    print(s.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
