"""Предсказательная сила признаков напрямую, без модели (information coefficient).

Зачем: прежде чем смотреть, «улучшила ли модель», надо знать, есть ли в новых
признаках сигнал вообще. IC — ранговая корреляция Спирмена признака с будущей
доходностью за горизонт разметки (в ATR). |IC| 0.02–0.03 на часовых барах —
уже заметно, 0.05+ — сильно; 0.00 — признак для направления бесполезен (но
может быть полезен мета-модели как признак «когда модели верить»).

Считается на train, чтобы не подглядывать в test/sim. Будущие доходности на
соседних барах пересекаются (горизонт 10 баров), поэтому рядом даётся IC по
разреженной выборке (каждый 10-й бар) — если он того же знака и порядка,
эффект не артефакт пересечения.

    python scripts/feature_ic.py -c configs/lkoh_2024_orderflow.yaml --prefix of_
    python scripts/feature_ic.py -c configs/silver.yaml --top 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (путь до пакета)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forexmodel.config import load_config
from forexmodel.logging_utils import get_logger, setup_logging
from forexmodel.pipelines.dataset import build_dataset

log = get_logger(__name__)

#: технические признаки для сравнения масштаба — чтобы IC нового признака
#: было с чем сопоставить
REFERENCE = ["rsi_4", "macd_atr", "ext_from_low_24", "ext_from_high_24", "er_24", "adx_4", "rel_vol_20", "trend_4h"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IC признаков относительно будущей доходности")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--prefix", default=None, help="считать только признаки с этим префиксом (напр. of_)")
    parser.add_argument("--split", default="train", choices=["train", "test", "sim"])
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.WARNING))
    cfg = load_config(Path(args.config))
    ds = build_dataset(cfg)
    df = ds.splits[args.split].reset_index(drop=True)

    h = cfg.labeling.horizon
    fwd = (df["close"].shift(-h) - df["close"]) / df[cfg.atr_col].replace(0, np.nan)
    label_dir = df["label"].map({0: -1.0, 1: 0.0, 2: 1.0}) if "label" in df.columns else None

    features = [c for c in ds.features if not args.prefix or c.startswith(args.prefix)]
    if args.prefix:
        features += [c for c in REFERENCE if c in ds.features]

    rows = []
    for col in features:
        x = df[col]
        ok = x.notna() & fwd.notna()
        if ok.sum() < 100:
            continue
        ic, _ = spearmanr(x[ok], fwd[ok])
        sparse = ok & (np.arange(len(df)) % h == 0)
        ic_sparse, _ = spearmanr(x[sparse], fwd[sparse])

        # монотонность: средняя будущая доходность (в ATR) по квинтилям признака
        q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
        by_q = fwd[ok].groupby(q).mean()
        top_bottom = float(by_q.iloc[-1] - by_q.iloc[0])

        row = {
            "признак": col,
            "IC": round(float(ic), 4),
            "IC_разреж": round(float(ic_sparse), 4),
            "n": int(ok.sum()),
            "Q5−Q1 (ATR)": round(top_bottom, 3),
            "группа": "orderflow" if col.startswith("of_") else "база",
        }
        if label_dir is not None:
            ic_lab, _ = spearmanr(x[ok], label_dir[ok])
            row["IC_метка"] = round(float(ic_lab), 4)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("IC", key=lambda s: s.abs(), ascending=False)
    print(f"\nIC признаков vs будущая доходность за {h} баров (в ATR), выборка {args.split}, {len(df)} баров")
    print(f"IC_разреж — по каждому {h}-му бару (непересекающиеся горизонты); Q5−Q1 — разница средней")
    print("доходности между верхним и нижним квинтилем признака\n")
    print(out.head(args.top).to_string(index=False))

    of = out[out["группа"] == "orderflow"]
    if len(of):
        print(f"\nOrderflow: средний |IC| = {of['IC'].abs().mean():.4f}, признаков с |IC| >= 0.02: {int((of['IC'].abs() >= 0.02).sum())} из {len(of)}")
    base = out[out["группа"] == "база"]
    if len(base):
        print(f"База (для сравнения): средний |IC| = {base['IC'].abs().mean():.4f}")

    output = Path(args.output) if args.output else cfg.reports_path() / f"feature_ic_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"\nТаблица сохранена: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
