"""Отдельная модель на каждый паттерн: паттерн задаёт сторону, модель — брать ли сделку.

Идея со встречи 2026-09-28: «три-четыре модели, каждая ориентируется на свой паттерн».
Паттерны и исходы сделок — из `pattern_lab.py` (тот же кэш баров, тот же выход). Для
паттерна P события — бары, где P срабатывает; метка — сделка в сторону P после комиссии
в плюсе. Признаки — на закрытии бара, со знаком относительно стороны сделки (ход за 1/3/6
баров, растяжение от EMA, наклон EMA, положение в диапазоне, гейт, волатильность в ATR и
в % цены, час, день недели).

Walk-forward по годам: модель учится на событиях, сделка которых закрылась до начала
года (последние 20% — для ранней остановки и порога), торгует год. Порог — квантиль
прогноза на этой валидации: берём долю `--keep` лучших. Сравнение — с тем же паттерном
без модели на тех же годах. Модель своя на каждый инструмент и общая на все инструменты
(`pooled`, с признаком инструмента) — у общей в 3 раза больше примеров.

    python scripts/pattern_models.py --exit 48:4:4:2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from entry_rules_lab import sequential
from pattern_lab import INSTRUMENTS, add_features, bars, patterns

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["s_r1", "s_r3", "s_r6", "s_ext", "s_ema_slope", "s_pos72", "s_gate", "vol_ratio", "atr_pct",
            "range24", "range72", "hour", "dow"]
PATTERNS = ["по тренду", "импульс 6/3.0", "пробой 72", "откат в тренде", "серия 3"]


def events(h: pd.DataFrame, d: np.ndarray, inst: str) -> pd.DataFrame:
    """События паттерна: признаки со знаком стороны, исход сделки до комиссии, время выхода."""
    idx = np.flatnonzero(d != 0)
    s = d[idx]
    x = h.iloc[idx]
    atr = x["atr_14"].to_numpy()
    e = pd.DataFrame({
        "i": idx, "time": x["time"].to_numpy(), "side": s, "inst": inst,
        "s_r1": s * x["r1"].to_numpy(), "s_r3": s * x["r3"].to_numpy(), "s_r6": s * x["r6"].to_numpy(),
        "s_ext": s * x["ext"].to_numpy(),
        "s_ema_slope": s * ((x["ema20"] - x["ema50"]).to_numpy() / atr),
        "s_pos72": s * ((x["close"] - (x["hi72"] + x["lo72"]) / 2).to_numpy() / atr),
        "s_gate": s * x["gate"].to_numpy(),
        "vol_ratio": x["vol_ratio"].to_numpy(),
        "atr_pct": atr / x["close"].to_numpy() * 100,
        "range24": x["range24"].to_numpy(), "range72": x["range72"].to_numpy(),
        "hour": x["time"].dt.hour.to_numpy(), "dow": x["time"].dt.dayofweek.to_numpy(),
    })
    e["gross"] = np.where(s > 0, x["g_long"].to_numpy(), x["g_short"].to_numpy())
    e["exit"] = np.where(s > 0, x["exit_long"].to_numpy(), x["exit_short"].to_numpy())
    return e.dropna(subset=["gross"])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, commission: float, seed: int, feats):
    """Прогноз P(сделка в плюсе после комиссии) на test и прогнозы на валидации (для порогов)."""
    from catboost import CatBoostClassifier

    y = (train["gross"] - commission > 0).astype(int)
    cut = int(len(train) * 0.8)
    m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=4, l2_leaf_reg=10, random_seed=seed,
                           verbose=False, early_stopping_rounds=100,
                           cat_features=["inst"] if "inst" in feats else None)
    m.fit(train[feats].iloc[:cut], y.iloc[:cut], eval_set=(train[feats].iloc[cut:], y.iloc[cut:]))
    return m.predict_proba(test[feats])[:, 1], m.predict_proba(train[feats].iloc[cut:])[:, 1]


def trades_from(h, e_sel: pd.DataFrame, commission: float) -> pd.DataFrame:
    d = np.zeros(len(h))
    d[e_sel["i"].to_numpy()] = e_sel["side"].to_numpy()
    return sequential(h, d, np.ones(len(h), bool), commission)


def tstat(x):
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exit", default=None, help="выход, как в pattern_lab (часы:стоп:трейлинг:активация)")
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2026])
    ap.add_argument("--keep", nargs="+", type=float, default=[0.5, 0.3])
    ap.add_argument("--patterns", nargs="+", default=PATTERNS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="pattern_models")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    hs, comm = {}, {}
    for inst in INSTRUMENTS:
        h, cfg = bars(inst, exit_spec=args.exit)
        hs[inst], comm[inst] = add_features(h), cfg.simulation.commission_pct
    pats = {inst: patterns(h) for inst, h in hs.items()}

    rows, all_trades = [], []
    for pname in args.patterns:
        ev = {inst: events(hs[inst], pats[inst][pname], inst) for inst in hs}
        pooled = pd.concat(ev.values()).sort_values("time").reset_index(drop=True)
        for year in range(args.years[0], args.years[1] + 1):
            start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
            for mode in ("own", "pooled"):
                for inst, e in ev.items():
                    src = e if mode == "own" else pooled
                    train = src[pd.to_datetime(src["exit"]) < start]       # сделка закрылась до года теста
                    test = e[(e["time"] >= start) & (e["time"] < end)]
                    if len(train) < 200 or test.empty:
                        continue
                    feats = FEATURES + (["inst"] if mode == "pooled" else [])
                    p, pv = fit_predict(train, test, comm[inst], args.seed, feats)
                    thr_by_keep = {k: float(np.quantile(pv, 1 - k)) for k in args.keep}
                    variants = {"без модели": test}
                    variants.update({f"модель {k:.0%}": test[p >= thr_by_keep[k]] for k in args.keep})
                    for vname, sel in variants.items():
                        if mode == "pooled" and vname == "без модели":
                            continue
                        tr = trades_from(hs[inst], sel, comm[inst])
                        tr = tr[(tr["time"] >= start) & (tr["time"] < end)]
                        lab = vname if vname == "без модели" else f"{vname} ({'своя' if mode == 'own' else 'общая'})"
                        all_trades.append(tr.assign(паттерн=pname, инструмент=inst, год=year, вариант=lab))
            print(f"[{pname}] {year} готов", flush=True)

    t = pd.concat(all_trades)
    out_dir = ROOT / "reports" / "patterns"
    t.to_csv(out_dir / f"{args.name}_trades.csv", index=False)
    for (pname, inst, var), g in t.groupby(["паттерн", "инструмент", "вариант"]):
        yearly = g.groupby("год")["net"].sum()
        rows.append({"паттерн": pname, "инструмент": inst, "вариант": var, "сделок": len(g),
                     "до": g["gross"].mean(), "нетто": g["net"].mean(), "t": tstat(g["net"]),
                     "лет_в_плюсе": f"{int((yearly > 0).sum())}/{len(yearly)}"})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"{args.name}.csv", index=False)
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
