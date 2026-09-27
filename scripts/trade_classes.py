"""Классы сделок, стоп по классу, «спасательный» вход и режим рынка — на готовом прогоне.

Проверка идей из обсуждения (docs/results/silver_1h_cb/trade_review.md, часть II):

  1. «Вход на откате — другой риск»: сделки делятся по ходу цены за 3 бара до
     сигнала относительно направления сделки — продолжение (≥ 0.3 ATR по ходу),
     плоско, откат (0.3–1 ATR против), нож (> 1 ATR против) — и для каждого класса
     считается результат при стопе 0.5 / 0.75 / 1 / 1.5 / 2 ATR (трейлинг как в
     конфиге). Затем политики «стоп по классу» гоняются последовательно.
  2. «Если подтверждение на нескольких уровнях — не жди 4 часа»: спасательный
     вход при нейтральном гейте, если бар сигнала ≥ 0.5 ATR и ход за 3 бара
     ≥ 1.5 ATR в одну сторону.
  3. Режим рынка: скользящие variance ratio и автокорреляция часовых доходностей
     за 10 и 30 дней (каузально) — лучше ли входы по тренду в «трендовом» режиме.

Направление на истории (train) — гейт C (без «ADX растёт», каждый час, удержание
2 ч), на test/sim — сигналы модели, совпавшие с гейтом. Исходы сделки на каждом
баре при каждом стопе кэшируются в reports/<run>/entry_rules_fwd_sl<стоп>.pkl.

    python scripts/trade_classes.py --run silver_1h_cb
    python scripts/trade_classes.py --run lkoh_2024_base_reg_cb --stops 1 2
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from entry_rules_lab import _stats, forward_outcomes
from forexmodel.config import load_config
from forexmodel.data.loader import load_minute_compact
from forexmodel.features.trend import trend_gate_values

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["продолжение", "плоско", "откат", "нож"]


def classify(d: np.ndarray, r3_atr: np.ndarray) -> np.ndarray:
    m = d * r3_atr
    return np.where(d == 0, "нет", np.where(m >= 0.3, "продолжение", np.where(m <= -1, "нож", np.where(m <= -0.3, "откат", "плоско"))))


def sequential_policy(h: pd.DataFrame, outs: dict, d: np.ndarray, cls: np.ndarray, sl_of_class: dict, commission: float) -> pd.DataFrame:
    """Сделки без перекрытия; стоп берётся по классу входа."""
    t = h["time"].to_numpy()
    last_exit, rows = np.datetime64("1970-01-01"), []
    for i in np.flatnonzero(d != 0):
        if t[i] <= last_exit:
            continue
        o = outs[sl_of_class[cls[i]]]
        k = "long" if d[i] > 0 else "short"
        g = o[f"g_{k}"].iat[i]
        if not np.isfinite(g):
            continue
        rows.append((t[i], cls[i], g))
        last_exit = o[f"exit_{k}"].iat[i]
    r = pd.DataFrame(rows, columns=["time", "класс", "gross"])
    r["net"] = r["gross"] - commission
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--stops", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.5, 2.0])
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    cfg = load_config(ROOT / "artifacts" / args.run / "config.yaml")
    reports = ROOT / "reports" / args.run
    splits = cfg.data.splits
    test_start, sim_start = pd.Timestamp(splits["test_start"]), pd.Timestamp(splits["sim_start"])
    comm = cfg.simulation.commission_pct

    outs = {}
    for sl in args.stops:
        cache = reports / f"entry_rules_fwd_sl{sl:g}.pkl"
        if not cache.exists():
            print(f"Считаю исходы при стопе {sl:g} ATR (один раз)...")
            minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
            c = copy.deepcopy(cfg)
            c.simulation.sl_atr = sl
            forward_outcomes(minute, c).to_pickle(cache)
            del minute
            gc.collect()
        outs[sl] = pd.read_pickle(cache)
    h = outs[args.stops[0]]
    period = np.where(h["time"] < test_start, "train", np.where(h["time"] < sim_start, "test", "sim"))
    atr = h["atr_14"]
    bar = ((h["close"] - h["open"]) / atr).to_numpy()
    r3 = ((h["close"] - h["close"].shift(3)) / atr).to_numpy()
    gate = trend_gate_values(h, dataclasses.replace(cfg.trend, require_adx_rising=False, update_every_bar=True, hold_bars=2)).astype(int)
    hidx = pd.Index(h["time"])

    def direction(split: str) -> np.ndarray:
        if split == "train":
            return np.where(period == "train", gate, 0)
        s = pd.read_csv(reports / f"signals_{split}.csv", parse_dates=["time"], usecols=["time", "final_class"])
        idx = hidx.get_indexer(s["time"])
        md = np.where(s["final_class"] == 2, 1, np.where(s["final_class"] == 0, -1, 0))
        d = np.zeros(len(h), int)
        d[idx] = np.where(md == gate[idx], md, 0)
        return d

    def gross(sl: float, d: np.ndarray) -> np.ndarray:
        o = outs[sl]
        return np.where(d > 0, o["g_long"], np.where(d < 0, o["g_short"], np.nan))

    # 1. класс × стоп: средний результат входа на баре, до комиссии
    rows = []
    for split in ("train", "test", "sim"):
        d = direction(split)
        cls = classify(d, r3)
        for sl in args.stops:
            g = gross(sl, d)
            for c in CLASSES:
                m = cls == c
                rows.append({"выборка": split, "класс": c, "стоп": sl, "n": int(m.sum()), "до_комиссии": round(float(np.nanmean(g[m])), 3)})
    t1 = pd.DataFrame(rows)
    print("\n1. КЛАСС × СТОП: средний результат входа до комиссии, % (train — по гейту C без модели; test/sim — сигналы модели)")
    for split in ("train", "test", "sim"):
        x = t1[t1["выборка"] == split]
        print(f"\n  {split}, n по классам: {x[x['стоп'] == args.stops[0]].set_index('класс')['n'].to_dict()}")
        print(x.pivot_table(index="класс", columns="стоп", values="до_комиссии", sort=False).to_string())
    t1.to_csv(reports / "trade_classes_stops.csv", index=False)

    # 2. политики стопа по классу, последовательно
    s0, s1 = args.stops[0], args.stops[-1]
    policies = {
        f"стоп {s0:g} всем": {c: s0 for c in CLASSES},
        f"стоп {s1:g} всем": {c: s1 for c in CLASSES},
        f"откат/нож {s0:g}, остальным {s1:g}": {"продолжение": s1, "плоско": s1, "откат": s0, "нож": s0},
        f"откат/нож {s1:g}, остальным {s0:g}": {"продолжение": s0, "плоско": s0, "откат": s1, "нож": s1},
    }
    if 1.0 in args.stops:
        policies["стоп 1 всем (как в конфиге)"] = {c: 1.0 for c in CLASSES}
    rows = []
    for split in ("train", "test", "sim"):
        d = direction(split)
        cls = classify(d, r3)
        for name, pol in policies.items():
            r = sequential_policy(h, outs, d, cls, pol, comm)
            rows.append({"политика": name, "выборка": split, **_stats(r["net"], r["gross"])})
    t2 = pd.DataFrame(rows)
    print("\n2. СТОП ПО КЛАССУ, последовательные сделки")
    print(t2.pivot_table(index="политика", columns="выборка", values=["сделок", "до_комиссии", "итог_%", "PF"], sort=False).to_string())
    t2.to_csv(reports / "trade_classes_policies.csv", index=False)

    # 3. спасательный вход по импульсу при нейтральном гейте
    mom = np.where((np.abs(r3) >= 1.5) & (np.sign(bar) == np.sign(r3)) & (np.abs(bar) >= 0.5), np.sign(r3), 0).astype(int)
    rows = []
    one = {c: 1.0 if 1.0 in outs else args.stops[0] for c in CLASSES}
    for split in ("train", "test", "sim"):
        base = direction(split)
        if split == "train":
            resc = np.where((base == 0) & (period == "train"), mom, base)
        else:
            s = pd.read_csv(reports / f"signals_{split}.csv", parse_dates=["time"], usecols=["time", "final_class"])
            idx = hidx.get_indexer(s["time"])
            md = np.where(s["final_class"] == 2, 1, np.where(s["final_class"] == 0, -1, 0))
            resc = base.copy()
            resc[idx] = np.where((gate[idx] == 0) & (md != 0) & (md == mom[idx]), md, base[idx])
        only = np.where((base == 0) & (resc != 0), resc, 0)
        for name, d in (("гейт C", base), ("гейт C + спасательный вход", resc), ("только спасательные входы", only)):
            r = sequential_policy(h, outs, d, classify(d, r3), one, comm)
            rows.append({"правило": name, "выборка": split, **_stats(r["net"], r["gross"])})
    t3 = pd.DataFrame(rows)
    print("\n3. СПАСАТЕЛЬНЫЙ ВХОД: гейт нейтрален, но бар ≥ 0.5 ATR и 3 бара ≥ 1.5 ATR в одну сторону")
    print(t3.pivot_table(index="правило", columns="выборка", values=["сделок", "до_комиссии", "итог_%", "PF"], sort=False).to_string())
    t3.to_csv(reports / "trade_classes_rescue.csv", index=False)

    # 4. режим рынка: скользящие VR и автокорреляция (каузально), терцили по train
    lr = np.log(h["close"])
    cont = h["time"].diff() <= pd.Timedelta("12h")
    r1 = lr.diff().where(cont)
    r4 = lr.diff(4).where(cont.rolling(4).sum() == 4)
    feats = {}
    for win in (240, 720):
        feats[f"vr4_{win}"] = r4.rolling(win, min_periods=win // 2).var() / (4 * r1.rolling(win, min_periods=win // 2).var())
        feats[f"acf1_{win}"] = r1.rolling(win, min_periods=win // 2).corr(r1.shift(1))
    g1 = gross(one["откат"], gate)
    rows = []
    for fname, f in feats.items():
        tr = (period == "train") & (gate != 0) & f.notna()
        q = f[tr].quantile([1 / 3, 2 / 3]).to_numpy()
        bucket = np.where(f < q[0], "низкий", np.where(f < q[1], "средний", "высокий"))
        row = {"метрика": fname, "границы": f"{q[0]:.3f} / {q[1]:.3f}"}
        for b in ("низкий", "средний", "высокий"):
            m = tr & (bucket == b)
            row[f"train {b}"] = round(float(np.nanmean(g1[m])), 3)
        for split in ("test", "sim"):
            d = direction(split)
            gm = gross(one["откат"], d)
            for b in ("низкий", "высокий"):
                m = (d != 0) & (bucket == b) & f.notna()
                row[f"{split} {b}"] = round(float(np.nanmean(gm[m])), 3) if m.any() else np.nan
        rows.append(row)
    t4 = pd.DataFrame(rows)
    print("\n4. РЕЖИМ: результат входа по тренду до комиссии по терцилям скользящей метрики (терцили — по train)")
    print(t4.to_string(index=False))
    t4.to_csv(reports / "trade_classes_regime.csv", index=False)
    print(f"\nТаблицы: {reports}/trade_classes_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
