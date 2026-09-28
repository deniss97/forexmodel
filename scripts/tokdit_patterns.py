"""TokDiT как модель паттерна: паттерн даёт момент и сторону, диффузия — брать ли сделку.

Паттерн — импульс «6 ч ≥ 3 ATR по тренд-гейту» (docs/results/patterns.md), выход 48 ч.
TokDiT (nano, с нуля на CPU) учится продолжать 96 часов цены на `--horizon` часов вперёд;
одна модель на все инструменты: окно нормируется по контексту (ZNorm по наблюдённым
позициям), масштаб цены ей неважен. Обучающие окна целиком, вместе с горизонтом,
заканчиваются до `--split-year`, импульсы для проверки — с `--split-year`.

Для каждого импульса модель сэмплирует S путей; P(продолжение) — доля путей, которые
к концу горизонта ушли в сторону сделки. Проверки:
  * AUC P(продолжение) против «сделка в плюсе» (по фактическому исходу выхода 48 ч);
  * сделки только при P ≥ медианы P на обучающих импульсах (порог известен заранее)
    против всех импульсов;
  * воспроизводит ли модель импульс: средний предсказанный ход в сторону сделки.

    python scripts/tokdit_patterns.py --exit 48:4:4:2 --horizon 48
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import tokdit_direction as td
from entry_rules_lab import sequential
from pattern_lab import INSTRUMENTS, add_features, bars, patterns

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exit", default="48:4:4:2")
    ap.add_argument("--pattern", default="импульс 6/3.0")
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--split-year", type=int, default=2021)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--sampler-steps", type=int, default=10)
    ap.add_argument("--pred-batch", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--tokdit-size", default="nano")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    import torch

    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(td.TOKDIT))
    L, H = args.context, args.horizon
    split = np.datetime64(f"{args.split_year}-01-01")

    hs, train_paths, events = {}, [], []
    for inst in INSTRUMENTS:
        h, cfg = bars(inst, exit_spec=args.exit)
        h = add_features(h)
        hs[inst] = (h, cfg.simulation.commission_pct)
        paths, t_pred, t_end = td.real_windows(h[["time", "close"]], L, H)
        train_paths.append(paths[t_end < split])
        d = patterns(h)[args.pattern]
        ev = np.flatnonzero((d != 0) & (d == h["gate"].to_numpy()))
        # окно, у которого последний бар контекста — бар импульса: индекс окна = i − (L − 1)
        w = ev - (L - 1)
        ok = (w >= 0) & (w < len(paths))
        events.append(pd.DataFrame({"inst": inst, "i": ev[ok], "w": w[ok], "side": d[ev[ok]],
                                    "time": h["time"].to_numpy()[ev[ok]]}))
        events[-1]["paths"] = list(paths[w[ok]])
    trp = np.concatenate(train_paths)
    ev = pd.concat(events, ignore_index=True)
    print(f"обучение TokDiT: {len(trp)} окон до {args.split_year}; импульсов всего {len(ev)}", flush=True)

    t0 = time.time()
    model = td.fit_tokdit(trp, L, H, args, args.seed)
    print(f"обучено за {time.time() - t0:.0f} с", flush=True)
    P = np.stack(ev["paths"].to_numpy())
    p_up, mean = td.predict_tokdit(model, P, L, H, args)
    ev = ev.drop(columns="paths")
    ev["p_cont"] = np.where(ev["side"] > 0, p_up, 1 - p_up)
    ev["pred_move"] = ev["side"] * mean                       # предсказанный ход в сторону сделки, %
    ev["real_move"] = ev["side"] * P[:, -1]                    # фактический ход к концу горизонта, %
    gross = []
    for inst, (h, _) in hs.items():
        m = ev["inst"] == inst
        gl, gs = h["g_long"].to_numpy(), h["g_short"].to_numpy()
        i = ev.loc[m, "i"].to_numpy()
        gross.append(pd.Series(np.where(ev.loc[m, "side"] > 0, gl[i], gs[i]), index=ev.index[m]))
    ev["gross"] = pd.concat(gross)
    out = ROOT / "reports" / "patterns"
    ev.to_csv(out / "tokdit_patterns_events.csv", index=False)

    from sklearn.metrics import roc_auc_score

    ev["период"] = np.where(ev["time"] < split, "выбор", "проверка")
    rows = []
    for inst, g in ev.groupby("inst"):
        h, comm = hs[inst]
        thr = float(g.loc[g["период"] == "выбор", "p_cont"].median())      # порог — по импульсам периода выбора
        for per, x in g.groupby("период"):
            y = (x["gross"] - comm > 0).astype(int)
            row = {"инструмент": inst, "период": per, "импульсов": len(x),
                   "auc": roc_auc_score(y, x["p_cont"]) if 0 < y.sum() < len(y) else np.nan,
                   "пред_ход": x["pred_move"].mean(), "факт_ход": x["real_move"].mean(), "порог": thr}
            for name, sel in (("все", x), ("P≥порога", x[x["p_cont"] >= thr])):
                dd = np.zeros(len(h))
                dd[sel["i"].to_numpy()] = sel["side"].to_numpy()
                tr = sequential(h, dd, np.ones(len(h), bool), comm)
                row[f"{name}_сделок"] = len(tr)
                row[f"{name}_нетто"] = tr["net"].mean() if len(tr) else np.nan
                row[f"{name}_итог"] = tr["net"].sum()
            rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(out / "tokdit_patterns.csv", index=False)
    print(r.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
