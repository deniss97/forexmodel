"""Честный бумажный прогон сигналов S42: минутки обрезаны по моменту запуска — как они лежали у живого контура (03.11).

Прежний прогон (scripts/paper_replay.py, 22.10) звал live_signals.analyse на полном файле: будущие минуты были в данных,
и разметка triple-barrier у свежей свечи уже известна. Поэтому он не увидел, что живой контур выбрасывал последнюю
свечу и узнавал о входе S42 на 1–3 дня позже (протокол 30.10). Здесь загрузчик минуток подменён: build_dataset видит
только минуты до момента запуска.

Для каждой сделки эталона (S42 одним прогоном на полном файле, как paper_replay.reference) — решение контура в
моменты запуска вокруг входа HH:01:
  * HH:01 — новый порядок (S42 первым шагом): в данных минуты до HH:00, вход ещё впереди — «ОТКРЫТЬ в HH:01»;
  * HH:05 — нынешний cron: минута входа уже в данных — «ОТКРЫТЬ» по окну 30 минут;
  * HH:05 со старым кодом (разметка) — как контур работал до 03.11.
И случайные часы вне сделок эталона: контур должен молчать (проверка, что вход «в HH:01» не даёт лишних сигналов).
Один вызов analyse — 15–20 с; решения кэшируются в reports/paper_replay/lag_decisions.pkl.

    python scripts/live_lag_replay.py [--random 40]
"""
from __future__ import annotations

import argparse
import logging
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

import live_signals as ls
import paper_replay as pr
from experiments import ROOT
from forexmodel.data import loader
from forexmodel.pipelines import dataset

OUT = ROOT / "reports" / "paper_replay"
DOC = ROOT / "docs" / "results" / "live_lag_replay.md"
CSV = ROOT / "docs" / "results" / "live_lag_replay" / "decisions.csv"
VARIANTS = {"HH:01, без разметки": (1, False), "HH:05, без разметки": (5, False), "HH:05, с разметкой (до 03.11)": (5, True)}

_MIN: dict = {}            # путь CSV → минутки целиком (читаются один раз)
_CUT: list = [None]        # момент запуска: build_dataset видит только минуты раньше него


def _cut_loader(orig):
    def load(path, *a, **kw):
        key = str(path)
        if key not in _MIN:
            _MIN[key] = orig(path, *a, **kw)
        m = _MIN[key]
        return m[m["time"] < _CUT[0]].reset_index(drop=True) if _CUT[0] is not None else m
    return load


dataset.load_minute_compact = _cut_loader(loader.load_minute_compact)
dataset.load_minute_csv = _cut_loader(loader.load_minute_csv)


def decide(cache: dict, inst: str, run: pd.Timestamp, labels: bool, live: dict) -> dict:
    """Решение analyse в момент run: в данных минуты с началом до run − 1 мин (минута run − 1 ещё не закрыта — не берём)."""
    key = (inst, run, labels)
    if key not in cache:
        _CUT[0] = run - pd.Timedelta(minutes=1)
        orig = ls.build_dataset
        ls.build_dataset = lambda cfg, with_labels=True: orig(cfg, with_labels=labels)
        try:
            r = ls.analyse(inst, live["instruments"][inst], live, run - pd.Timedelta(minutes=1))     # окно 40 дней — от момента запуска
        finally:
            ls.build_dataset, _CUT[0] = orig, None
        m = r["_м"]
        cache[key] = {"действие": r["действие"], "решение": m["решение"], "вход_в": m.get("вход_в"), "цена": m.get("цена"),
                      "стоп": m.get("стоп"), "данные_до": m.get("данные_до")}
        pd.to_pickle(cache, OUT / "lag_decisions.pkl")
        print(f"   [{inst}] {run:%m-%d %H:%M} {'разметка' if labels else 'без'}: {r['действие']}", flush=True)
    return cache[key]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--random", type=int, default=40, help="случайных часов вне сделок на бумагу (проверка молчания)")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    cache = pd.read_pickle(OUT / "lag_decisions.pkl") if (OUT / "lag_decisions.pkl").exists() else {}
    rows, silent, rng = [], [], np.random.default_rng(11)
    for inst in pr.STOCKS:
        ref, last = pr.reference(inst, live)
        print(f"[{inst}] эталон: {len(ref)} сделок с {pr.FRESH:%d.%m}, данные до {last}", flush=True)
        for r in ref.itertuples():
            # вход в HH:01 — запуск в тот же час; вход на первой минуте утренней сессии (06:59, 09:59) после сигнала на
            # последней свече прошлой сессии — первый запуск после него
            hour = r.open_dt.floor("h") if r.open_dt.minute <= 1 else r.open_dt.ceil("h")
            row = {"бумага": inst.upper(), "вход эталона": r.open_dt, "цена эталона": r.open_price,
                   "эталон, %": r.gross_unit_pct, "причина": pr.REASON.get(r.exit_reason, r.exit_reason)}
            for name, (minute, labels) in VARIANTS.items():
                d = decide(cache, inst, hour + pd.Timedelta(minutes=minute), labels, live)
                ok = d["решение"] == "открыть" and abs(pd.Timestamp(d["вход_в"]) - r.open_dt) <= pd.Timedelta(minutes=5)
                row[name] = "ОТКРЫТЬ" if ok else d["решение"]
                if name.startswith("HH:01"):
                    row["цена HH:01 к эталону, %"] = (d["цена"] / r.open_price - 1) * 100 if ok else np.nan
            rows.append(row)
        # молчание: часы с минутками, вдали от входов эталона (±3 ч), запуск в HH:01
        m = next(iter(_MIN.values()))                   # минутки этой бумаги (в кэше только она)
        hours = pd.DatetimeIndex(m.loc[m["time"] >= pr.FRESH, "time"]).floor("h").unique() + pd.Timedelta(hours=1)
        far = [h for h in hours if h <= last and all(abs((h - e).total_seconds()) > 3 * 3600 for e in ref["open_dt"])]
        for h in sorted(rng.choice(far, size=min(a.random, len(far)), replace=False)):
            d = decide(cache, inst, pd.Timestamp(h) + pd.Timedelta(minutes=1), False, live)
            if d["решение"] == "открыть":
                silent.append((inst.upper(), pd.Timestamp(h), d["действие"]))
        _MIN.clear()                                    # минутки бумаги больше не нужны — память под следующую
    t = pd.DataFrame(rows).sort_values("вход эталона").reset_index(drop=True)
    t.to_csv(CSV, index=False)
    write_doc(t, silent, a.random)
    return 0


def write_doc(t: pd.DataFrame, silent: list, n_random: int) -> None:
    n = len(t)
    got = {k: int((t[k] == "ОТКРЫТЬ").sum()) for k in VARIANTS}
    px = t["цена HH:01 к эталону, %"].dropna()
    md = ["# Честный бумажный прогон сигналов S42: данные обрезаны по моменту запуска", "",
          f"SBER, GAZP, LKOH, сделки эталона S42 с {pr.FRESH:%d.%m.%Y} ({n} сделок). Эталон — один прогон пайплайна на "
          "полном файле. Контур — `live_signals.analyse` на минутках, обрезанных по моменту запуска, как у живого cron. "
          "Расчёт — `scripts/live_lag_replay.py`, решения — [live_lag_replay/decisions.csv](live_lag_replay/decisions.csv).", "",
          "| запуск | «ОТКРЫТЬ» вовремя | из сделок эталона |", "|---|---|---|"]
    md += [f"| {k} | {v} | {v / n:.0%} |" for k, v in got.items()]
    big = t[t["цена HH:01 к эталону, %"].abs() > 5]
    md += ["", f"* Цена «ОТКРЫТЬ» в HH:01 (закрытие часа) против цены входа эталона (открытие минуты HH:01): медиана "
           f"{px.median():+.3f}%, от {px.min():+.2f}% до {px.max():+.2f}%." if len(px) else "",
           f"* Разница больше 5% (сделок: {len(big)}) — SBER до 18.07: эталон в ценах с поправкой на дивиденд 18.07 (37.64 ₽), "
           "контур на тот момент — в сырых (дивиденд ещё впереди). Результат сделки в процентах это не меняет." if len(big) else "",
           "* Вход не в HH:01 — первая минута утренней сессии (06:59, 09:59) после сигнала на последней свече прошлой "
           "сессии: такой вход проверяется первым запуском после него (07:01 и 07:05, 10:01 и 10:05).",
           f"* Случайные часы вне сделок эталона (по {n_random} на бумагу, запуск в HH:01): контур открыл бы сделку "
           f"{len(silent)} раз" + (": " + "; ".join(f"{i} {h:%d.%m %H:%M}" for i, h, _ in silent) if silent else "."), "",
           "## Сделки", "", "| бумага | вход эталона | эталон, % | причина | " + " | ".join(VARIANTS) + " |",
           "|---|---|---|---|" + "---|" * len(VARIANTS)]
    for _, r in t.iterrows():
        md.append(f"| {r['бумага']} | {r['вход эталона']:%d.%m %H:%M} | {r['эталон, %']:+.2f} | {r['причина']} | "
                  + " | ".join(str(r[k]) for k in VARIANTS) + " |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
