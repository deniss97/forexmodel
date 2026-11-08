"""Скептик к H63, H64, H41 (docs/results/skeptic_2026-11-08.md): перепроверка модели LKOH на исправленном измерении.

Скептик нашёл, что база L10 вырождена: ранняя остановка оставляла 1–10 деревьев, и в 2021 и 2024 годах модель не торговала
(0 сделок на всех seed) — варианты сравнивались с ней по разным годам. Исправление: фиксированное число деревьев без ранней
остановки (catboost.val_size=0, iterations=N); N из {100, 300} выбирается только по 2018–2020 — наибольший средний PF выбора
при условии, что база торгует во всех 9 годах на всех seed. Нейросеть (H64) сравнивается при равном числе сделок: порог
уверенности сети (nnconf) выбирается по 2018–2020 так, чтобы сделок было как у базы (±20%), и дальше не меняется.

    python scripts/skeptic_models.py --pick-n      # для очереди: напечатать выбранное N
    python scripts/skeptic_models.py               # сводка → docs/results/skeptic_models.md
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import metrics

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "skeptic_models.md"
WF = ROOT / "reports" / "walk_forward"
SEEDS = (42, 7, 1)
NS = (100, 300)
CONF = ("0.4", "0.45", "0.5", "0.55", "0.6")
PER = ("выбор", "проверка")


def runs_exist(pat: str) -> bool:
    return all((WF / f"{pat.format(sd)}_trades.csv").exists() for sd in SEEDS)


def stats(pat: str, inst: str = "lkoh") -> dict:
    out = {p: {"PF": [], "сделок": [], "итог": [], "до издержек": []} for p in PER}
    years_ok = True
    for sd in SEEDS:
        run = pat.format(sd)
        t = load_trades({"id": run, "sources": {inst: [run]}, "costs": {"profile": "exchange"}, "dividends": True})
        yrs = set(t["open_dt"].dt.year)
        years_ok &= all(y in yrs for y in range(2018, 2027))
        for p in PER:
            m = metrics(t, [inst], p)
            x = t[(t["open_dt"] >= pd.Timestamp("2021-01-01")) == (p == "проверка")]
            out[p]["PF"].append(m.get("PF", np.nan))
            out[p]["сделок"].append(m.get("сделок", 0))
            out[p]["итог"].append(m.get("итог", np.nan))
            out[p]["до издержек"].append(float(x["gross_pct"].mean()) if len(x) else np.nan)
    return {"years_ok": years_ok, **{p: {k: float(np.nanmean(v)) for k, v in d.items()} | {"PF по seed": d["PF"]} for p, d in out.items()}}


def pick_n() -> int:
    best, best_pf = None, -1.0
    for n in NS:
        pat = f"lkoh_sk_mc_n{n}_s{{}}"
        if not runs_exist(pat):
            continue
        s = stats(pat)
        pf = s["выбор"]["PF"]
        print(f"N={n}: PF выбора {pf:.3f}, торгует все годы: {s['years_ok']}", file=sys.stderr)
        if s["years_ok"] and pf > best_pf:
            best, best_pf = n, pf
    if best is None:                                   # ни одно N не торгует все годы — берём лучшее по выбору
        best = max(NS, key=lambda n: stats(f"lkoh_sk_mc_n{n}_s{{}}")["выбор"]["PF"] if runs_exist(f"lkoh_sk_mc_n{n}_s{{}}") else -1)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick-n", action="store_true")
    a = ap.parse_args(argv)
    if a.pick_n:
        print(pick_n())
        return 0
    n = pick_n()
    rows = []
    variants = {"база: 3 класса (L10, ранняя остановка)": "lkoh_h63_multiclass_s{}",
                f"база: 3 класса, {n} деревьев": f"lkoh_sk_mc_n{n}_s{{}}",
                f"H63 пара «вверх» / «вниз», {n} деревьев": f"lkoh_sk_bp_n{n}_s{{}}",
                f"H63 «ход × куда», {n} деревьев": f"lkoh_sk_md_n{n}_s{{}}",
                f"H41 + наклон тренда, {n} деревьев": f"lkoh_sk_sl_n{n}_s{{}}"}
    for name, pat in variants.items():
        if runs_exist(pat):
            rows.append({"вариант": name, **{f"{p}: {k}": v for p in PER for k, v in stats(pat)[p].items() if k != "PF по seed"},
                         "все 9 лет": stats(pat)["years_ok"]})
    # нейросеть: порог, при котором сделок на 2018–2020 как у базы с N деревьями (±20%), выбирается по выбору
    base = stats(f"lkoh_sk_mc_n{n}_s{{}}")
    nn_pick = None
    for c in CONF:
        pat = f"lkoh_h64_nn_s{{}}__nnconf_{c}"
        if not runs_exist(pat):
            continue
        s = stats(pat)
        rows.append({"вариант": f"H64 нейросеть, порог уверенности {c}", **{f"{p}: {k}": v for p in PER for k, v in s[p].items() if k != "PF по seed"},
                     "все 9 лет": s["years_ok"]})
        ratio = s["выбор"]["сделок"] / max(base["выбор"]["сделок"], 1)
        if 0.8 <= ratio <= 1.2 and (nn_pick is None or abs(ratio - 1) < nn_pick[1]):
            nn_pick = (c, abs(ratio - 1))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("N =", n, "; порог сети:", nn_pick, "\n", R.round(3).to_string())
    bpf = {p: base[p]["PF"] for p in PER}
    verdicts = {}
    for name, key in ((f"H63 пара «вверх» / «вниз», {n} деревьев", "H63 пара"), (f"H63 «ход × куда», {n} деревьев", "H63 ход × куда"),
                      (f"H41 + наклон тренда, {n} деревьев", "H41")):
        r = R[R["вариант"] == name]
        if len(r):
            verdicts[key] = bool(all(float(r[f"{p}: PF"].iloc[0]) > bpf[p] for p in PER))
    if nn_pick:
        r = R[R["вариант"] == f"H64 нейросеть, порог уверенности {nn_pick[0]}"]
        verdicts["H64 при равном числе сделок"] = bool(all(float(r[f"{p}: PF"].iloc[0]) > bpf[p] for p in PER))
    print(verdicts)
    md = ["# Скептик к модели: H63, H64, H41 на исправленном измерении", "",
          "Разбор и правило — `docs/results/skeptic_2026-11-08.md`. Скептик нашёл два изъяна прежнего сравнения: база L10 с ранней "
          "остановкой в 2021 и 2024 годах обучалась до 1–10 деревьев и не торговала совсем (варианты сравнивались с ней по разным "
          "годам), а у нейросети был выключен фильтр ожидаемой ценности (она торговала в 1.7–5 раз чаще). Здесь: фиксированное число "
          f"деревьев без ранней остановки — N = {n} (выбрано по 2018–2020 из 100 и 300 при условии, что база торгует все 9 лет); "
          "нейросеть — с порогом уверенности, при котором сделок на 2018–2020 столько же, сколько у базы (±20%). LKOH, гейт C, "
          "годовой walk-forward 2018–2026, seed 42, 7, 1, издержки биржи. Расчёт — `scripts/queues/skeptic_models.sh`, "
          "`scripts/skeptic_models.py`.", "",
          "**Правило (до расчёта):** вариант лучше базы, если PF выше в обоих периодах (среднее по seed); «да» по гипотезе — только "
          "если это повторится на SBER с теми же настройками.", "",
          "| вариант | выбор: сделок | выбор: PF | выбор: до издержек, % | проверка: сделок | проверка: PF | проверка: до издержек, % | все 9 лет |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['вариант']} | {r['выбор: сделок']:.0f} | {r['выбор: PF']:.2f} | {r['выбор: до издержек']:+.3f} | "
                  f"{r['проверка: сделок']:.0f} | {r['проверка: PF']:.2f} | {r['проверка: до издержек']:+.3f} | {'да' if r['все 9 лет'] else 'нет'} |")
    md += ["", f"Порог сети, выбранный по 2018–2020: {nn_pick[0] if nn_pick else 'нет подходящего (±20% сделок)'}.", "",
           "Лучше базы в обоих периодах: " + ", ".join(f"{k} — {'да' if v else 'нет'}" for k, v in verdicts.items()) + "."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
