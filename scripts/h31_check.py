"""Проверка H31 (19.10): действительно ли обучение CatBoost только внутри тренд-гейта лучше базы — со всеми издержками B01.

Сделки — те же прогоны, что в журнале: M01 (база, 3 seed) и M02 (обучение внутри гейта, 4 seed) на серебре,
L07 / L08 на LKOH (1 seed). Прогоны 27.09 считались до двух поправок B01, поэтому три уровня издержек:

  1. «как в банке 09.10» — только комиссия 0.04% (результат из файлов прогона);
  2. «журнал сейчас» — пересчёт по профилю base: комиссия 0.04%, проскальзывание 0.05% на сторону, свопы;
  3. «все издержки B01» — плюс стоп по гэпу: если минута открылась уже за стопом, выход по её open (как
     simulation.stop_fill: gap, с 04.10), у LKOH ещё дивиденды — лонг через отсечку получает дивиденд,
     шорт его платит (как скорректированные цены B01).

Метрики — тем же кодом, что журнал; разница M02 − M01 — по месяцам (среднее по seed), t по месяцам.

    python scripts/h31_check.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from experiments import ROOT, WF, fmt
from forexmodel.data.dividends import load_dividends
from forexmodel.evaluation.experiments import PERIODS, metrics, standardize
from forexmodel.simulation.costs import profile_entry, recost

DOC = ROOT / "docs" / "results" / "h31_check.md"
RUNS = {"M01": ("silver", {42: "silver_base", 7: "silver_base_s7", 1: "silver_base_s1"}),
        "M02": ("silver", {42: "silver_base_gateonly", 7: "silver_base_gateonly_s7", 1: "silver_base_gateonly_s1",
                           2: "silver_base_gateonly_s2"}),
        "L07": ("lkoh", {42: "lkoh_gate"}), "L08": ("lkoh", {42: "lkoh_gateonly"})}
MINUTES = {"silver": "XAGUSD_1m_full_1m.pkl", "lkoh": "LKOH_1m_15-26_1m.pkl"}
LEVELS = ("как в банке 09.10", "журнал сейчас", "все издержки B01")
STOPS = ("stop_loss", "trailing_stop", "early_stop")
KEYS = ("сделок", "итог", "PF", "без_2_лучших", "t")
PER = ("выбор", "проверка")


def gap_fill(t: pd.DataFrame, opens: pd.Series) -> tuple[pd.DataFrame, int]:
    """Стоп по гэпу: минута выхода открылась за уровнем стопа — исполнение по её open."""
    t = t.copy()
    stop = t["exit_reason"].isin(STOPS).to_numpy()
    o = opens.reindex(t.loc[stop, "close_dt"]).to_numpy(float)
    d = np.where(t.loc[stop, "side"].to_numpy() == "buy", 1.0, -1.0)
    ex = t.loc[stop, "exit_price"].to_numpy(float)
    worse = np.isfinite(o) & ((o - ex) * d < 0)
    t.loc[stop, "exit_price"] = np.where(worse, o, ex)
    return t, int(worse.sum())


def dividends(t: pd.DataFrame, divs: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Поправка на дивиденды, % цены входа: лонг через отсечку получает дивиденд, шорт платит."""
    adj = np.zeros(len(t))
    for ex, amount in zip(divs["ex_time"], divs["amount"]):
        on = ((t["open_dt"] < ex) & (t["close_dt"] >= ex)).to_numpy()
        d = np.where(t["side"].to_numpy() == "buy", 1.0, -1.0)
        adj += np.where(on, d * amount / t["open_price"].to_numpy(float) * 100, 0.0)
    return adj, int((adj != 0).sum())


def load(exp: str) -> tuple[dict[str, pd.DataFrame], dict]:
    inst, runs = RUNS[exp]
    prof = profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst)
    m = pd.read_pickle(ROOT / "reports" / "_cache" / MINUTES[inst])
    opens = pd.Series(m["open"].to_numpy(float), index=m["time"])
    del m
    divs = load_dividends(ROOT / "data" / "dividends" / "moex_dividends.csv", inst) if inst == "lkoh" else None
    out, info = {lv: [] for lv in LEVELS}, {"гэп": 0, "стопов": 0, "дивиденд": 0, "сделок": 0, "Δгэп": 0.0}
    for seed, run in runs.items():
        raw = pd.read_csv(WF / f"{run}_trades.csv", parse_dates=["open_dt", "close_dt"])
        raw["size"] = 1.0
        d = np.where(raw["side"] == "buy", 1.0, -1.0)
        # проверка: результат до издержек в файле — ровно ход цены от входа до выхода
        assert np.allclose(raw["gross_pct"], d * (raw["exit_price"] / raw["open_price"] - 1) * 100, atol=1e-6), run
        out[LEVELS[0]].append(standardize(raw, inst, str(seed)))
        raw["gross_unit_pct"] = raw["gross_pct"]
        out[LEVELS[1]].append(standardize(recost(raw, prof), inst, str(seed)))
        g, n = gap_fill(raw, opens)
        g["gross_unit_pct"] = d * (g["exit_price"] / g["open_price"] - 1) * 100
        info["Δгэп"] += float((g["gross_unit_pct"] - raw["gross_pct"]).sum())
        if divs is not None:
            adj, k = dividends(g, divs)
            g["gross_unit_pct"] += adj
            info["дивиденд"] += k
        out[LEVELS[2]].append(standardize(recost(g, prof), inst, str(seed)))
        info["гэп"] += n
        info["стопов"] += int(raw["exit_reason"].isin(STOPS).sum())
        info["сделок"] += len(raw)
    return {lv: pd.concat(v, ignore_index=True) for lv, v in out.items()}, info


def monthly(t: pd.DataFrame) -> pd.Series:
    """Результат по месяцам закрытия, среднее по seed (как журнал: прогоны поровну)."""
    runs = t["run"].nunique()
    return t.groupby(t["close_dt"].dt.to_period("M"))["profit_pct"].sum() / runs


def diff_t(a: pd.DataFrame, b: pd.DataFrame, per: str) -> tuple[float, float]:
    """Разница b − a по месяцам периода: сумма, % и t."""
    lo, hi = PERIODS[per]
    sel = lambda x: x[((x["open_dt"] >= lo) if lo is not None else True) & ((x["open_dt"] < hi) if hi is not None else True)]  # noqa: E731
    j = pd.concat([monthly(sel(a)).rename("a"), monthly(sel(b)).rename("b")], axis=1).fillna(0.0)
    d = j["b"] - j["a"]
    return float(d.sum()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main() -> int:
    data, info = {}, {}
    for exp in RUNS:
        data[exp], info[exp] = load(exp)
        print(exp, info[exp], flush=True)

    L = ["# Проверка H31: обучение CatBoost только внутри тренд-гейта (19.10)", "",
         "Вопрос Дениса: действительно ли H31 подтвердилась и учтены ли все издержки, которые есть в базе B01. "
         "Сделки — те же прогоны, что в журнале: M01 (база, 3 seed) и M02 (обучение внутри гейта, 4 seed) на серебре, "
         "для справки L07 / L08 на LKOH (1 seed). Расчёт — `scripts/h31_check.py`.", "",
         "<!-- вывод -->", "<!-- /вывод -->", "",
         "## Какие издержки были учтены", "",
         "Прогоны M01 и M02 считались 27.09 — до двух поправок, которые теперь входят в базу B01 (11.10):", "",
         "| издержка B01 | в файлах прогона 27.09 | в журнале сейчас | здесь, уровень 3 |", "|---|---|---|---|",
         "| комиссия 0.04% | да | да | да |",
         "| проскальзывание 0.05% на сторону | нет | да (пересчёт по профилю base) | да |",
         "| свопы за перенос | нет | да | да |",
         "| стоп по гэпу (минута открылась за стопом — выход по open) | нет: стоп по уровню (добавлен 04.10) | **нет** | да, по минуткам |",
         "| дивиденды (LKOH) | нет: цены без поправки | **нет** | да: лонг через отсечку получает дивиденд, шорт платит |", "",
         "Цифры в банке («t 1.8 против 1.2, итог +84% против +55%») — из журнала 09.10, где ещё была только комиссия.", ""]
    for exp in RUNS:
        i = info[exp]
        L.append(f"* {exp}: сделок {i['сделок']} (все seed), выходов по стопу {i['стопов']}, из них с гэпом {i['гэп']} "
                 f"(сумма поправки {i['Δгэп']:+.1f}% по всем seed)" + (f", сделок через дивидендную отсечку {i['дивиденд']}" if exp[0] == "L" else ""))
    L.append("")

    L += ["## Серебро: M02 против M01 — все seed вместе (как журнал)", "",
          "| издержки | эксперимент | " + " | ".join(f"{p}: {k}" for p in PER for k in ("сделок", "итог %", "PF", "без 2 лучших", "t")) + " | разница M02 − M01 по месяцам: выбор (t) | проверка (t) |",
          "|" + "---|" * (2 + 2 * len(KEYS) + 2)]
    for lv in LEVELS:
        for exp in ("M01", "M02"):
            row = [fmt(metrics(data[exp][lv], ["silver"], p).get(k), k) for p in PER for k in KEYS]
            tail = ["", ""]
            if exp == "M02":
                tail = [f"{s:+.1f} ({tv:.2f})" for s, tv in (diff_t(data["M01"][lv], data["M02"][lv], p) for p in PER)]
            L.append(f"| {lv} | {exp} | " + " | ".join(row) + " | " + " | ".join(tail) + " |")
    L += ["", "## Серебро: по seed (проверка 2021–2026)", "",
          "| издержки | seed | M01: сделок | итог % | PF | t | M02: сделок | итог % | PF | t | M02 лучше по PF и t |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    wins = {}
    for lv in LEVELS:
        w = 0
        for seed in ("42", "7", "1", "2"):
            a = data["M01"][lv]
            b = data["M02"][lv]
            ma = metrics(a[a["run"] == seed], ["silver"], "проверка") if seed in set(a["run"]) else {}
            mb = metrics(b[b["run"] == seed], ["silver"], "проверка")
            better = "—"
            if ma:
                ok = (mb["PF"] > ma["PF"]) and (mb["t"] > ma["t"])
                w += ok
                better = "да" if ok else "нет"
            L.append(f"| {lv} | {seed} | " + " | ".join(fmt(ma.get(k), k) if ma else "—" for k in ("сделок", "итог", "PF", "t")) + " | "
                     + " | ".join(fmt(mb.get(k), k) for k in ("сделок", "итог", "PF", "t")) + f" | {better} |")
        wins[lv] = w
    L.append("")
    L += ["## Серебро: по годам, все издержки B01 (итог %, среднее по seed)", "",
          "| год | M01 | M02 | разница |", "|---|---|---|---|"]
    lv = LEVELS[2]
    ya = data["M01"][lv].groupby(data["M01"][lv]["open_dt"].dt.year)["profit_pct"].sum() / 3
    yb = data["M02"][lv].groupby(data["M02"][lv]["open_dt"].dt.year)["profit_pct"].sum() / 4
    for y in sorted(set(ya.index) | set(yb.index)):
        L.append(f"| {y} | {ya.get(y, 0):+.1f} | {yb.get(y, 0):+.1f} | {yb.get(y, 0) - ya.get(y, 0):+.1f} |")
    L.append("")
    L += ["## LKOH: L08 против L07 (1 seed)", "",
          "| издержки | эксперимент | " + " | ".join(f"{p}: {k}" for p in PER for k in ("сделок", "итог %", "PF", "без 2 лучших", "t")) + " |",
          "|" + "---|" * (2 + 2 * len(KEYS))]
    for lv in LEVELS:
        for exp in ("L07", "L08"):
            L.append(f"| {lv} | {exp} | " + " | ".join(fmt(metrics(data[exp][lv], ["lkoh"], p).get(k), k) for p in PER for k in KEYS) + " |")
    L += ["", "Совпадение прогонов: сделки M09 (размер позиции) и M13 (варианты тренд-гейта) на seed 42 и 7 совпадают с M02 "
          "на 100% — после 27.09 обучение внутри гейта включено в `configs/silver.yaml`, и все последующие «базы» модели "
          "серебра — это уже M02. Разброс «M01 +55% против M09 +86%», которым в журнале объясняли сомнения в H31, — "
          "это и есть разница M01 и M02, а не шум между базами.", ""]

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find("<!-- вывод -->"), old.find("<!-- /вывод -->")
    text = "\n".join(L) + "\n"
    if a >= 0 and b > a:
        text = text.replace("<!-- вывод -->\n<!-- /вывод -->", old[a:b + len("<!-- /вывод -->")])
    DOC.write_text(text, encoding="utf-8")
    print("seed, где M02 лучше по PF и t на проверке:", wins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
