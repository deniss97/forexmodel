"""Гипотеза H73 (Тимур, созвон 04.11): «три белых солдата» в итераторе параметров.

Фигура — по описанию Тимура в банке: три растущие свечи подряд, открытие каждой — внутри тела предыдущей, закрытие —
выше максимума предыдущей, тела примерно равны, верхние тени короткие; свечи крупнее обычной волатильности; работает
после продолжительного снижения. Зеркально — «три чёрные вороны» (шорт; только металлы — акции у нас в лонг).

Входы (9 = 3 варианта × 3 таймфрейма — 1 ч, 4 ч, дни; сигнал — на закрытии третьей свечи, вход — открытие следующего часа):
  * «строго, после снижения» — все правила + перед фигурой цена за 20 свечей упала ≥ 2 ATR;
  * «строго, где угодно» — все правила без условия места (Тимур: «в целом подобный паттерн вне зависимости от места»);
  * «мягко» — три растущие свечи, закрытие выше предыдущего, открытие внутри тела предыдущей; без правил размера и теней.
Правила «строго»: тела отличаются не больше чем вдвое, верхняя тень ≤ 50% тела, три тела вместе ≥ 1.5 ATR(14)
своего таймфрейма («больше волатильности»). Первая версия (в 1.5 раза, тень ≤ 25%, каждое тело ≥ 0.5 ATR) давала 0–6
фигур за 10 лет на инструмент — проверять нечего; узкие места — «тела равны» и «короткие тени».

Итератор — движок перебора H50/H54 (scripts/strategy_search.py, second_search.py): набор для запуска — серебро и золото
через CFD в обе стороны (издержки B01), GAZP, SBER, LKOH на бирже только в лонг; все 12 фильтров, все 36 выходов, ATR 14,
размер 1 и «по риску». Выбор — 2016–2020, проверка 2021 — июнь 2026. Для сравнения — те же пробы со случайными входами
той же частоты (на каждую пробу — вход в случайные часы, столько же событий по инструменту).

Ещё до перебора — событийная проверка: ход цены через 6 / 24 / 72 часа после фигуры в ATR против всех часов.

    python scripts/h73_soldiers.py            # → reports/h73/*.csv, docs/results/h73_soldiers.md
"""
from __future__ import annotations

import logging
import sys
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import second_search as s2
import strategy_search as ss

ROOT = ss.ROOT
OUT = ROOT / "reports" / "h73"
DOC = ROOT / "docs" / "results" / "h73_soldiers.md"
VARIANTS = ("строго, после снижения", "строго, где угодно", "мягко")
TFS = ("1ч", "4ч", "дни")
EQ, SHADOW, BODY_ATR, DECLINE_N, DECLINE_ATR = 2.0, 0.5, 1.5, 20, 2.0
HORIZONS = (6, 24, 72)


def name(variant: str, tf: str) -> str:
    return f"3 солдата {tf}, {variant}"


def bars(ins: ss.Inst, tf: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Свечи таймфрейма и индекс последнего часа каждой (на нём свеча закрыта — там сигнал)."""
    t = pd.Series(pd.to_datetime(ins.t))
    if tf == "1ч":
        g = np.arange(ins.n)
    elif tf == "4ч":
        g = pd.factorize(t.dt.floor("4h"))[0]
    else:
        g = pd.factorize(pd.Series(ss.day_key(t, ins.inst)))[0]
    df = pd.DataFrame({"g": g, "o": ins.o, "h": ins.h, "l": ins.l, "c": ins.c, "k": np.arange(ins.n)})
    b = df.groupby("g", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), last=("k", "max"))
    pc = b["c"].shift(1)
    tr = np.maximum(b["h"] - b["l"], np.maximum((b["h"] - pc).abs(), (b["l"] - pc).abs()))
    b["atr"] = tr.rolling(14).mean().shift(1)            # волатильность ДО фигуры
    return b.reset_index(drop=True), b["last"].to_numpy()


def pattern(b: pd.DataFrame, variant: str, sgn: int) -> np.ndarray:
    """Фигура на закрытии свечи j (j, j−1, j−2 — три свечи). sgn = −1 — зеркально (вороны)."""
    o, h, l, c = (b[k].to_numpy(float) for k in ("o", "h", "l", "c"))
    if sgn < 0:                                           # зеркало: вороны = солдаты на −цене
        o, c, h, l = -o, -c, -l, -h
    atr = b["atr"].to_numpy(float)
    body = c - o
    sh = lambda x, k: x if k == 0 else np.r_[np.full(k, np.nan), x[:-k]]  # noqa: E731
    with np.errstate(invalid="ignore"):
        up = (body > 0) & (sh(body, 1) > 0) & (sh(body, 2) > 0)
        # открытие внутри тела предыдущей (для свечей 2 и 3)
        inside = lambda k: (sh(o, k) >= sh(o, k + 1)) & (sh(o, k) <= sh(c, k + 1))  # noqa: E731
        m = up & inside(0) & inside(1)
        if variant == "мягко":
            return m & (c > sh(c, 1)) & (sh(c, 1) > sh(c, 2))
        m &= (c > sh(h, 1)) & (sh(c, 1) > sh(h, 2))       # закрытие выше максимума предыдущей
        b3 = np.vstack([body, sh(body, 1), sh(body, 2)])
        m &= np.nanmax(b3, axis=0) <= EQ * np.nanmin(b3, axis=0)
        upper = np.vstack([h - c, sh(h - c, 1), sh(h - c, 2)])
        m &= (upper <= SHADOW * b3).all(axis=0)
        m &= b3.sum(axis=0) >= BODY_ATR * atr               # три тела вместе — крупнее обычной волатильности
        if variant == "строго, после снижения":
            c0 = sh(c, 3)                                  # закрытие перед первой свечой
            m &= (c0 - sh(c, 3 + DECLINE_N)) <= -DECLINE_ATR * atr
    return np.nan_to_num(m).astype(bool)


def add_entries(ins: ss.Inst) -> None:
    for tf in TFS:
        b, last = bars(ins, tf)
        for v in VARIANTS:
            ev = np.zeros(ins.n, np.int8)
            ev[last[pattern(b, v, 1)]] = 1
            ev[last[pattern(b, v, -1)]] = -1
            ins.entries[name(v, tf)] = ev


def event_study(data: dict) -> pd.DataFrame:
    """Ход после фигуры (в ATR 14 часа сигнала) через 6 / 24 / 72 часа против всех часов, по периодам."""
    rows = []
    for inst, ins in data.items():
        a = np.where(ins.atr[14] > 0, ins.atr[14], np.nan)     # ATR = 0 (застывшая цена) давал ±inf в средних
        fwd = {hh: (np.r_[ins.c[hh:], np.full(hh, np.nan)] - ins.c) / a for hh in HORIZONS}
        sel = ins.year < ss.SEL_END
        for nm, ev in ins.entries.items():
            if not nm.startswith("3 солдата"):
                continue
            for side in (1, -1):
                if side < 0 and inst in s2.STOCKS:
                    continue
                for per, pm in (("выбор", sel & ins.ok), ("проверка", ~sel & ins.ok)):
                    m = (ev == side) & pm
                    r = {"инструмент": inst, "вход": nm, "сторона": "лонг" if side > 0 else "шорт", "период": per, "событий": int(m.sum())}
                    for hh in HORIZONS:
                        x = side * fwd[hh][m]
                        base = side * fwd[hh][pm]
                        r[f"ход {hh}ч"] = float(np.nanmean(x)) if m.any() else np.nan
                        r[f"все часы {hh}ч"] = float(np.nanmean(base))
                    rows.append(r)
    return pd.DataFrame(rows)


def shuffled(ins: ss.Inst, ev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Случайные входы той же частоты и стороны в разрешённые часы (тот же год — чтобы число событий по годам совпадало)."""
    out = np.zeros_like(ev)
    for y in np.unique(ins.year[ev != 0]):
        cand = np.flatnonzero((ins.year == y) & ins.ok)
        for side in (1, -1):
            k = int(((ev == side) & (ins.year == y)).sum())
            if k:
                out[rng.choice(cand, size=min(k, len(cand)), replace=False)] = side
    return out


def run() -> int:
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = s2.load_data()
    for ins in data.values():
        add_entries(ins)
    es = event_study(data)
    es.to_csv(OUT / "event_study.csv", index=False)
    print(f"событийная проверка: {time.time() - t0:.0f} с", flush=True)
    ref = s2.s42_monthly()
    entries = [name(v, tf) for tf in TFS for v in VARIANTS]
    rng = np.random.default_rng(73)
    # случайные входы: по одной копии на вход (тот же календарь событий по годам), имя «случайно …»
    for nm in entries:
        for ins in data.values():
            ins.entries["случайно · " + nm] = shuffled(ins, ins.entries[nm], rng)
    rows = []
    for nm in entries:
        for kind in ("фигура", "случайно"):
            ent = nm if kind == "фигура" else "случайно · " + nm
            for f1 in ss.FILTERS:
                for ex in ss.EXITS:
                    for size in ("1", "по риску"):
                        prm = {"entry": ent, "f1": f1, "f2": "нет", "exit": ex, "atr": 14, "size": size}
                        r = s2.evaluate(data, prm)
                        j = s2.judge(r, ref)
                        pf_s, net_s, n_s, _ = s2.part(r["years"], range(2016, 2021))
                        rows.append({"вход": nm, "что": kind, "фильтр": f1, "выход": ex, "размер": size,
                                     "PF выбор": min(pf_s, 99), "итог выбор": net_s, "сделок выбор": n_s,
                                     "PF проверка": j["test_pf"], "итог проверка": j["test_net"], "сделок проверка": j["test_n"],
                                     "связь с S42": j["corr"], "t портфеля выбор": j["pt_sel"], "t портфеля проверка": j["pt_test"]})
        print(f"{nm}: {len(rows)} проб, {time.time() - t0:.0f} с", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "trials.parquet", index=False)
    return 0


CRITERION = ("H73 подтвердится, если (1) лучшая по выбору 2016–2020 проба с входом «три белых солдата» (не меньше 50 сделок "
             "на выборе, связь с S42 ниже 0.3 — правило H54) на проверке даёт PF > 1.1 и не меньше 10 сделок в год на набор, и (2) "
             "фигура лучше случайных входов той же частоты: медианный PF проб с фигурой выше, чем у тех же проб со случайными "
             "входами, и на выборе, и на проверке.")


def f2(v, d=2):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def report() -> int:
    logging.disable(logging.WARNING)
    data = s2.load_data()                     # событийная проверка — секунды, пересчитываем при каждой сборке отчёта
    for ins in data.values():
        add_entries(ins)
    es = event_study(data)
    es.to_csv(OUT / "event_study.csv", index=False)
    T = pd.read_parquet(OUT / "trials.parquet")
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find("<!-- вывод -->"), old.find("<!-- /вывод -->")
    verdict = old[a:b + len("<!-- /вывод -->")] if a >= 0 and b > a else "<!-- вывод -->\n<!-- /вывод -->"
    md = ["# «Три белых солдата» в итераторе (H73)", "",
          "Гипотеза Тимура (банк, 31.10; в работу 04.11): фигура «три белых солдата» — разворот после снижения; «в итераторе "
          "параметров посмотреть, как зароялят». Фигура — по описанию в банке; зеркально — «три чёрные вороны» (шорт, только "
          "металлы). Набор для запуска: серебро и золото через CFD в обе стороны, GAZP, SBER, LKOH на бирже в лонг. Выбор "
          "2016–2020, проверка 2021 — июнь 2026. Расчёт — `scripts/h73_soldiers.py`.", "",
          "**Критерий (записан до расчёта).** " + CRITERION, "", verdict, "",
          "## Как определена фигура", "",
          "Три свечи одного таймфрейма (1 ч, 4 ч, день; сигнал — на закрытии третьей, вход — открытие следующего часа):", "",
          "* **строго** — все три растут; открытие каждой внутри тела предыдущей; закрытие выше максимума предыдущей; тела "
          f"отличаются не больше чем в {EQ:g} раза; верхняя тень ≤ {SHADOW:.0%} тела; три тела вместе ≥ {BODY_ATR:g} ATR(14) "
          "своего таймфрейма («крупнее обычной волатильности»);",
          f"* **строго, после снижения** — то же, и перед фигурой цена за {DECLINE_N} свечей упала не меньше чем на {DECLINE_ATR:g} ATR;",
          "* **мягко** — три растущие свечи, каждая закрывается выше предыдущей и открывается внутри её тела; без правил размера и теней.",
          "", "Первая версия «строго» — тела в 1.5 раза, тень ≤ 25%, каждое тело ≥ 0.5 ATR — давала 0–6 фигур за 10 лет на "
          "инструмент; сильнее всего отсекают «тела почти равны» и «короткие тени». Пороги ослаблены до расчёта результатов.", "",
          "## Сколько раз встречается и что бывает после", "",
          "Ход цены после фигуры в ATR(14) часа сигнала, в сторону сделки, среднее по событиям против среднего по всем часам "
          "(«все»). Лонг — пять рынков, шорт — металлы.", "",
          "| вход | сторона | период | событий | через 6 ч | все | через 24 ч | все | через 72 ч | все |", "|---|---|---|---|---|---|---|---|---|---|"]
    g = es.groupby(["вход", "сторона", "период"])
    for (nm, side, per), x in g:
        w = x["событий"].to_numpy(float)
        if w.sum() == 0:
            md.append(f"| {nm} | {side} | {per} | 0 | — | — | — | — | — | — |")
            continue
        avg = lambda c: float(np.nansum(x[c].fillna(0) * w) / w.sum())  # noqa: E731
        base = lambda c: float(x[c].mean())  # noqa: E731
        md.append(f"| {nm} | {side} | {per} | {int(w.sum())} | {avg('ход 6ч'):+.2f} | {base('все часы 6ч'):+.2f} | {avg('ход 24ч'):+.2f} | "
                  f"{base('все часы 24ч'):+.2f} | {avg('ход 72ч'):+.2f} | {base('все часы 72ч'):+.2f} |")
    md += ["", "## Итератор: фигура против случайных входов той же частоты", "",
           "На каждый вход — 12 фильтров × 36 выходов × 2 размера = 864 пробы; те же пробы со случайными входами (то же число "
           "событий по годам и сторонам). PF — набор, рынки поровну.", "",
           "| вход | медиана PF выбор: фигура | случайно | медиана PF проверка: фигура | случайно | доля проб с PF проверки > 1: фигура | случайно |",
           "|---|---|---|---|---|---|---|"]
    for nm, x in T.groupby("вход", sort=False):
        a, r = x[x["что"] == "фигура"], x[x["что"] == "случайно"]
        md.append(f"| {nm} | {f2(a['PF выбор'].median())} | {f2(r['PF выбор'].median())} | {f2(a['PF проверка'].median())} | "
                  f"{f2(r['PF проверка'].median())} | {(a['PF проверка'] > 1).mean():.0%} | {(r['PF проверка'] > 1).mean():.0%} |")
    md += ["", "## Лучшая по выбору проба каждого входа и её проверка", "",
           "Отбор как в H54: не меньше 50 сделок на выборе, связь месяцев с S42 ниже 0.3, наибольший PF выбора.", "",
           "| вход | фильтр | выход | размер | сделок выбор | PF выбор | сделок проверка | в год | PF проверка | итог проверка, % | t с S42 проверка |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    picks = []
    for nm, x in T[T["что"] == "фигура"].groupby("вход", sort=False):
        ok = x[(x["сделок выбор"] >= s2.MIN_SEL_TRADES) & (x["связь с S42"] < s2.MAX_CORR)]
        if not len(ok):
            md.append(f"| {nm} | — | — | — | {int(x['сделок выбор'].max())} (макс.) | — | — | — | — | — | — |")
            continue
        b = ok.sort_values("PF выбор", ascending=False).iloc[0]
        picks.append(b)
        md.append(f"| {nm} | {b['фильтр']} | {b['выход']} | {b['размер']} | {int(b['сделок выбор'])} | {f2(b['PF выбор'])} | "
                  f"{int(b['сделок проверка'])} | {b['сделок проверка'] / 5.5:.0f} | {f2(b['PF проверка'])} | {b['итог проверка']:+.1f} | "
                  f"{f2(b['t портфеля проверка'])} |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(picks).to_csv(OUT / "picks.csv", index=False)
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(report() if sys.argv[1:] == ["report"] else run())
