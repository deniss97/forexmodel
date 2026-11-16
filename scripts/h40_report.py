"""H40: отчёт docs/results/h40_trades.md из посчитанных таблиц (reports/h40_trades/*.csv).

Собирает из готовых файлов: группы «сделка под уровнем» против остальных (по стратегиям и половинам истории),
проверку критерия плана, сетку устойчивости, опору снизу, остановку цены у уровня, перестановочный тест и сверку
«лента против минуток». Текст между <!-- вывод --> и <!-- /вывод --> сохраняется.

    python scripts/h40_report.py min_main
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from h40_trades import STRATS, criterion, groups  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "h40_trades"
DOC = ROOT / "docs" / "results" / "h40_trades.md"
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
HEAD = """# Сделки под большим уровнем объёма (H40, план 14.11)

Проверка из банка гипотез: не «уровень как сигнал» (это было 08.11, [h40_profile.md](h40_profile.md)), а разбор сделок
журнала — **сделки наших стратегий, открытые прямо под большим свежим уровнем объёма, идут хуже остальных**, потому что
цена упирается в уровень. План подтверждён в банке 14.11, считалось по нему и по ответам автора на вопросы.

**Сделки** (журнал экспериментов, издержки биржи, дивиденды учтены, цены переведены в масштаб биржевых свечей):
S42 (импульс) — S42 на SBER, GAZP, LKOH и S51 на 16 акциях индекса; S46 (пробой 55 дней) — S46 и S53; S48 (пробой
20 дней) — S48 и S55; сделки модели по LKOH (L10, CatBoost на 3 класса, seed 42) в обе стороны — по ответу автора,
«не ограничиваться лонгами стратегий». Период — входы 2022–2026 (половины 2022–2024 и 2025–2026, как в критерии).

**Уровни.** Профиль объёма по ценам с шагом 0.1% (уровень = floor(log(цена) / log(1.001))). Основной источник — минутки
(объём минуты попадает в уровень её средней цены, `scripts/h40_levels_min.py`): они есть на всю историю сделок.
Лента обезличенных сделок (ALGOPACK с Диска, `scripts/h40_levels.py`) выгружается часами, поэтому на ней сделана
сверка: совпадают ли уровни двух источников на днях, где есть оба (раздел «Лента против минуток»).

**Большой уровень** — локальный максимум сглаженного профиля, где объём в X раз выше среднего по соседним ±2% цены
(X = 2 в основном расчёте, 1.5 и 3 — в сетке устойчивости). Профиль — по прошлым N торговым дням (N = 20; 10 и 40 в
сетке) и только по часам **строго до часа входа**: объём часа, в котором открыта сделка, в расчёт не идёт (иначе
собственный объём пробоя рисует «уровень» ровно на цене входа).

**Свежесть** — правило автора: уровень живёт, пока цена не вернулась к нему с расстояния k·ATR; после возврата объём на
уровне копится заново (k = 1, 2, 3 в сетке; 0 — весь объём окна).

**Группы.** Для каждой сделки — расстояние от входа до ближайшего большого уровня по ходу сделки (для лонга сверху, для
шорта снизу) в ATR(14 ч). «Уровень ближе 1 ATR» против «остальных» (у остальных уровень дальше или его нет).

**Критерий плана:** разница PF между группами одного знака в 2022–2024 и в 2025–2026 и хотя бы у двух стратегий из трёх;
в меньшей группе не меньше 30 сделок.

Расчёт: `scripts/h40_trades.py`, `scripts/h40_levels_min.py`, `scripts/h40_levels.py`, `scripts/h40_sig.py`,
`scripts/h40_stall.py`, `scripts/h40_cmp.py`, отчёт — `scripts/h40_report.py`.
"""


def md_table(d: pd.DataFrame, cols: dict[str, str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols)
    lines = [head]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append("да" if v else "нет")
            elif isinstance(v, float) and not np.isfinite(v):
                cells.append("—")
            elif c in fmt:
                cells.append(format(v, fmt[c]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section_groups(gr: pd.DataFrame, thr: float) -> str:
    near = f"уровень ближе {thr:g} ATR"
    order = [*STRATS, "модель LKOH", "все"]
    rows = []
    for s in order:
        for h in ("2022–2024", "2025–2026", "2022–2026"):
            sel = gr[(gr["стратегия"] == s) & (gr["период"] == h)]
            a = sel[sel["группа"] == near]
            b = sel[sel["группа"] == "остальные"]
            if not len(a) or not len(b):
                continue
            rows.append({"стратегия": s, "период": h,
                         "n_рядом": int(a["сделок"].iloc[0]), "PF_рядом": float(a["PF"].iloc[0]),
                         "ср_рядом": float(a["средняя"].iloc[0]), "выигр_рядом": float(a["выигрышей"].iloc[0]),
                         "n_ост": int(b["сделок"].iloc[0]), "PF_ост": float(b["PF"].iloc[0]),
                         "ср_ост": float(b["средняя"].iloc[0]), "выигр_ост": float(b["выигрышей"].iloc[0]),
                         "без_2_лучших_рядом": float(a["без_2_лучших"].iloc[0]),
                         "без_2_лучших_ост": float(b["без_2_лучших"].iloc[0])})
    t = pd.DataFrame(rows)
    cols = {"стратегия": "стратегия", "период": "период", "n_рядом": "сделок у уровня", "PF_рядом": "PF у уровня",
            "ср_рядом": "средняя у уровня, %", "выигр_рядом": "выигрышей у уровня, %", "n_ост": "сделок остальных",
            "PF_ост": "PF остальных", "ср_ост": "средняя остальных, %", "выигр_ост": "выигрышей остальных, %"}
    f = {"PF_рядом": ".2f", "PF_ост": ".2f", "ср_рядом": "+.2f", "ср_ост": "+.2f", "выигр_рядом": ".0f",
         "выигр_ост": ".0f"}
    return md_table(t, cols, f)


def main(argv: list[str]) -> int:
    tag = argv[0] if argv else "min_main"
    thr = 1.0
    gr = pd.read_csv(REP / f"groups_{tag}.csv")
    f = pd.read_csv(REP / f"trades_{tag}.csv", parse_dates=["open_dt"])
    ok, why, tbl = criterion(gr, thr)
    parts = [HEAD, MARK[0], "\n## Вывод\n\n(заполняется вручную)\n", MARK[1]]
    if DOC.exists():
        old = DOC.read_text(encoding="utf-8")
        if MARK[0] in old and MARK[1] in old:
            parts[1:4] = [old.split(MARK[0])[1].split(MARK[1])[0].join(MARK)]

    parts.append(f"\nСделок в разборе: {len(f)} ({f['open_dt'].min():%Y-%m-%d} — {f['open_dt'].max():%Y-%m-%d}), "
                 f"бумаг {f['tk'].nunique()}; большой уровень по ходу сделки ближе 1 ATR — у "
                 f"{100 * float((f['d_block'] <= thr).mean()):.0f}% сделок, уровня в окне нет вовсе — у "
                 f"{100 * float(f['d_block'].isna().mean()):.0f}%.\n")
    parts.append("## Сделки у уровня против остальных\n\n" + section_groups(gr, thr) + "\n")
    parts.append(f"\n![PF по группам](h40_trades/groups_{tag}.png)\n")
    parts.append("\n## Критерий плана\n\n" + md_table(
        tbl, {"стратегия": "стратегия", "PF_рядом_2022–2024": "PF рядом 22–24", "PF_ост_2022–2024": "PF ост. 22–24",
              "PF_рядом_2025–2026": "PF рядом 25–26", "PF_ост_2025–2026": "PF ост. 25–26",
              "n_рядом_2022–2024": "n рядом 22–24", "n_рядом_2025–2026": "n рядом 25–26",
              "знак_один": "разница одного знака", "сделок_хватает": "сделок ≥ 30", "прошла": "прошла"},
        {"PF_рядом_2022–2024": ".2f", "PF_ост_2022–2024": ".2f", "PF_рядом_2025–2026": ".2f",
         "PF_ост_2025–2026": ".2f"}) + f"\n\n**{why}; критерий {'выполнен' if ok else 'не выполнен'}.**\n")

    g = REP / f"grid_{tag}.csv"
    if g.exists():
        d = pd.read_csv(g)
        parts.append("\n## Устойчивость к настройкам\n\n" + md_table(
            d, {"X": "X (во сколько раз выше соседей)", "окно_дней": "окно, дней", "свежесть_ATR": "свежесть, ATR",
                "порог_ATR": "группа «рядом», ATR", "сделок_рядом": "сделок рядом", "PF_рядом": "PF рядом",
                "PF_остальные": "PF остальных", "критерий": "критерий"},
            {"PF_рядом": ".2f", "PF_остальные": ".2f", "X": ".1f", "свежесть_ATR": ".0f", "порог_ATR": ".1f"}) + "\n")

    supp = groups(f, "d_supp", thr)
    supp.to_csv(REP / f"supp_{tag}.csv", index=False)
    parts.append("\n## Опора за спиной (шаг 5 плана): большой уровень с другой стороны входа\n\n"
                 + section_groups(supp, thr) + "\n")

    st = REP / f"stall_{tag}.csv"
    if st.exists():
        d = pd.read_csv(st)
        parts.append("\n## Останавливается ли цена у уровня\n\nДоля сделок, где лучший ход (MFE) совпал с уровнем "
                     "(±0.2 ATR), и то же для «пустышки» — точки на 0.5 и 1 ATR дальше уровня, где уровня нет.\n\n"
                     + md_table(d, {c: c for c in d.columns}) + "\n")

    sg = REP / f"sig_{tag}.csv"
    if sg.exists():
        d = pd.read_csv(sg)
        parts.append("\n## Случайна ли разница (перестановки внутри года и бумаги)\n\n"
                     + md_table(d, {c: c for c in d.columns}) + "\n")

    names = {f"cmp_{tag}_tape_both.csv": "19 акций, лента ALGOPACK с Диска (выгружено на момент расчёта)",
             f"cmp_{tag}_lkoh_both.csv": "LKOH, секундные агрегаты ленты (data/raw/orderflow_lkoh_1s.parquet)"}
    cm = [(lab, REP / f) for f, lab in names.items() if (REP / f).exists()]
    if cm:
        parts.append("\n## Лента против минуток\n\nПлан называл источником уровней ленту обезличенных сделок. Профиль по "
                     "минуткам даёт те же уровни: признак «большой уровень по ходу ближе 1 ATR» совпадает почти всегда.\n")
        for lab, f in cm:
            d = pd.read_csv(f, header=None, names=["что", "значение"])
            parts.append(f"\n**{lab}**\n\n" + md_table(d, {"что": "что", "значение": "значение"}) + "\n")

    lk = sorted(REP.glob("groups_lkoh_*.csv"))
    if lk:
        rows = []
        for f in lk:
            side = {"both": "все сделки", "buy": "только покупки", "sell": "только продажи"}[f.stem.split("_")[-1]]
            g = pd.read_csv(f)
            g = g[(g["стратегия"] == "все") & (g["период"] == "2025–2026")]
            a = g[g["группа"] == "уровень ближе 1 ATR"]
            b = g[g["группа"] == "остальные"]
            rows.append({"объём уровня": side, "сделок рядом": int(a["сделок"].iloc[0]), "PF рядом": float(a["PF"].iloc[0]),
                         "сделок остальных": int(b["сделок"].iloc[0]), "PF остальных": float(b["PF"].iloc[0])})
        parts.append("\n## Покупки и продажи по уровням отдельно (вопрос автора), лента LKOH 2025–2026\n\n"
                     + md_table(pd.DataFrame(rows), {c: c for c in rows[0]},
                                {"PF рядом": ".2f", "PF остальных": ".2f"}) + "\n")

    parts.append(f"\n## Графики\n\n![Расстояние до уровня и результат сделки](h40_trades/deciles_{tag}.png)\n"
                 f"\n![По годам](h40_trades/years_{tag}.png)\n"
                 f"\n![Примеры сделок](h40_trades/examples_{tag}.png)\n\n"
                 "Все ситуации входов 2025–2026 с профилем объёма на момент входа (замечание автора к плану) — "
                 f"`docs/results/h40_trades/entries_<бумага>_{tag}_*.png`.\n")
    DOC.write_text("\n".join(parts), encoding="utf-8")
    print(f"готово: {DOC}, критерий {'да' if ok else 'нет'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
