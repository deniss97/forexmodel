"""Набор для запуска (гипотезы H52 и H07, задача T15): отчёт docs/results/launch_set.md с графиками.

Сравнивает на одних и тех же пяти инструментах (серебро, золото, GAZP, SBER, LKOH):
  B01 — база (всё через CFD в обе стороны), S42 — набор для запуска (металлы через CFD, акции на бирже
  в лонг), S43 — то же при шаге цены серебра 0.0001, S44 — размер позиции по волатильности.
Метрики — тем же кодом, что журнал экспериментов (scripts/experiments.py).

    python scripts/launch_set.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import TEST_START, metrics  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, fmt, load_catalog, load_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "launch_set"
DOC = ROOT / "docs" / "results" / "launch_set.md"
SET = ["silver", "gold", "gazp", "sber", "lkoh"]
METALS, STOCKS = ["silver", "gold"], ["gazp", "sber", "lkoh"]
NAMES = {"silver": "серебро", "gold": "золото", "gazp": "GAZP", "sber": "SBER", "lkoh": "LKOH"}
RUNS = {"B01": "база B01 (всё через CFD)", "S42": "S42 набор для запуска", "S43": "S43 серебро с шагом 0.0001",
        "S44": "S44 размер по волатильности"}
COLORS = {"B01": GRAY, "S42": BLUE, "S43": "#8a5cc4", "S44": "#c8641e"}
KEYS = ("сделок", "итог", "PF", "без_2_лучших", "просадка", "t", "лет_в_плюсе")
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def weighted(t: pd.DataFrame, inst: list[str]) -> pd.DataFrame:
    x = t[t["instrument"].isin(inst)].sort_values("close_dt").copy()
    x["wp"] = x["profit_pct"] / len(inst)
    return x


def equity_png(tr: dict, path) -> None:
    f, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    for k in ("B01", "S43", "S44", "S42"):
        x = weighted(tr[k], SET)
        ax.plot(x["close_dt"], x["wp"].cumsum(), color=COLORS[k], linewidth=2.6 if k == "S42" else 1.6, label=RUNS[k])
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=BLUE, fontsize=9)
    style(ax, "накопленный итог, % (5 инструментов поровну)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    f.tight_layout()
    f.savefig(path, dpi=110, facecolor=BG)
    plt.close(f)


def rolling_pf(t: pd.DataFrame, inst: list[str], months: int = 12) -> pd.Series:
    """PF сделок, закрытых за последние `months` месяцев, на конец каждого месяца."""
    x = weighted(t, inst).set_index("close_dt")
    ends = pd.date_range(x.index.min() + pd.DateOffset(months=months), x.index.max(), freq="ME")
    out = {}
    for e in ends:
        w = x.loc[e - pd.DateOffset(months=months):e, "wp"]
        loss = -w[w < 0].sum()
        out[e] = w[w > 0].sum() / loss if loss > 0 else np.nan
    return pd.Series(out)


def rolling_png(tr: dict, path) -> None:
    f, ax = plt.subplots(figsize=(9.5, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    for k, inst, label, color in (("S42", SET, "набор целиком", BLUE), ("S42", METALS, "металлы", "#8a5cc4"),
                                  ("S42", STOCKS, "акции в лонг", "#1d7a55")):
        s = rolling_pf(tr[k], inst)
        ax.plot(s.index, s.clip(upper=4), color=color, linewidth=2.4 if inst is SET else 1.4, label=label)
    ax.axhline(1, color=INK2, linewidth=1)
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    style(ax, "PF за последние 12 месяцев")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    f.tight_layout()
    f.savefig(path, dpi=110, facecolor=BG)
    plt.close(f)


def style(ax, ylabel: str) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def row(name: str, m: dict) -> str:
    return f"| {name} | " + " | ".join(fmt(m.get(k), k) for k in KEYS) + " |"


def main() -> int:
    _, exps = load_catalog()
    tr = {k: load_trades(exps[k]) for k in RUNS}
    OUT.mkdir(parents=True, exist_ok=True)
    equity_png(tr, OUT / "equity.png")
    rolling_png(tr, OUT / "rolling_pf.png")

    # вывод пишется руками между маркерами и переживает пересборку
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    head = "| | сделок | итог % | PF | без 2 лучших | просадка | t | лет в плюсе |\n|---|---|---|---|---|---|---|---|"
    md = ["# Набор для запуска: металлы через CFD, акции на бирже в лонг", "",
          "Гипотезы **H52** (набор как единый портфель) и **H07** (размер позиции по волатильности), задача **T15** — "
          "последний проход по импульсу перед запуском. Сделки — штатные прогоны журнала, метрики — тем же кодом, что "
          "журнал экспериментов: пять инструментов поровну, проверка — сделки, открытые с 2021 года. Расчёт — "
          "`scripts/launch_set.py`, эксперименты S42–S44 в журнале.", "",
          "![Накопленный итог](launch_set/equity.png)", "", verdict, ""]
    for period, title in (("проверка", "Проверка 2021 — июнь 2026"), ("выбор", "Выбор 2016–2020")):
        md += [f"## {title}", "", head]
        md += [row(RUNS[k], metrics(tr[k], SET, period)) for k in RUNS]
        md += [row("S42 — только металлы", metrics(tr["S42"], METALS, period)),
               row("S42 — только акции в лонг", metrics(tr["S42"], STOCKS, period)), ""]

    md += ["## По инструментам — S42, проверка", "", head]
    md += [row(NAMES[i], metrics(tr["S42"], [i], "проверка")) for i in SET]
    md.append("")

    years = {}
    for k in RUNS:
        x = weighted(tr[k], SET)
        years[k] = x.groupby(x["open_dt"].dt.year)["wp"].sum()
    ys = sorted(set().union(*[set(v.index) for v in years.values()]))
    md += ["## По годам — итог %, пять инструментов поровну", "",
           "| год | " + " | ".join(RUNS[k].split(" ")[0] for k in RUNS) + " |", "|---|" + "---|" * len(RUNS)]
    md += [f"| {y} | " + " | ".join(f"{years[k].get(y, 0):+.1f}" for k in RUNS) + " |" for y in ys]
    md.append("")

    s = tr["S44"]
    md += ["## Размер позиции по волатильности (S44)", "",
           "Средний размер сделки: " + ", ".join(
               f"{NAMES[i]} — выбор {s[(s.instrument == i) & (s.open_dt < TEST_START)]['size'].mean():.2f}, "
               f"проверка {s[(s.instrument == i) & (s.open_dt >= TEST_START)]['size'].mean():.2f}" for i in SET) + ".", "",
           "## PF за скользящие 12 месяцев — S42", "", "![Скользящий PF](launch_set/rolling_pf.png)", ""]
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
