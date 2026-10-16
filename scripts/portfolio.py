"""Портфель «набор для запуска + самостоятельная стратегия» (гипотеза H53, задачи T18–T20, T22).

Кандидаты — самостоятельные стратегии журнала на тех же пяти инструментах, что набор для запуска S42:
металлы через CFD в обе стороны, акции GAZP / SBER / LKOH на бирже только в лонг (шорты кандидатов по
акциям убираются из готовых сделок), издержки — профиль exchange. Капитал поровну между стратегиями.
Отчёт — docs/results/portfolio.md с графиками; вывод между маркерами пишется руками и переживает
пересборку.

    python scripts/portfolio.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import TEST_START  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, load_catalog, load_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "portfolio"
DOC = ROOT / "docs" / "results" / "portfolio.md"
SET = ["silver", "gold", "gazp", "sber", "lkoh"]
STOCKS = ["gazp", "sber", "lkoh"]
BASE = "S42"
CANDIDATES = ["S31", "S30", "S32", "S29", "S36", "S37", "S38", "S39", "S40", "S41"]
MIN_PER_YEAR = 10  # T19: меньше 10 сделок в год на пять инструментов — кандидат не рассматривается
PF_MIN = 1.1
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def candidate_trades(exp: dict) -> pd.DataFrame:
    """Сделки кандидата на наборе для запуска: пять инструментов, издержки exchange, акции только в лонг."""
    e = {**exp, "sources": {i: exp["sources"][i] for i in SET}, "costs": {"profile": "exchange"}, "long_only": STOCKS}
    return load_trades(e)


def f(v, k: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if k in ("сделок", "лет_в_плюсе", "макс_позиций"):
        return str(v)
    if k in ("PF", "t_мес", "итог_к_просадке", "корр_мес"):
        return f"{v:.2f}"
    if k in ("совпадение", "в_ту_же_сторону"):
        return f"{v:.0%}"
    if k in ("в_год", "ср_позиций"):
        return f"{v:.1f}"
    return f"{v:+.1f}"


def equity_png(curves: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [GRAY, BLUE, "#c8641e", "#1d7a55", "#8a5cc4"]
    for (label, x), c in zip(curves.items(), colors):
        x = x.sort_values("close_dt")
        ax.plot(x["close_dt"], (x["w"] * x["profit_pct"]).cumsum(), color=c, linewidth=2.4 if c == BLUE else 1.6, label=label)
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=BLUE, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, % капитала", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def main() -> int:
    _, exps = load_catalog()
    base = load_trades(exps[BASE])
    alone = pf.combine({BASE: base}, SET)
    rows, port, passed = [], {}, []
    for cid in CANDIDATES:
        c = candidate_trades(exps[cid])
        solo = pf.combine({cid: c}, SET)
        both = pf.combine({BASE: base, cid: c}, SET)
        port[cid] = both
        r = {"id": cid, "title": exps[cid]["title"]}
        for per in ("выбор", "проверка"):
            r[per] = {"solo": pf.metrics(solo, per), "port": pf.metrics(both, per), "ov": pf.overlap(solo, alone, per)}
        per_year = pf.metrics(solo, "всё").get("в_год", 0)
        r["в_год"] = per_year
        ok_solo = all(r[p]["solo"].get("PF", 0) > PF_MIN for p in ("выбор", "проверка")) and per_year >= MIN_PER_YEAR
        ok_port = all(r[p]["port"].get("t_мес", -9) > pf.metrics(alone, p)["t_мес"] and
                      r[p]["port"].get("итог_к_просадке", -9) > pf.metrics(alone, p)["итог_к_просадке"] for p in ("выбор", "проверка"))
        r["ok_solo"], r["ok_port"] = ok_solo, ok_port
        if ok_solo and ok_port:
            passed.append(cid)
        rows.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    best = max(rows, key=lambda r: r["проверка"]["port"].get("t_мес", -9))["id"]
    # сочетания: лучший по t на проверке, прошедшие отбор и все они вместе
    combos = [[best]] + [[c] for c in passed if c != best]
    if len(combos) > 1:
        combos.append(sorted({c for cs in combos for c in cs}, key=CANDIDATES.index))
    curves = {f"{BASE} один": alone}
    for cs in combos:
        curves[f"{BASE} + " + " + ".join(cs)] = pf.combine({BASE: base, **{c: candidate_trades(exps[c]) for c in cs}}, SET)
    equity_png(curves, OUT / "equity.png")

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    md = ["# Портфель: набор для запуска + самостоятельная стратегия", "",
          "Гипотеза **H53**, задачи **T18–T20, T22**. Новую стратегию не сравниваем с B01 (созвон 16.10): важно, "
          "прибыльна ли она сама по себе и что даёт вместе с набором для запуска S42. Все стратегии — на тех же пяти "
          "инструментах: серебро и золото через CFD в обе стороны, GAZP, SBER, LKOH на бирже только в лонг (шорты "
          "кандидатов по акциям убраны из готовых сделок), издержки — профиль exchange. Капитал поровну между "
          "стратегиями, внутри — поровну между инструментами. Сумма процентов портфеля — среднее двух стратегий, "
          "поэтому портфель сравнивается с S42 по **t месячных результатов** и **итогу к просадке**. "
          "Расчёт — `scripts/portfolio.py`, модуль — `forexmodel/evaluation/portfolio.py`.", "",
          f"Отбор кандидата: PF > {PF_MIN} сам по себе на выборе и проверке, не меньше {MIN_PER_YEAR} сделок в год (T19); "
          "портфель лучше S42 по t и итогу к просадке в обоих периодах.", "",
          "![Накопленный итог](portfolio/equity.png)", "", verdict, ""]
    sa = {p: pf.metrics(alone, p) for p in ("выбор", "проверка")}
    for per, title in (("проверка", "Проверка 2021 — июнь 2026"), ("выбор", "Выбор 2016–2020")):
        md += [f"## {title}", "",
               "| стратегия | сделок в год | сама: итог % | PF | без 2 лучших | t мес | портфель с S42: итог % | t мес | итог / просадка | совпадение по времени с S42 | корр. месяцев |",
               "|---|---|---|---|---|---|---|---|---|---|---|",
               f"| **{BASE} один** | {f(sa[per]['в_год'], 'в_год')} | {f(sa[per]['итог'], 'итог')} | {f(sa[per]['PF'], 'PF')} | "
               f"{f(sa[per]['без_2_лучших'], 'итог')} | {f(sa[per]['t_мес'], 't_мес')} | — | {f(sa[per]['t_мес'], 't_мес')} | "
               f"{f(sa[per]['итог_к_просадке'], 'итог_к_просадке')} | — | — |"]
        for r in rows:
            s, p, o = r[per]["solo"], r[per]["port"], r[per]["ov"]
            md.append(f"| {r['id']} {r['title']} | {f(s.get('в_год'), 'в_год')} | {f(s.get('итог'), 'итог')} | {f(s.get('PF'), 'PF')} | "
                      f"{f(s.get('без_2_лучших'), 'итог')} | {f(s.get('t_мес'), 't_мес')} | {f(p.get('итог'), 'итог')} | "
                      f"{f(p.get('t_мес'), 't_мес')} | {f(p.get('итог_к_просадке'), 'итог_к_просадке')} | "
                      f"{f(o.get('совпадение'), 'совпадение')} | {f(o.get('корр_мес'), 'корр_мес')} |")
        md.append("")
    md += ["## Отбор", "", "| стратегия | сделок в год (весь период) | сама в плюсе в обоих периодах | портфель лучше S42 в обоих периодах |",
           "|---|---|---|---|"]
    md += [f"| {r['id']} | {r['в_год']:.1f} | {'да' if r['ok_solo'] else 'нет'} | {'да' if r['ok_port'] else 'нет'} |" for r in rows]
    md += ["", "Финалисты перебора S36–S41 отобраны по результатам 2016–2020, поэтому их «выбор» — не проверка, а "
           "подгонка: честная часть для них — только проверка 2021–2026.", "",
           "## Сочетания", "", "| портфель | период | итог % | PF | без 2 лучших | просадка | итог / просадка | t мес | лет в плюсе |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, v in curves.items():
        for per in ("проверка", "выбор"):
            m = pf.metrics(v, per)
            md.append(f"| {k} | {per} | {f(m['итог'], 'итог')} | {f(m['PF'], 'PF')} | {f(m['без_2_лучших'], 'итог')} | "
                      f"{f(m['просадка'], 'итог')} | {f(m['итог_к_просадке'], 'итог_к_просадке')} | {f(m['t_мес'], 't_мес')} | {m['лет_в_плюсе']} |")
    cc = {k: pf.concurrency(v, "проверка") for k, v in curves.items()}
    md += ["", "## Одновременные позиции на проверке", "", "| портфель | максимум | в среднем, когда что-то открыто |", "|---|---|---|"]
    md += [f"| {k} | {v.get('макс_позиций', '—')} | {f(v.get('ср_позиций'), 'ср_позиций')} |" for k, v in cc.items()]
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"прошли отбор: {passed or 'никто'}; лучший портфель на проверке по t: {BASE} + {best} → {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
