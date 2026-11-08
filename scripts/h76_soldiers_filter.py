"""Гипотеза H76 («скептик» к H73, созвон 05.11): лучше ли входы S42, перед которыми сложились «три белых солдата».

Фигуры — определения H73 (scripts/h73_soldiers.py) на часовых свечах: «строго, где угодно», «строго, после снижения» и
«мягко». Вход S42 «с фигурой», если фигура закрылась не раньше чем за 12 часов до свечи сигнала (включительно).
Критерий (банк, до расчёта): с фигурой лучше, чем без, по PF и средней сделке и на выборе 2016–2020, и на проверке
2021–2026, сделок с фигурой не меньше 20 в каждом периоде. Отчёт — docs/results/h76_soldiers_filter.md.

    python scripts/h76_soldiers_filter.py
"""
from __future__ import annotations

import logging

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import h73_soldiers as H
import strategy_search as ss
from forexmodel.evaluation.experiments import TEST_START

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "h76_soldiers_filter.md"
STOCKS = ["gazp", "sber", "lkoh"]
LOOK = 12            # часов до свечи сигнала
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x: pd.Series) -> float:
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def main() -> int:
    logging.disable(logging.WARNING)
    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    t = t[t["instrument"].isin(STOCKS)].reset_index(drop=True)
    flags = {v: np.zeros(len(t), bool) for v in H.VARIANTS}
    for inst in STOCKS:
        ins = ss.Inst(inst)
        H.add_entries(ins)
        tt = pd.to_datetime(ins.t)
        for v in H.VARIANTS:
            ev_t = tt[ins.entries[H.name(v, "1ч")] == 1]            # свеча, на которой фигура закрылась
            for k in np.flatnonzero(t["instrument"].to_numpy() == inst):
                sig = t.at[k, "open_dt"] - pd.Timedelta(minutes=61)   # начало свечи сигнала S42
                lo = sig - pd.Timedelta(hours=LOOK)
                flags[v][k] = bool(((ev_t >= lo) & (ev_t <= sig)).any())
    sel = (t["open_dt"] < TEST_START).to_numpy()
    rows = []
    for v in H.VARIANTS:
        for per, pm in (("выбор", sel), ("проверка", ~sel)):
            for kind, m in (("с фигурой", flags[v] & pm), ("без фигуры", ~flags[v] & pm)):
                x = t.loc[m, "profit_pct"]
                rows.append({"фигура": v, "период": per, "вход S42": kind, "сделок": len(x), "PF": pf(x),
                             "средняя %": x.mean() if len(x) else np.nan, "в плюсе %": (x > 0).mean() * 100 if len(x) else np.nan})
    R = pd.DataFrame(rows)
    print(R.round(2).to_string())
    verdicts = {}
    for v in H.VARIANTS:
        ok = True
        for per in ("выбор", "проверка"):
            a = R[(R["фигура"] == v) & (R["период"] == per) & (R["вход S42"] == "с фигурой")].iloc[0]
            b = R[(R["фигура"] == v) & (R["период"] == per) & (R["вход S42"] == "без фигуры")].iloc[0]
            ok &= a["сделок"] >= 20 and a["PF"] > b["PF"] and a["средняя %"] > b["средняя %"]
        verdicts[v] = ok
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# «Скептик» к H73: «три белых солдата» как подтверждение входа S42 (H76)", "",
          "H73 отклонена как отдельная стратегия: строгая фигура на часах лучше случайного входа, но встречается слишком редко. "
          "Шаг в сторону (идея «скептика», созвон 05.11): не торговать по фигуре, а проверить, лучше ли входы S42, перед которыми "
          f"она сложилась. Вход «с фигурой» — фигура закрылась не раньше чем за {LOOK} ч до свечи сигнала S42. Фигуры — "
          "определения H73 на часах. Сделки S42 — из журнала (GAZP, SBER, LKOH). Расчёт — `scripts/h76_soldiers_filter.py`.", "",
          "**Критерий (банк, до расчёта):** с фигурой лучше, чем без, по PF и средней сделке в обоих периодах, сделок с фигурой "
          "не меньше 20 в каждом.", "", verdict, "",
          "| фигура | период | вход S42 | сделок | PF | средняя % | в плюсе |", "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['фигура']} | {r['период']} | {r['вход S42']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} | "
                  f"{r['в плюсе %']:.0f}% |")
    md += ["", "Прошла ли каждая фигура критерий: " + ", ".join(f"{v} — {'да' if ok else 'нет'}" for v, ok in verdicts.items()) + "."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(verdicts, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
