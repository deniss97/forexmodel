"""Скептик к H79 (docs/results/skeptic_2026-11-08.md): попытка сломать «не открывать S42 за сутки до решения ЦБ». Отчёт —
docs/results/skeptic_h79.md.

1. Исправленный календарь: только плановые заседания (13:30 МСК); у внеочередных решение выходило около 10:30 и объявлялось
   меньше чем за сутки — их нельзя было знать заранее. Пересчёт S42 на GAZP, SBER, LKOH (банковский критерий H79).
2. Правило S42 на 18 других акциях (готовые сделки <бумага>_lo__impulse_6_3_long, издержки биржи, без наложений): сделки,
   открытые за 24 ч до решения, против остальных; сравнение со случайными календарями — 2000 наборов пятниц 13:30 того же
   размера, не совпадающих с заседаниями (единица — событие, а не сделка). То же — на GAZP, SBER, LKOH.
Правило (до расчёта): на 2021–2026 средняя «до решения» ниже остальных, а случайные календари дают такую же или худшую
среднюю не чаще 5%; на 2016–2020 знак не противоположный.

    python scripts/skeptic_h79.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import h79_calendar as H
from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "skeptic_h79.md"
OTHERS = ["afks", "aflt", "alrs", "chmf", "gmkn", "irao", "magn", "moex", "mtss", "nlmk", "nvtk", "plzl", "rosn", "rtkm",
          "sngsp", "tatn", "trnfp", "vtbr"]
MAIN = ["gazp", "sber", "lkoh"]
WIN = 24
REPS = 2000


def trades(insts: list[str]) -> pd.DataFrame:
    parts = []
    for i in insts:
        run = f"{i}_lo__impulse_6_3_long"
        t = load_trades({"id": run, "sources": {i: [run]}, "costs": {"profile": "exchange"}, "dividends": True})
        parts.append(no_overlaps(t))
    t = pd.concat(parts, ignore_index=True)
    t["stop"] = t["exit_reason"].str.contains("stop", case=False)
    return t


def before(t: pd.DataFrame, events: np.ndarray) -> np.ndarray:
    """Сделка открыта за ≤ WIN ч до ближайшего события."""
    e = np.sort(events)
    od = t["open_dt"].to_numpy()
    k = np.searchsorted(e, od, side="left")
    nxt = np.where(k < len(e), e[np.minimum(k, len(e) - 1)], np.datetime64("NaT"))
    gap = (nxt - od) / np.timedelta64(1, "h")
    return np.nan_to_num(gap, nan=1e9) <= WIN


def test(t: pd.DataFrame, ev: pd.DataFrame, fridays: pd.DatetimeIndex, rng, label: str) -> list[dict]:
    rows = []
    for per, a, b in (("выбор 2016–2020", pd.Timestamp("2016-01-01"), TEST_START), ("проверка 2021–2026", TEST_START, pd.Timestamp("2100-01-01"))):
        x = t[(t["open_dt"] >= a) & (t["open_dt"] < b)]
        e_real = ev.loc[(ev["event"] >= a) & (ev["event"] < b), "event"].to_numpy()
        fr = fridays[(fridays >= a) & (fridays < b)].to_numpy()
        m = before(x, e_real)
        v = x["profit_pct"].to_numpy()
        real = float(v[m].mean()) if m.any() else np.nan
        sims = []
        for _ in range(REPS):
            fake = rng.choice(fr, size=min(len(e_real), len(fr)), replace=False)
            mm = before(x, fake)
            if mm.any():
                sims.append(v[mm].mean())
        sims = np.array(sims)
        rows.append({"набор": label, "период": per, "заседаний": len(e_real), "сделок до решения": int(m.sum()),
                     "средняя до решения %": real, "средняя остальных %": float(v[~m].mean()),
                     "стопов до решения %": float(x["stop"].to_numpy()[m].mean() * 100) if m.any() else np.nan,
                     "стопов остальных %": float(x["stop"].to_numpy()[~m].mean() * 100),
                     "случайные календари: медиана %": float(np.median(sims)) if len(sims) else np.nan,
                     "p (случайный не лучше)": float((sims <= real).mean()) if len(sims) and np.isfinite(real) else np.nan})
    return rows


def main() -> int:
    ev_all = H.events()
    ev = ev_all[ev_all["scheduled"] == 1].reset_index(drop=True)            # только плановые: их знали заранее
    # 1. S42 на GAZP, SBER, LKOH по исправленному календарю — те же расчёты, что в отчёте H79
    S, _ = H.strategy(ev)
    print(S.round(2).to_string())
    s = S.set_index(["вариант", "период"])
    v = "S42 без входов за 24 ч до решения"
    ok1 = all(s.loc[(v, p), "PF"] > s.loc[("S42", p), "PF"] and s.loc[(v, p), "итог / просадка"] > s.loc[("S42", p), "итог / просадка"]
              for p in ("выбор", "проверка"))
    # 2. случайные календари: пятницы 13:30, кроме дней заседаний
    days = pd.date_range("2015-06-01", "2026-11-08", freq="W-FRI")
    meet = set(ev_all["date"].dt.normalize())
    fridays = pd.DatetimeIndex([d + pd.Timedelta(hours=13, minutes=30) for d in days if d not in meet])
    rng = np.random.default_rng(79)
    rows = test(trades(OTHERS), ev, fridays, rng, "18 других акций")
    rows += test(trades(MAIN), ev, fridays, rng, "GAZP, SBER, LKOH")
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())
    o = R[R["набор"] == "18 других акций"].set_index("период")
    ok2 = bool(o.loc["проверка 2021–2026", "средняя до решения %"] < o.loc["проверка 2021–2026", "средняя остальных %"]
               and o.loc["проверка 2021–2026", "p (случайный не лучше)"] <= 0.05
               and not (o.loc["выбор 2016–2020", "средняя до решения %"] > o.loc["выбор 2016–2020", "средняя остальных %"] + 0.5))
    md = ["# Скептик к H79: исправленный календарь и 18 других акций", "",
          "Правило и разбор — `docs/results/skeptic_2026-11-08.md`. Расчёт — `scripts/skeptic_h79.py`.", "",
          "## 1. S42 на GAZP, SBER, LKOH, только плановые заседания", "",
          "Внеочередные заседания (28.02.2022, 08.04.2022, 26.05.2022, 15.08.2023) убраны из фильтра: решение выходило около 10:30 "
          "и объявлялось меньше чем за сутки.", "",
          "| вариант | период | сделок | PF | средняя % | итог, % депозита | просадка, % | итог / просадка |", "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} | {r['итог % депозита']:+.1f} | "
                  f"{r['просадка %']:.1f} | {r['итог / просадка']:.2f} |")
    md += ["", f"Банковский критерий H79 на исправленном календаре: {'выполнен' if ok1 else 'не выполнен'}.", "",
           "## 2. Сделки за 24 ч до решения против случайных календарей", "",
           f"Случайный календарь — столько же пятниц 13:30, сколько заседаний в периоде, из пятниц без заседаний; {REPS} наборов. "
           "«p» — доля случайных календарей, у которых средняя «до события» такая же или хуже.", "",
           "| набор | период | заседаний | сделок до решения | средняя до решения, % | средняя остальных, % | стопов до решения | стопов остальных | случайные: медиана, % | p |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['набор']} | {r['период']} | {r['заседаний']} | {r['сделок до решения']} | {r['средняя до решения %']:+.2f} | "
                  f"{r['средняя остальных %']:+.2f} | {r['стопов до решения %']:.0f}% | {r['стопов остальных %']:.0f}% | "
                  f"{r['случайные календари: медиана %']:+.2f} | {r['p (случайный не лучше)']:.3f} |")
    md += ["", f"Повтор на 18 других акциях по правилу скептика: {'да' if ok2 else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерии:", ok1, ok2, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
