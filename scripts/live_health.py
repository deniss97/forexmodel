"""Здоровье стратегии на сейчас против порогов тревоги (гипотеза H43, задачи T16–T17).

Первый этап запуска — GAZP, SBER, LKOH на бирже в лонг. Сделки: история из журнала экспериментов (S42, акции)
до конца истории и свежие — тем же кодом (scripts/fresh_check.py) до последней минуты в data/raw/live.
Сейчас считаются: PF сделок за последние 12 месяцев, сколько месяцев подряд он ниже 1, просадка от пика,
итог за 12 месяцев. Пороги — худшее, что было в истории до её конца (scripts/launch_set.py, alarm).
Отчёт — reports/live/health.md; в живой торговле сюда же добавятся сделки счёта.

    python scripts/live_health.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from experiments import ROOT, load_catalog, load_trades
from fresh_check import STOCKS, run
from launch_set import alarm, rolling_pf, weighted

OUT = ROOT / "reports" / "live" / "health.md"
INST = list(STOCKS)


def status(t: pd.DataFrame, end: pd.Timestamp) -> dict:
    x = weighted(t, INST).set_index("close_dt")["wp"]
    last12 = x.loc[end - pd.DateOffset(months=12):end]
    loss = -last12[last12 < 0].sum()
    pf = rolling_pf(t, INST).dropna()
    streak = 0
    for v in pf.to_numpy()[::-1]:
        if v >= 1:
            break
        streak += 1
    eq = x.cumsum()
    return {"PF12": float(last12[last12 > 0].sum() / loss) if loss > 0 else np.inf, "подряд < 1": streak,
            "просадка": float(eq.iloc[-1] - eq.max()), "12 мес": float(last12.sum()), "сделок за 12 мес": int(len(last12))}


def main() -> int:
    _, exps = load_catalog()
    hist = load_trades(exps["S42"])
    hist = hist[hist["instrument"].isin(INST)]
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    fresh, ends = [], []
    for inst in INST:
        t, hist_end, last = run(inst, live, "impulse:6:3:long")
        f = t[t["период"] == "свежий"]
        fresh.append(pd.DataFrame({"instrument": inst, "run": "fresh", "open_dt": f["open_dt"], "close_dt": f["close_dt"],
                                   "profit_pct": f["profit_pct"]}))
        hist = hist[~((hist["instrument"] == inst) & (hist["open_dt"] > hist_end))]
        ends.append(last)
    now = min(ends)
    both = pd.concat([hist[["instrument", "run", "open_dt", "close_dt", "profit_pct"]], *fresh], ignore_index=True)
    lim = alarm(hist, INST)
    st = status(both, now)
    checks = [("PF за 12 месяцев", f"{st['PF12']:.2f}", f"< {lim['PF12 мин']:.2f}", st["PF12"] < lim["PF12 мин"]),
              ("месяцев подряд с PF за 12 мес < 1", str(st["подряд < 1"]), f"> {lim['PF12 < 1 подряд, мес']}", st["подряд < 1"] > lim["PF12 < 1 подряд, мес"]),
              ("просадка от пика", f"{st['просадка']:+.1f}%", f"глубже {lim['просадка']:+.1f}%", st["просадка"] < lim["просадка"]),
              ("итог за 12 месяцев", f"{st['12 мес']:+.1f}%", f"хуже {lim['худшие 12 мес, %']:+.1f}%", st["12 мес"] < lim["худшие 12 мес, %"])]
    bad = [c for c in checks if c[3]]
    md = [f"# Здоровье стратегии на {now:%d.%m.%Y %H:%M}", "",
          "Первый этап — GAZP, SBER, LKOH на бирже в лонг, поровну. Сделки: журнал (история) + свежие (тот же код). "
          f"Сделок за последние 12 месяцев: {st['сделок за 12 мес']}.", "",
          f"**{'ТРЕВОГА: ' + ', '.join(c[0] for c in bad) + ' — остановиться и разобраться' if bad else 'В норме: всё внутри того, что бывало в истории'}**", "",
          "| метрика | сейчас | тревога, если | |", "|---|---|---|---|"]
    md += [f"| {n} | {v} | {thr} | {'⚠' if b else 'ок'} |" for n, v, thr, b in checks]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
