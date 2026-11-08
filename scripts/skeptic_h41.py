"""Скептик к H41 (docs/results/skeptic_2026-11-08.md): единственная устойчивая находка H41 — cos угла недельного наклона
(168 ч) связан с размером следующего хода — не проверялась там, где могла пригодиться: как фильтр готовых сделок. Отчёт —
docs/results/skeptic_h41.md.

Сделки импульса (правило S42/B01) на серебре, LKOH, GAZP, SBER; признак — slope_cos_168 на часе сигнала (час до входа);
трети по порогам, посчитанным только по сделкам 2016–2020. Правило (до расчёта): средняя сделка до издержек в трети с самым
крутым трендом (малый cos) выше, чем в самой плоской, в обоих периодах на 3 рынках из 4.

    python scripts/skeptic_h41.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import h41_slope as S
from forexmodel.evaluation.experiments import TEST_START

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "skeptic_h41.md"
MARKETS = ["silver", "lkoh", "gazp", "sber"]


def main() -> int:
    _, exps = load_catalog()
    t_all = load_trades(exps["S42"])
    rows, okc = [], 0
    for inst in MARKETS:
        f = S.frame(inst).set_index("time")["slope_cos_168"]
        t = t_all[t_all["instrument"] == inst].copy()
        sig = t["open_dt"].dt.floor("1h") - pd.Timedelta(hours=1)
        t["cos"] = f.reindex(sig.to_numpy()).to_numpy()
        t = t.dropna(subset=["cos"])
        sel = t["open_dt"] < TEST_START
        q1, q2 = np.quantile(t.loc[sel, "cos"], [1 / 3, 2 / 3])
        t["треть"] = np.where(t["cos"] <= q1, "крутой тренд", np.where(t["cos"] <= q2, "средний", "плоский"))
        res = {}
        for per, pm in (("выбор", sel), ("проверка", ~sel)):
            for g in ("крутой тренд", "средний", "плоский"):
                x = t[pm & (t["треть"] == g)]
                rows.append({"рынок": inst, "период": per, "треть": g, "сделок": len(x),
                             "до издержек %": float(x["gross_pct"].mean()) if len(x) else np.nan,
                             "после издержек %": float(x["profit_pct"].mean()) if len(x) else np.nan})
            st = t[pm & (t["треть"] == "крутой тренд")]["gross_pct"].mean()
            fl = t[pm & (t["треть"] == "плоский")]["gross_pct"].mean()
            res[per] = st > fl
        okc += all(res.values())
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(), "\nрынков, где крутой лучше плоского в обоих периодах:", okc)
    md = ["# Скептик к H41: крутизна недельного тренда как фильтр сделок импульса", "",
          "Разбор и правило — `docs/results/skeptic_2026-11-08.md`. Сделки импульса (S42: металлы через CFD, акции в лонг), признак — "
          "cos угла наклона за 168 ч на часе сигнала; трети по порогам 2016–2020. Расчёт — `scripts/skeptic_h41.py`.", "",
          "| рынок | период | треть | сделок | до издержек, % | после издержек, % |", "|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['рынок']} | {r['период']} | {r['треть']} | {r['сделок']} | {r['до издержек %']:+.2f} | {r['после издержек %']:+.2f} |")
    md += ["", f"Рынков, где в трети с крутым трендом средняя выше, чем в плоской, в обоих периодах: {okc} из 4 (нужно 3)."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
