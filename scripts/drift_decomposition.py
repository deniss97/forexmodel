"""Сколько дохода импульсной стратегии даёт дрейф инструмента, а сколько — сам сигнал.

Серебро, золото и акции за 2015–2026 выросли. Лонг, удерживаемый в среднем 5 суток, получает долю
этого роста независимо от сигнала. Разложение по каждой сделке:

    дрейф   = сторона × (средний часовой лог-рост инструмента за этот год) × часов в сделке
    остаток = результат − дрейф

Дрейф года считается по факту (это диагностика, а не торговое правило). Если остаток близок к
нулю, стратегия — переупакованный «купи и держи»; если остаток велик и положителен и у лонгов, и у
шортов, сигнал есть сверх дрейфа. Дополнительно: корреляция месячных результатов стратегии с
месячными доходностями инструмента и доля времени в лонге/шорте.

Сделки — reports/audit/audit_trades.csv (гейт C, исполнение по уровню стопа).

    python scripts/drift_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audit"


def main() -> int:
    pd.set_option("display.width", 250)
    T = pd.read_csv(OUT / "audit_trades.csv", parse_dates=["signal"])
    T = T[(T["гейт"] == "гейт C") & (T["исполнение"] == "уровень")].copy()
    rows, monthly = [], []
    for inst, tr in T.groupby("инструмент"):
        h = pd.read_pickle(ROOT / "reports" / "_cache" / f"hourly_{inst}.pkl")
        h["год"] = h["time"].dt.year
        lr = np.log(h["close"]).diff()
        drift_h = lr.groupby(h["год"]).mean() * 100            # % за торговый час, по годам
        # длительность — в торговых часах: число часовых баров от входа до выхода (не календарные часы:
        # у акций ночи и выходные, у форекса выходные)
        ht = h["time"].to_numpy()
        t_in = (tr["signal"] + pd.Timedelta("1h")).to_numpy()
        t_out = t_in + pd.to_timedelta(tr["hours"], unit="h").to_numpy()
        tr = tr.assign(bars=np.searchsorted(ht, t_out, "right") - np.searchsorted(ht, t_in, "left"))
        tr = tr.assign(дрейф=tr["side"] * tr["год"].map(drift_h) * tr["bars"])
        tr["остаток"] = tr["net"] - tr["дрейф"]
        for side, name in ((1, "лонг"), (-1, "шорт"), (0, "все")):
            x = tr if side == 0 else tr[tr["side"] == side]
            rows.append({"инструмент": inst, "сторона": name, "сделок": len(x), "итог": x["net"].sum(),
                         "из_них_дрейф": x["дрейф"].sum(), "остаток": x["остаток"].sum(),
                         "t_остатка": x["остаток"].mean() / (x["остаток"].std() / np.sqrt(len(x))) if len(x) > 2 else np.nan})
        # месяцы: результат стратегии против доходности инструмента
        tr["месяц"] = tr["signal"].dt.to_period("M")
        sm = tr.groupby("месяц")["net"].sum()
        hm = h.set_index("time")["close"].resample("ME").last().pct_change() * 100
        hm.index = hm.index.to_period("M")
        both = pd.concat([sm.rename("стратегия"), hm.rename("инструмент")], axis=1).dropna()
        hours_long = tr.loc[tr["side"] > 0, "bars"].sum()
        hours_short = tr.loc[tr["side"] < 0, "bars"].sum()
        monthly.append({"инструмент": inst, "корр_месячных": both.corr().iloc[0, 1],
                        "бета_к_инструменту": np.polyfit(both["инструмент"], both["стратегия"], 1)[0],
                        "часов_в_лонге_%": 100 * hours_long / len(h), "часов_в_шорте_%": 100 * hours_short / len(h),
                        "рост_инструмента_%": 100 * (h["close"].iloc[-1] / h["close"].iloc[0] - 1)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "drift_decomposition.csv", index=False)
    piv = r.pivot_table(index="инструмент", columns="сторона", values=["итог", "из_них_дрейф", "остаток", "t_остатка"])
    print("РАЗЛОЖЕНИЕ ИТОГА (% после комиссии, гейт C, 2015–2026): итог = дрейф инструмента за время в сделке + остаток")
    for side in ("лонг", "шорт", "все"):
        print(f"\n--- {side}")
        print(piv.xs(side, axis=1, level=1)[["итог", "из_них_дрейф", "остаток", "t_остатка"]].round(2).to_string())
    tot = r[r["сторона"] == "все"][["итог", "из_них_дрейф", "остаток"]].sum() / 8
    print(f"\nпортфель 8 (1/8): итог {tot['итог']:+.1f}%, из них дрейф {tot['из_них_дрейф']:+.1f}%, остаток {tot['остаток']:+.1f}%")
    m = pd.DataFrame(monthly)
    m.to_csv(OUT / "drift_monthly.csv", index=False)
    print("\nСВЯЗЬ С ИНСТРУМЕНТОМ: корреляция месячных результатов и бета стратегии к инструменту; доля времени в позиции")
    print(m.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
