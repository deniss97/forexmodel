"""H63 (Тимур): модель LKOH на 3 класса против пары бинарных моделей — отчёт docs/results/h63_binary.md.

Прогоны — scripts/queues/h63_binary.sh: одинаковые настройки (CatBoost без NN и меты, сигнал по модели, тренд-гейт C,
разметка atr_asym +1.5 / −0.75 ATR за 10 ч), годовой walk-forward 2018–2026, seed 42, 7, 1. Метрики — кодом журнала,
издержки биржи; сравнение по периодам «выбор» (сделки 2018–2020) и «проверка» (2021–2026), среднее по seed.
Доля верных направлений — сделка закрылась с плюсом до издержек.

    python scripts/h63_binary.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START, metrics

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "h63_binary.md"
SEEDS = (42, 7, 1)
OBJ = {"multiclass": "3 класса (как сейчас)", "binary_pair": "пара бинарных (H63)"}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def main() -> int:
    rows = []
    for obj in OBJ:
        for sd in SEEDS:
            t = load_trades({"id": f"{obj} s{sd}", "sources": {"lkoh": [f"lkoh_h63_{obj}_s{sd}"]}, "costs": {"profile": "exchange"},
                             "dividends": True})
            for per in ("выбор", "проверка"):
                m = metrics(t, ["lkoh"], per)
                x = t[(t["open_dt"] >= TEST_START) == (per == "проверка")]
                rows.append({"модель": OBJ[obj], "seed": sd, "период": per, "сделок": m.get("сделок", 0), "итог %": m.get("итог", np.nan),
                             "PF": m.get("PF", np.nan), "без 2 лучших": m.get("без_2_лучших", np.nan), "просадка": m.get("просадка", np.nan),
                             "верное направление %": float((x["gross_pct"] > 0).mean() * 100) if len(x) else np.nan,
                             "лонгов %": float((x["side"].astype(str).isin(["buy", "1", "2"])).mean() * 100) if len(x) else np.nan})
    R = pd.DataFrame(rows)
    A = R.groupby(["модель", "период"], sort=False)[["сделок", "итог %", "PF", "без 2 лучших", "просадка", "верное направление %", "лонгов %"]].mean()
    print(R.round(2).to_string(), "\n", A.round(2).to_string())
    g = lambda m, p, k: float(A.loc[(OBJ[m], p), k])  # noqa: E731
    ok = all(g("binary_pair", p, "PF") > g("multiclass", p, "PF") and g("binary_pair", p, "верное направление %") > g("multiclass", p, "верное направление %")
             for p in ("выбор", "проверка"))
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Пара однонаправленных моделей вместо модели на 3 класса (H63)", "",
          "Тимур: модель на 3 классах хорошо предсказывает волатильность, но плохо — направление; две модели «рост дальше барьера: "
          "да / нет» и «падение: да / нет» могут дать лучшее качество. Сравнение A/B на LKOH при одинаковых настройках: CatBoost без "
          "NN и меты, сигнал по модели, тренд-гейт C, разметка atr_asym (+1.5 / −0.75 ATR за 10 часов), годовой walk-forward "
          "2018–2026, seed 42, 7, 1, издержки биржи. Пара собирается в те же три вероятности: p_flat = (1 − p_up)(1 − p_down), "
          "затем та же балансировка классов, что у модели на 3 класса (без неё пара торговала бы втрое чаще). Расчёт — "
          "`scripts/queues/h63_binary.sh`, `scripts/h63_binary.py`; код — `forexmodel/models/catboost_primary.py` "
          "(catboost.objective: binary_pair).", "",
          "**Критерий (банк, до расчёта):** PF пары выше и на 2018–2020, и на 2021–2026 (среднее по 3 seed), и доля верных "
          "направлений выше; «полезна для торговли» — если вдобавок PF на проверке > 1.", "", verdict, "",
          "## Среднее по 3 seed", "", "| модель | период | сделок | итог % | PF | без 2 лучших | просадка | верное направление | лонгов |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (m, p), r in A.iterrows():
        md.append(f"| {m} | {p} | {r['сделок']:.0f} | {r['итог %']:+.1f} | {r['PF']:.2f} | {r['без 2 лучших']:+.1f} | {r['просадка']:.1f} | "
                  f"{r['верное направление %']:.1f}% | {r['лонгов %']:.0f}% |")
    md += ["", "## По seed", "", "| модель | seed | период | сделок | итог % | PF | верное направление |", "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['модель']} | {r['seed']} | {r['период']} | {r['сделок']} | {r['итог %']:+.1f} | {r['PF']:.2f} | {r['верное направление %']:.1f}% |")
    md += ["", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
