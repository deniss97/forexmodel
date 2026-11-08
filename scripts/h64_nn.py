"""H64 (Тимур): нейросеть CNN-BiLSTM против CatBoost на LKOH — отчёт docs/results/h64_nn.md.

Прогоны — scripts/queues/h64_nn.sh (сеть) и scripts/queues/h63_binary.sh (CatBoost на 3 класса, база L10): одинаковые
настройки — без меты, тренд-гейт C, разметка atr_asym (+1.5 / −0.75 ATR за 10 ч), годовой walk-forward 2018–2026,
seed 42, 7, 1. Метрики — кодом журнала, издержки биржи; «выбор» — сделки 2018–2020, «проверка» — 2021–2026.

    python scripts/h64_nn.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START, metrics

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "h64_nn.md"
SEEDS = (42, 7, 1)
RUNS = {"CatBoost (L10)": "lkoh_h63_multiclass_s{}", "нейросеть CNN-BiLSTM (H64)": "lkoh_h64_nn_s{}"}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def main() -> int:
    rows = []
    for name, pat in RUNS.items():
        for sd in SEEDS:
            run = pat.format(sd)
            t = load_trades({"id": run, "sources": {"lkoh": [run]}, "costs": {"profile": "exchange"}, "dividends": True})
            for per in ("выбор", "проверка"):
                m = metrics(t, ["lkoh"], per)
                x = t[(t["open_dt"] >= TEST_START) == (per == "проверка")]
                rows.append({"модель": name, "seed": sd, "период": per, "сделок": m.get("сделок", 0), "итог %": m.get("итог", np.nan),
                             "PF": m.get("PF", np.nan), "просадка": m.get("просадка", np.nan),
                             "до издержек, % на сделку": float(x["gross_pct"].mean()) if len(x) else np.nan,
                             "верное направление %": float((x["gross_pct"] > 0).mean() * 100) if len(x) else np.nan,
                             "лонгов %": float((x["side"] == 1).mean() * 100) if len(x) else np.nan})
    R = pd.DataFrame(rows)
    cols = ["сделок", "итог %", "PF", "просадка", "до издержек, % на сделку", "верное направление %", "лонгов %"]
    A = R.groupby(["модель", "период"], sort=False)[cols].mean()
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(), "\n", A.round(3).to_string())
    nn, cb = list(RUNS)[1], list(RUNS)[0]
    ok = all(A.loc[(nn, p), "PF"] > A.loc[(cb, p), "PF"] for p in ("выбор", "проверка"))

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Нейросеть вместо CatBoost (H64)", "",
          "Тимур: в его опыте нейросеть (torch) по результатам и «адаптивности» превосходила CatBoost. В конвейере уже есть "
          "нейросеть CNN-BiLSTM (`forexmodel/models/nn_cnn_bilstm.py`: окно 20 часов, свёртка + двунаправленный LSTM, 40 эпох с "
          "ранней остановкой, те же 81 признак и та же разметка на 3 класса). Сравнение A/B на LKOH при настройках H63: без "
          "меты, тренд-гейт C, разметка atr_asym, годовой walk-forward 2018–2026, seed 42, 7, 1, издержки биржи. Сигнал — "
          "только сеть против только CatBoost (база L10). Расчёт — `scripts/queues/h64_nn.sh`, `scripts/h64_nn.py`.", "",
          "**Критерий (банк, до расчёта):** PF сети выше CatBoost и на 2018–2020, и на 2021–2026 (среднее по 3 seed); "
          "«полезна для торговли» — если вдобавок PF на проверке > 1.", "", verdict, "",
          "## Среднее по 3 seed", "", "| модель | период | сделок | итог % | PF | просадка | до издержек, % на сделку | верное направление | лонгов |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (m, p), r in A.iterrows():
        md.append(f"| {m} | {p} | {r['сделок']:.0f} | {r['итог %']:+.1f} | {r['PF']:.2f} | {r['просадка']:.1f} | "
                  f"{r['до издержек, % на сделку']:+.3f} | {r['верное направление %']:.1f}% | {r['лонгов %']:.0f}% |")
    md += ["", "## По seed", "", "| модель | seed | период | сделок | итог % | PF | до издержек, % на сделку |", "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['модель']} | {r['seed']} | {r['период']} | {r['сделок']} | {r['итог %']:+.1f} | {r['PF']:.2f} | "
                  f"{r['до издержек, % на сделку']:+.3f} |")
    md += ["", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
