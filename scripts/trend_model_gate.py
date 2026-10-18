"""Трёхслойный тренд-фильтр как фильтр сделок модели CatBoost (гипотеза H58, созвон 18.10).

Сделки модели — прогоны журнала (M13 — серебро, 3 seed; L07 — LKOH); фильтр только решает, брать ли сделку:
лонг — если на последнем закрытом часе до входа фильтр показывает вверх, шорт — если вниз. Варианты:
трёхслойный фильтр (режим по дневной EMA200, как в H55), согласие двух уровней (трёхслойный + быстрый v2 Тимура),
для справки — гейт C. Метрики — тем же кодом, что журнал (seed поровну, t — среднее по прогонам).

    python scripts/trend_model_gate.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_grid as tg  # noqa: E402
from forexmodel.evaluation.experiments import metrics  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, fmt, load_catalog, load_trades  # noqa: E402
from trend_labeling import timur  # noqa: E402

OUT = ROOT / "docs" / "results" / "trend_model_gate"
DOC = ROOT / "docs" / "results" / "trend_model_gate.md"
MODELS = {"M13": "silver", "L07": "lkoh"}
NAMES = {"silver": "серебро", "lkoh": "LKOH"}
KEYS = ("сделок", "итог", "PF", "без_2_лучших", "просадка", "t", "лет_в_плюсе")
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def states(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    three = np.nan_to_num(tg.gate_values(bars, {"mode": "score", "score_regime_tf": "24h"}))
    fast = timur(bars, 5, 9, 20.0, 1)
    gate = np.nan_to_num(tg.gate_values(bars, {}))
    both = np.where((three == fast) & (three != 0), three, 0)
    return {"трёхслойный": three, "трёхслойный + v2 Тимура": both, "гейт C": gate}


def at_entry(t: pd.DataFrame, bars: pd.DataFrame, g: np.ndarray) -> np.ndarray:
    """Состояние фильтра на последнем часе, закрытом до входа (бар [t, t+1ч) закрыт к t+1ч)."""
    bt = bars["time"].to_numpy()
    i = np.searchsorted(bt, (t["open_dt"] - pd.Timedelta(hours=1)).to_numpy(), side="right") - 1
    return np.where(i >= 0, g[np.clip(i, 0, len(g) - 1)], 0)


def main() -> int:
    tg._init()
    _, exps = load_catalog()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for mid, inst in MODELS.items():
        t = load_trades(exps[mid])
        bars = tg._W["data"][inst]["bars"]
        variants = {"все сделки модели": t}
        for name, g in states(bars).items():
            s = at_entry(t, bars, g)
            variants[f"{name} согласен"] = t[s == t["side"].to_numpy()]
        for v, x in variants.items():
            for per in ("выбор", "проверка"):
                m = metrics(x, [inst], per)
                rows.append({"модель": f"{mid} ({NAMES[inst]})", "вариант": v, "период": per,
                             "доля сделок": len(x) / len(t), **{k: m.get(k) for k in KEYS + ("с", "по")}})
        curves[mid] = {k: v for k, v in variants.items() if k in ("все сделки модели", "трёхслойный согласен", "трёхслойный + v2 Тимура согласен")}
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.8), facecolor=BG)
    for ax, (mid, cv) in zip(axes, curves.items()):
        ax.set_facecolor(BG)
        for (k, x), c in zip(cv.items(), (GRAY, BLUE, "#1d7a55")):
            x = x.sort_values("close_dt")
            ax.plot(x["close_dt"], (x["profit_pct"] / x["run"].nunique()).cumsum(), color=c, linewidth=2 if c == BLUE else 1.5, label=k)
        ax.axvline(pd.Timestamp("2021-01-01" if mid == "M13" else "2022-01-01"), color=INK2, linewidth=0.8, linestyle="--")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{mid} — {NAMES[MODELS[mid]]}", loc="left", fontsize=10, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    axes[0].set_ylabel("накопленный итог, % (seed поровну)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    md = ["# Тренд-фильтр как фильтр сделок модели CatBoost (H58)", "",
          "Созвон 18.10: находки по тренд-фильтру применить к текущей модели. Сделки модели не меняются — фильтр решает, "
          "брать ли сделку: лонг, если на последнем закрытом часе до входа фильтр показывает вверх, шорт — если вниз. "
          "Модели — прогоны журнала: **M13** (серебро, 3 seed; выбор 2019–2020, проверка 2021–2026) и **L07** (LKOH, годовой "
          "walk-forward; выбор 2018–2020, проверка 2022–2026). Издержки — как у прогонов (B01). Трёхслойный фильтр — с "
          "режимом по дневной EMA200 (H55); «+ v2 Тимура» — оба уровня показывают одно направление. Расчёт — "
          "`scripts/trend_model_gate.py`.", "",
          "![Накопленный итог](trend_model_gate/equity.png)", "", verdict, ""]
    for mdl, x in df.groupby("модель", sort=False):
        md += [f"## {mdl}", "", "| вариант | период | доля сделок | сделок | итог % | PF | без 2 лучших | просадка | t | лет в плюсе |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['вариант']} | {r['период']} {r['с'] or ''}–{r['по'] or ''} | {r['доля сделок']:.0%} | "
               + " | ".join(fmt(r[k], k) for k in KEYS) + " |" for _, r in x.iterrows()]
        md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
