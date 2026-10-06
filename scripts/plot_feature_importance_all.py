"""Важность ВСЕХ признаков модели CatBoost (не только топ-20): серебро и LKOH, на обучении и вне выборки.

Читает таблицы scripts/feature_importance.py (reports/feature_importance/<имя>.csv и _groups.csv) и рисует:
  * all_<имя>.png — все признаки, по пользе вне выборки: серый — доля в прогнозе на обучении,
    синий / оранжевый — польза / вред на следующем году;
  * groups_silver_vs_lkoh.png — группы признаков у двух инструментов рядом.
Полные таблицы копируются в docs/results/feature_importance/.

    python scripts/plot_feature_importance_all.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "feature_importance"
DOC = ROOT / "docs" / "results" / "feature_importance"
RUNS = {"silver": "серебро", "lkoh": "LKOH"}
BG, INK, INK2, GRID = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc"
BLUE, ORANGE, GRAY = "#2459b3", "#c8641e", "#9aa3ae"


def plot_all(name: str) -> None:
    t = pd.read_csv(SRC / f"{name}.csv").sort_values("вне_выборки")
    n = len(t)
    f, ax = plt.subplots(figsize=(11, 0.26 * n + 1.5), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(n)
    ax.barh(y + 0.2, t["на_обучении"], 0.4, color=GRAY, label="доля в прогнозе на обучении, %")
    ax.barh(y - 0.2, t["вне_выборки"], 0.4, color=np.where(t["вне_выборки"] >= 0, BLUE, ORANGE),
            label="польза на следующем году, % (оранжевый — вред)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}  ({int(w)}/9)" if name == "lkoh" else f"{a}  ({int(w)}/8)"
                        for a, w in zip(t["признак"], t["окон_помог"])], fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.grid(axis="x", color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{RUNS[name]}: все {n} признаков (в скобках — в скольких окнах признак помог вне выборки)",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.set_ylim(-1, n)
    f.tight_layout()
    f.savefig(DOC / f"all_{name}.png", dpi=130, facecolor=BG)
    plt.close(f)


def plot_groups() -> None:
    g = {k: pd.read_csv(SRC / f"{k}_groups.csv").set_index("группа") for k in RUNS}
    groups = list(dict.fromkeys(list(g["silver"].index) + list(g["lkoh"].index)))
    f, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG, sharey=True)
    y = np.arange(len(groups))
    for ax, col, title in zip(axes, ("на_обучении", "вне_выборки"), ("доля в прогнозе на обучении, %", "польза на следующем году, %")):
        ax.set_facecolor(BG)
        for j, (k, color) in enumerate((("silver", GRAY), ("lkoh", BLUE))):
            v = g[k].reindex(groups)[col].fillna(0).to_numpy()
            ax.barh(y + (0.2 if j == 0 else -0.2), v, 0.4, color=color, label=RUNS[k])
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.grid(axis="x", color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=10)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(groups, fontsize=10, color=INK)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=11, labelcolor=INK, loc="lower right")
    f.tight_layout()
    f.savefig(DOC / "groups_silver_vs_lkoh.png", dpi=140, facecolor=BG)
    plt.close(f)


def main() -> int:
    DOC.mkdir(parents=True, exist_ok=True)
    for name in RUNS:
        plot_all(name)
        for suffix in ("", "_groups", "_by_window"):
            shutil.copy(SRC / f"{name}{suffix}.csv", DOC / f"{name}{suffix}.csv")
    plot_groups()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
