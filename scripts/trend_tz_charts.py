"""Графики к docs/results/trend_filter_tz.md (ТЗ 15.10, гипотезы H44–H47, H49).

  selection_vs_test.png — PF каждого из 128 фильтров на выборе (2016–2020) и проверке (2021–26): все рынки и металлы;
  variants.png          — итог проверки вариантов S26–S32 против базы B01 (металлы, портфель 4, все рынки);
  quality.png           — метрики ТЗ: задержка разворота и ложные переключения у всех фильтров.

    python scripts/trend_tz_charts.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import metrics, scopes, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "reports" / "trend_grid"
WF = ROOT / "reports" / "walk_forward"
OUT = ROOT / "docs" / "results" / "trend_tz"
INST = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
BG, INK, INK2, GRID_C, BLUE, ORANGE, GRAY = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc", "#2459b3", "#c8641e", "#9aa3ae"

VARIANTS = [("B01 · гейт C (база)", "costs__impulse_6_3_h240_tr6"),
            ("S26 · трёхслойный по ТЗ", "tz__impulse_6_3_gatetz"),
            ("S27 · трёхслойный, лучший на выборе", "tz__impulse_6_3_gatex4"),
            ("S28 · Дончиан 55/20 4 ч, лучший классический", "tz__impulse_6_3_gatedon55"),
            ("S29 · без фильтра, только лонг", "tz__impulse_6_3_notrend_long"),
            ("S30 · вход NR7 + inside bar", "tz__pattern_nr7ib"),
            ("S31 · вход пробой 55 дней", "tz__pattern_donch_55"),
            ("S32 · вход ложный пробой 20 дней", "tz__pattern_soup_20")]


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID_C, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def wide() -> pd.DataFrame:
    df = pd.read_csv(GRID / "grid.csv")
    w = df.pivot_table(index=["группа", "фильтр"], columns=["срез", "период"], values=["итог", "PF"])
    w.columns = [f"{s}|{p}|{v}" for v, s, p in w.columns]
    return w.reset_index()


def selection_vs_test() -> None:
    w = wide()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=BG)
    marks = {"гейт C (база)": (BLUE, "гейт C (база B01)"), "скор: ТЗ как есть": (ORANGE, "трёхслойный по ТЗ"),
             "всегда лонг": (INK, "всегда лонг")}
    for ax, scope, title in ((axes[0], "все", "Все 8 рынков"), (axes[1], "металлы", "Металлы (серебро + золото)")):
        style(ax)
        x, y = w[f"{scope}|выбор|PF"], w[f"{scope}|проверка|PF"]
        ax.scatter(x, y, s=22, color=GRAY, alpha=0.7, lw=0, label="128 фильтров ТЗ (H44, H45)")
        for name, (col, lab) in marks.items():
            r = w[w["фильтр"] == name].iloc[0]
            ax.scatter(r[f"{scope}|выбор|PF"], r[f"{scope}|проверка|PF"], s=70, color=col, edgecolor=BG, lw=2, zorder=3)
            ax.annotate(lab, (r[f"{scope}|выбор|PF"], r[f"{scope}|проверка|PF"]), xytext=(7, 4),
                        textcoords="offset points", fontsize=9, color=INK)
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.axvline(1, color=INK2, lw=0.8, ls="--")
        ax.set_xlabel("PF на выборе 2016–2020", color=INK2, fontsize=9)
        ax.set_ylabel("PF на проверке 2021–2026", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    fig.suptitle("Лучший на выборе фильтр не лучший на проверке: связи нет", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "selection_vs_test.png", dpi=130, facecolor=BG)
    plt.close(fig)


def load(run: str) -> pd.DataFrame:
    parts = []
    for i in INST:
        p = WF / f"{i}_{run}_trades.csv"
        if p.exists():
            t = recost(pd.read_csv(p), profile_entry(ROOT / "configs" / "costs" / "base.yaml", i))
            parts.append(standardize(t, i, run))
    return pd.concat(parts)


def variants() -> pd.DataFrame:
    rows = []
    for label, run in VARIANTS:
        t = load(run)
        for scope, mem in scopes(INST).items():
            if scope in ("все", "портфель 4", "металлы"):
                for per in ("выбор", "проверка"):
                    m = metrics(t, mem, per)
                    rows.append({"вариант": label, "срез": scope, "период": per,
                                 **{k: m.get(k) for k in ("сделок", "итог", "PF", "без_2_лучших", "t", "лет_в_плюсе")}})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "variants.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=BG, sharey=True)
    labels = [v[0] for v in VARIANTS][::-1]
    for ax, scope in zip(axes, ("металлы", "портфель 4", "все")):
        style(ax)
        d = df[(df["срез"] == scope) & (df["период"] == "проверка")].set_index("вариант").loc[labels]
        cols = [BLUE if lab.startswith("B01") else GRAY for lab in labels]
        ax.barh(range(len(labels)), d["итог"], color=cols, height=0.62)
        for k, v in enumerate(d["итог"]):
            ax.annotate(f"{v:+.0f}%", (v, k), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title({"металлы": "Металлы", "портфель 4": "Портфель 4", "все": "Все 8 рынков"}[scope],
                     color=INK, fontsize=11, loc="left")
        ax.set_xlabel("итог на проверке 2021–26, % (издержки B01)", color=INK2, fontsize=9)
        lo, hi = d["итог"].min(), d["итог"].max()
        ax.set_xlim(min(lo, 0) * 1.35 - 5, max(hi, 0) * 1.35 + 5)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=9, color=INK)
    fig.suptitle("Ни один фильтр и вход из ТЗ не лучше базы на металлах", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "variants.png", dpi=130, facecolor=BG)
    plt.close(fig)
    return df


def quality() -> None:
    q = pd.read_csv(GRID / "quality.csv")
    w = wide()[["фильтр", "все|проверка|PF"]]
    q = q.merge(w, on="фильтр")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=BG)
    style(ax)
    ax.scatter(q["проверка|задержка_медиана_ч"], q["проверка|ложных_на_ногу"], s=22, color=GRAY, alpha=0.7, lw=0,
               label="фильтры ТЗ")
    h44 = q[q["группа"].str.startswith("H44")]
    ax.scatter(h44["проверка|задержка_медиана_ч"], h44["проверка|ложных_на_ногу"], s=30, color=ORANGE, lw=0,
               label="трёхслойные (H44)")
    c = q[q["фильтр"] == "гейт C (база)"].iloc[0]
    ax.scatter(c["проверка|задержка_медиана_ч"], c["проверка|ложных_на_ногу"], s=70, color=BLUE, edgecolor=BG, lw=2,
               zorder=3)
    ax.annotate("гейт C (база)", (c["проверка|задержка_медиана_ч"], c["проверка|ложных_на_ногу"]), xytext=(7, 4),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("задержка разворота, ч (медиана, от экстремума до смены знака)", color=INK2, fontsize=9)
    ax.set_ylabel("ложных переключений на ногу тренда", color=INK2, fontsize=9)
    ax.set_title("Метрики ТЗ на проверке 2021–26: эталон — ZigZag 6 ATR", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "quality.png", dpi=130, facecolor=BG)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    selection_vs_test()
    quality()
    d = variants()
    pd.set_option("display.width", 250)
    print(d.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
