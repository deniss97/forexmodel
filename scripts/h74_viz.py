"""Графики к H74: примеры размеченных фигур на свечах, число событий по годам, ход цены после пробоя, подбор порогов.

Оформление — как в scripts/viz_examples.py: светлый фон, синий / оранжевый, подписи по-русски, заголовок-вывод.
Считает заодно сводку хода цены после пробоя (reports/h74/forward_summary.csv).

    .venv/bin/py scripts/h74_viz.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import h74_patterns as H  # noqa: E402

ROOT = H.ROOT
REP = H.REP
OUT = H.OUT
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#8a8884"
PATS = H.PATS
ORDER = ["db", "tb", "ihs", "htf", "tri", "barr"]
SHORT = {"db": "двойное\nдно", "tb": "тройное\nдно", "ihs": "перевёрнутая\n«голова и плечи»", "htf": "high and\ntight flag",
         "tri": "восходящий\nтреугольник", "barr": "bump-and-run\nreversal"}
SHORT2 = {"db": "2 дно", "tb": "3 дно", "ihs": "ГиП", "htf": "флаг", "tri": "треуг.", "barr": "BARR"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def candles(ax, b: H.Bars, a: int, z: int) -> None:
    x = np.arange(a, z)
    up = b.c[a:z] >= b.o[a:z]
    ax.vlines(x[up], b.l[a:z][up], b.h[a:z][up], color=C1, lw=0.7)
    ax.vlines(x[~up], b.l[a:z][~up], b.h[a:z][~up], color=C2, lw=0.7)
    for m, c in ((up, C1), (~up, C2)):
        lo = np.minimum(b.o[a:z][m], b.c[a:z][m])
        ax.bar(x[m], np.maximum(np.abs(b.c[a:z][m] - b.o[a:z][m]), 1e-9), bottom=lo, width=0.75, color=c, lw=0)
    ticks = [i for i in range(a, z) if i % 25 == 0]
    ax.set_xticks(ticks, [pd.Timestamp(b.dt[i]).strftime("%m.%y") for i in ticks])
    ax.set_xlim(a - 1, z)


def examples(bars: dict[str, H.Bars], evs: pd.DataFrame, pat: str, n: int = 6) -> Path:
    """До шести примеров фигуры: сначала основные бумаги, потом разные годы и бумаги."""
    g = evs[evs["pat"] == pat].copy()
    g["main"] = ~g["ticker"].isin(H.MAIN)
    g = g.sort_values(["main", "year", "ticker"])
    pick, seen = [], set()
    for row in g.itertuples():
        key = (row.ticker, row.year // 3)
        if key in seen:
            continue
        seen.add(key)
        pick.append(row)
        if len(pick) == n:
            break
    for row in g.itertuples():                                     # если выборка не набралась — добираем подряд
        if len(pick) >= n:
            break
        if row.Index not in {p.Index for p in pick}:
            pick.append(row)
    rows = (len(pick) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.4 * rows), facecolor=SURF, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, e in zip(axes.ravel(), pick):
        ax.set_visible(True)
        b = bars[e.ticker]
        a, z = max(0, e.i0 - 20), min(b.n, e.brk + 35)
        candles(ax, b, a, z)
        for i, p in e.pts:
            ax.plot([i], [p], "o", ms=5, mfc="none", mec=INK, mew=1.2)
        for i1, p1, i2, p2 in e.lines:
            ax.plot([i1, i2], [p1, p2], color=INK, lw=1.1, ls=(0, (4, 3)))
        ax.axvline(e.brk, color=C1, lw=1.2)
        ax.annotate("пробой", (e.brk, b.h[a:z].max()), xytext=(3, -8), textcoords="offset points",
                    color=C1, fontsize=8, ha="left", va="top")
        style(ax)
        note = f"\n{e.note}" if e.note else ""
        ax.set_title(f"{e.ticker}, пробой {e.date:%d.%m.%Y} ({e.part}){note}", color=INK, fontsize=8.5, loc="left")
    fig.suptitle(f"H74, фигура «{PATS[pat]}»: примеры разметки на дневных свечах "
                 f"(круги — опорные точки, пунктир — уровень или линия пробоя)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = OUT / f"h74_examples_{pat}.png"
    fig.savefig(p, dpi=140, facecolor=SURF)
    plt.close(fig)
    return p


def counts_png(evs: pd.DataFrame, meta: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURF,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(ORDER))
    sel = [int(((evs["pat"] == p) & (evs["part"] == "выбор")).sum()) for p in ORDER]
    val = [int(((evs["pat"] == p) & (evs["part"] == "проверка")).sum()) for p in ORDER]
    axes[0].bar(x - 0.2, sel, width=0.4, color=C1, label="выбор 2015–2020")
    axes[0].bar(x + 0.2, val, width=0.4, color=C2, label="проверка 2021–2026")
    axes[0].axhline(H.MIN_EVENTS, color=INK, lw=1, ls=(0, (4, 3)))
    axes[0].annotate("порог плана: 50 событий на выборе", (len(ORDER) - 0.5, H.MIN_EVENTS), xytext=(0, 4),
                     textcoords="offset points", color=INK, fontsize=8, ha="right")
    axes[0].set_xticks(x, [SHORT[p] for p in ORDER], fontsize=7)
    axes[0].set_ylabel("число найденных фигур", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    style(axes[0])
    by_year = evs.pivot_table(index="year", columns="pat", values="date", aggfunc="count").fillna(0)
    bottom = np.zeros(len(by_year))
    colors = [C1, C2, "#7fb0e8", "#f2a279", C3, "#c9c7c2"]
    for p, c in zip(ORDER, colors):
        v = by_year[p].to_numpy() if p in by_year else np.zeros(len(by_year))
        axes[1].bar(by_year.index, v, bottom=bottom, width=0.8, color=c, label=PATS[p])
        bottom += v
    axes[1].axvline(2020.5, color=INK2, lw=1, ls=(0, (3, 3)))
    axes[1].annotate("выбор | проверка", (2020.5, bottom.max() * 0.45), xytext=(4, 0), textcoords="offset points",
                     color=INK2, fontsize=8, va="center")
    axes[1].set_ylabel("фигур за год, все 19 бумаг", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2)
    style(axes[1])
    enough = [p for p in ORDER if meta["chosen"][p]["enough"]]
    fig.suptitle(f"H74: разметка даёт достаточно фигур для {len(enough)} видов из 6 (не хватает только тройного дна) "
                 f"и держится по годам", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "h74_counts.png"
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p


def relax_png(relax: pd.DataFrame, meta: dict) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), facecolor=SURF)
    for ax, pat in zip(axes.ravel(), ORDER):
        g = relax[relax["pat"] == pat]
        x = np.arange(len(g))
        ch = meta["chosen"][pat]["tier"]
        cols = [C1 if t == ch else C3 for t in g["tier"]]
        ax.bar(x, g["выбор"], width=0.6, color=cols)
        ax.axhline(H.MIN_EVENTS, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(x, [t.replace(": ", ":\n").replace(", ", ",\n") for t in g["tier"]], fontsize=6.5)
        ax.set_title(PATS[pat], color=INK, fontsize=9, loc="left")
        style(ax)
    axes[0][0].set_ylabel("фигур на выборе 2015–2020", color=INK2, fontsize=9)
    axes[1][0].set_ylabel("фигур на выборе 2015–2020", color=INK2, fontsize=9)
    fig.suptitle("H74, шаг 4 плана: строгие («книжные») пороги дают 0–18 фигур за 6 лет на 19 бумагах — "
                 "синим отмечен порог, на котором событий хватает", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "h74_relax.png"
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p


def forward_png(fwd: pd.DataFrame, base: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    """Ход цены после пробоя в ATR против «среднего часа», по периодам."""
    periods = [("выбор 2015–2020", 2015, 2020), ("проверка 2021–2023", 2021, 2023), ("проверка 2024–2026", 2024, 2026)]
    hz = [6, 24, 72]
    rows = []
    for lab, a, z in periods:
        g = fwd[(fwd["year"] >= a) & (fwd["year"] <= z)]
        part = "выбор" if a < 2021 else "проверка"
        bs = base[base["part"] == part]
        for pat in ORDER:
            gg = g[g["pat"] == pat]
            for q in hz:
                m = gg[f"h{q}"]
                bq = bs[bs["h"] == q]
                b_mean = float(np.average(bq["mean"], weights=bq["n"])) if len(bq) else np.nan
                rows.append({"период": lab, "фигура": pat, "часов": q, "n": len(m),
                             "ход_ATR": float(m.mean()) if len(m) else np.nan,
                             "средний_час_ATR": b_mean,
                             "t": float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 else np.nan})
    s = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURF)
    x = np.arange(len(ORDER))
    for ax, (lab, a, z) in zip(axes, periods):
        g = s[s["период"] == lab]
        for k, (q, off, c) in enumerate(zip(hz, (-0.26, 0, 0.26), (C1, C2, C3))):
            v = [float(g[(g["фигура"] == p) & (g["часов"] == q)]["ход_ATR"].iloc[0]) for p in ORDER]
            ax.bar(x + off, v, width=0.24, color=c, label=f"{q} ч" if ax is axes[0] else None)
        bq = g[g["часов"] == 24]["средний_час_ATR"]
        ax.axhline(float(bq.iloc[0]) if len(bq) else 0, color=INK, lw=1, ls=(0, (4, 3)))
        ax.axhline(0, color=INK2, lw=0.8)
        ns = [int(g[(g["фигура"] == p) & (g["часов"] == 24)]["n"].iloc[0]) for p in ORDER]
        ax.set_xticks(x, [f"{SHORT2[p]}\n{n} шт." for p, n in zip(ORDER, ns)], fontsize=7.5)
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
        style(ax)
    axes[0].set_ylabel("ход цены после пробоя, в ATR часа", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, title="через", title_fontsize=8)
    axes[2].annotate("пунктир — обычный час рынка за 24 ч", (len(ORDER) - 0.5, 0), xytext=(0, 6),
                     textcoords="offset points", color=INK, fontsize=8, ha="right")
    fig.suptitle("H74: после пробоя цена идёт лучше обычного часа только у двойного дна, у флага и BARR — хуже\n"
                 "ГиП — перевёрнутая «голова и плечи», треуг. — восходящий треугольник, BARR — bump-and-run reversal",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "h74_forward.png"
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p, s


def main() -> None:
    import json

    OUT.mkdir(parents=True, exist_ok=True)
    evs = pd.read_pickle(REP / "events.pkl")
    meta = json.loads((REP / "meta.json").read_text(encoding="utf-8"))
    relax = pd.read_csv(REP / "relax.csv")
    fwd = pd.read_csv(REP / "forward.csv", parse_dates=["date"])
    base = pd.read_csv(REP / "baseline.csv")

    bars = {}
    for t in meta["tickers"]:
        bars[t] = H.Bars(t, H.to_daily(H.hourly(t)))
    print("свечи загружены", flush=True)

    for pat in ORDER:
        if (evs["pat"] == pat).any():
            print(examples(bars, evs, pat), flush=True)
    print(counts_png(evs, meta), flush=True)
    print(relax_png(relax, meta), flush=True)
    p, s = forward_png(fwd, base)
    s.to_csv(REP / "forward_summary.csv", index=False)
    print(p, flush=True)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)


if __name__ == "__main__":
    main()
