"""Графики еженедельной презентации: крупные подписи, размеры под слайд 1920×1080.

Неделя 2026-09-28 – 2026-10-05 (отчёт №1): импульсная стратегия, проверки и запас прочности.
Источники — уже посчитанные прогоны:
  * сделки штатного walk-forward с исполнением стопа по гэпу, с проскальзыванием и без
    (reports/walk_forward/*_robust__impulse_6_3*_trades.csv);
  * сводка 8 инструментов (docs/results/walk_forward/universe_summary.csv);
  * запас прочности (reports/audit/robustness.csv);
  * 300 случайных тренд-гейтов (reports/trend_lab/impulse_gate_search.npz / .json);
  * важность групп признаков CatBoost серебра (docs/results/feature_importance/silver_groups.csv);
  * кэш баров лаборатории паттернов (reports/_cache/pattern_bars_*.pkl) — для «импульс продолжается».

    python scripts/weekly_charts.py [--out docs/results/weekly/2026-10-05]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "reports" / "walk_forward"
CORE = ("silver", "lkoh", "gazp", "sber")
NAMES = {"silver": "Серебро", "gold": "Золото", "lkoh": "LKOH", "gazp": "GAZP", "sber": "SBER",
         "moex": "MOEX", "mtss": "MTSS", "eth": "ETH"}

# палитра презентации
BG, INK, INK2, GRID = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc"
BLUE, ORANGE, GRAY = "#2459b3", "#c8641e", "#9aa3ae"
DPI = 200
FS = 18          # пунктов: на слайде ≈ 25 px


def figure(w_px: int, h_px: int, ncols: int = 1):
    f, ax = plt.subplots(1, ncols, figsize=(w_px / 100, h_px / 100), facecolor=BG)
    return f, ax


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=FS - 1, length=0, pad=8)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def save(f, out: Path, name: str):
    f.savefig(out / name, dpi=DPI, facecolor=BG)
    plt.close(f)


def core_trades(variant: str) -> pd.DataFrame:
    frames = []
    for inst in CORE:
        t = pd.read_csv(WF / f"{inst}_robust__{variant}_trades.csv", parse_dates=["open_dt", "close_dt"])
        frames.append(t.assign(inst=inst))
    return pd.concat(frames, ignore_index=True).sort_values("close_dt")


def equity(out: Path) -> dict:
    """Портфель 4 инструментов по 1/4: накопленный итог, % (простая сумма, без реинвестирования)."""
    f, ax = figure(1664, 600)
    style(ax)
    stats = {}
    for variant, label, color in (("impulse_6_3", "без проскальзывания", BLUE),
                                  ("impulse_6_3_slip0.1_xslip0.1", "проскальзывание 0.1% на входе и выходе", ORANGE)):
        t = core_trades(variant)
        eq = (t["profit_pct"] / len(CORE)).cumsum()
        dd = (eq - eq.cummax()).min()
        ax.plot(t["close_dt"], eq, color=color, linewidth=3, label=label)
        ax.text(t["close_dt"].iloc[-1], eq.iloc[-1], f"  {eq.iloc[-1]:+.0f}%", color=color, fontsize=FS + 2,
                va="center", fontweight="bold")
        by_year = (t.groupby("год")["profit_pct"].sum() / len(CORE))
        stats[variant] = {"итог": eq.iloc[-1], "просадка": dd, "лет_в_плюсе": int((by_year > 0).sum()),
                          "лет": len(by_year), "по_годам": by_year.round(1).to_dict(), "сделок": len(t)}
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("накопленный итог, %", color=INK2, fontsize=FS)
    ax.legend(frameon=False, fontsize=FS, labelcolor=INK, loc="upper left")
    ax.margins(x=0.01)
    f.subplots_adjust(left=0.07, right=0.92, top=0.96, bottom=0.1)
    save(f, out, "equity.png")
    return stats


def years(out: Path, by_year: dict) -> None:
    f, ax = figure(1664, 560)
    style(ax)
    ys = sorted(by_year)
    v = np.array([by_year[y] for y in ys])
    bars = ax.bar([str(y) for y in ys], v, color=[BLUE if x > 0 else ORANGE for x in v], width=0.62)
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, x + 1, f"{x:+.0f}%", ha="center", va="bottom", color=INK,
                fontsize=FS + 1, fontweight="bold")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(0, max(v) * 1.18)
    ax.set_ylabel("итог года, %", color=INK2, fontsize=FS)
    ax.grid(axis="x", visible=False)
    f.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.1)
    save(f, out, "years.png")


def instruments(out: Path) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "docs" / "results" / "walk_forward" / "universe_summary.csv")
    s = s[(s["вариант"] == "без") & s["инструмент"].isin(NAMES)].set_index("инструмент")
    order = s["итог"].sort_values().index
    f, ax = figure(1000, 660)
    style(ax)
    v = s.loc[order, "итог"].to_numpy()
    colors = [ORANGE if x < 0 else (BLUE if i in CORE else GRAY) for i, x in zip(order, v)]
    ax.barh([NAMES[i] for i in order], v, color=colors, height=0.62)
    for k, x in enumerate(v):
        ax.text(x + (6 if x >= 0 else -6), k, f"{x:+.0f}%", va="center", ha="left" if x >= 0 else "right",
                color=INK, fontsize=FS + 1, fontweight="bold")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-75, 310)
    ax.tick_params(axis="y", labelsize=FS + 1, colors=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("итог 2016–2026 после комиссии, %", color=INK2, fontsize=FS)
    f.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.13)
    save(f, out, "instruments.png")
    return s


def robustness(out: Path) -> pd.Series:
    r = pd.read_csv(ROOT / "reports" / "audit" / "robustness.csv")
    p = r[r["инструмент"].isin(CORE)].groupby("проскальзывание")["итог"].mean()
    f, ax = figure(1000, 660)
    style(ax)
    ax.axvspan(-0.02, 0.1, color=BLUE, alpha=0.08, lw=0)
    ax.text(0.04, -45, "рабочая зона", ha="center", va="center", color=BLUE, fontsize=FS, fontweight="bold")
    ax.plot(p.index, p.values, color=BLUE, linewidth=3.5, marker="o", markersize=11)
    for x, y in p.items():
        ax.text(x + 0.006, y + 7, f"{y:+.0f}%", ha="left", va="bottom", color=INK, fontsize=FS + 1, fontweight="bold")
    ax.axhline(0, color=INK2, linewidth=1.2)
    be = np.interp(0, p.values[::-1], p.index[::-1])              # пересечение нуля
    ax.annotate(f"безубыток ≈ {round(be * 20) / 20:.2f}%", xy=(be, 0), xytext=(be - 0.12, -60), color=ORANGE,
                fontsize=FS, fontweight="bold", arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    ax.set_xlim(-0.02, 0.33)
    ax.set_ylim(-80, p.max() * 1.15)
    ax.set_xticks(p.index)
    ax.set_xticklabels([f"{x:g}%" for x in p.index])
    ax.set_xlabel("проскальзывание на входе и на выходе, % на сторону", color=INK2, fontsize=FS)
    ax.set_ylabel("итог портфеля 4, %", color=INK2, fontsize=FS)
    f.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.14)
    save(f, out, "robustness.png")
    return p


def gates(out: Path) -> dict:
    """Итог портфеля 8 инструментов (импульс 6 ч / 3 ATR) у 300 случайных тренд-гейтов."""
    meta = json.loads((ROOT / "reports" / "trend_lab" / "impulse_gate_search.json").read_text(encoding="utf-8"))
    S = np.load(ROOT / "reports" / "trend_lab" / "impulse_gate_search.npz")["S"]
    q = [tuple(k) for k in meta["KX"]].index((6, 3.0))
    names = meta["names"]
    tot = S[:, 0, q].sum(axis=(-2, -1)) / S.shape[3]
    rnd = np.array([tot[i] for i, n in enumerate(names) if n.startswith("r")])
    no_gate, gate_c = tot[names.index("без гейта")], tot[names.index("гейт C")]
    f, ax = figure(1000, 660)
    style(ax)
    ax.hist(rnd, bins=np.arange(0, 160, 6), color=GRAY, edgecolor=BG, linewidth=1.5, label="случайные фильтры")
    for x, label, color, ls in ((np.median(rnd), f"медиана случайных: {np.median(rnd):+.0f}%", INK, ":"),
                                (no_gate, f"без фильтра: {no_gate:+.0f}%", ORANGE, "--"),
                                (gate_c, f"фильтр C: {gate_c:+.0f}%", BLUE, "-")):
        ax.axvline(x, color=color, linewidth=3.5, linestyle=ls, label=label)
    ax.legend(frameon=False, fontsize=FS, labelcolor=INK, loc="upper left")
    ax.set_xlabel("итог портфеля 8 инструментов 2015–2026, %", color=INK2, fontsize=FS)
    ax.set_ylabel("число фильтров", color=INK2, fontsize=FS)
    ax.grid(axis="x", visible=False)
    f.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.14)
    save(f, out, "gates.png")
    return {"медиана": float(np.median(rnd)), "без": float(no_gate), "C": float(gate_c),
            "процентиль_C": float((rnd < gate_c).mean() * 100), "n": len(rnd)}


def features(out: Path) -> pd.DataFrame:
    g = pd.read_csv(ROOT / "docs" / "results" / "feature_importance" / "silver_groups.csv")
    short = {"время (час, день недели)": "время суток, день",
             "волатильность (ATR, ширина канала, размах свечи)": "волатильность",
             "положение в диапазоне, растяжение, z-оценки": "положение в диапазоне",
             "тренд-фильтр 4ч": "тренд-фильтр 4ч", "MACD": "MACD",
             "скользящие средние (отклонение, спред)": "скользящие средние"}
    g = g[g["группа"].isin(short)].set_index("группа").loc[list(short)][::-1]
    f, ax = figure(1000, 660)
    style(ax)
    y = np.arange(len(g))
    ax.barh(y + 0.19, g["на_обучении"], height=0.36, color=GRAY, label="доля в прогнозе")
    ax.barh(y - 0.19, g["вне_выборки"], height=0.36, color=BLUE, label="польза на новых данных")
    for k, (a, b) in enumerate(zip(g["на_обучении"], g["вне_выборки"])):
        ax.text(max(a, 0) + 1, k + 0.19, f"{a:.0f}%", va="center", color=INK2, fontsize=FS - 1)
        ax.text(max(b, 0) + 1 if b >= 0 else 1, k - 0.19, f"{b:+.0f}%" if abs(b) >= 1 else f"{b:+.1f}%", va="center",
                color=ORANGE if b < 0 else BLUE, fontsize=FS - 1, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([short[i] for i in g.index], color=INK, fontsize=FS)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-8, 55)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=FS, labelcolor=INK, loc="lower right")
    ax.set_xlabel("%", color=INK2, fontsize=FS)
    f.subplots_adjust(left=0.37, right=0.98, top=0.97, bottom=0.12)
    save(f, out, "features.png")
    return g


def impulse_continues(out: Path) -> dict:
    """Средний ход после импульса в его сторону против обычного момента (те же стороны), в ATR."""
    from pattern_lab import add_features, bars, patterns

    f, axes = figure(1664, 560, 3)
    hours = np.arange(0, 121)
    res = {}
    for ax, inst in zip(axes, ("silver", "gold", "lkoh")):
        h, _ = bars(inst)
        h = add_features(h)
        d = patterns(h)["импульс 6/3.0"]
        g = h["gate"].to_numpy()
        ev = np.flatnonzero((d != 0) & (d == g))
        c, atr = h["close"].to_numpy(), h["atr_14"].to_numpy()
        n = len(c)
        pat, ctrl = [], []
        for k in hours:
            ok = ev[ev + k < n]
            pat.append(np.nanmean(d[ok] * (c[ok + k] - c[ok]) / atr[ok]))
            ctrl.append(np.nanmean(d[ok]) * np.nanmean((c[k:] - c[: n - k]) / atr[: n - k]))
        style(ax)
        ax.plot(hours, pat, color=BLUE, linewidth=3, label="после импульса")
        ax.plot(hours, ctrl, color=GRAY, linewidth=3, linestyle="--", label="обычный момент")
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(NAMES[inst], color=INK, fontsize=FS + 3, loc="left", fontweight="bold")
        ax.set_xticks([0, 40, 80, 120])
        ax.set_xlabel("часов после входа", color=INK2, fontsize=FS)
        ax.set_ylim(-0.05, 1.1)
        res[inst] = (pat[-1], ctrl[-1])
    axes[0].set_ylabel("ход в сторону сделки, ATR", color=INK2, fontsize=FS)
    axes[0].legend(frameon=False, fontsize=FS, labelcolor=INK, loc="upper left")
    f.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17, wspace=0.18)
    save(f, out, "impulse_continues.png")
    return res


def rule_trade(out: Path) -> dict:
    """Одна типичная удачная сделка серебра: импульс, вход, трейлинг-стоп, выход."""
    from pattern_lab import bars

    h, _ = bars("silver")
    h = h.set_index("time")
    t = pd.read_csv(WF / "silver_exit_240h6__impulse_6_3_trades.csv", parse_dates=["signal_dt", "open_dt", "close_dt"])
    t = t[t["год"] >= 2021]
    wins = t[t["profit_pct"] > 0].sort_values("profit_pct")
    tr = wins.iloc[int(len(wins) * 0.8)]
    d = 1 if tr["side"] == "buy" else -1
    sig = tr["signal_dt"]
    w = h.loc[sig - pd.Timedelta("30h"): tr["close_dt"] + pd.Timedelta("24h")]
    f, ax = figure(900, 640)
    style(ax)
    x = np.arange(len(w))                         # торговые часы подряд, без выходных
    pos = {ts: i for i, ts in enumerate(w.index)}
    ax.plot(x, w["close"], color=INK, linewidth=2.5, label="цена")
    i_sig = pos[sig.floor("h")] if sig.floor("h") in pos else int(np.searchsorted(w.index, sig))
    ax.axvspan(i_sig - 5, i_sig + 1, color=ORANGE, alpha=0.22, lw=0, label="импульс 6 ч")
    atr, entry = tr["atr_at_entry"], tr["open_price"]
    path = h.loc[tr["open_dt"].floor("h"): tr["close_dt"]]
    best = path["high"].cummax().clip(lower=entry) if d > 0 else path["low"].cummin().clip(upper=entry)
    init = entry - d * 6 * atr
    on = d * (best - entry) >= 3 * atr
    stop = np.where(on, np.maximum(init, best - 6 * atr) if d > 0 else np.minimum(init, best + 6 * atr), init)
    xp = np.array([pos[ts] for ts in path.index if ts in pos])
    ax.plot(xp, stop[: len(xp)], color=BLUE, linewidth=2.5, linestyle="--", label="стоп, подтягивается")
    i_in = int(np.searchsorted(w.index, tr["open_dt"]))
    i_out = min(int(np.searchsorted(w.index, tr["close_dt"])), len(w) - 1)
    ax.scatter([i_in], [entry], s=180, color=BLUE, zorder=5, edgecolor=BG, linewidth=2.5)
    ax.scatter([i_out], [tr["exit_price"]], s=180, color=INK, zorder=5, edgecolor=BG, linewidth=2.5)
    ax.annotate("вход", (i_in, entry), xytext=(12, -28 * d), textcoords="offset points", color=BLUE,
                fontsize=FS + 1, fontweight="bold")
    ax.annotate(f"выход {tr['profit_pct']:+.1f}%", (i_out, tr["exit_price"]), xytext=(-150, 22 * d),
                textcoords="offset points", color=INK, fontsize=FS + 1, fontweight="bold")
    ax.set_xticks([])
    ax.set_xlabel(f"торговые часы · серебро, {sig:%d.%m.%Y}", color=INK2, fontsize=FS)
    ax.legend(frameon=False, fontsize=FS - 1, labelcolor=INK, loc="upper left" if d > 0 else "lower left")
    f.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.08)
    save(f, out, "rule_trade.png")
    return {"дата": str(sig), "сторона": tr["side"], "итог": float(tr["profit_pct"]),
            "часов": float(tr["minutes_in_trade"] / 60)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "weekly" / "2026-10-05"))
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = equity(out)
    for k, v in st.items():
        print(k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items()})
    years(out, st["impulse_6_3"]["по_годам"])
    print("инструменты:", instruments(out)["итог"].round(1).to_dict())
    print("запас прочности:", robustness(out).round(1).to_dict())
    print("случайные гейты:", gates(out))
    print("признаки:\n", features(out).round(1).to_string())
    print("импульс продолжается (после / обычно, ATR за 120 ч):", impulse_continues(out))
    print("пример сделки:", rule_trade(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
