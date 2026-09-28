"""Картинки для docs/results/patterns_explained.md (простое объяснение импульсной стратегии).

Нужны: кэш баров лаборатории паттернов (reports/_cache/pattern_bars_*.pkl, создаёт
scripts/pattern_lab.py) и сделки walk-forward выхода 240 ч / 6 ATR
(reports/walk_forward/*_exit_240h6__impulse_6_3*_trades.csv, создаёт очередь из
docs/results/patterns.md §10).

    python scripts/plot_patterns_explained.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pattern_lab import add_features, bars, patterns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "patterns_explained"
WF = ROOT / "reports" / "walk_forward"

# палитра: справочные цвета (dataviz), светлая тема
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NAMES = {"silver": "Серебро", "gold": "Золото", "lkoh": "LKOH"}


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def fig(w, h, n=1):
    f, ax = plt.subplots(1, n, figsize=(w, h), facecolor=SURFACE)
    return f, ax


def example_trades():
    """Удачная и неудачная сделка серебра: цена, окно импульса, вход, трейлинг-стоп, выход."""
    h, _ = bars("silver")
    h = h.set_index("time")
    t = pd.read_csv(WF / "silver_exit_240h6__impulse_6_3_trades.csv", parse_dates=["signal_dt", "open_dt", "close_dt"])
    t = t[t["год"] >= 2021]
    win = t[t["profit_pct"] > 0].sort_values("profit_pct").iloc[int(len(t[t["profit_pct"] > 0]) * 0.8)]
    loss = t[t["exit_reason"] == "stop_loss"].sort_values("profit_pct").iloc[len(t[t["exit_reason"] == "stop_loss"]) // 2]
    f, axes = fig(12, 4.2, 2)
    for ax, tr, title in ((axes[0], win, "Удачная сделка"), (axes[1], loss, "Неудачная сделка (стоп)")):
        d = 1 if tr["side"] == "buy" else -1
        sig = tr["signal_dt"]
        w = h.loc[sig - pd.Timedelta("30h"): tr["close_dt"] + pd.Timedelta("24h")]
        ax.plot(w.index, w["close"], color=BLUE, linewidth=2, label="цена (закрытие часа)")
        ax.axvspan(sig - pd.Timedelta("5h"), sig + pd.Timedelta("1h"), color=ORANGE, alpha=0.15, lw=0)
        ax.annotate("импульс 6 ч", xy=(sig - pd.Timedelta("2h"), ax.get_ylim()[1]), xytext=(4, -4),
                    textcoords="offset points", color=INK2, fontsize=8, ha="left", va="top")
        # трейлинг-стоп по часовым барам (иллюстрация; в расчётах — по минутам)
        atr, entry = tr["atr_at_entry"], tr["open_price"]
        path = h.loc[tr["open_dt"].floor("h"): tr["close_dt"]]
        best = (path["high"] if d > 0 else path["low"]).cummax() if d > 0 else path["low"].cummin()
        best = best.clip(lower=entry) if d > 0 else best.clip(upper=entry)
        init = entry - d * 6 * atr
        on = d * (best - entry) >= 3 * atr
        stop = np.where(on, np.maximum(init, best - 6 * atr) if d > 0 else np.minimum(init, best + 6 * atr), init)
        ax.plot(path.index, stop, color=RED, linewidth=1.5, linestyle="--", label="трейлинг-стоп (6 ATR)")
        ax.scatter([tr["open_dt"]], [entry], s=70, color=AQUA, zorder=5, edgecolor=SURFACE, linewidth=2,
                   label="вход")
        ax.scatter([tr["close_dt"]], [tr["exit_price"]], s=70, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2,
                   label="выход")
        side = "покупка" if d > 0 else "продажа"
        style(ax, f"{title}: {side}, {tr['profit_pct']:+.2f}% за {tr['minutes_in_trade'] / 60:.0f} ч")
        ax.tick_params(axis="x", rotation=30)
    hnd, lab = axes[0].get_legend_handles_labels()
    f.legend(hnd, lab, frameon=False, fontsize=9, labelcolor=INK2, loc="lower center", ncol=4)
    f.tight_layout(rect=(0, 0.07, 1, 1))
    f.savefig(OUT / "1_example_trades.png", dpi=130, facecolor=SURFACE)


def impulse_continues():
    """Средний ход после импульса (в его сторону) против обычного момента, в ATR."""
    f, axes = fig(12, 3.8, 3)
    hours = np.arange(0, 121)
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
            fwd = (c[ok + k] - c[ok]) / atr[ok]
            pat.append(np.nanmean(d[ok] * fwd))
            allfwd = (c[k:] - c[: n - k]) / atr[: n - k]
            ctrl.append(np.nanmean(d[ok]) * np.nanmean(allfwd))     # тот же набор сторон, обычный момент
        ax.plot(hours, pat, color=BLUE, linewidth=2, label="после импульса")
        ax.plot(hours, ctrl, color=INK2, linewidth=2, linestyle="--", label="обычный момент, те же стороны")
        ax.axhline(0, color=GRID, linewidth=1)
        style(ax, NAMES[inst])
        ax.set_xlabel("часов после входа", color=INK2, fontsize=9)
        ax.text(hours[-1], pat[-1], f" {pat[-1]:+.1f} ATR", color=INK, fontsize=9, va="center")
    axes[0].set_ylabel("средний ход в сторону сделки, ATR", color=INK2, fontsize=9)
    hnd, lab = axes[0].get_legend_handles_labels()
    f.legend(hnd, lab, frameon=False, fontsize=9, labelcolor=INK2, loc="lower center", ncol=2)
    f.tight_layout(rect=(0, 0.07, 1, 1))
    f.savefig(OUT / "2_impulse_continues.png", dpi=130, facecolor=SURFACE)


def portfolio(variant: str) -> pd.DataFrame:
    ts = []
    for inst in ("silver", "gold", "lkoh"):
        t = pd.read_csv(WF / f"{inst}_exit_240h6__{variant}_trades.csv", parse_dates=["close_dt"])
        ts.append(t[["close_dt", "profit_pct", "год"]])
    a = pd.concat(ts).sort_values("close_dt")
    a["p"] = a["profit_pct"] / 3
    return a


def equity():
    f, ax = fig(11, 4)
    for var, color, lab in (("impulse_6_3", BLUE, "без проскальзывания"),
                            ("impulse_6_3_slip0.1", ORANGE, "проскальзывание 0.1% на входе")):
        a = portfolio(var)
        eq = a["p"].cumsum()
        ax.plot(a["close_dt"], eq, color=color, linewidth=2, label=lab)
        ax.text(a["close_dt"].iloc[-1], eq.iloc[-1], f" {eq.iloc[-1]:+.0f}%", color=INK, fontsize=9, va="center")
    ax.axhline(0, color=GRID, linewidth=1)
    style(ax, "Портфель серебро + золото + LKOH (по 1/3), накопленный результат после комиссии, %")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    f.tight_layout()
    f.savefig(OUT / "3_portfolio_equity.png", dpi=130, facecolor=SURFACE)


def years():
    a = portfolio("impulse_6_3")
    y = a.groupby("год")["p"].sum()
    f, ax = fig(11, 3.6)
    colors = [BLUE if v > 0 else RED for v in y.to_numpy()]
    ax.bar(y.index.astype(str), y.to_numpy(), color=colors, width=0.6)
    for x, v in zip(y.index.astype(str), y.to_numpy()):
        ax.text(x, v + (1 if v > 0 else -1), f"{v:+.1f}%", ha="center", va="bottom" if v > 0 else "top",
                color=INK, fontsize=9)
    ax.axhline(0, color=INK2, linewidth=1)
    style(ax, "Портфель по годам, % (2026 — до мая)")
    f.tight_layout()
    f.savefig(OUT / "4_portfolio_years.png", dpi=130, facecolor=SURFACE)


def main() -> int:
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    example_trades()
    impulse_continues()
    equity()
    years()
    print(f"готово: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
