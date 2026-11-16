"""H40, картинки по замечанию Тимура (созвон 17.11): не гистограмма, а сами случаи.

Три наши акции (SBER, GAZP, LKOH), входы 2025–2026, все сделки журнала, посчитанные в H40 (scripts/h40_trades.py,
reports/h40_trades/trades_min_main.csv: S42, S46, S48 и модель LKOH):

* overview_2025_2026.png — цена по часам и все сделки на ней. Сделка «под уровнем» (большой свежий уровень объёма по
  ходу сделки ближе 1 ATR) — крупный треугольник + сам уровень оранжевым отрезком от входа до выхода; остальные —
  мелкие точки. Цвет — итог сделки. В заголовке строки — число сделок и PF обеих групп.
* cases_2025_2026_<n>.png — каждая сделка под уровнем крупно: свечи, профиль объёма на момент входа, уровень, вход,
  выход, прошла ли цена уровень за время сделки.

    .venv/bin/python scripts/h40_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import h40_trades as h  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TICKERS = ["SBER", "GAZP", "LKOH"]
YEARS = (2025, 2026)
THR = 1.0                    # «под уровнем»: уровень по ходу сделки ближе 1 ATR (как в основном расчёте H40)
DAYS, X, W, HVN, RESET = 20, 2.0, 20, 0.0, 0.0   # параметры основного расчёта (tag min_main)
WIN, LOSS, FAINT = "#1f9d55", "#d64545", "#b9b7b1"
OUT = h.VIZ


def pf(x: pd.Series) -> str:
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    if len(x) < 5:
        return "мало для PF"
    return "PF ∞" if neg == 0 else f"PF {pos / neg:.2f}"


def load() -> pd.DataFrame:
    f = pd.read_csv(h.REP / "trades_min_main.csv", parse_dates=["open_dt", "close_dt"])
    f = f[f["tk"].isin(TICKERS) & f["year"].between(*YEARS)].copy()
    f["near"] = f["d_block"] <= THR
    f["lvl"] = np.where(f["side"] > 0, f["lvl_up"], f["lvl_dn"])
    return f.sort_values("open_dt").reset_index(drop=True)


def overview(f: pd.DataFrame, path: Path) -> Path:
    rows = [("SBER", False), ("GAZP", False), ("LKOH", False), ("LKOH", True)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 3.3 * len(rows)), facecolor=h.SURF)
    for ax, (tk, model) in zip(axes, rows):
        d = h.bars(tk)
        d = d[(d.index.year >= YEARS[0]) & (d.index.year <= YEARS[1])]
        g = f[(f["tk"] == tk) & ((f["strategy"] == "модель LKOH") == model)]
        ax.plot(d.index, d["close"], color=h.INK2, lw=0.6)
        oth, near = g[~g["near"]], g[g["near"]]
        ax.scatter(oth["open_dt"], oth["price"], s=9, color=np.where(oth["profit_pct"] > 0, WIN, LOSS), alpha=0.45,
                   lw=0, zorder=3)
        for r in near.itertuples():
            col = WIN if r.profit_pct > 0 else LOSS
            ax.plot([r.open_dt, r.close_dt], [r.lvl, r.lvl], color=h.C2, lw=2.2, solid_capstyle="butt", zorder=4)
            ax.scatter([r.open_dt], [r.price], s=70, marker="^" if r.side > 0 else "v", color=col,
                       edgecolor=h.INK, lw=0.7, zorder=5)
        name = f"{tk}, модель (лонг и шорт)" if model else f"{tk}, стратегии S42 / S46 / S48 (лонг)"
        ax.set_title(f"{name}: {len(g)} сделок · под уровнем {len(near)} ({pf(near['profit_pct'])}, "
                     f"в плюсе {int((near['profit_pct'] > 0).sum())}) · остальные {len(oth)} ({pf(oth['profit_pct'])}, "
                     f"в плюсе {int((oth['profit_pct'] > 0).sum())})", color=h.INK, fontsize=10, loc="left")
        h.style(ax)
        ax.margins(x=0.005)
    handles = [plt.Line2D([], [], marker="^", ls="", color=WIN, mec=h.INK, ms=8, label="сделка под уровнем, в плюсе"),
               plt.Line2D([], [], marker="^", ls="", color=LOSS, mec=h.INK, ms=8, label="сделка под уровнем, в минусе"),
               plt.Line2D([], [], color=h.C2, lw=2.2, label="большой уровень объёма (от входа до выхода)"),
               plt.Line2D([], [], marker="o", ls="", color=FAINT, ms=4, label="остальные сделки (цвет — итог)")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.975))
    fig.suptitle("H40: все сделки 2025–2026 на трёх акциях и те, что открыты под большим уровнем объёма "
                 "(ближе 1 ATR по ходу сделки)", color=h.INK, fontsize=12, x=0.01, y=0.997, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=120, facecolor=h.SURF)
    plt.close(fig)
    return path


def case_panel(axc, axp, d, r, res):
    i_e, i_x, prof, b_lo, big, p, a = res
    i0, i1 = max(0, i_e - 60), min(len(d), i_x + 12)
    h.candles(axc, d, i0, i1)
    pb = np.exp((np.arange(b_lo, b_lo + len(prof)) + 0.5) * h.STEP)
    for lp in pb[big]:
        axc.axhline(lp, color=h.C2, lw=0.6, alpha=0.35)
    axc.axhline(r.lvl, color=h.C2, lw=2.0)
    axc.axvspan(i_e - i0, min(i_x, i1 - 1) - i0, color=h.C1, alpha=0.07, lw=0)
    axc.plot([i_e - i0], [p], marker="^" if r.side > 0 else "v", color=h.INK, ms=8)
    xe = min(i_x, i1 - 1)
    axc.plot([xe - i0], [d["close"].to_numpy()[xe]], marker="X", ms=8,
             color=WIN if r.profit_pct > 0 else LOSS, mec=h.INK, mew=0.6)
    hi = d["high"].to_numpy()[i_e:i_x + 1].max() if i_x > i_e else p
    lo = d["low"].to_numpy()[i_e:i_x + 1].min() if i_x > i_e else p
    passed = hi > r.lvl if r.side > 0 else lo < r.lvl
    h.style(axc)
    axc.grid(False)
    axc.set_xticks([])
    side = "лонг" if r.side > 0 else "шорт"
    strat = "модель" if r.strategy.startswith("модель") else r.strategy.split()[0]
    axc.set_title(f"{r.tk} {strat} {side} {pd.Timestamp(r.open_dt):%d.%m.%y} · {r.profit_pct:+.1f}%\n"
                  f"уровень {r.d_block:.1f} ATR {'выше' if r.side > 0 else 'ниже'} · "
                  f"{'цена прошла уровень' if passed else 'не прошла уровень'}", color=h.INK, fontsize=8, loc="left")
    ylo = min(d["low"].to_numpy()[i0:i1].min(), r.lvl)
    yhi = max(d["high"].to_numpy()[i0:i1].max(), r.lvl)
    pad = 0.06 * (yhi - ylo)
    axc.set_ylim(ylo - pad, yhi + pad)
    sel = (pb >= ylo - pad) & (pb <= yhi + pad)
    axp.barh(pb[sel], prof[sel], height=pb[sel] * h.STEP, color=np.where(big[sel], h.C2, "#c9d6ea"), lw=0)
    axp.set_ylim(*axc.get_ylim())
    h.style(axp)
    axp.grid(False)
    axp.set_xticks([])
    axp.tick_params(labelleft=False)
    return passed


def cases(f: pd.DataFrame, per: int = 12) -> tuple[list[Path], pd.DataFrame]:
    near = f[f["near"]]
    items, rec = [], []
    for tk in TICKERS:
        g = near[near["tk"] == tk]
        if g.empty:
            continue
        d = h.bars(tk)
        st = h.Store(h.levels_min(tk), d.index)
        for r in g.itertuples():
            res = h.one_profile(d, st, r, DAYS, X, W, HVN, RESET)
            if res:
                items.append((d, r, res))
    items.sort(key=lambda t: t[1].open_dt)
    paths = []
    for s in range(0, len(items), per):
        chunk = items[s:s + per]
        nr = int(np.ceil(len(chunk) / 3))
        fig = plt.figure(figsize=(13, 3.0 * nr + 0.6), facecolor=h.SURF)
        gs = fig.add_gridspec(nr, 6, width_ratios=[3, 0.8, 3, 0.8, 3, 0.8], hspace=0.55, wspace=0.12)
        for j, (d, r, res) in enumerate(chunk):
            passed = case_panel(fig.add_subplot(gs[j // 3, 2 * (j % 3)]), fig.add_subplot(gs[j // 3, 2 * (j % 3) + 1]),
                                d, r, res)
            rec.append({"бумага": r.tk, "вход": f"{pd.Timestamp(r.open_dt):%d.%m.%Y %H:%M}", "стратегия": r.strategy,
                        "сторона": "лонг" if r.side > 0 else "шорт", "итог %": round(r.profit_pct, 2),
                        "до уровня, ATR": round(r.d_block, 2), "уровень": round(r.lvl, 2), "прошла уровень": passed})
        fig.suptitle(f"H40: сделки под большим уровнем объёма, SBER / GAZP / LKOH, 2025–2026 — лист {s // per + 1} "
                     f"из {int(np.ceil(len(items) / per))}.  Жирная оранжевая линия — уровень, бледные — другие большие "
                     f"уровни; ▲/▼ — вход, ✕ — выход (цвет — итог); голубая заливка — время в сделке; справа — профиль "
                     f"объёма за 20 дней до входа", color=h.INK, fontsize=9.5, x=0.01, ha="left", wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = OUT / f"cases_2025_2026_{s // per + 1}.png"
        fig.savefig(p, dpi=120, facecolor=h.SURF)
        plt.close(fig)
        paths.append(p)
    return paths, pd.DataFrame(rec)


def main() -> int:
    f = load()
    print(overview(f, OUT / "overview_2025_2026.png"))
    paths, tab = cases(f)
    for p in paths:
        print(p)
    tab.to_csv(h.REP / "cases_2025_2026.csv", index=False)
    print(f"сделок {len(f)}, под уровнем {int(f['near'].sum())}, на листах {len(tab)}, "
          f"прошли уровень {int(tab['прошла уровень'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
