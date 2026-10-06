"""Импульс с реальными издержками брокера и дивидендами: что остаётся от стратегии.

Сделки — прогоны очереди reports/walk_forward/logs/queue_costs.sh (walk-forward 2016–2026, стоп по
гэпу, у акций цены скорректированы на дивиденды, сетка выхода «удержание × стоп/трейлинг»). Путь
сделки от издержек не зависит, поэтому издержки пересчитываются по готовым сделкам
(forexmodel/simulation/costs.recost) для нескольких профилей:
  * «прежний расчёт» — комиссия 0.04% за круг, без свопов;
  * «Альфа-Форекс» — спред вместо комиссии и свопы за перенос (configs/costs/alfaforex.yaml), серебро
    с шагом цены 0.001;
  * «Альфа-Форекс, серебро 0.0001» — то же, если серебро котируется с 4 знаками.

Что считается:
  1. эффект дивидендов (4 основных инструмента, выход 240 ч / 6 ATR, прежние издержки): прогоны
     *_robust (без корректировки) против *_costs (с корректировкой);
  2. по инструментам при выходе 240 ч / 6 ATR: итог, профит-фактор, без 2 лучших, куда уходит доход
     (спред, свопы лонгов и шортов), переносов на сделку;
  3. сетка выхода под издержками Альфа-Форекс: портфель, выбор по 2016–2020, проверка 2021–2026.
Сравнение — по профит-фактору и итогу без 2 лучших сделок, не только по сумме процентов.

    python scripts/broker_costs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "reports" / "walk_forward"
OUT = ROOT / "reports" / "costs"
DOC = ROOT / "docs" / "results" / "costs"
PROFILE = ROOT / "configs" / "costs" / "alfaforex.yaml"
INSTRUMENTS = ["silver", "lkoh", "gazp", "sber", "gold", "moex", "mtss", "eth"]
CORE = ["silver", "lkoh", "gazp", "sber"]
NAMES = {"silver": "Серебро", "gold": "Золото", "lkoh": "LKOH", "gazp": "GAZP", "sber": "SBER",
         "moex": "MOEX", "mtss": "MTSS", "eth": "ETH"}
EXITS = [(h, tr) for h in (48, 72, 120, 168, 240) for tr in (3, 4, 6)]
BASE_EXIT = (240, 6)
OLD = {"spread_pct": 0.04, "swap_long_pct": 0.0, "swap_short_pct": 0.0, "rollover_hour": 0, "triple_weekday": 2}

BG, INK, INK2, GRID = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc"
BLUE, ORANGE, GRAY, LIGHT = "#2459b3", "#c8641e", "#9aa3ae", "#8db3f2"
FS = 15


def profiles() -> dict[str, dict[str, dict]]:
    alfa = {i: profile_entry(PROFILE, i) for i in INSTRUMENTS}
    alfa4 = dict(alfa, silver=profile_entry(PROFILE, "silver_4dp"))
    return {"прежний расчёт": {i: OLD for i in INSTRUMENTS}, "Альфа-Форекс": alfa,
            "Альфа-Форекс, серебро 0.0001": alfa4}


def trades(inst: str, h: int, tr: int, run: str = "costs") -> pd.DataFrame | None:
    p = WF / f"{inst}_{run}__impulse_6_3_h{h}_tr{tr}_trades.csv"
    if run == "robust":
        p = WF / f"{inst}_robust__impulse_6_3_trades.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p, parse_dates=["open_dt", "close_dt"])
    return t.rename(columns={"год": "year"})


def stats(p: pd.Series) -> dict:
    loss = -p[p < 0].sum()
    ex2 = p.drop(p.nlargest(2).index) if len(p) > 2 else p
    return {"сделок": len(p), "итог": p.sum(), "без_2_лучших": ex2.sum(),
            "PF": p[p > 0].sum() / loss if loss > 0 else np.inf,
            "t": p.mean() / (p.std() / np.sqrt(len(p))) if len(p) > 2 and p.std() > 0 else np.nan}


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=FS - 1, length=0, pad=6)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main() -> int:
    pd.set_option("display.width", 250)
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    prof = profiles()
    have = [i for i in INSTRUMENTS if trades(i, *BASE_EXIT) is not None]
    print("инструменты с готовыми прогонами:", have)
    lines = []

    # 1. дивиденды: прежние издержки, выход 240/6, без корректировки (robust) и с ней (costs)
    rows = []
    for inst in [i for i in CORE if i != "silver" and i in have]:
        a, b = trades(inst, 0, 0, "robust"), trades(inst, *BASE_EXIT)
        if a is None:
            continue
        sa, sb = stats(a["profit_pct"]), stats(b["profit_pct"])
        rows.append({"инструмент": inst, "без дивидендов": sa["итог"], "с дивидендами": sb["итог"],
                     "PF без": sa["PF"], "PF с": sb["PF"], "сделок без": sa["сделок"], "сделок с": sb["сделок"]})
    div = pd.DataFrame(rows)
    if len(div):
        div.to_csv(OUT / "dividends_effect.csv", index=False)
        lines.append("1. ДИВИДЕНДЫ (выход 240 ч / 6 ATR, комиссия 0.04%, стоп по гэпу), итог %:\n" + div.round(2).to_string(index=False))

    # 2. по инструментам при базовом выходе: профили издержек
    rows, years = [], []
    for name, pr in prof.items():
        for inst in have:
            t = recost(trades(inst, *BASE_EXIT), pr[inst])
            s = stats(t["profit_pct"])
            long_ = t["side"] == "buy"
            rows.append({"профиль": name, "инструмент": inst, **s, "до_издержек": t["gross_unit_pct"].sum(),
                         "спред": -t["commission_pct"].sum(), "своп_лонгов": t.loc[long_, "swap_pct"].sum(),
                         "своп_шортов": t.loc[~long_, "swap_pct"].sum(), "переносов_на_сделку": t["rollovers"].mean(),
                         "лет_в_плюсе": f"{int((t.groupby('year')['profit_pct'].sum() > 0).sum())}/{t['year'].nunique()}"})
            years.append(t.groupby("year")["profit_pct"].sum().rename((name, inst)))
    inst_tab = pd.DataFrame(rows)
    inst_tab.to_csv(OUT / "by_instrument.csv", index=False)
    by_year = pd.concat(years, axis=1)
    lines.append("\n2. ПО ИНСТРУМЕНТАМ, выход 240 ч / 6 ATR, 2016–2026 (ETH 2021–2026), % после издержек:")
    for name in prof:
        x = inst_tab[inst_tab["профиль"] == name].drop(columns="профиль").set_index("инструмент")
        port4 = x.loc[[i for i in CORE if i in x.index], "итог"].mean()
        port8 = x["итог"].mean()
        lines.append(f"\n--- {name}: портфель 4 {port4:+.1f}%, портфель {len(x)} {port8:+.1f}%\n" + x.round(2).to_string())

    # 3. сетка выхода под издержками Альфа-Форекс: портфель 4 и все, выбор 2016–2020, проверка 2021–2026
    grid = []
    for name in ("прежний расчёт", "Альфа-Форекс", "Альфа-Форекс, серебро 0.0001"):
        pr = prof[name]
        for h, trl in EXITS:
            per = {}
            for inst in have:
                t = trades(inst, h, trl)
                if t is None:
                    continue
                per[inst] = recost(t, pr[inst])
            if not per:
                continue
            for pname, members in (("портфель 4", [i for i in CORE if i in per]), ("все", list(per))):
                pool = pd.concat([per[i].assign(inst=i) for i in members])
                w = 1 / len(members)
                for period, q in (("2016–2020", pool["year"] < 2021), ("2021–2026", pool["year"] >= 2021), ("всё", slice(None))):
                    p = pool.loc[q, "profit_pct"] * w
                    s = stats(p)
                    grid.append({"профиль": name, "портфель": pname, "выход": f"{h} ч / {trl} ATR", "h": h, "tr": trl,
                                 "период": period, "итог": s["итог"], "без_2_лучших": s["без_2_лучших"], "PF": s["PF"],
                                 "t": s["t"], "сделок": s["сделок"],
                                 "переносов_на_сделку": pool.loc[q, "rollovers"].mean()})
    G = pd.DataFrame(grid)
    G.to_csv(OUT / "exit_grid.csv", index=False)
    for name in ("Альфа-Форекс", "Альфа-Форекс, серебро 0.0001", "прежний расчёт"):
        for pname in ("портфель 4", "все"):
            g = G[(G["профиль"] == name) & (G["портфель"] == pname)]
            if not len(g):
                continue
            piv = g.pivot_table(index="выход", columns="период", values=["итог", "PF"], sort=False)
            sel = g[g["период"] == "2016–2020"].sort_values("PF", ascending=False).iloc[0]
            chk = g[(g["период"] == "2021–2026") & (g["выход"] == sel["выход"])].iloc[0]
            base = g[(g["период"] == "2021–2026") & (g["h"] == BASE_EXIT[0]) & (g["tr"] == BASE_EXIT[1])].iloc[0]
            lines.append(f"\n3. СЕТКА ВЫХОДА · {name} · {pname}: итог % и PF по периодам\n" + piv.round(2).to_string()
                         + f"\n   выбор по PF 2016–2020: {sel['выход']} (PF {sel['PF']:.2f}) → 2021–2026 итог {chk['итог']:+.1f}%, "
                           f"PF {chk['PF']:.2f}; базовый 240 ч / 6 ATR в 2021–2026: {base['итог']:+.1f}%, PF {base['PF']:.2f}")
    # 4. порог безубыточности по переносу: при каком свопе (одинаковом для лонга и шорта, % в день)
    # сделка в среднем в нуле — при спреде Альфа-Форекс и при спреде 0.04% (биржа, своя позиция)
    alfa = prof["Альфа-Форекс"]
    rows = []
    for inst in have:
        t = recost(trades(inst, *BASE_EXIT), alfa[inst])
        g, n = t["gross_unit_pct"].mean(), t["rollovers"].mean()
        longs = (t["side"] == "buy").mean()
        rows.append({"инструмент": inst, "до_издержек_на_сделку": g, "переносов_на_сделку": n, "доля_лонгов": longs,
                     "спред_Альфа": alfa[inst]["spread_pct"],
                     "своп_Альфа_лонг": alfa[inst]["swap_long_pct"], "своп_Альфа_шорт": alfa[inst]["swap_short_pct"],
                     "безубыток_своп_при_спреде_Альфа": (g - alfa[inst]["spread_pct"]) / n,
                     "безубыток_своп_при_0.04": (g - 0.04) / n,
                     # биржа: лонг своей позицией без переноса, платит только шорт
                     "безубыток_своп_шорта_если_лонг_бесплатно": (g - 0.04) / (n * (1 - longs)) if longs < 1 else np.nan})
    be = pd.DataFrame(rows).set_index("инструмент")
    be.to_csv(OUT / "breakeven_swap.csv")
    lines.append("\n4. БЕЗУБЫТОК ПО ПЕРЕНОСУ (выход 240 ч / 6 ATR), % цены в день:\n" + be.round(4).to_string())
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    plots(inst_tab, G, div, by_year, be)
    return 0


def plots(inst_tab: pd.DataFrame, G: pd.DataFrame, div: pd.DataFrame, by_year: pd.DataFrame, be: pd.DataFrame) -> None:
    # E. сколько может стоить перенос: безубыточный своп против свопа Альфа-Форекс
    order = [i for i in INSTRUMENTS if i in be.index]
    f, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    style(ax)
    k = np.arange(len(order))
    alfa_cost = -(be.loc[order, "своп_Альфа_лонг"] * be.loc[order, "доля_лонгов"]
                  + be.loc[order, "своп_Альфа_шорт"] * (1 - be.loc[order, "доля_лонгов"]))
    ax.bar(k - 0.2, be.loc[order, "безубыток_своп_при_спреде_Альфа"].clip(lower=0), 0.4, color=BLUE,
           label="стратегия в нуле при таком свопе (спред Альфа-Форекс)")
    ax.bar(k + 0.2, alfa_cost, 0.4, color=ORANGE, label="своп Альфа-Форекс (средний по сторонам сделок)")
    for kk, (a, b) in enumerate(zip(be.loc[order, "безубыток_своп_при_спреде_Альфа"], alfa_cost)):
        ax.text(kk - 0.2, max(a, 0) + 0.002, f"{a:.3f}", ha="center", color=BLUE, fontsize=FS - 2, fontweight="bold")
        ax.text(kk + 0.2, b + 0.002, f"{b:.3f}", ha="center", color=ORANGE, fontsize=FS - 2)
    ax.set_xticks(k)
    ax.set_xticklabels([NAMES[i] for i in order], fontsize=FS, color=INK)
    ax.set_ylabel("% цены за перенос (день)", color=INK2, fontsize=FS)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=FS - 1, labelcolor=INK, loc="upper right")
    f.tight_layout()
    f.savefig(DOC / "breakeven_swap.png", dpi=150, facecolor=BG)
    plt.close(f)

    # A. куда уходит доход: до издержек → спред → свопы → итог, по инструментам (Альфа-Форекс)
    x = inst_tab[inst_tab["профиль"] == "Альфа-Форекс"].set_index("инструмент")
    old = inst_tab[inst_tab["профиль"] == "прежний расчёт"].set_index("инструмент")["итог"]
    order = [i for i in INSTRUMENTS if i in x.index]
    f, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    style(ax)
    k = np.arange(len(order))
    w = 0.2
    ax.bar(k - 1.5 * w, x.loc[order, "до_издержек"], w, color=GRAY, label="до издержек")
    ax.bar(k - 0.5 * w, old.loc[order], w, color=LIGHT, label="прежний расчёт (комиссия 0.04%)")
    ax.bar(k + 0.5 * w, x.loc[order, "спред"] + x.loc[order, "своп_лонгов"] + x.loc[order, "своп_шортов"], w,
           color=ORANGE, label="спред и свопы Альфа-Форекс")
    ax.bar(k + 1.5 * w, x.loc[order, "итог"], w, color=BLUE, label="итог с издержками Альфа-Форекс")
    for kk, v in zip(k, x.loc[order, "итог"]):
        ax.text(kk + 1.5 * w, v + (6 if v >= 0 else -6), f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top",
                color=BLUE, fontsize=FS - 1, fontweight="bold")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(k)
    ax.set_xticklabels([NAMES[i] for i in order], fontsize=FS, color=INK)
    ax.set_ylabel("% за 2016–2026 (ETH — с 2021)", color=INK2, fontsize=FS)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=FS - 1, labelcolor=INK, loc="upper right", ncol=2)
    f.tight_layout()
    f.savefig(DOC / "where_it_goes.png", dpi=150, facecolor=BG)
    plt.close(f)

    # B. сетка выхода: PF портфеля 4 по периодам (Альфа-Форекс)
    for name, fname in (("Альфа-Форекс", "exit_grid_pf.png"), ("Альфа-Форекс, серебро 0.0001", "exit_grid_pf_4dp.png")):
        g = G[(G["профиль"] == name) & (G["портфель"] == "портфель 4")]
        if not len(g):
            continue
        f, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
        for ax, period in zip(axes, ("2016–2020", "2021–2026")):
            p = g[g["период"] == period].pivot_table(index="tr", columns="h", values="PF")
            im = ax.imshow(p.to_numpy(), cmap="RdBu", vmin=0.6, vmax=1.4, aspect="auto")
            for (r, c), v in np.ndenumerate(p.to_numpy()):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=FS, color=INK if 0.8 < v < 1.25 else BG,
                        fontweight="bold")
            ax.set_xticks(range(p.shape[1]))
            ax.set_xticklabels([f"{h} ч" for h in p.columns], fontsize=FS - 1, color=INK2)
            ax.set_yticks(range(p.shape[0]))
            ax.set_yticklabels([f"{t} ATR" for t in p.index], fontsize=FS - 1, color=INK2)
            ax.set_title(f"{period}: профит-фактор", fontsize=FS + 1, color=INK, loc="left")
            ax.set_xlabel("максимальное удержание", fontsize=FS - 1, color=INK2)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_ylabel("стоп и трейлинг", fontsize=FS - 1, color=INK2)
        f.patch.set_facecolor(BG)
        f.tight_layout()
        f.savefig(DOC / fname, dpi=150, facecolor=BG)
        plt.close(f)

    # C. портфель 4 по годам: прежний расчёт и Альфа-Форекс (базовый выход)
    cols = {name: [c for c in by_year.columns if c[0] == name and c[1] in CORE] for name in ("прежний расчёт", "Альфа-Форекс")}
    if all(cols.values()):
        f, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
        style(ax)
        yrs = by_year.index.to_numpy()
        for j, (name, color) in enumerate((("прежний расчёт", LIGHT), ("Альфа-Форекс", BLUE))):
            v = by_year[cols[name]].sum(axis=1) / len(cols[name])
            ax.bar(yrs + (j - 0.5) * 0.38, v, 0.38, color=color, label=name)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(yrs)
        ax.set_ylabel("итог года, портфель 4, %", color=INK2, fontsize=FS)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=FS, labelcolor=INK)
        f.tight_layout()
        f.savefig(DOC / "years_old_vs_alfa.png", dpi=150, facecolor=BG)
        plt.close(f)

    # D. дивиденды
    if len(div):
        f, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
        style(ax)
        k = np.arange(len(div))
        ax.bar(k - 0.2, div["без дивидендов"], 0.4, color=GRAY, label="цены без корректировки")
        ax.bar(k + 0.2, div["с дивидендами"], 0.4, color=BLUE, label="скорректированы на дивиденды")
        for kk, (a, b) in enumerate(zip(div["без дивидендов"], div["с дивидендами"])):
            ax.text(kk - 0.2, a + 3, f"{a:+.0f}", ha="center", color=INK2, fontsize=FS - 1)
            ax.text(kk + 0.2, b + 3, f"{b:+.0f}", ha="center", color=BLUE, fontsize=FS - 1, fontweight="bold")
        ax.set_xticks(k)
        ax.set_xticklabels([NAMES[i] for i in div["инструмент"]], fontsize=FS, color=INK)
        ax.set_ylabel("итог 2016–2026, %", color=INK2, fontsize=FS)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=FS - 1, labelcolor=INK)
        f.tight_layout()
        f.savefig(DOC / "dividends_effect.png", dpi=150, facecolor=BG)
        plt.close(f)


if __name__ == "__main__":
    raise SystemExit(main())
