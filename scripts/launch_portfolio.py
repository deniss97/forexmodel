"""Портфель для запуска на бирже: S42 + S46 + S48 на широком наборе акций и капитал под целые лоты (гипотеза H67).

S42 — импульс на SBER, GAZP, LKOH (сделки журнала); S46 «пробой 55 дней» и S48 «пробой 20 дней + 0.25 ATR» — на 19
акциях: SBER, GAZP, LKOH и 16 новых из H66 (журнал S46/S48 и S53/S55). Только лонг, издержки биржи, дивиденды.
Капитал поровну между стратегиями, внутри стратегии — бумаги поровну (у каждой стратегии свой список бумаг, поэтому
веса считаются здесь, а не в portfolio.combine). Метрики — forexmodel/evaluation/portfolio.py, как в
iterator_walkthrough.md: портфель сравнивается с одним S42 по t месяцев и итогу к просадке.

Целые лоты: лот — по Т-Инвестициям (lots.json, 24.10), стоимость лота — лот × цена входа сделки (цены с поправкой на
сплиты и дивиденды — в сегодняшних единицах бумаги). Доля бумаги в капитале постоянна; сделка, на которую не хватает
лота, пропускается; остаток меньше лота лежит деньгами. Загрузка капитала — сумма позиций по часам (у S46 размер по
риску, до 3× доли бумаги): больше 1 — нужно плечо.

    python scripts/launch_portfolio.py
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, NEW_STOCKS, ROOT, load_catalog, load_trades  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import TEST_START  # noqa: E402

OUT = ROOT / "docs" / "results" / "launch_portfolio"
DOC = ROOT / "docs" / "results" / "launch_portfolio.md"
LAUNCH = ["sber", "gazp", "lkoh"]
WIDE = LAUNCH + NEW_STOCKS
END = pd.Timestamp("2026-05-01")          # общий конец истории S42 в журнале: сделки, открытые раньше
VARIANTS = {"A: S42 один": {"S42": 1.0},
            "B: S42 + S46": {"S42": 0.5, "S46": 0.5},
            "C: S42 + S48": {"S42": 0.5, "S48": 0.5},
            "D: S42 + S46 + S48": {"S42": 1 / 3, "S46": 1 / 3, "S48": 1 / 3},
            "D16: то же, S46 и S48 только на 16 новых": {"S42": 1 / 3, "S46·16": 1 / 3, "S48·16": 1 / 3}}
CAPITALS = [20_000, 50_000, 100_000, 300_000, 1_000_000]
PERIODS = ("выбор", "проверка")
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def strategies() -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Сделки стратегий и их бумаги (вес бумаги — от этого списка, а не от того, были ли на ней сделки)."""
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    out = {"S42": (s42[s42["instrument"].isin(LAUNCH)], LAUNCH)}
    for code, k in (("S46", 2), ("S48", 4)):
        t = load_trades({"id": code, "sources": {i: [f"{i}_wt{k}__eng"] for i in WIDE}, "costs": {"profile": "exchange"}})
        out[code] = (t, WIDE)
        out[f"{code}·16"] = (t[t["instrument"].isin(NEW_STOCKS)], NEW_STOCKS)
    return {k: (t[t["open_dt"] < END].copy(), insts) for k, (t, insts) in out.items()}


def portfolio(parts: dict, shares: dict) -> pd.DataFrame:
    """Сделки портфеля с весом w — доля капитала стратегии на одну её бумагу."""
    xs = []
    for name, share in shares.items():
        t, insts = parts[name]
        xs.append(t.assign(strategy=name, w=share / len(insts)))
    return pd.concat(xs, ignore_index=True)


def exposure(x: pd.DataFrame) -> dict:
    """Загрузка капитала по часам: сумма w × размер открытых позиций (1 — весь капитал в бумагах)."""
    grid = pd.date_range(x["open_dt"].min().floor("h"), x["close_dt"].max().ceil("h"), freq="h")
    load = np.zeros(len(grid))
    a = grid.searchsorted(x["open_dt"].dt.floor("h"))
    b = grid.searchsorted(x["close_dt"].dt.floor("h"))
    size = x["size"].fillna(1.0).to_numpy(float) if "size" in x else np.ones(len(x))
    for i, j, v in zip(a, b, x["w"].to_numpy() * size):
        load[i:max(j, i + 1)] += v
    on = load[load > 0]
    return {"макс": float(load.max()), "95%": float(np.percentile(on, 95)) if len(on) else 0.0,
            "ср": float(on.mean()) if len(on) else 0.0, "часов > 1": float((load > 1).mean() * 100)}


def in_lots(x: pd.DataFrame, capital: float, lots: dict) -> pd.DataFrame:
    """Те же сделки целыми лотами: позиция = доля бумаги × размер, округлённая вниз до лота; меньше лота — пропуск."""
    lot = x["instrument"].str.upper().map({k: v["lot"] for k, v in lots.items()}).to_numpy(float)
    size = x["size"].fillna(1.0).to_numpy(float) if "size" in x else np.ones(len(x))
    want = capital * x["w"].to_numpy() * size                       # рублей в позицию
    cost = lot * x["open_price"].to_numpy(float)                    # рублей за лот
    got = np.floor(want / cost) * cost
    return x.assign(profit_pct=x["profit_pct"].to_numpy() * np.where(want > 0, got / want, 0), taken=got > 0)


def f(v, d: int = 2, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lots = json.loads((OUT / "lots.json").read_text(encoding="utf-8"))
    parts = strategies()
    rows, curves = [], {}
    for name, shares in VARIANTS.items():
        x = portfolio(parts, shares)
        r = {"портфель": name}
        for per in PERIODS:
            m = pf.metrics(x, per)
            r.update({f"{per}: {k}": m.get(k) for k in ("в_год", "итог", "PF", "без_2_лучших", "просадка", "итог_к_просадке", "t_мес")})
        others = {k: v for k, v in shares.items() if k != "S42"}
        if others:
            second = portfolio(parts, {k: v / sum(others.values()) for k, v in others.items()})
            base = portfolio(parts, {"S42": 1.0})
            for per in PERIODS:
                r[f"{per}: корр. месяцев с S42"] = pf.overlap(second, base, per).get("корр_мес")
        r.update({f"загрузка: {k}": v for k, v in exposure(x).items()})
        rows.append(r)
        m = pf.monthly(x).cumsum()
        curves[name] = m
    res = pd.DataFrame(rows)

    # капитал и целые лоты — для портфелей A и D
    cap_rows = []
    for name in ("A: S42 один", "D: S42 + S46 + S48"):
        x = portfolio(parts, VARIANTS[name])
        for c in CAPITALS:
            y = in_lots(x, c, lots)
            r = {"портфель": name, "капитал": c, "сделок взято, %": y["taken"].mean() * 100}
            for per in PERIODS:
                m = pf.metrics(y[y["taken"]], per)
                r.update({f"{per}: итог": m.get("итог"), f"{per}: t_мес": m.get("t_мес"), f"{per}: итог_к_просадке": m.get("итог_к_просадке")})
            cap_rows.append(r)
    cap = pd.DataFrame(cap_rows)
    # бумаги, на которые при каждом капитале не хватает лота даже на одну позицию D
    w_wide = 1 / 3 / len(WIDE)
    need = {t.upper(): v["lot_rub"] / w_wide for t, v in lots.items() if t.lower() in WIDE}

    res.to_csv(OUT / "variants.csv", index=False)
    cap.to_csv(OUT / "capital.csv", index=False)
    plot_equity(curves)
    plot_capital(cap)
    write_doc(res, cap, lots, need)
    return 0


def _axes(ax) -> None:
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)


def plot_equity(curves: dict) -> None:
    """Накопленный результат по месяцам: один S42 против портфеля D (и D16 — без бумаг, где выбирали S46 и S48)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=BG)
    _axes(ax)
    style = {"A: S42 один": (GRAY, "-", 2.0), "D: S42 + S46 + S48": (BLUE, "-", 2.4),
             "D16: то же, S46 и S48 только на 16 новых": (BLUE, "--", 1.6)}
    for name, (color, ls, lw) in style.items():
        m = curves[name]
        x = m.index.to_timestamp()
        ax.plot(x, m.to_numpy(), color=color, linestyle=ls, linewidth=lw, label=name)
        ax.annotate(name.split(":")[0], (x[-1], m.iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=8)
    ax.axvline(TEST_START, color=INK2, linewidth=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=INK2, fontsize=8.5)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("накоплено, % капитала", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)


def plot_capital(cap: pd.DataFrame) -> None:
    """Итог портфеля D на проверке целыми лотами при разном капитале; линия — без ограничения лотами."""
    d = cap[cap["портфель"] == "D: S42 + S46 + S48"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=BG)
    _axes(ax)
    ax.grid(False, axis="x")
    labels = [f"{c // 1000:,} тыс.".replace(",", " ") if c < 1_000_000 else "1 млн" for c in d["капитал"]]
    bars = ax.bar(labels, d["проверка: итог"], color=BLUE, width=0.6)
    for b, share in zip(bars, d["сделок взято, %"]):
        ax.annotate(f"{share:.0f}% сделок", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK2, fontsize=8)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("итог на проверке, % капитала", color=INK2, fontsize=9)
    ax.set_title("Портфель D целыми лотами: итог 2021–2026 при разном капитале", color=INK, fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "capital.png", dpi=110, facecolor=BG)
    plt.close(fig)


def write_doc(res: pd.DataFrame, cap: pd.DataFrame, lots: dict, need: dict) -> None:
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    A = res.iloc[0]
    D = res[res["портфель"] == "D: S42 + S46 + S48"].iloc[0]
    crit = {f"t месяцев лучше S42: {p}": D[f"{p}: t_мес"] > A[f"{p}: t_мес"] for p in PERIODS}
    crit.update({f"итог к просадке лучше S42: {p}": D[f"{p}: итог_к_просадке"] > A[f"{p}: итог_к_просадке"] for p in PERIODS})
    crit.update({f"корр. месяцев с S42 < 0.5: {p}": (D[f"{p}: корр. месяцев с S42"] or 1) < 0.5 for p in PERIODS})
    md = ["# Портфель для запуска: S42 + S46 + S48 на широком наборе акций (H67)", "",
          "S42 — импульс на SBER, GAZP, LKOH; S46 «пробой 55 дней» и S48 «пробой 20 дней + 0.25 ATR» — на 19 акциях (SBER, "
          "GAZP, LKOH и 16 новых из [H66](new_stocks.md)). Только лонг, издержки биржи, цены с поправкой на дивиденды. Капитал "
          "поровну между стратегиями, внутри стратегии — бумаги поровну. Выбор 2016–2020, проверка 2021 — апрель 2026 (общий "
          "конец истории S42). Критерий записан в банк (H67) до расчёта. Расчёт — `scripts/launch_portfolio.py`.", "",
          "![Накопленный результат: один S42 против портфеля D](launch_portfolio/equity.png)", "", verdict, "",
          "## Критерий H67", "", "| пункт | выполнен |", "|---|---|"]
    md += [f"| {k} | {'✅' if v else '❌'} |" for k, v in crit.items()]
    md += ["", "## Портфели", "",
           "| портфель | выбор: сделок в год | итог % | PF | просадка | итог / просадка | t мес | проверка: сделок в год | итог % | PF | "
           "без 2 лучших | просадка | итог / просадка | t мес | корр. месяцев с S42 (выбор / проверка) | загрузка капитала: макс / 95% |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        corr = "—" if "выбор: корр. месяцев с S42" not in r or pd.isna(r.get("выбор: корр. месяцев с S42")) else \
            f"{f(r['выбор: корр. месяцев с S42'])} / {f(r['проверка: корр. месяцев с S42'])}"
        md.append(f"| {r['портфель']} | {f(r['выбор: в_год'], 0)} | {f(r['выбор: итог'], 1, True)} | {f(r['выбор: PF'])} | "
                  f"{f(r['выбор: просадка'], 1)} | {f(r['выбор: итог_к_просадке'])} | {f(r['выбор: t_мес'])} | {f(r['проверка: в_год'], 0)} | "
                  f"{f(r['проверка: итог'], 1, True)} | {f(r['проверка: PF'])} | {f(r['проверка: без_2_лучших'], 1, True)} | "
                  f"{f(r['проверка: просадка'], 1)} | {f(r['проверка: итог_к_просадке'])} | {f(r['проверка: t_мес'])} | {corr} | "
                  f"{f(r['загрузка: макс'])} / {f(r['загрузка: 95%'])} |")
    md += ["", "Итог — % капитала портфеля без реинвестирования (у портфеля из нескольких стратегий итог — среднее их итогов, поэтому "
           "сравниваем по t месяцев и итогу к просадке). Загрузка капитала — какая доля капитала в бумагах по часам; больше 1 — "
           "позиции больше капитала (плечо).", "",
           "## Капитал и целые лоты", "",
           "Доля бумаги в капитале постоянна; позиция округляется вниз до целого лота; сделка, на которую лота не хватает, "
           "пропускается. Лоты и цены — Т-Инвестиции 24.10.", "",
           "![Портфель D целыми лотами при разном капитале](launch_portfolio/capital.png)", "",
           "| портфель | капитал, ₽ | сделок взято | выбор: итог % | t мес | проверка: итог % | t мес | итог / просадка |",
           "|---|---|---|---|---|---|---|---|"]
    for r in cap.to_dict("records"):
        md.append(f"| {r['портфель']} | {r['капитал']:,} | {f(r['сделок взято, %'], 0)}% | {f(r['выбор: итог'], 1, True)} | "
                  f"{f(r['выбор: t_мес'])} | {f(r['проверка: итог'], 1, True)} | {f(r['проверка: t_мес'])} | {f(r['проверка: итог_к_просадке'])} |".replace(",", " "))
    md += ["", "Капитал, при котором на долю бумаги в портфеле D (1/3 ÷ 19 = 1/57 капитала) помещается хотя бы один лот при "
           "размере 1:", "", "| бумага | лот, шт. | цена 24.10, ₽ | лот, ₽ | капитал D от, ₽ |", "|---|---|---|---|---|"]
    for t, v in sorted(lots.items(), key=lambda kv: -kv[1]["lot_rub"]):
        md.append(f"| {t} | {v['lot']} | {v['price']:g} | {v['lot_rub']:,.0f} | {need[t]:,.0f} |".replace(",", " "))
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    print(res.round(2).to_string(index=False))
    print(cap.round(2).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
