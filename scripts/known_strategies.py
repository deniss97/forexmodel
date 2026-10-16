"""Известные стратегии (задача T26): проверка на наших акциях и вклад в портфель с импульсом.

Каждая стратегия — функция от дневных свечей одной бумаги, возвращающая сделки (день входа, день выхода).
Дневки — из часовых свечей перебора с поправкой на дивиденды (reports/night_search/feat); вход и выход —
по открытию или закрытию дня; издержки биржи — тем же recost, что журнал (configs/costs/exchange.yaml).
Оценка — модулем портфеля (forexmodel/evaluation/portfolio.py): сама по себе и вместе с акциями S42 на общем
капитале. Чтобы проверить стратегию Тимура — описать её в банке по шаблону и добавить функцию в STRATEGIES.

  * H56 — RSI(2) Коннорса: цена выше SMA200 и RSI(2) < 10 → лонг на открытии следующего дня; выход на открытии
    после закрытия выше SMA5, не дольше 20 дней;
  * H57 — смена месяца: лонг с открытия последнего торгового дня месяца до закрытия третьего дня следующего.

    python scripts/known_strategies.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import TEST_START, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, load_catalog, load_trades  # noqa: E402

FEAT = ROOT / "reports" / "night_search" / "feat"
COSTS = ROOT / "configs" / "costs" / "exchange.yaml"
OUT = ROOT / "docs" / "results" / "known_strategies"
DOC = ROOT / "docs" / "results" / "known_strategies.md"
STOCKS = ["gazp", "sber", "lkoh"]
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def daily(inst: str) -> pd.DataFrame:
    b = pd.read_pickle(FEAT / f"{inst}.pkl")["bars"]
    b = b[b["time"] >= "2015-01-01"]
    g = b.groupby(b["time"].dt.floor("D"))
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                         "t_open": g["time"].first(), "t_close": g["time"].last() + pd.Timedelta(minutes=59)}).reset_index(drop=True)


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def connors_rsi2(d: pd.DataFrame) -> list[tuple[int, int, str, str]]:
    """(день входа, день выхода, вход по 'open'/'close', выход по 'open'/'close')."""
    sma200, sma5, r = d["close"].rolling(200).mean(), d["close"].rolling(5).mean(), rsi(d["close"], 2)
    trades, i, n = [], 200, len(d)
    while i < n - 1:
        if d["close"].iat[i] > sma200.iat[i] and r.iat[i] < 10:
            e = i + 1
            x = e
            while x < n - 1 and x - e < 20 and not d["close"].iat[x] > sma5.iat[x]:
                x += 1
            trades.append((e, min(x + 1, n - 1), "open", "open"))
            i = x + 1
        else:
            i += 1
    return trades


def turn_of_month(d: pd.DataFrame) -> list[tuple[int, int, str, str]]:
    month = d["t_open"].dt.to_period("M").to_numpy()
    last = [i for i in range(len(d) - 1) if month[i] != month[i + 1]]
    return [(i, i + 3, "open", "close") for i in last if i + 3 < len(d)]


STRATEGIES = {"H56 RSI(2) Коннорса": connors_rsi2, "H57 смена месяца": turn_of_month}


def trades_for(fn, inst: str) -> pd.DataFrame:
    d = daily(inst)
    rows = []
    for e, x, ein, xout in fn(d):
        ep = d[ein].iat[e]
        xp = d[xout].iat[x]
        rows.append({"open_dt": d["t_open"].iat[e] if ein == "open" else d["t_close"].iat[e],
                     "close_dt": d["t_open"].iat[x] if xout == "open" else d["t_close"].iat[x],
                     "side": "buy", "open_price": ep, "exit_price": xp, "size": 1.0,
                     "gross_unit_pct": (xp / ep - 1) * 100, "gross_pct": (xp / ep - 1) * 100, "exit_reason": "правило"})
    t = pd.DataFrame(rows)
    t = t[t["open_dt"] >= "2016-01-01"]
    t = recost(t, profile_entry(COSTS, inst))
    return standardize(t, inst, fn.__name__)


def f(v, k: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if k in ("PF", "t", "corr"):
        return f"{v:.2f}"
    if k == "share":
        return f"{v:.0%}"
    if k == "n":
        return f"{v:.1f}"
    return f"{v:+.1f}"


def main() -> int:
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    s42 = s42[s42["instrument"].isin(STOCKS)]
    alone = pf.combine({"S42": s42}, STOCKS)
    cand = {name: pd.concat([trades_for(fn, i) for i in STOCKS], ignore_index=True) for name, fn in STRATEGIES.items()}
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    curves = {"S42 акции в лонг": alone, **{f"{k} сама": pf.combine({k: v}, STOCKS) for k, v in cand.items()},
              **{f"S42 + {k.split()[0]}": pf.combine({"S42": s42, k: v}, STOCKS) for k, v in cand.items()}}
    for (k, x), c in zip(curves.items(), [GRAY, "#8a5cc4", "#c8641e", BLUE, "#1d7a55"]):
        x = x.sort_values("close_dt")
        ax.plot(x["close_dt"], (x["w"] * x["profit_pct"]).cumsum(), color=c, linewidth=2.2 if k.startswith("S42 +") else 1.5, label=k)
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, % капитала", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    md = ["# Известные стратегии на наших акциях (T26)", "",
          "Задача **T26**: известные публичные стратегии — по шаблону в банке, на тех же акциях, что первый этап запуска "
          "(GAZP, SBER, LKOH на бирже, только лонг), с издержками биржи (комиссия 0.1% за круг, проскальзывание 0.05% "
          "на сторону), цены с поправкой на дивиденды. Параметры — как в первоисточнике, без подбора. Оценка — сама по себе "
          "и вместе с акциями S42 на общем капитале (капитал поровну). Расчёт — `scripts/known_strategies.py`.", "",
          "* **H56 RSI(2) Коннорса** — цена выше SMA200 и RSI(2) < 10 на закрытии → лонг на открытии следующего дня; выход "
          "на открытии после закрытия выше SMA5, не дольше 20 дней (Connors & Alvarez).",
          "* **H57 смена месяца** — лонг с открытия последнего торгового дня месяца до закрытия третьего дня следующего "
          "(Lakonishok & Smidt 1988, McConnell & Xu 2008).", "",
          "![Накопленный итог](known_strategies/equity.png)", "", verdict, ""]
    for per, title in (("проверка", "Проверка 2021 — июнь 2026"), ("выбор", "Выбор 2016–2020")):
        sa = pf.metrics(alone, per)
        md += [f"## {title}", "", "| стратегия | сделок в год | итог % | PF | без 2 лучших | просадка | t мес | с S42: итог % | t мес | итог / просадка | корр. месяцев с S42 |",
               "|---|---|---|---|---|---|---|---|---|---|---|",
               f"| **S42 акции в лонг** | {f(sa['в_год'], 'n')} | {f(sa['итог'], 'x')} | {f(sa['PF'], 'PF')} | {f(sa['без_2_лучших'], 'x')} | "
               f"{f(sa['просадка'], 'x')} | {f(sa['t_мес'], 't')} | — | {f(sa['t_мес'], 't')} | {f(sa['итог_к_просадке'], 't')} | — |"]
        for k, v in cand.items():
            solo = pf.combine({k: v}, STOCKS)
            m, p, o = pf.metrics(solo, per), pf.metrics(pf.combine({"S42": s42, k: v}, STOCKS), per), pf.overlap(solo, alone, per)
            md.append(f"| {k} | {f(m['в_год'], 'n')} | {f(m['итог'], 'x')} | {f(m['PF'], 'PF')} | {f(m['без_2_лучших'], 'x')} | "
                      f"{f(m['просадка'], 'x')} | {f(m['t_мес'], 't')} | {f(p['итог'], 'x')} | {f(p['t_мес'], 't')} | "
                      f"{f(p['итог_к_просадке'], 't')} | {f(o.get('корр_мес'), 'corr')} |")
        md.append("")
    md += ["## По бумагам — проверка", "", "| стратегия | бумага | сделок | итог % | PF | средняя сделка % |", "|---|---|---|---|---|---|"]
    for k, v in cand.items():
        for i in STOCKS:
            x = v[(v["instrument"] == i) & (v["open_dt"] >= TEST_START)]
            loss = -x.loc[x["profit_pct"] < 0, "profit_pct"].sum()
            md.append(f"| {k} | {i.upper()} | {len(x)} | {x['profit_pct'].sum():+.1f} | "
                      f"{(x.loc[x['profit_pct'] > 0, 'profit_pct'].sum() / loss) if loss > 0 else float('nan'):.2f} | {x['profit_pct'].mean():+.2f} |")
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
