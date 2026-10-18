"""Сделки по самому тренд-фильтру — самостоятельная стратегия (гипотеза H59, созвон 18.10).

Простейшее «следовать фильтру» на наборе для запуска: вход по открытию часа после того, как фильтр включил направление
(+1 — лонг, −1 — шорт, шорт только у металлов), выход по открытию часа после того, как фильтр вышел из этого
состояния. Без стопа и без подбора. Фильтры — трёхслойный (режим по дневной EMA200, H55) и для сравнения гейт C.
Издержки — тем же recost, что журнал: металлы — B01 (configs/costs/base.yaml), акции — биржа (exchange.yaml).
Оценка — модулем портфеля: сама по себе и вместе с S42 на общем капитале.

    python scripts/trend_follow.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_grid as tg  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import TEST_START, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, load_catalog, load_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "trend_follow"
DOC = ROOT / "docs" / "results" / "trend_follow.md"
SET = ["silver", "gold", "gazp", "sber", "lkoh"]
STOCKS = {"gazp", "sber", "lkoh"}
TL = {"mode": "score", "score_regime_tf": "24h"}
# (параметры фильтра, держать через нейтраль); последний вариант придуман после основного расчёта — для разбора, не доказательство
FILTERS = {"трёхслойный": (TL, False), "гейт C": ({}, False), "трёхслойный, держать до противоположного сигнала": (TL, True)}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def follow(bars: pd.DataFrame, g: np.ndarray, long_only: bool, hold: bool = False) -> pd.DataFrame:
    """Сделки «держать позицию в сторону фильтра»: решение на закрытии часа i, исполнение по открытию часа i+1.
    hold — нейтраль фильтра не закрывает позицию (держим до противоположного сигнала)."""
    g = np.nan_to_num(g).astype(int)
    if hold:
        g = pd.Series(np.where(g == 0, np.nan, g)).ffill().fillna(0).to_numpy(int)
    if long_only:
        g = np.maximum(g, 0)
    t, o = bars["time"].to_numpy(), bars["open"].to_numpy(float)
    yr = bars["time"].dt.year.to_numpy()
    rows, pos, e = [], 0, 0
    for i in range(len(g) - 1):
        if yr[i] < 2016:
            continue
        if g[i] != pos:
            if pos != 0:
                rows.append((e, i + 1, pos))
            pos, e = g[i], i + 1
    if pos != 0:
        rows.append((e, len(g) - 1, pos))
    if not rows:
        return pd.DataFrame()
    e, x, s = (np.array(v) for v in zip(*rows))
    gross = s * (o[x] / o[e] - 1) * 100
    return pd.DataFrame({"open_dt": pd.to_datetime(t[e]) + pd.Timedelta(minutes=1), "close_dt": pd.to_datetime(t[x]) + pd.Timedelta(minutes=1),
                         "side": np.where(s > 0, "buy", "sell"), "open_price": o[e], "exit_price": o[x], "size": 1.0,
                         "gross_unit_pct": gross, "gross_pct": gross, "exit_reason": "фильтр"})


def strategy(name: str, data: dict) -> pd.DataFrame:
    parts = []
    for inst in SET:
        bars = data[inst]["bars"]
        params, hold = FILTERS[name]
        t = follow(bars, tg.gate_values(bars, params), inst in STOCKS, hold)
        prof = "exchange" if inst in STOCKS else "base"
        t = recost(t, profile_entry(ROOT / "configs" / "costs" / f"{prof}.yaml", inst))
        parts.append(standardize(t, inst, name))
    return pd.concat(parts, ignore_index=True)


def f(v, d=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.1f}" if d == 1 else f"{v:.{d}f}")


def main() -> int:
    tg._init()
    data = tg._W["data"]
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    alone = pf.combine({"S42": s42}, SET)
    cand = {k: strategy(k, data) for k in FILTERS}
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    curves = {"S42 один": alone, **{f"по фильтру «{k}» сама": pf.combine({k: v}, SET) for k, v in cand.items()},
              "S42 + трёхслойный": pf.combine({"S42": s42, "тф": cand["трёхслойный"]}, SET)}
    for (k, x), c in zip(curves.items(), (GRAY, "#8a5cc4", "#c8641e", BLUE)):
        x = x.sort_values("close_dt")
        ax.plot(x["close_dt"], (x["w"] * x["profit_pct"]).cumsum(), color=c, linewidth=2.3 if c == BLUE else 1.5, label=k)
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
    md = ["# Сделки по самому тренд-фильтру (H59)", "",
          "Созвон 18.10: «а что, если открытие сделок по тренд-фильтру — совсем простенько». Набор для запуска: серебро и "
          "золото через CFD в обе стороны, GAZP, SBER, LKOH на бирже только в лонг. Вход по открытию часа после того, как "
          "фильтр включил направление, выход — после того, как вышел из него; без стопа и без подбора. Издержки — как в "
          "журнале (металлы B01, акции — биржа). Расчёт — `scripts/trend_follow.py`.", "",
          "![Накопленный итог](trend_follow/equity.png)", "", verdict, ""]
    for per, title in (("проверка", "Проверка 2021 — июнь 2026"), ("выбор", "Выбор 2016–2020")):
        sa = pf.metrics(alone, per)
        md += [f"## {title}", "", "| стратегия | сделок в год | итог % | PF | без 2 лучших | просадка | t мес | с S42: итог % | t мес | итог / просадка | корр. месяцев с S42 |",
               "|---|---|---|---|---|---|---|---|---|---|---|",
               f"| **S42 один** | {sa['в_год']:.1f} | {f(sa['итог'], 1)} | {f(sa['PF'])} | {f(sa['без_2_лучших'], 1)} | {f(sa['просадка'], 1)} | "
               f"{f(sa['t_мес'])} | — | {f(sa['t_мес'])} | {f(sa['итог_к_просадке'])} | — |"]
        for k, v in cand.items():
            solo = pf.combine({k: v}, SET)
            m, p, o = pf.metrics(solo, per), pf.metrics(pf.combine({"S42": s42, k: v}, SET), per), pf.overlap(solo, alone, per)
            md.append(f"| по фильтру «{k}» | {m['в_год']:.1f} | {f(m['итог'], 1)} | {f(m['PF'])} | {f(m['без_2_лучших'], 1)} | {f(m['просадка'], 1)} | "
                      f"{f(m['t_мес'])} | {f(p['итог'], 1)} | {f(p['t_мес'])} | {f(p['итог_к_просадке'])} | {f(o.get('корр_мес'))} |")
        md.append("")
    md += ["## По рынкам — трёхслойный, проверка", "", "| рынок | сделок | итог % | PF | средняя сделка % | средняя длительность, дней |", "|---|---|---|---|---|---|"]
    v = cand["трёхслойный"]
    for i in SET:
        x = v[(v["instrument"] == i) & (v["open_dt"] >= TEST_START)]
        loss = -x.loc[x["profit_pct"] < 0, "profit_pct"].sum()
        md.append(f"| {i} | {len(x)} | {x['profit_pct'].sum():+.1f} | {(x.loc[x['profit_pct'] > 0, 'profit_pct'].sum() / loss) if loss > 0 else float('nan'):.2f} | "
                  f"{x['profit_pct'].mean():+.2f} | {((x['close_dt'] - x['open_dt']).dt.total_seconds() / 86400).mean():.1f} |")
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
