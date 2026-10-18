"""Акции в лонг по трёхслойному фильтру против «купить и держать» (гипотеза H60, разбор H59 18.10).

Правило — вариант H59 «держать до противоположного сигнала»: GAZP, SBER, LKOH в лонг, пока трёхслойный фильтр
(режим по дневной EMA200) не покажет вниз; вход и выход по открытию часа после сигнала, нейтраль не закрывает.
Без стопа и подбора. Цены с поправкой на дивиденды, издержки биржи (configs/costs/exchange.yaml) — на каждый вход.

Сравнение по риску — на капитале с реинвестированием, как принято для правила «в рынке / вне рынка» против
держания: у обоих стоимость позиции на закрытии каждого часа (сделки правила — от дней до 2–3 месяцев, по закрытым
сделкам просадка выглядела бы меньше настоящей), просадка — от пика капитала. Без реинвестирования «купить и держать»
мерил бы просадку от накопленного роста в пунктах начального капитала, а правило — «откладывало» прибыль после
каждой сделки; сравнение было бы нечестным в пользу правила. Бумаги поровну (у каждой своя треть капитала).
«Купить и держать» в каждом периоде — покупка в его начале; правило в начале периода тоже входит заново, если
фильтр уже смотрит вверх.

Портфель с импульсом: акции S42 (журнал) и правило поровну, ребалансировка раз в месяц; против одних акций S42.

    python scripts/trend_hold.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_grid as tg  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, load_catalog, load_trades  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from trend_follow import TL, follow  # noqa: E402

OUT = ROOT / "docs" / "results" / "trend_hold"
DOC = ROOT / "docs" / "results" / "trend_hold.md"
STOCKS = ["gazp", "sber", "lkoh"]
NAMES = {"gazp": "GAZP", "sber": "SBER", "lkoh": "LKOH"}
PERIODS = {"выбор": ("2016-01-01", "2021-01-01"), "проверка": ("2021-01-01", None)}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def marked(bars: pd.DataFrame, g: np.ndarray, cost: float, a: str, b: str | None) -> pd.DataFrame:
    """Капитал бумаги на закрытии каждого часа периода (в начале — 1): правило и «купить и держать».
    Покупка по открытию часа на весь капитал, издержки биржи — на вход; позиция на часе j — решение на закрытии
    часа j−1. «Купить и держать» — одна покупка по открытию первого часа периода."""
    t = bars["time"].to_numpy()
    sel = (t >= np.datetime64(a)) & ((t < np.datetime64(b)) if b else True)
    idx = np.flatnonzero(sel)
    o, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    gh = pd.Series(np.where(np.nan_to_num(g) == 0, np.nan, np.nan_to_num(g))).ffill().fillna(0).to_numpy()
    pos = np.r_[0.0, np.maximum(gh, 0)[:-1]]
    eq, inn = np.empty(len(idx)), np.zeros(len(idx), bool)
    cap, px = 1.0, None
    for n, j in enumerate(idx):
        if pos[j] > 0 and px is None:
            px, cap = o[j], cap * (1 - cost / 100)
        elif pos[j] == 0 and px is not None:
            cap, px = cap * o[j] / px, None
        eq[n] = cap * (c[j] / px) if px is not None else cap
        inn[n] = px is not None
    hold = (1 - cost / 100) * c[idx] / o[idx[0]]
    return pd.DataFrame({"rule": eq, "hold": hold, "in": inn}, index=pd.to_datetime(t[idx]))


def daily(eq: pd.Series) -> pd.Series:
    return eq.resample("D").last().dropna()


def risk(eq: pd.Series) -> dict:
    """По капиталу (в начале 1): итог %, t месячных доходностей, просадка от пика % и самая долгая (дней), итог к просадке."""
    e = eq.to_numpy()
    peak = np.maximum.accumulate(np.maximum(e, 1.0))
    dd = e / peak - 1
    longest, start = 0, None
    for d, x in zip(eq.index, dd):
        if x < 0 and start is None:
            start = d
        elif x >= 0 and start is not None:
            longest, start = max(longest, (d - start).days), None
    if start is not None:
        longest = max(longest, (eq.index[-1] - start).days)
    m = monthly_ret(eq)
    total, depth = float((e[-1] - 1) * 100), float(dd.min() * 100)
    return {"итог": total, "t": t_of(m), "просадка": depth, "дней": longest, "и/п": total / -depth if depth < 0 else np.inf}


def monthly_ret(eq: pd.Series) -> pd.Series:
    """Месячные доходности капитала, % (первый месяц — от 1)."""
    m = eq.groupby(eq.index.to_period("M")).last()
    return (m / m.shift(1).fillna(1.0) - 1) * 100


def from_monthly(m: pd.Series) -> dict:
    """Метрики капитала, собранного из месячных доходностей (портфель с S42)."""
    eq = (1 + m / 100).cumprod()
    peak = np.maximum.accumulate(np.maximum(eq.to_numpy(), 1.0))
    depth = float((eq.to_numpy() / peak - 1).min() * 100)
    total = float((eq.iloc[-1] - 1) * 100)
    return {"t": t_of(m), "итог": total, "просадка": depth, "и/п": total / -depth if depth < 0 else np.inf}


def t_of(m: pd.Series) -> float:
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m))))


def f(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.1f}" if d == 1 else f"{v:.{d}f}"


def main() -> int:
    tg._init()
    data = {i: tg._W["data"][i]["bars"][["time", "open", "high", "low", "close"]].copy() for i in STOCKS}
    gates = {i: tg.gate_values(tg._W["data"][i]["bars"], TL) for i in STOCKS}
    tg._W.clear()
    cost = {i: profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", i)["spread_pct"] for i in STOCKS}

    # сделки правила — как в H59 (для PF, без 2 лучших и числа сделок)
    parts = []
    for i in STOCKS:
        t = follow(data[i], gates[i], long_only=True, hold=True)
        parts.append(standardize(recost(t, profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", i)), i, "H60"))
    trades = pd.concat(parts, ignore_index=True)

    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    s42 = s42[s42["instrument"].isin(STOCKS)]

    res, per_stock, curves = {}, {}, {}
    for per, (a, b) in PERIODS.items():
        hrs = {i: marked(data[i], gates[i], cost[i], a, b) for i in STOCKS}
        day = {k: pd.concat([daily(h[k]) for h in hrs.values()], axis=1).ffill().fillna(1.0).mean(axis=1) for k in ("rule", "hold")}
        res[per] = {k: risk(v) for k, v in day.items()}
        res[per]["в рынке"] = float(np.mean([h["in"].mean() for h in hrs.values()]))
        per_stock[per] = {i: {k: risk(daily(h[k])) for k in ("rule", "hold")} for i, h in hrs.items()}
        curves[per] = day
        # портфель с акциями S42: поровну, ребалансировка раз в месяц; S42 — месячный результат журнала (% капитала)
        m42 = pf.monthly(pf._period(pf.combine({"S42": s42}, STOCKS), per))
        mr = monthly_ret(day["rule"])
        m42.index, mr.index = m42.index.astype(str), mr.index.astype(str)
        j = pd.concat([m42.rename("s42"), mr.rename("rule")], axis=1).fillna(0.0).sort_index()
        res[per]["порт"] = {"S42": from_monthly(j["s42"]), "S42 + правило": from_monthly(0.5 * j["s42"] + 0.5 * j["rule"]),
                            "корр": float(j["s42"].corr(j["rule"]))}
        res[per]["сделки"] = pf.metrics(pf.combine({"H60": trades}, STOCKS), per)

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=BG, gridspec_kw={"width_ratios": [5, 5.5]})
    for ax, (per, day) in zip(axes, curves.items()):
        ax.set_facecolor(BG)
        ax.plot(day["hold"].index, (day["hold"] - 1) * 100, color=GRAY, linewidth=1.5, label="купить и держать")
        ax.plot(day["rule"].index, (day["rule"] - 1) * 100, color=BLUE, linewidth=2, label="в лонг по трёхслойному фильтру")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{per}: капитал, % к началу периода (бумаги поровну)", loc="left", fontsize=9.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)

    ok_hold = all(res[p]["rule"]["t"] > res[p]["hold"]["t"] and res[p]["rule"]["и/п"] > res[p]["hold"]["и/п"] for p in PERIODS)
    ok_port = all(res[p]["порт"]["S42 + правило"]["t"] > res[p]["порт"]["S42"]["t"]
                  and res[p]["порт"]["S42 + правило"]["и/п"] > res[p]["порт"]["S42"]["и/п"] for p in PERIODS)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    md = ["# Акции в лонг по трёхслойному фильтру против «купить и держать» (H60)", "",
          "Вариант из разбора H59 ([trend_follow.md](trend_follow.md)): GAZP, SBER, LKOH в лонг, пока трёхслойный фильтр не "
          "покажет вниз (нейтраль не закрывает), без стопа и подбора; цены с поправкой на дивиденды, издержки биржи на каждый "
          "вход. Оба варианта — правило и «купить и держать» — считаются на капитале с реинвестированием по стоимости позиции "
          "на закрытии каждого часа (сделки правила — от дней до 2–3 месяцев, по закрытым сделкам просадка выглядела бы "
          "меньше настоящей), просадка — от пика капитала; у каждой бумаги своя треть капитала. В каждом периоде — покупка в его "
          "начале. Расчёт — `scripts/trend_hold.py`.", "",
          f"Критерий H60: на единицу риска (t месяцев и итог к просадке) лучше держания в обоих периодах — "
          f"**{'да' if ok_hold else 'нет'}**; портфель с акциями S42 лучше одних акций S42 по t и итогу к просадке в обоих "
          f"периодах — **{'да' if ok_port else 'нет'}**.", "",
          "![Правило против купить и держать](trend_hold/equity.png)", "", verdict, "",
          "## Правило против «купить и держать»", "",
          "| период | вариант | итог % | t мес | просадка % | самая долгая просадка, дней | итог / просадка | в рынке |",
          "|---|---|---|---|---|---|---|---|"]
    for per in PERIODS:
        for k, name in (("rule", "по фильтру"), ("hold", "купить и держать")):
            r = res[per][k]
            share = f"{res[per]['в рынке']:.0%}" if k == "rule" else "100%"
            md.append(f"| {per} | {name} | {f(r['итог'], 1)} | {f(r['t'])} | {f(r['просадка'], 1)} | {r['дней']} | {f(r['и/п'])} | {share} |")
    md += ["", "## По бумагам", "", "| период | бумага | по фильтру: итог % | просадка % | итог / просадка | держать: итог % | просадка % | итог / просадка |",
           "|---|---|---|---|---|---|---|---|"]
    for per in PERIODS:
        for i in STOCKS:
            r, h = per_stock[per][i]["rule"], per_stock[per][i]["hold"]
            md.append(f"| {per} | {NAMES[i]} | {f(r['итог'], 1)} | {f(r['просадка'], 1)} | {f(r['и/п'])} | {f(h['итог'], 1)} | "
                      f"{f(h['просадка'], 1)} | {f(h['и/п'])} |")
    md += ["", "## Сделки правила (как в журнале, по закрытым сделкам)", "",
           "| период | сделок в год | итог % | PF | без 2 лучших | лет в плюсе |", "|---|---|---|---|---|---|"]
    for per in PERIODS:
        m = res[per]["сделки"]
        md.append(f"| {per} | {m['в_год']:.1f} | {f(m['итог'], 1)} | {f(m['PF'])} | {f(m['без_2_лучших'], 1)} | {m['лет_в_плюсе']} |")
    md += ["", "## Вместе с импульсом: акции S42 и правило поровну", "",
           "Поровну, ребалансировка раз в месяц: S42 — месячный результат журнала (сделки до 10 дней), правило — месячная "
           "доходность по стоимости позиции. Сравниваем по t месяцев и итогу к просадке.", "",
           "| период | портфель | итог % | t мес | просадка % (по концам месяцев) | итог / просадка | корр. месяцев |", "|---|---|---|---|---|---|---|"]
    for per in PERIODS:
        p = res[per]["порт"]
        for k in ("S42", "S42 + правило"):
            md.append(f"| {per} | {'акции S42 одни' if k == 'S42' else 'акции S42 + правило'} | {f(p[k]['итог'], 1)} | {f(p[k]['t'])} | "
                      f"{f(p[k]['просадка'], 1)} | {f(p[k]['и/п'])} | {f(p['корр']) if k != 'S42' else '—'} |")
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"держание: {ok_hold}, портфель: {ok_port}")
    for per in PERIODS:
        print(per, {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in res[per][k].items()} for k in ("rule", "hold")})
        print("   порт", {k: (round(v["t"], 2), round(v["и/п"], 2)) if isinstance(v, dict) else round(v, 2) for k, v in res[per]["порт"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
