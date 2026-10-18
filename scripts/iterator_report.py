"""Отчёт T36 — прозрачность итератора: четыре стратегии перебора H54 на наборе для запуска.

Как перебор их собрал, как они торгуют (сделки на графике), наложение на трёхслойный тренд-фильтр (H61) и
согласие двух стратегий (H62), портфель «S42 + стратегия». Сделки — журнал S45–S48 (движок перебора, издержки
как у S42) и S42; метрики — тем же кодом, что журнал (рынки поровну).

    python scripts/iterator_walkthrough.py report
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_grid as tg  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, REASON, ROOT, fmt, load_catalog, load_trades  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import metrics  # noqa: E402
from iterator_walkthrough import EXP_ID, STRATS  # noqa: E402
from trend_model_gate import at_entry  # noqa: E402

OUT = ROOT / "docs" / "results" / "iterator_walkthrough"
DOC = ROOT / "docs" / "results" / "iterator_walkthrough.md"
JOURNAL = "https://claude.ai/artifact/E4teCbCVvojCf96CJy9UQU"
INST = ["silver", "gold", "lkoh", "gazp", "sber"]
NAMES = {"silver": "серебро", "gold": "золото", "lkoh": "LKOH", "gazp": "GAZP", "sber": "SBER"}
PER = ("выбор", "проверка")
KEYS = ("сделок", "итог", "PF", "без_2_лучших", "просадка", "t")
GREEN, RED = "#1d7a55", "#b3412f"
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
WINDOW = pd.Timedelta(hours=24)
# окна для картинок сделок — полтора года проверки, чтобы сделки было видно (весь период — в журнале)
WINDOWS = {"sber": ("2023-01-01", "2024-07-01"), "silver": ("2024-01-01", "2025-07-01")}
# финалисты H54 (docs/results/second_strategy.md): место, устойчивая оценка, корреляция месяцев с S42 на выборе
RANK = {1: (1, 2.10, 0.08), 2: (3, 1.78, 0.10), 3: (4, 2.09, 0.10), 4: (5, 1.72, 0.09)}
PLAIN = {
    1: ("NR7 → пробой", "если вчерашний дневной диапазон — самый узкий за 7 дней (NR7), первый час сегодня, закрывшийся "
        "выше вчерашнего максимума, — сигнал в лонг (ниже минимума — в шорт, но шорты отсекает фильтр)",
        "только лонг; сила тренда ADX(14) на 1 ч выше 25",
        "закрытие ниже минимума последних 20 дней (канал Дончиана); защитный стоп 6 ATR(10)"),
    2: ("пробой 55 дней", "первый за день час, закрывшийся выше максимума последних 55 дней, — лонг; ниже минимума — шорт "
        "(«Черепахи»; у акций шорты не берутся)",
        "цена на 4 ч по ту же сторону от EMA200; ATR14/ATR100 > 1 — волатильность выше обычной",
        "разворот Supertrend (10, 3) на 4 ч против позиции; защитный стоп 3 ATR(20)"),
    3: ("канал Кельтнера на дневках", "дневное закрытие вышло за EMA20 ± 2 ATR10 — лонг вверх, шорт вниз (сигнал — смена "
        "состояния канала; оно держится, пока цена не вернётся к EMA20)",
        "дневная цена по ту же сторону от EMA200 дней",
        "обратный сигнал того же канала; защитный стоп 6 ATR(14)"),
    4: ("пробой 20 дней + 0.25 ATR", "первый за день час, закрывшийся выше максимума 20 дней с запасом 0.25 ATR, — лонг; "
        "ниже минимума — шорт",
        "дневная цена по ту же сторону от EMA200 дней",
        "закрытие за минимумом (максимумом для шорта) 20 дней; защитный стоп 3 ATR(10)"),
}


def kept_conclusion() -> list[str]:
    """Вывод пишется руками между метками и при пересборке отчёта сохраняется."""
    if DOC.exists():
        text = DOC.read_text(encoding="utf-8")
        if MARK[0] in text and MARK[1] in text:
            return [MARK[0] + text.split(MARK[0], 1)[1].split(MARK[1], 1)[0] + MARK[1]]
    return [MARK[0], MARK[1]]


def three_layer() -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Часовые свечи и состояние трёхслойного фильтра (режим по дневной EMA200, как в H55 и H58) по рынкам набора."""
    tg._init()
    out = {}
    for inst in INST:
        full = tg._W["data"][inst]["bars"]
        g = np.nan_to_num(tg.gate_values(full, {"mode": "score", "score_regime_tf": "24h"}))
        out[inst] = (full[["time", "close"]].copy(), g)
    tg._W.clear()
    return out


def m_of(t: pd.DataFrame, per: str) -> dict:
    return metrics(t, INST, per) if len(t) else {"сделок": 0}


def agree(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Сделки a, на входе которых b держит позицию на том же рынке в ту же сторону (или открыла её за 24 ч)."""
    ok = np.zeros(len(a), bool)
    for (inst, side), g in a.groupby(["instrument", "side"]):
        h = b[(b["instrument"] == inst) & (b["side"] == side)]
        if h.empty:
            continue
        t0 = g["open_dt"].to_numpy()[:, None]
        o, c = h["open_dt"].to_numpy()[None, :], h["close_dt"].to_numpy()[None, :]
        hit = (o <= t0) & ((c > t0) | (o >= t0 - WINDOW.to_timedelta64()))
        ok[a.index.get_indexer(g.index)] = hit.any(axis=1)
    return ok


def row(name: str, t: pd.DataFrame, base: pd.DataFrame | None = None) -> dict:
    r = {"вариант": name}
    for per in PER:
        m = m_of(t, per)
        r[per] = m
        if base is not None:
            n0 = m_of(base, per).get("сделок") or 0
            r[per + "_доля"] = (m.get("сделок") or 0) / n0 if n0 else np.nan
    return r


def cells(r: dict, keys=KEYS) -> list[str]:
    out = []
    for per in PER:
        m = r[per]
        out += [fmt(m.get(k), k) for k in keys]
    return out


def describe(t: pd.DataFrame) -> dict:
    """Как торгует стратегия: сделок в год, удержание, доля выигрышей, причины выхода, доля лонгов."""
    hold = (t["close_dt"] - t["open_dt"]).dt.total_seconds() / 86400
    years = (t["close_dt"].max() - t["open_dt"].min()).days / 365.25
    reasons = t["exit_reason"].map(lambda r: REASON.get(r, r)).value_counts(normalize=True)
    return {"в год": len(t) / years, "дней": float(hold.median()), "выигрышей": float((t["profit_pct"] > 0).mean() * 100),
            "лонгов": float((t["side"] > 0).mean() * 100),
            "выходы": ", ".join(f"{k} {v:.0%}" for k, v in reasons.items() if v >= 0.05)}


def equity_png(trades: dict[str, pd.DataFrame], path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(BG)
    colors = {"S42": INK, "S45": GRAY, "S46": BLUE, "S47": "#c28a12", "S48": GREEN}
    for k, t in trades.items():
        x = t.sort_values("close_dt")
        ax.plot(x["close_dt"], (x["profit_pct"] / len(INST)).cumsum(), color=colors[k], linewidth=2 if k in ("S42", "S46") else 1.3,
                label=k if k == "S42" else f"{k} · {STRATS[int(k[1:]) - 44]['entry']}")
    ax.axvline(pd.Timestamp("2021-01-01"), color=INK2, linewidth=0.8, linestyle="--")
    ax.text(pd.Timestamp("2021-02-01"), ax.get_ylim()[1] * 0.92, "проверка →", color=INK2, fontsize=8)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_title("Накопленный итог, % капитала (пять рынков поровну)", loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def trades_png(inst: str, trades: dict[str, pd.DataFrame], bars: pd.DataFrame, g: np.ndarray, start: str, end: str, path) -> None:
    """Сделки четырёх стратегий на дневной цене рынка; фон — трёхслойный фильтр (зелёный вверх, красный вниз)."""
    s = pd.Series(g, index=pd.to_datetime(bars["time"]))
    px = pd.Series(bars["close"].to_numpy(), index=s.index)
    sel = (s.index >= start) & (s.index < end)
    s, px = s[sel], px[sel]
    day = px.resample("D").last().dropna()
    state = s.resample("D").last().reindex(day.index).fillna(0)
    ids = [k for k in trades if k != "S42"]
    fig, axes = plt.subplots(len(ids), 1, figsize=(11, 2.3 * len(ids)), facecolor=BG, sharex=True)
    lo, hi = day.min() * 0.97, day.max() * 1.03
    for ax, k in zip(axes, ids):
        ax.set_facecolor(BG)
        ax.fill_between(day.index, lo, hi, where=state.to_numpy() > 0, color=GREEN, alpha=0.08, step="post", linewidth=0)
        ax.fill_between(day.index, lo, hi, where=state.to_numpy() < 0, color=RED, alpha=0.08, step="post", linewidth=0)
        ax.plot(day.index, day.to_numpy(), color=GRAY, linewidth=0.9)
        t = trades[k]
        t = t[(t["instrument"] == inst) & (t["open_dt"] >= start) & (t["open_dt"] < end)]
        for o, c, side, p in zip(t["open_dt"], t["close_dt"], t["side"], t["profit_pct"]):
            y0, y1 = px.asof(o), px.asof(c)
            col = GREEN if p > 0 else RED
            ax.plot([o, c], [y0, y1], color=col, linewidth=2.2, solid_capstyle="butt")
            ax.plot([o], [y0], marker="^" if side > 0 else "v", color=col, markersize=5)
        n = len(t)
        ax.set_title(f"{k} · {STRATS[int(k[1:]) - 44]['entry']} — {n} сделок, из них в плюсе {int((t['profit_pct'] > 0).sum())}",
                     loc="left", fontsize=9, color=INK)
        ax.set_ylim(lo, hi)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.tick_params(colors=INK2, labelsize=8)
    span = f"{pd.Timestamp(start):%m.%Y} – {pd.Timestamp(end) - pd.Timedelta(days=1):%m.%Y}"
    fig.suptitle(f"{NAMES[inst]}, {span}: сделки (линия от входа до выхода, зелёная — в плюсе) "
                 f"и трёхслойный фильтр (фон: зелёный — вверх, красный — вниз)", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def report() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, exps = load_catalog()
    ids = ["S42"] + [EXP_ID[n] for n in STRATS]
    trades = {k: load_trades(exps[k]).reset_index(drop=True) for k in ids}
    filt = three_layer()

    # H61: наложение на трёхслойный фильтр
    h61 = {}
    for k in ids:
        t = trades[k]
        s = np.zeros(len(t))
        for inst, (bars, g) in filt.items():
            sel = (t["instrument"] == inst).to_numpy()
            if sel.any():
                s[sel] = at_entry(t[sel], bars, g)
        on = s == t["side"].to_numpy()
        h61[k] = [row("все сделки", t), row("фильтр согласен", t[on], t), row("фильтр против или нейтрален", t[~on], t)]

    # H62: согласие двух стратегий
    h62 = {}
    for a in ids:
        for b in ids:
            if a != b:
                ok = agree(trades[a], trades[b])
                h62[(a, b)] = row(f"{a}, когда {b} в позиции", trades[a][ok], trades[a])

    # портфель «S42 + стратегия» на общем капитале
    port = {}
    for k in ids:
        parts = {"S42": trades["S42"]} if k == "S42" else {"S42": trades["S42"], k: trades[k]}
        x = pf.combine(parts, INST)
        port[k] = {per: pf.metrics(x, per) for per in PER}
        if k != "S42":
            port[k]["корр"] = {per: pf.overlap(trades[k].assign(w=1 / len(INST)), trades["S42"].assign(w=1 / len(INST)), per)
                               for per in PER}

    equity_png(trades, OUT / "equity.png")
    for inst, (start, end) in WINDOWS.items():
        bars, g = filt[inst]
        trades_png(inst, trades, bars, g, start, end, OUT / f"trades_{inst}.png")

    # ---------- документ ----------
    L = ["# Прозрачность итератора: четыре стратегии перебора на графике (T36)", "",
         "Созвон 18.10: Тимур хочет увидеть на 3–4 конкретных стратегиях, как перебор их находит и как они торгуют, "
         "посмотреть сделки «глазками», наложить их на тренд-фильтр (H61) и попробовать входить, когда две стратегии "
         "согласны (H62). Стратегии — финалисты честного перебора H54 ([second_strategy.md](second_strategy.md)) на "
         "наборе для запуска: серебро и золото через CFD в обе стороны, GAZP, SBER, LKOH на бирже только в лонг. "
         f"Сделки каждой — в журнале экспериментов: **S45–S48** ({JOURNAL}, карточка → «Сделки на графике»).", "",
         "Сделки посчитаны тем же часовым движком, которым считал сам перебор (сверен с минутным симулятором: сделок "
         "±0.5%), — это ровно то, что «видел» итератор; издержки как у S42. Расчёт — `scripts/iterator_walkthrough.py`.", "",
         *kept_conclusion(), ""]

    L += ["## Как итератор собирает стратегию", "",
          "Стратегия — это шесть блоков: **вход** (одно из 128 правил), до **двух фильтров** (можно ли сейчас лонг или шорт), "
          "**выход**, **период ATR** (им меряются стоп и отступы) и **размер** (1 или «по риску» — объём обратно "
          "пропорционален расстоянию до стопа). Одна проба перебора проходит так "
          "([путь пробы по коду](strategy_search/guide.md#путь-одной-пробы-по-коду)):", "",
          "1. **Блоки** — случайные (разведка) или лучшая проба с заменой 1–2 блоков (мутация).",
          "2. **Входы** — часы, где вход дал сигнал и оба фильтра его пропускают; вход по открытию следующего часа.",
          "3. **Сделки** — стоп, трейлинг или выход по сигналу, не дольше 240 ч; пока сделка открыта, новые входы "
          "пропускаются (одна позиция на рынок).",
          "4. **Оценка** — PF на выборе 2016–2020 по набору (рынки поровну); проверку 2021–2026 перебор не видит.",
          "5. **Устойчивость** — претенденту считается медиана PF пробы и 6 соседей (один блок заменён): одиночный "
          "выброс так не проходит.",
          "6. **Отбор H54** — не меньше 50 сделок за выбор, корреляция месяцев с S42 ниже 0.3, устойчивая оценка не "
          "ниже 1.2; финалисты — по приросту t месяцев портфеля «S42 + стратегия» на выборе.", "",
          "Из 1 029 487 проб отбор прошли 588; ниже — четыре разные по устройству из пяти финалистов "
          "(второй финалист — почти копия пробоя 55 дней).", ""]

    L += ["## Четыре стратегии простыми словами", ""]
    for n, prm in STRATS.items():
        k = EXP_ID[n]
        name, entry, filters, exit_ = PLAIN[n]
        place, robust, corr = RANK[n]
        d_all = describe(trades[k])
        ms, mt = m_of(trades[k], "выбор"), m_of(trades[k], "проверка")
        L += [f"### {k}. {name}", "",
              f"* **Вход:** {entry}.", f"* **Фильтры:** {filters}.", f"* **Выход:** {exit_}.",
              f"* **Размер:** {'по риску' if prm['size'] != '1' else 'одинаковый'}.", "",
              f"Как нашёлся: {place}-й из пяти финалистов H54, устойчивая оценка {robust:.2f}, корреляция месяцев с S42 на "
              f"выборе {corr:.2f}. Сделок в год — {d_all['в год']:.0f}, удержание (медиана) — {d_all['дней']:.1f} дн., "
              f"в плюсе — {d_all['выигрышей']:.0f}% сделок, лонгов — {d_all['лонгов']:.0f}%; чем закрываются: "
              f"{d_all['выходы']} (время — предел 240 ч).", "",
              f"**Выбор → проверка:** PF {fmt(ms.get('PF'), 'PF')} → **{fmt(mt.get('PF'), 'PF')}**, итог "
              f"{fmt(ms.get('итог'), 'итог')}% → {fmt(mt.get('итог'), 'итог')}%, без 2 лучших "
              f"{fmt(ms.get('без_2_лучших'), 'без_2_лучших')} → {fmt(mt.get('без_2_лучших'), 'без_2_лучших')}, t "
              f"{fmt(ms.get('t'), 't')} → {fmt(mt.get('t'), 't')}.", ""]
    L += ["| стратегия | рынок | выбор: сделок | PF | итог % | проверка: сделок | PF | итог % |", "|---|---|---|---|---|---|---|---|"]
    for k in ids:
        for inst in INST:
            a, b = metrics(trades[k], [inst], "выбор"), metrics(trades[k], [inst], "проверка")
            L.append(f"| {k} | {NAMES[inst]} | {fmt(a.get('сделок'), 'сделок')} | {fmt(a.get('PF'), 'PF')} | {fmt(a.get('итог'), 'итог')} | "
                     f"{fmt(b.get('сделок'), 'сделок')} | {fmt(b.get('PF'), 'PF')} | {fmt(b.get('итог'), 'итог')} |")
    L += ["", "![Накопленный итог четырёх стратегий и S42](iterator_walkthrough/equity.png)", ""]

    L += ["## Сделки на графике", "",
          "Линия — сделка от входа до выхода (зелёная — в плюсе, красная — в минусе), треугольник — вход. Фон — "
          "трёхслойный тренд-фильтр (H55): зелёный — вверх, красный — вниз. Интерактивно, со свечами, стопами и "
          f"каждой сделкой — в журнале ({JOURNAL}), эксперименты S45–S48.", "",
          "![SBER: сделки четырёх стратегий, 2023 — середина 2024](iterator_walkthrough/trades_sber.png)", "",
          "![Серебро: сделки четырёх стратегий, 2024 — середина 2025](iterator_walkthrough/trades_silver.png)", ""]

    hdr = "| вариант | доля сделок (выбор / проверка) | " + " | ".join(f"{p}: {k}" for p in PER for k in ("сделок", "итог %", "PF", "без 2 лучших", "просадка", "t")) + " |"
    sep = "|" + "---|" * (2 + 2 * len(KEYS))
    L += ["## H61 — наложение на тренд-фильтр", "",
          "Сделка берётся, только если трёхслойный фильтр на последнем закрытом часе до входа смотрит в её сторону (лонг — "
          "вверх, шорт — вниз). Критерий: у ≥ 3 из 4 стратегий PF и итог без 2 лучших выше в обоих периодах при доле "
          "сделок не меньше трети. S42 — для справки.", ""]
    wins = 0
    for k in ids:
        L += [f"**{k}**", "", hdr, sep]
        for r in h61[k]:
            share = f"{r['выбор_доля']:.0%} / {r['проверка_доля']:.0%}" if "выбор_доля" in r else "100%"
            L.append(f"| {r['вариант']} | {share} | " + " | ".join(cells(r)) + " |")
        L.append("")
        if k != "S42":
            a, f = h61[k][0], h61[k][1]
            better = all((f[p].get("PF") or 0) > (a[p].get("PF") or 0) and (f[p].get("без_2_лучших") or -1e9) > (a[p].get("без_2_лучших") or -1e9)
                         and f[p + "_доля"] >= 1 / 3 for p in PER)
            wins += better
    h61_ok = wins >= 3
    L += [f"Стратегий, где фильтр улучшил PF и итог без 2 лучших в обоих периодах при доле ≥ 1/3: **{wins} из 4** — "
          f"H61 **{'подтвердилась' if h61_ok else 'не подтвердилась'}**.", ""]

    L += ["## H62 — согласие двух стратегий", "",
          "Сделка стратегии A берётся, если на её входе стратегия B держит позицию на том же рынке в ту же сторону (или "
          "открыла её за последние 24 ч). Критерий: PF и t выше, чем у всех сделок A, в обоих периодах, сделок не меньше "
          "30 на период.", "", "| A | B | доля сделок A (выбор / проверка) | " + " | ".join(f"{p}: {k}" for p in PER for k in ("сделок", "PF", "t")) + " | вывод |",
          "|" + "---|" * 10]
    ok62 = []
    for (a, b), r in h62.items():
        base = h61[a][0]
        good = all((r[p].get("сделок") or 0) >= 30 and (r[p].get("PF") or 0) > (base[p].get("PF") or 0)
                   and (r[p].get("t") or -9) > (base[p].get("t") or -9) for p in PER)
        if good:
            ok62.append((a, b))
        L.append(f"| {a} | {b} | {r['выбор_доля']:.0%} / {r['проверка_доля']:.0%} | " + " | ".join(cells(r, ("сделок", "PF", "t"))) +
                 f" | {'лучше' if good else '—'} |")
    L += ["", "Для сравнения — все сделки: " + "; ".join(
        f"{k} PF {fmt(h61[k][0]['выбор'].get('PF'), 'PF')} / {fmt(h61[k][0]['проверка'].get('PF'), 'PF')}, t "
        f"{fmt(h61[k][0]['выбор'].get('t'), 't')} / {fmt(h61[k][0]['проверка'].get('t'), 't')}" for k in ids) + " (выбор / проверка).", ""]
    s42_pairs = [p for p in ok62 if "S42" in p]
    L += [f"Пар, где согласие лучше в обоих периодах при ≥ 30 сделках: **{len(ok62)} из {len(h62)}**, из них с S42 — "
          f"{len(s42_pairs)} ({', '.join(f'{a} при {b}' for a, b in s42_pairs) or 'нет'}).", ""]

    L += ["## Вместе с импульсом — портфель «S42 + стратегия»", "",
          "Капитал поровну между S42 и второй стратегией, внутри — рынки поровну (forexmodel/evaluation/portfolio.py). "
          "Итог портфеля из двух — среднее их результатов, поэтому сравниваем не по сумме, а по t месяцев и итогу к просадке.", "",
          "| портфель | выбор: t мес | итог/просадка | просадка | проверка: t мес | итог/просадка | просадка | корр. месяцев с S42 (выбор / проверка) |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ids:
        p = port[k]
        corr = (f"{p['корр']['выбор'].get('корр_мес', np.nan):.2f} / {p['корр']['проверка'].get('корр_мес', np.nan):.2f}"
                if "корр" in p else "—")
        L.append(f"| {'S42 один' if k == 'S42' else 'S42 + ' + k} | {p['выбор']['t_мес']:.2f} | {p['выбор']['итог_к_просадке']:.2f} | "
                 f"{p['выбор']['просадка']:.1f} | {p['проверка']['t_мес']:.2f} | {p['проверка']['итог_к_просадке']:.2f} | "
                 f"{p['проверка']['просадка']:.1f} | {corr} |")
    L.append("")

    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    # краткая сводка в консоль для вывода
    print(f"H61: {wins}/4; H62: {len(ok62)}/{len(h62)} пар, с S42: {s42_pairs}")
    for k in ids:
        p = port[k]
        print(k, "порт t", round(p["выбор"]["t_мес"], 2), round(p["проверка"]["t_мес"], 2),
              "и/п", round(p["выбор"]["итог_к_просадке"], 2), round(p["проверка"]["итог_к_просадке"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
