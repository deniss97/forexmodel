"""H59, уточнение Тимура 04.11: стратегия по тренд-фильтру с выходом раньше смены тренда — отдельный эксперимент
(S49 не трогаем). Отчёт — docs/results/h59_trend_exits.md, сделки лучшего варианта — журнал (S63).

Тимур: S49 «недобирает возможной прибыли, но генерит практически максимально возможное количество убытков»; нужен
выход ближе к пику или при небольшом откате от него и ранний выход при убытке; «похоже на трейлинг-стоп» или правило
вроде «закрытие ниже EMA 5, но не напрямую» (в чистом виде ухудшает).

Вход — как у S49: по открытию часа после того, как трёхслойный фильтр (режим по дневной EMA200) включил направление
(металлы — обе стороны через CFD, акции — только лонг). Выход — что раньше:
  * фильтр вышел из направления (как у S49) — по открытию следующего часа;
  * начальный стоп sl ATR от входа (0 — нет);
  * трейлинг trail ATR от лучшей цены сделки (0 — нет);
  * «мягкая EMA 5»: после того как сделка прошла +ema ATR в плюс, закрытие часа по другую сторону EMA(5) — выход по
    открытию следующего часа (0 — нет).
Стоп и трейлинг — по максимуму / минимуму часа, при гэпе — по открытию. После выхода по стопу новый вход — только
по новому включению фильтра (сигнал прежний, вход в ту же сторону — нет).

Сетка 3 × 5 × 3 = 45 вариантов. Выбор — только по 2016–2020: среди вариантов с ≥ 10 сделок в год и итогом без
2 лучших > 0 — наибольший PF. Проверка 2021 — июнь 2026 — один раз, для выбранного; остальные — для картины.
Критерий H59 (записан 18.10): PF > 1.1 и без 2 лучших > 0 в обоих периодах, ≥ 10 сделок в год; с S42 лучше S42 по t
месяцев и итогу к просадке в обоих периодах.

    python scripts/h59_trend_exits.py
"""
from __future__ import annotations

import itertools

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_follow as tfw  # noqa: E402
import trend_grid as tg  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import TEST_START, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, WF, load_catalog, load_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "h59_trend_exits"
DOC = ROOT / "docs" / "results" / "h59_trend_exits.md"
SET, STOCKS = tfw.SET, tfw.STOCKS
SL, TRAIL, EMA = (0, 2, 3), (0, 2, 3, 4, 6), (0, 1, 2)
RUN = "tf3x"                     # имя прогона лучшего варианта в reports/walk_forward


def atr14(b: pd.DataFrame) -> np.ndarray:
    pc = b["close"].shift(1)
    tr = np.maximum(b["high"] - b["low"], np.maximum((b["high"] - pc).abs(), (b["low"] - pc).abs()))
    return tr.rolling(14).mean().to_numpy(float)


def trades(b: pd.DataFrame, g: np.ndarray, long_only: bool, sl: float, trail: float, ema: float) -> pd.DataFrame:
    g = np.nan_to_num(g).astype(int)
    if long_only:
        g = np.maximum(g, 0)
    t = b["time"].to_numpy()
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    e5 = b["close"].ewm(span=5, adjust=False).mean().to_numpy(float)
    a = atr14(b)
    yr = b["time"].dt.year.to_numpy()
    n, rows, i = len(g), [], 1
    while i < n - 1:
        if yr[i] < 2016 or g[i] == 0 or g[i] == g[i - 1] or not np.isfinite(a[i]):
            i += 1
            continue
        s, e, A = g[i], i + 1, a[i]                          # сигнал на закрытии i, вход по открытию i+1
        ent = o[e]
        # зеркало шорта: работаем с ценой × s, логика лонга
        stop = ent * s - sl * A if sl else -np.inf
        best, x, px, why, nxt = ent * s, None, None, None, None
        for j in range(e, n):
            hi, lo, op = (h[j] if s > 0 else -l[j]), (l[j] if s > 0 else -h[j]), o[j] * s
            cur = max(stop, best - trail * A) if trail else stop
            if lo <= cur:
                x, px, why = j, min(op, cur) * s, "трейлинг" if trail and cur > stop else "стоп"
                break
            best = max(best, hi)
            if j + 1 >= n:
                x, px, why = j, c[j], "конец данных"
                break
            if g[j] != s:
                # переворот +1 → −1 на этом же часе — вход в обратную сторону по тому же сигналу (как S49)
                x, px, why, nxt = j + 1, o[j + 1], "фильтр", j
                break
            if ema and best - ent * s >= ema * A and c[j] * s < e5[j] * s:
                x, px, why = j + 1, o[j + 1], "EMA 5 после прибыли"
                break
        rows.append((e, x, s, ent, px, why))
        # новый вход — по новому включению фильтра; после выхода по фильтру сигнал часа выхода проверяется сразу
        i = nxt if nxt is not None else max(x, i + 1)
    if not rows:
        return pd.DataFrame()
    e, x, s, ent, px, why = (np.array(v) for v in zip(*rows))
    gross = s * (px / ent - 1) * 100
    return pd.DataFrame({"open_dt": pd.to_datetime(t[e]) + pd.Timedelta(minutes=1), "close_dt": pd.to_datetime(t[x]) + pd.Timedelta(minutes=1),
                         "side": np.where(s > 0, "buy", "sell"), "open_price": ent, "exit_price": px, "size": 1.0,
                         "gross_unit_pct": gross, "gross_pct": gross, "exit_reason": why})


def variant(data: dict, gates: dict, sl, trail, ema, raw: bool = False):
    parts, rawp = [], {}
    for inst in SET:
        t = trades(data[inst]["bars"], gates[inst], inst in STOCKS, sl, trail, ema)
        rawp[inst] = t
        t = recost(t, profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))
        parts.append(standardize(t, inst, f"sl{sl} tr{trail} ema{ema}"))
    out = pd.concat(parts, ignore_index=True)
    return (out, rawp) if raw else out


def name(sl, trail, ema) -> str:
    p = [f"стоп {sl} ATR" if sl else None, f"трейлинг {trail} ATR" if trail else None, f"EMA 5 после +{ema} ATR" if ema else None]
    return ", ".join(x for x in p if x) or "как S49 (только фильтр)"


def f(v, d=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.1f}" if d == 1 else f"{v:.{d}f}")


def main() -> int:
    tg._init()
    data = tg._W["data"]
    gates = {inst: tg.gate_values(data[inst]["bars"], tfw.TL) for inst in SET}
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    alone = pf.combine({"S42": s42}, SET)
    rows, cache = [], {}
    for sl, trail, ema in itertools.product(SL, TRAIL, EMA):
        v = variant(data, gates, sl, trail, ema)
        cache[(sl, trail, ema)] = v
        solo = pf.combine({"v": v}, SET)
        r = {"вариант": name(sl, trail, ema), "sl": sl, "trail": trail, "ema": ema}
        for per in ("выбор", "проверка"):
            m, p = pf.metrics(solo, per), pf.metrics(pf.combine({"S42": s42, "v": v}, SET), per)
            r.update({f"{per}: {k}": m[k] for k in ("в_год", "итог", "PF", "без_2_лучших", "просадка", "t_мес")})
            r.update({f"{per}: с S42 t": p["t_мес"], f"{per}: с S42 итог/просадка": p["итог_к_просадке"]})
        rows.append(r)
        print(r["вариант"], f"выбор PF {r['выбор: PF']:.2f}", f"проверка PF {r['проверка: PF']:.2f}", flush=True)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "grid.csv", index=False)
    # выбираем среди вариантов С ранним выходом (сам фильтр — это S49); никто не прошёл отбор — лучший по PF выбора
    new = R[(R.sl > 0) | (R.trail > 0) | (R.ema > 0)]
    ok = new[(new["выбор: в_год"] >= 10) & (new["выбор: без_2_лучших"] > 0)]
    best = ok.sort_values("выбор: PF", ascending=False).iloc[0] if len(ok) else new.sort_values("выбор: PF", ascending=False).iloc[0]
    key = (int(best["sl"]), int(best["trail"]), int(best["ema"]))
    # сделки выбранного варианта — в журнал (издержки пересчитает сборка журнала по профилю эксперимента)
    _, rawp = variant(data, gates, *key, raw=True)
    for inst, t in rawp.items():
        t.to_csv(WF / f"{inst}_{RUN}__follow_trades.csv", index=False)
    a = alone
    sa = {per: pf.metrics(a, per) for per in ("выбор", "проверка")}
    s49 = R[(R.sl == 0) & (R.trail == 0) & (R.ema == 0)].iloc[0]

    fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    for lab, v, col, lw in (("S49: только фильтр", cache[(0, 0, 0)], GRAY, 1.6), (f"выбранный: {best['вариант']}", cache[key], BLUE, 2.4)):
        x = pf.combine({"v": v}, SET).sort_values("close_dt")
        ax.plot(x["close_dt"], (x["w"] * x["profit_pct"]).cumsum(), color=col, lw=lw, label=lab)
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=BLUE, fontsize=9)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, % капитала (5 рынков)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    va, vb = old.find("<!-- вывод -->"), old.find("<!-- /вывод -->")
    verdict = old[va:vb + len("<!-- /вывод -->")] if va >= 0 and vb > va else "<!-- вывод -->\n<!-- /вывод -->"
    md = ["# Стратегия по тренд-фильтру: выход раньше смены тренда (H59, уточнение 04.11)", "",
          "Тимур: S49 «недобирает прибыль, но генерит почти максимально возможное количество убытков» — нужен выход ближе "
          "к пику и ранний выход при убытке. Вход как у S49 (трёхслойный фильтр включил направление; металлы через CFD в обе "
          "стороны, акции в лонг); выход — что раньше: фильтр, начальный стоп, трейлинг от лучшей цены, «мягкая EMA 5» "
          "(закрытие по другую сторону EMA 5 — только после того, как сделка прошла +X ATR). Сетка 45 вариантов; выбор — "
          "среди вариантов с ранним выходом только по 2016–2020 (≥ 10 сделок в год, без 2 лучших > 0, наибольший PF; не прошёл "
          "никто — лучший по PF), проверка — один раз. Издержки как в "
          "журнале. Расчёт — `scripts/h59_trend_exits.py`.", "",
          verdict, "", "![Накопленный итог](h59_trend_exits/equity.png)", ""]
    for per in ("выбор", "проверка"):
        md += [f"## {'Выбор 2016–2020' if per == 'выбор' else 'Проверка 2021 — июнь 2026'}", "",
               "| вариант | сделок в год | итог % | PF | без 2 лучших | просадка | t мес | с S42: t мес | итог / просадка |",
               "|---|---|---|---|---|---|---|---|---|",
               f"| S42 один | {sa[per]['в_год']:.1f} | {f(sa[per]['итог'], 1)} | {f(sa[per]['PF'])} | {f(sa[per]['без_2_лучших'], 1)} | "
               f"{f(sa[per]['просадка'], 1)} | {f(sa[per]['t_мес'])} | — | {f(sa[per]['итог_к_просадке'])} |"]
        for lab, r in (("S49: только фильтр", s49), (f"**выбранный: {best['вариант']}**", best)):
            md.append(f"| {lab} | {r[f'{per}: в_год']:.1f} | {f(r[f'{per}: итог'], 1)} | {f(r[f'{per}: PF'])} | {f(r[f'{per}: без_2_лучших'], 1)} | "
                      f"{f(r[f'{per}: просадка'], 1)} | {f(r[f'{per}: t_мес'])} | {f(r[f'{per}: с S42 t'])} | {f(r[f'{per}: с S42 итог/просадка'])} |")
        md.append("")
    md += ["## Вся сетка", "", "PF на выборе → на проверке; жирным — прошёл отбор по выбору.", "",
           "| вариант | выбор: сделок в год | выбор: PF | выбор: без 2 лучших | проверка: PF | проверка: без 2 лучших |", "|---|---|---|---|---|---|"]
    for _, r in R.sort_values("выбор: PF", ascending=False).iterrows():
        passed = r["выбор: в_год"] >= 10 and r["выбор: без_2_лучших"] > 0
        lab = f"**{r['вариант']}**" if passed else r["вариант"]
        md.append(f"| {lab} | {r['выбор: в_год']:.1f} | {f(r['выбор: PF'])} | {f(r['выбор: без_2_лучших'], 1)} | {f(r['проверка: PF'])} | "
                  f"{f(r['проверка: без_2_лучших'], 1)} |")
    v = cache[key]
    md += ["", "## Выбранный вариант по рынкам — проверка", "", "| рынок | сделок | итог % | PF | средняя сделка % | выходы |", "|---|---|---|---|---|---|"]
    for i in SET:
        x = v[(v["instrument"] == i) & (v["open_dt"] >= TEST_START)]
        loss = -x.loc[x["profit_pct"] < 0, "profit_pct"].sum()
        why = ", ".join(f"{k} {n}" for k, n in x["exit_reason"].value_counts().items())
        md.append(f"| {i} | {len(x)} | {x['profit_pct'].sum():+.1f} | {(x.loc[x['profit_pct'] > 0, 'profit_pct'].sum() / loss) if loss > 0 else float('nan'):.2f} | "
                  f"{x['profit_pct'].mean():+.2f} | {why} |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("выбран:", best["вариант"], "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
