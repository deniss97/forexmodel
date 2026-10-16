"""Тренд-фильтр как разметка тренда для модели (гипотеза H55, задачи T23–T24).

Не гейт импульса, а разметка: насколько фильтр верно показывает направление текущего тренда. Эталон
тренда задним числом — ноги ZigZag по закрытиям часа на трёх масштабах: 3 ATR (короткие движения),
6 ATR (масштаб сделки импульса), 12 ATR (долгий тренд). Метрики по масштабу и периоду:
  * верно      — доля часов, где знак фильтра совпадает с направлением ноги (среди часов, где фильтр не 0);
  * говорит    — доля часов, где фильтр не нейтрален;
  * задержка   — медиана часов от начала ноги до первого бара фильтра в её сторону;
  * ложных     — входов фильтра в состояние против ноги внутри неё, в среднем на ногу;
  * ход 10 / 72 — средний ход цены за 10 и 72 бара в сторону фильтра, в ATR (у «всегда лонг» — дрейф).
Фильтры — как есть, без подбора; значения только по закрытым барам. Свечи — reports/trend_grid/candidates.pkl.

    python scripts/trend_labeling.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_grid as tg  # noqa: E402
from experiments import BG, GRID, INK, INK2, ROOT  # noqa: E402

OUT = ROOT / "docs" / "results" / "trend_labeling"
DOC = ROOT / "docs" / "results" / "trend_labeling.md"
SET = ["silver", "gold", "gazp", "sber", "lkoh"]
SCALES = (3.0, 6.0, 12.0)
HORIZONS = (10, 72)
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def adx(b: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up, dn = b["high"].diff(), -b["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=b.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=b.index)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - b["close"].shift()).abs(), (b["low"] - b["close"].shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, n)
    pdi, ndi = 100 * wilder(pdm, n) / atr, 100 * wilder(ndm, n) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return wilder(dx, n), pdi, ndi


def timur(bars: pd.DataFrame, ema: int, adx_n: int, thr: float, max_ffill: int | None) -> np.ndarray:
    """Фильтр из ноутбука Тимура S_TrendFiter_dev: знак (close − EMA) на 4 ч, если ADX > порога; иначе старое
    направление (v1 — без ограничения, v2 — не дольше max_ffill 4-часовых баров, потом 0). Значение 4-часового
    бара доступно после его закрытия (как _merge_trend_no_leak в ноутбуке)."""
    b4 = bars.set_index("time")[["open", "high", "low", "close"]].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    e = b4["close"].ewm(span=ema, adjust=False).mean()
    a, pdi, ndi = adx(b4, adx_n)
    conf = (a > thr) & (pdi != ndi)
    raw = pd.Series(np.where(conf, np.where(b4["close"] > e, 1.0, -1.0), np.nan), index=b4.index)
    tr = raw.ffill() if max_ffill is None else raw.ffill(limit=max_ffill).fillna(0.0)
    s = pd.DataFrame({"time": b4.index + pd.Timedelta("4h"), "v": tr.to_numpy()})
    m = pd.merge_asof(bars[["time"]].sort_values("time"), s, on="time", direction="backward")
    return m["v"].fillna(0.0).to_numpy()


FILTERS = {
    "гейт C (база)": ("grid", {}),
    "трёхслойный ТЗ как есть": ("grid", {"mode": "score"}),
    "трёхслойный, режим по дневной EMA200": ("grid", {"mode": "score", "score_regime_tf": "24h"}),
    "Тимур v1: EMA21 4ч + ADX14": ("timur", (21, 14, 20.0, None)),
    "Тимур v2: EMA5 4ч + ADX9, протяжка 1 бар": ("timur", (5, 9, 20.0, 1)),
    "Дончиан 55/20 на 4 ч": ("grid", {"mode": "classic", "rule": "donchian:55:20", "timeframe": "4h", "update_every_bar": True, "hold_bars": 0}),
    "EMA 20/50 на 4 ч": ("grid", {"mode": "classic", "rule": "ema_cross:20:50", "timeframe": "4h", "update_every_bar": True, "hold_bars": 0}),
    "всегда лонг": ("grid", {"always": 1}),
}


def values(bars: pd.DataFrame, spec) -> np.ndarray:
    kind, p = spec
    v = tg.gate_values(bars, p) if kind == "grid" else timur(bars, *p)
    return np.nan_to_num(np.asarray(v, dtype=float))


def score_one(bars: pd.DataFrame, g: np.ndarray) -> list[dict]:
    """Метрики одного фильтра на одном рынке: строки (масштаб, период)."""
    c = bars["close"].to_numpy(float)
    atr = (bars["high"] - bars["low"]).rolling(14).mean().to_numpy()
    year = bars["time"].dt.year.to_numpy()
    sel = (year >= 2016) & (year < 2021)
    test = year >= 2021
    enter = np.r_[False, (g[1:] != g[:-1]) & (g[1:] != 0)]
    rows = []
    for k in SCALES:
        leg_dir = np.zeros(len(c))
        legs = tg.zigzag(c, atr, k)
        lag, false, start = [], [], []
        for s0, e0, d in legs:
            leg_dir[s0 + 1:e0 + 1] = d
            seg = g[s0 + 1:e0 + 1]
            hit = np.flatnonzero(seg == d)
            lag.append(int(hit[0]) + 1 if hit.size else e0 - s0)
            false.append(int((enter[s0 + 1:e0 + 1] & (seg == -d)).sum()))
            start.append(year[s0])
        lag, false, start = np.array(lag), np.array(false), np.array(start)
        for per, m in (("выбор", sel), ("проверка", test)):
            on = m & (leg_dir != 0) & (g != 0)
            lm = (start >= 2016) & (start < 2021) if per == "выбор" else start >= 2021
            rows.append({"масштаб": k, "период": per, "верно": float((g[on] == leg_dir[on]).mean()) if on.any() else np.nan,
                         "говорит": float((g[m] != 0).mean()), "задержка": float(np.median(lag[lm])) if lm.any() else np.nan,
                         "ложных": float(false[lm].mean()) if lm.any() else np.nan, "ног": int(lm.sum())})
    for per, m in (("выбор", sel), ("проверка", test)):
        for h in HORIZONS:
            fwd = np.full(len(c), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd[:-h] = (c[h:] - c[:-h]) / atr[:-h]
            ok = m & np.isfinite(fwd)
            on, up, dn = ok & (g != 0), ok & (g > 0), ok & (g < 0)
            # разница «после вверх − после вниз» не зависит от дрейфа рынка: у фильтра без умения она около нуля
            rows.append({"масштаб": f"ход {h}", "период": per, "ход": float(np.mean(g[on] * fwd[on])) if on.any() else np.nan,
                         "разница": float(fwd[up].mean() - fwd[dn].mean()) if up.any() and dn.any() else np.nan})
    return rows


def labeling_png(bars: pd.DataFrame, gs: dict, path, start="2025-06-01", end="2026-05-29") -> None:
    """Цена серебра и разметка трёх фильтров: зелёный — вверх, красный — вниз, без цвета — нейтрально."""
    names = ["трёхслойный ТЗ как есть", "гейт C (база)", "Тимур v2: EMA5 4ч + ADX9, протяжка 1 бар"]
    m = (bars["time"] >= start) & (bars["time"] <= end)
    t, c = bars.loc[m, "time"].to_numpy(), bars.loc[m, "close"].to_numpy()
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 7.5), sharex=True, facecolor=BG)
    for ax, n in zip(axes, names):
        g = gs[n][m.to_numpy()]
        ax.set_facecolor(BG)
        ax.plot(t, c, color=INK, linewidth=1.0)
        lo, hi = np.nanmin(c), np.nanmax(c)
        ax.fill_between(t, lo, hi, where=g > 0, color="#1d7a55", alpha=0.18, step="post", lw=0)
        ax.fill_between(t, lo, hi, where=g < 0, color="#b4462a", alpha=0.18, step="post", lw=0)
        ax.set_title(n, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def accuracy_png(df: pd.DataFrame, path) -> None:
    x = df[(df["период"] == "проверка") & df["масштаб"].isin(SCALES)].pivot_table(index="фильтр", columns="масштаб", values="верно")
    x = x.reindex([n for n in FILTERS if n in x.index])
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    w, pos = 0.26, np.arange(len(x))
    for j, (k, col) in enumerate(zip(SCALES, ("#9aa3ae", "#2459b3", "#8a5cc4"))):
        ax.barh(pos + (j - 1) * w, x[k] * 100, height=w, color=col, label=f"ZigZag {k:g} ATR")
    ax.axvline(50, color=INK2, linewidth=1)
    ax.set_yticks(pos, x.index, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(40, max(75, float(np.nanmax(x.to_numpy())) * 100 + 3))
    ax.set_xlabel("верно показанных часов на проверке 2021–2026, %", color=INK2, fontsize=9)
    ax.grid(True, axis="x", color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def main() -> int:
    tg._init()
    data = tg._W["data"]
    rows, silver = [], {}
    for name, spec in FILTERS.items():
        for inst in SET:
            bars = data[inst]["bars"]
            g = values(bars, spec)
            if inst == "silver":
                silver[name] = g
            rows += [{"фильтр": name, "рынок": inst, **r} for r in score_one(bars, g)]
        print(name, flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby(["фильтр", "масштаб", "период"], sort=False).mean(numeric_only=True).reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT / "metrics.csv", index=False)
    accuracy_png(agg, OUT / "accuracy.png")
    labeling_png(data["silver"]["bars"], silver, OUT / "silver_labeling.png")

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    md = ["# Тренд-фильтр как разметка тренда для модели", "",
          "Гипотеза **H55**, задачи **T23–T24**. Фильтр оцениваем не как гейт импульса, а как разметку: верно ли он "
          "показывает направление текущего тренда. Эталон тренда задним числом — ноги ZigZag по закрытиям часа на "
          "трёх масштабах: **3 ATR** (короткие движения), **6 ATR** (масштаб сделки импульса), **12 ATR** (долгий "
          "тренд). Рынки — серебро, золото, GAZP, SBER, LKOH, среднее по рынкам. Фильтры — как есть, без подбора, "
          "только по закрытым барам. Расчёт — `scripts/trend_labeling.py`.", "",
          "* **верно** — доля часов, где знак фильтра совпадает с направлением ноги (среди часов, где фильтр не нейтрален);",
          "* **говорит** — доля часов, где фильтр не нейтрален;",
          "* **задержка** — медиана часов от начала ноги до первого бара фильтра в её сторону;",
          "* **ложных** — входов фильтра в состояние против ноги внутри неё, в среднем на ногу;",
          "* **ход 10 / 72** — средний ход цены за 10 и 72 бара в сторону фильтра, в ATR (у «всегда лонг» — просто дрейф).", "",
          "![Доля верных часов](trend_labeling/accuracy.png)", "", verdict, ""]
    for k in SCALES:
        md += [f"## ZigZag {k:g} ATR", "", "| фильтр | период | верно | говорит | задержка, ч | ложных на ногу | ног |", "|---|---|---|---|---|---|---|"]
        for _, r in agg[agg["масштаб"] == k].iterrows():
            md.append(f"| {r['фильтр']} | {r['период']} | {r['верно']:.1%} | {r['говорит']:.0%} | {r['задержка']:.0f} | {r['ложных']:.1f} | {r['ног']:.0f} |")
        md.append("")
    md += ["## Ход цены после сигнала, ATR", "",
           "«В сторону фильтра» включает дрейф рынка (у «всегда лонг» — только дрейф). Умение фильтра без дрейфа — "
           "**разница**: средний ход после «вверх» минус после «вниз»; у фильтра без умения она около нуля.", "",
           "| фильтр | период | в сторону, 10 баров | в сторону, 72 бара | разница, 10 баров | разница, 72 бара |", "|---|---|---|---|---|---|"]
    for (f, per), x in agg[agg["масштаб"].astype(str).str.startswith("ход")].groupby(["фильтр", "период"], sort=False):
        v, d = dict(zip(x["масштаб"], x["ход"])), dict(zip(x["масштаб"], x["разница"]))
        fd = lambda z: "—" if not np.isfinite(z) else f"{z:+.3f}"  # noqa: E731
        md.append(f"| {f} | {per} | {v['ход 10']:+.3f} | {v['ход 72']:+.3f} | {fd(d['ход 10'])} | {fd(d['ход 72'])} |")
    md += ["", "## Как фильтры размечают серебро (июнь 2025 — май 2026)", "",
           "Зелёный — фильтр показывает тренд вверх, красный — вниз, без цвета — нейтрально.", "",
           "![Разметка серебра](trend_labeling/silver_labeling.png)", ""]
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
