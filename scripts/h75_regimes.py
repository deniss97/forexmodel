"""Гипотеза H75 (созвон 05.11, идея Дениса по образцу карты рядов TokDiT): карта режимов рынка и S42 только в «своих»
режимах. Отчёт — docs/results/h75_regimes.md, графики — docs/results/h75_regimes/.

Окно — 70 часовых свечей до сигнала (≈ неделя торгов; без будущего: последняя свеча окна — свеча сигнала S42).
Признаки окна (без масштаба цены): доходность, волатильность часовых доходностей, ATR%, наклон тренда (t) и R²,
эффективность хода (|ход| / путь), автокорреляция, доля растущих часов, асимметрия, эксцесс, просадка в окне, положение
закрытия в диапазоне, ускорение волатильности (последние 15 часов к окну), ход за ~3 месяца (1000 свечей) и
волатильность окна к волатильности за 3 месяца.

Режимы: окна на конец каждого торгового дня по GAZP, SBER, LKOH; масштаб (медиана и разброс), обрезка выбросов и KMeans
k = 5 — только по 2016–2020. Карта — PCA на 2D. Режим сделки S42 — ближайший центр к окну перед её сигналом.
Правило (записано в банке до расчёта): режимы, где S42 на выборе с PF < 1, не торгуем. Критерий: на проверке
2021–2026 с фильтром PF и итог к просадке выше, без 2 лучших > 0, сделок не меньше 60%. k = 4 и 6 — для устойчивости.

    python scripts/h75_regimes.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from forexmodel.evaluation.experiments import TEST_START, metrics  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, WF, load_catalog, load_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "h75_regimes"
DOC = ROOT / "docs" / "results" / "h75_regimes.md"
FEAT = ROOT / "reports" / "night_search" / "feat"
STOCKS = ["gazp", "sber", "lkoh"]
W, LONG, K, SEED = 70, 1000, 5, 0
NAMES = ["ход окна", "волатильность", "ATR %", "наклон (t)", "R² тренда", "эффективность хода", "автокорреляция",
         "доля растущих часов", "асимметрия", "эксцесс", "просадка в окне", "положение в диапазоне",
         "ускорение волатильности", "ход за 3 месяца", "волатильность к 3 месяцам"]
COLORS = ["#2459b3", "#c8641e", "#1e7a4f", "#8a5cc4", "#b3261e", "#6b7785"]
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def features(c: np.ndarray, h: np.ndarray, lo: np.ndarray, end: int) -> np.ndarray | None:
    """Признаки окна, кончающегося свечой end (включительно)."""
    if end < LONG:
        return None
    cw, hw, lw = c[end - W + 1:end + 1], h[end - W + 1:end + 1], lo[end - W + 1:end + 1]
    lc = np.log(cw)
    r = np.diff(lc)
    vol = r.std()
    if not np.isfinite(vol) or vol <= 0:
        return None
    x = np.arange(W)
    fit = stats.linregress(x, lc)
    path = np.abs(r).sum()
    peak = np.maximum.accumulate(cw)
    rl = np.diff(np.log(c[end - LONG:end + 1]))
    rng = cw.max() - cw.min()
    return np.array([
        lc[-1] - lc[0], vol, np.mean((hw - lw) / cw), fit.slope / fit.stderr if fit.stderr > 0 else 0.0, fit.rvalue ** 2,
        abs(lc[-1] - lc[0]) / path if path > 0 else 0.0, np.corrcoef(r[:-1], r[1:])[0, 1], np.mean(r > 0),
        stats.skew(r), stats.kurtosis(r), (cw / peak - 1).min(), (cw[-1] - cw.min()) / rng if rng > 0 else 0.5,
        r[-15:].std() / vol, np.log(c[end] / c[end - LONG]), vol / rl.std()])


def bars(inst: str) -> pd.DataFrame:
    b = pd.read_pickle(FEAT / f"{inst}.pkl")["bars"][["time", "open", "high", "low", "close"]].reset_index(drop=True)
    return b


def daily_windows(inst: str, b: pd.DataFrame) -> pd.DataFrame:
    """Окна на последнюю свечу каждого торгового дня."""
    c, h, lo = (b[k].to_numpy(float) for k in ("close", "high", "low"))
    ends = b.groupby(b["time"].dt.date).tail(1).index.to_numpy()
    rows = []
    for e in ends:
        f = features(c, h, lo, int(e))
        if f is not None and np.all(np.isfinite(f)):
            rows.append([inst, b["time"].iloc[e], *f])
    return pd.DataFrame(rows, columns=["инструмент", "time", *NAMES])


def trade_windows(t: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Окно перед каждой сделкой S42: последняя свеча окна — свеча сигнала (начало ≤ вход − 61 мин)."""
    out = []
    for i, r in t.iterrows():
        bb = b[r["instrument"]]
        tt = bb["time"].to_numpy()
        e = np.searchsorted(tt, np.datetime64(r["open_dt"] - pd.Timedelta(minutes=61)), side="right") - 1
        c, h, lo = (bb[k].to_numpy(float) for k in ("close", "high", "low"))
        f = features(c, h, lo, int(e)) if e >= 0 else None
        out.append([i, *(f if f is not None else [np.nan] * len(NAMES))])
    return pd.DataFrame(out, columns=["i", *NAMES]).set_index("i")


class Regimes:
    """Масштаб (медиана / межквартильный размах), обрезка ±4 и KMeans — только по окнам выбора."""

    def __init__(self, sel: pd.DataFrame, k: int):
        x = sel[NAMES].to_numpy(float)
        self.med = np.median(x, axis=0)
        q = np.percentile(x, [25, 75], axis=0)
        self.iqr = np.where(q[1] - q[0] > 0, q[1] - q[0], 1.0)
        self.km = KMeans(k, n_init=20, random_state=SEED).fit(self.z(sel))
        self.pca = PCA(2, random_state=SEED).fit(self.z(sel))

    def z(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip((df[NAMES].to_numpy(float) - self.med) / self.iqr, -4, 4)

    def label(self, df: pd.DataFrame) -> np.ndarray:
        return self.km.predict(self.z(df))


def describe(rg: Regimes) -> list[str]:
    """Название режима по центру: ход окна и волатильность (в разбросах выбора)."""
    names = []
    for cz in rg.km.cluster_centers_:
        ret, vol, eff, r3 = cz[NAMES.index("ход окна")], cz[NAMES.index("волатильность")], cz[NAMES.index("эффективность хода")], cz[NAMES.index("ход за 3 месяца")]
        a = "рост" if ret > 0.5 else "падение" if ret < -0.5 else "боковик"
        v = "нервный" if vol > 0.6 else "тихий" if vol < -0.4 else "обычный"
        tr = ", ровный ход" if eff > 0.6 else ""
        lg = ", 3 мес. вверх" if r3 > 0.5 else ", 3 мес. вниз" if r3 < -0.5 else ""
        names.append(f"{a}, {v}{tr}{lg}")
    return names


def pf(x: pd.Series) -> float:
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def by_regime(t: pd.DataFrame, lab: np.ndarray, k: int) -> pd.DataFrame:
    rows = []
    sel = t["open_dt"] < TEST_START
    for c in range(k):
        for per, m in (("выбор", sel), ("проверка", ~sel)):
            x = t.loc[m & (lab == c), "profit_pct"]
            rows.append({"режим": c, "период": per, "сделок": len(x), "PF": pf(x), "средняя %": x.mean() if len(x) else np.nan,
                         "сумма %": x.sum()})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    t = t[t["instrument"].isin(STOCKS)].reset_index(drop=True)
    b = {i: bars(i) for i in STOCKS}
    days = pd.concat([daily_windows(i, b[i]) for i in STOCKS], ignore_index=True)
    days = days[days["time"] >= "2016-01-01"]
    sel_days = days[days["time"] < TEST_START]
    tw = trade_windows(t, b)
    ok = tw.notna().all(axis=1)
    t, tw = t[ok.values].reset_index(drop=True), tw[ok].reset_index(drop=True)
    print(f"окон-дней {len(days)} (выбор {len(sel_days)}), сделок S42 с окном {len(t)}", flush=True)

    results, main_rg = {}, None
    for k in (4, K, 6):
        rg = Regimes(sel_days, k)
        lab = rg.label(tw)
        tab = by_regime(t, lab, k)
        bad = [c for c in range(k) if tab[(tab["режим"] == c) & (tab["период"] == "выбор")]["PF"].iloc[0] < 1]
        keep = ~np.isin(lab, bad)
        full_m = {p: metrics(t, STOCKS, p) for p in ("выбор", "проверка")}
        filt_m = {p: metrics(t[keep], STOCKS, p) for p in ("выбор", "проверка")}
        results[k] = {"rg": rg, "lab": lab, "tab": tab, "bad": bad, "keep": keep, "full": full_m, "filt": filt_m}
        print(k, "исключены", bad, "проверка PF", round(full_m["проверка"]["PF"], 3), "→", round(filt_m["проверка"]["PF"], 3),
              "сделок", full_m["проверка"]["сделок"], "→", filt_m["проверка"]["сделок"], flush=True)
        if k == K:
            main_rg = rg
    r = results[K]
    names = describe(main_rg)
    days["режим"] = main_rg.label(days)
    xy = main_rg.pca.transform(main_rg.z(days))
    txy = main_rg.pca.transform(main_rg.z(tw))

    # 1. карта: дни по режимам, сверху — сделки S42 (зелёные / красные)
    f, ax = plt.subplots(1, 2, figsize=(16, 6.2), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(True, color=GRID)
    for c in range(K):
        m = days["режим"].to_numpy() == c
        ax[0].scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.35, color=COLORS[c], label=f"{c + 1}: {names[c]}")
    ax[0].set_title("Карта окон (каждый торговый день, 3 акции): цвет — режим", color=INK, fontsize=12, loc="left")
    ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, markerscale=4, loc="best")
    win = t["profit_pct"].to_numpy() > 0
    ax[1].scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.12, color=GRAY)
    ax[1].scatter(txy[win, 0], txy[win, 1], s=18, color="#1e7a4f", alpha=0.8, label="сделка S42 в плюс")
    ax[1].scatter(txy[~win, 0], txy[~win, 1], s=18, color="#b3261e", alpha=0.8, label="сделка S42 в минус")
    ax[1].set_title("Где на карте входит S42", color=INK, fontsize=12, loc="left")
    ax[1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    for a in ax:
        a.set_xlabel("главная компонента 1", color=INK2)
        a.set_ylabel("главная компонента 2", color=INK2)
    f.tight_layout()
    f.savefig(OUT / "map.png", dpi=120, facecolor=BG)
    plt.close(f)

    # 2. доли режимов по годам
    share = pd.crosstab(days["time"].dt.year, days["режим"], normalize="index")
    f, ax = plt.subplots(figsize=(16, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    bottom = np.zeros(len(share))
    for c in range(K):
        v = share.get(c, pd.Series(0, index=share.index)).to_numpy()
        ax.bar(share.index, v, bottom=bottom, color=COLORS[c], label=f"{c + 1}: {names[c]}")
        bottom += v
    ax.axvline(2021.5, color=INK, linestyle=":")
    ax.text(2021.6, 1.01, "2022 →", color=INK, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("доля торговых дней", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    f.tight_layout()
    f.savefig(OUT / "share_by_year.png", dpi=120, facecolor=BG)
    plt.close(f)

    # 3. PF S42 по режимам, выбор и проверка
    tab = r["tab"]
    f, ax = plt.subplots(figsize=(16, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    kk = np.arange(K)
    for off, per, col in ((-0.2, "выбор", GRAY), (0.2, "проверка", BLUE)):
        v = tab[tab["период"] == per].set_index("режим")["PF"].reindex(kk).to_numpy()
        n = tab[tab["период"] == per].set_index("режим")["сделок"].reindex(kk).to_numpy()
        ax.bar(kk + off, v, 0.38, color=col, label=per)
        for x, y, nn in zip(kk + off, v, n):
            ax.text(x, (y if np.isfinite(y) else 0) + 0.03, f"{y:.2f}\n{nn} сд." if np.isfinite(y) else f"—\n{nn} сд.", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(1, color=INK2, linestyle=":")
    ax.set_xticks(kk)
    ax.set_xticklabels([f"{c + 1}: {names[c]}" + ("\n(не торгуем)" if c in r["bad"] else "") for c in kk], fontsize=9, color=INK)
    ax.set_ylabel("PF сделок S42", color=INK2)
    ax.legend(frameon=False, labelcolor=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    f.tight_layout()
    f.savefig(OUT / "pf_by_regime.png", dpi=120, facecolor=BG)
    plt.close(f)

    # 4. накопленный итог: S42 и S42 с фильтром режимов (акции поровну)
    f, ax = plt.subplots(figsize=(16, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    for lab_, tt, col, lw in (("S42 (GAZP, SBER, LKOH)", t, GRAY, 2.2), ("S42 без режимов с PF < 1 на выборе", t[r["keep"]], BLUE, 2.2)):
        x = tt.sort_values("close_dt")
        ax.plot(x["close_dt"], (x["profit_pct"] / len(STOCKS)).cumsum(), color=col, linewidth=lw, label=lab_)
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    ax.set_ylabel("накопленный итог, %", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    f.tight_layout()
    f.savefig(OUT / "equity.png", dpi=120, facecolor=BG)
    plt.close(f)

    # сделки фильтра — в журнал (S64): строки исходных прогонов S42 по входам, которые фильтр оставил
    kept = t[r["keep"]]
    for inst in STOCKS:
        raw = pd.read_csv(WF / f"{inst}_lo__impulse_6_3_long_trades.csv", parse_dates=["open_dt", "close_dt"])
        keep_dt = set(kept.loc[kept["instrument"] == inst, "open_dt"])
        raw[raw["open_dt"].isin(keep_dt)].to_csv(WF / f"{inst}_lo_reg__impulse_6_3_long_trades.csv", index=False)

    # отчёт
    def mrow(name, m):
        return (f"| {name} | {m['сделок']} | {m['итог']:+.1f} | {m['PF']:.2f} | {m['без_2_лучших']:+.1f} | {m['просадка']:.1f} | "
                f"{m['итог'] / abs(m['просадка']) if m['просадка'] else float('nan'):.2f} | {m['t']:.2f} | {m['лет_в_плюсе']} |")
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Карта режимов рынка: S42 только в «своих» режимах (H75)", "",
          "Идея Дениса (созвон 05.11) по образцу карты рядов TokDiT: описать каждый момент окном часовых свечей, разложить окна "
          "на режимы и посмотреть, в каких режимах зарабатывает импульс S42. Тимур: режимы меняются быстро, история до 2022 года "
          "может мало что значить. Расчёт — `scripts/h75_regimes.py`; сделки S42 — из журнала (GAZP, SBER, LKOH в лонг, "
          "издержки биржи, дивиденды).", "",
          "**Как:** окно — 70 часовых свечей до сигнала S42 (последняя — свеча сигнала, без будущего); 15 признаков без масштаба "
          "цены (ход, волатильность, ATR%, наклон и R² тренда, эффективность хода, автокорреляция, доля растущих часов, асимметрия, "
          "эксцесс, просадка в окне, положение в диапазоне, ускорение волатильности, ход за 3 месяца, волатильность к 3 месяцам). "
          "Масштаб и KMeans (k = 5) — только по окнам 2016–2020 (каждый торговый день, 3 акции); карта — PCA на 2D.", "",
          "**Правило и критерий (банк, до расчёта):** режимы, где S42 на выборе с PF < 1, не торгуем; на проверке 2021–2026 с "
          "фильтром PF и итог к просадке выше, без 2 лучших > 0, сделок не меньше 60% от S42.", "", verdict, "",
          "![Карта](h75_regimes/map.png)", "", "## Режимы", "",
          "| режим | выбор: сделок | PF | средняя % | проверка: сделок | PF | средняя % | доля дней 2016–2021 | доля дней 2022–2026 |",
          "|---|---|---|---|---|---|---|---|---|"]
    pre = days[days["time"].dt.year <= 2021]["режим"].value_counts(normalize=True)
    post = days[days["time"].dt.year >= 2022]["режим"].value_counts(normalize=True)
    for c in range(K):
        s = tab[(tab["режим"] == c) & (tab["период"] == "выбор")].iloc[0]
        q = tab[(tab["режим"] == c) & (tab["период"] == "проверка")].iloc[0]
        md.append(f"| {c + 1}: {names[c]}{' — **не торгуем**' if c in r['bad'] else ''} | {s['сделок']} | {s['PF']:.2f} | {s['средняя %']:+.2f} | "
                  f"{q['сделок']} | {q['PF']:.2f} | {q['средняя %']:+.2f} | {pre.get(c, 0):.0%} | {post.get(c, 0):.0%} |")
    md += ["", "![PF по режимам](h75_regimes/pf_by_regime.png)", "", "![Доли режимов по годам](h75_regimes/share_by_year.png)", "",
           "## S42 с фильтром режимов", "",
           "| вариант | сделок | итог % | PF | без 2 лучших | просадка | итог / просадка | t | лет в плюсе |", "|---|---|---|---|---|---|---|---|---|"]
    for per in ("выбор", "проверка"):
        md += [mrow(f"S42 · {per}", r["full"][per]), mrow(f"**с фильтром · {per}**", r["filt"][per])]
    md += ["", "![Накопленный итог](h75_regimes/equity.png)", "", "## Устойчивость к числу режимов", "",
           "| k | исключены режимы | проверка: PF без фильтра → с фильтром | сделок | итог / просадка |", "|---|---|---|---|---|"]
    for k, v in results.items():
        a, bq = v["full"]["проверка"], v["filt"]["проверка"]
        md.append(f"| {k} | {', '.join(str(c + 1) for c in v['bad']) or '—'} | {a['PF']:.2f} → {bq['PF']:.2f} | {a['сделок']} → {bq['сделок']} | "
                  f"{a['итог'] / abs(a['просадка']):.2f} → {bq['итог'] / abs(bq['просадка']):.2f} |")
    md += ["", "## Центры режимов (в разбросах выбора)", "", "| признак | " + " | ".join(f"{c + 1}" for c in range(K)) + " |",
           "|---|" + "---|" * K]
    for j, n in enumerate(NAMES):
        md.append(f"| {n} | " + " | ".join(f"{main_rg.km.cluster_centers_[c][j]:+.2f}" for c in range(K)) + " |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
