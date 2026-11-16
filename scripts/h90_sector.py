"""H90: акции нефтегаза — тянет ли резкий ход одной бумаги соседей по сектору (план подтверждён в банке 15.11).

Ответы автора на вопросы плана: состав сектора не менять (LKOH, GAZP, ROSN, TATN, NVTK, SNGSP); брать все резкие
движения (новостей у нас нет) и, кроме того, случаи, когда связь бумаг отклоняется от обычной; считать обе стороны,
лонг — отдельной строкой.

Часть 1 — «толчок». Минутные закрытия основной сессии 10:00–18:39. Толчок: бумага за 15 минут ушла не меньше чем на
порог (1.0 / 1.5 / 2.0%), а среднее остальных пяти за те же 15 минут — меньше трети порога. Один толчок на бумагу в
60 минут. Через `delay` минут (2 и 5) смотрим ход соседей в сторону толчка: +15, +30, +60 минут, до 18:39.
Простая торговля: самый отстающий сосед (наименьший ход в сторону толчка за 15 минут толчка), вход через delay минут
по закрытию минуты, выход через 60 минут или по стопу 1% на закрытии минуты; издержки 0.1% за круг (0.05% на сторону).

Часть 2 — «связь разошлась». Дневные закрытия, индекс сектора — среднее лог-доходностей шести бумаг. Для бумаги: спред
= накопленная доходность бумаги минус сектора; z — отклонение спреда от 20-дневного среднего в 20-дневных σ.
Событие: |z| ≥ 2 и корреляция дневных доходностей бумаги с сектором за 20 дней ниже своей медианы за 250 дней на 0.2
и больше («связь ослабла»). Ход: бумага против сектора за следующие 1 / 3 / 5 дней в сторону схождения; торговля —
вход на закрытии дня события, выход через 5 дней, издержки 0.1% за круг (только бумага, без хеджа сектором).

Протокол: выбор порога — 2016–2020, проверка 2021–2026 (половины 2021–2023 и 2024–2026).
    python scripts/h90_sector.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "results" / "h90_sector"
OUT.mkdir(parents=True, exist_ok=True)
TK = ["LKOH", "GAZP", "ROSN", "TATN", "NVTK", "SNGSP"]
SRC = {"GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv"}
COST = 0.10  # % за круг
THRS = [1.0, 1.5, 2.0]
DELAYS = [2, 5]
HOR = [15, 30, 60]
STOP = 1.0


def period(ts: pd.Series) -> np.ndarray:
    y = ts.dt.year.to_numpy()
    return np.where(y <= 2020, "выбор 2016–2020", np.where(y <= 2023, "проверка 2021–2023", "проверка 2024–2026"))


def load_minutes() -> pd.DataFrame:
    cols = {}
    for tk in TK:
        src = SRC.get(tk, RAW / "moex" / f"{tk}_1m.csv")
        d = pd.read_csv(src, usecols=["close", "begin"], dtype={"close": "float32"})
        d["begin"] = pd.to_datetime(d["begin"])
        d = d[(d["begin"] >= "2016-01-01")]
        t = d["begin"].dt.hour * 60 + d["begin"].dt.minute
        d = d[(t >= 600) & (t < 1120)]  # 10:00–18:39
        cols[tk] = d.drop_duplicates("begin").set_index("begin")["close"]
        print(tk, len(d), flush=True)
    m = pd.DataFrame(cols).sort_index()
    # полный минутный ряд каждого дня, пропуски — последней ценой (не дальше 5 минут)
    days = m.index.normalize().unique()
    grid = pd.DatetimeIndex(np.concatenate([d + pd.to_timedelta(np.arange(600, 1120), "min") for d in days]))
    m = m.reindex(grid).groupby(grid.normalize()).ffill(limit=5)
    return m.astype("float32")


def shocks(m: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(m.to_numpy(dtype="float64"))
    idx = m.index
    day = idx.normalize().to_numpy()
    n = len(idx)
    rows = []
    r15 = np.full_like(lp, np.nan)
    r15[15:] = (lp[15:] - lp[:-15]) * 100
    same = np.zeros(n, bool)
    same[15:] = day[15:] == day[:-15]
    r15[~same] = np.nan
    last = {}
    for thr in THRS:
        for j, tk in enumerate(TK):
            others = [k for k in range(len(TK)) if k != j]
            om = np.nanmean(r15[:, others], axis=1)
            cand = np.where((np.abs(r15[:, j]) >= thr) & (np.abs(om) < thr / 3))[0]
            prev = -10**9
            for i in cand:
                if i - prev < 60:
                    continue
                prev = i
                sgn = np.sign(r15[i, j])
                for dl in DELAYS:
                    e = i + dl
                    if e >= n or day[e] != day[i]:
                        continue
                    nb = {}
                    for k in others:
                        rec = {"nb": TK[k], "lag15": sgn * r15[i, k]}
                        for h in HOR:
                            x = e + h
                            rec[f"h{h}"] = sgn * (lp[x, k] - lp[e, k]) * 100 if x < n and day[x] == day[i] else np.nan
                        # до конца дня
                        endi = e + (np.searchsorted(day[e:], day[i], side="right") - 1)
                        rec["eod"] = sgn * (lp[endi, k] - lp[e, k]) * 100
                        nb[TK[k]] = rec
                    # отстающий сосед и сделка
                    lag = min(nb.values(), key=lambda r: r["lag15"] if np.isfinite(r["lag15"]) else 1e9)
                    k = TK.index(lag["nb"])
                    endi = min(e + 60, e + (np.searchsorted(day[e:], day[i], side="right") - 1))
                    path = sgn * (lp[e + 1:endi + 1, k] - lp[e, k]) * 100
                    path = path[np.isfinite(path)]
                    hit = np.where(path <= -STOP)[0]
                    gross = path[hit[0]] if len(hit) else (path[-1] if len(path) else np.nan)
                    for r in nb.values():
                        rows.append({"thr": thr, "delay": dl, "t": idx[i], "src": tk, "dir": sgn,
                                     "shock": abs(r15[i, j]), **r, "trade": r["nb"] == lag["nb"],
                                     "gross": gross if r["nb"] == lag["nb"] else np.nan})
    d = pd.DataFrame(rows)
    d["per"] = period(d["t"])
    d["net"] = d["gross"] - COST
    return d


def md(df: pd.DataFrame, index: bool) -> str:
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def pf(x: pd.Series) -> float:
    x = x.dropna()
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else np.nan


def trade_stats(t: pd.DataFrame) -> dict:
    x = t["net"].dropna()
    s = x.sort_values()
    return {"сделок": len(x), "итог %": round(x.sum(), 1), "PF": round(pf(x), 2),
            "без 2 лучших %": round(s.iloc[:-2].sum(), 1) if len(s) > 2 else np.nan,
            "средняя %": round(x.mean(), 3), "без издержек средняя %": round(t["gross"].mean(), 3),
            "доля плюсовых": round((x > 0).mean(), 2)}


def part1(m: pd.DataFrame) -> list[str]:
    d = shocks(m)
    d.to_parquet(OUT / "shocks.parquet", index=False)
    lines = ["## Часть 1 — толчок одной бумаги", ""]
    # догоняют ли соседи: средний ход соседей в сторону толчка, доля от толчка
    g = d.groupby(["thr", "delay", "per"])
    tab = g.agg(толчков=("t", lambda s: s.nunique()), h15=("h15", "mean"), h30=("h30", "mean"),
                h60=("h60", "mean"), eod=("eod", "mean"), толчок=("shock", "mean")).reset_index()
    tab["доля догона 60 мин"] = (tab["h60"] / tab["толчок"]).round(2)
    lines += ["Средний ход соседа в сторону толчка, % (после входа через delay минут):", "",
              md(tab.round(3), False), ""]
    # торговля
    tr = d[d["trade"]]
    rows = []
    for (thr, dl, per), t in tr.groupby(["thr", "delay", "per"]):
        rows.append({"порог %": thr, "delay": dl, "период": per, **trade_stats(t)})
    for (thr, dl, per), t in tr[tr["dir"] > 0].groupby(["thr", "delay", "per"]):
        rows.append({"порог %": thr, "delay": dl, "период": per + " (лонг)", **trade_stats(t)})
    tt = pd.DataFrame(rows)
    tt.to_csv(OUT / "trades_summary.csv", index=False)
    lines += ["Торговля «отстающий сосед», выход через 60 минут или по стопу 1%, издержки 0.1% за круг:", "",
              md(tt, False), ""]
    # годы для лучшего по выбору варианта
    sel = tt[tt["период"] == "выбор 2016–2020"].sort_values("PF", ascending=False).iloc[0]
    best = tr[(tr["thr"] == sel["порог %"]) & (tr["delay"] == sel["delay"])].copy()
    best["год"] = best["t"].dt.year
    yr = best.groupby("год").agg(сделок=("net", "size"), итог=("net", "sum"), PF=("net", pf)).round(2)
    lines += [f"Выбран по 2016–2020: порог {sel['порог %']}%, вход через {sel['delay']} мин. По годам:", "",
              md(yr, True), ""]
    best.to_csv(OUT / "best_trades.csv", index=False)
    # график: средний путь соседей по минутам для выбранного варианта
    plot_paths(m, d[(d["thr"] == sel["порог %"]) & (d["delay"] == sel["delay"])], sel)
    plot_years(yr, sel)
    plot_examples(m, best)
    return lines


def plot_paths(m: pd.DataFrame, d: pd.DataFrame, sel) -> None:
    lp = np.log(m.to_numpy(dtype="float64"))
    pos = pd.Index(m.index)
    fig, ax = plt.subplots(figsize=(9, 5))
    span = np.arange(-15, 61)
    for per, col in [("выбор 2016–2020", "#888"), ("проверка 2021–2023", "#2a6fdb"), ("проверка 2024–2026", "#d9480f")]:
        ev = d[d["per"] == per].drop_duplicates(["t", "src"])
        src_p, nb_p = [], []
        for _, r in ev.iterrows():
            i = pos.get_loc(r["t"])
            if i - 15 < 0 or i + 60 >= len(pos) or pos[i - 15].normalize() != pos[i + 60].normalize():
                continue
            j = TK.index(r["src"])
            others = [k for k in range(len(TK)) if k != j]
            base = lp[i - 15]
            src_p.append(r["dir"] * (lp[i + span, j] - base[j]) * 100)
            nb_p.append(r["dir"] * np.nanmean(lp[i + span][:, others] - base[others], axis=1) * 100)
        if not src_p:
            continue
        ax.plot(span, np.nanmean(src_p, axis=0), color=col, lw=2, label=f"бумага с толчком, {per} ({len(src_p)})")
        ax.plot(span, np.nanmean(nb_p, axis=0), color=col, lw=2, ls="--", label=f"соседи, {per}")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(sel["delay"], color="k", lw=0.8, ls=":")
    ax.set_xlabel("минуты от конца 15-минутного толчка (0) — пунктир: вход")
    ax.set_ylabel("ход в сторону толчка, %")
    ax.set_title(f"H90: толчок ≥ {sel['порог %']}% за 15 мин у одной бумаги нефтегаза — что делают соседи")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "paths.png", dpi=120)
    plt.close(fig)


def plot_years(yr: pd.DataFrame, sel) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    c = ["#2f9e44" if v > 0 else "#e03131" for v in yr["итог"]]
    ax.bar(yr.index.astype(str), yr["итог"], color=c)
    for i, (v, n) in enumerate(zip(yr["итог"], yr["сделок"])):
        ax.text(i, v, f"{n}", ha="center", va="bottom" if v > 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(4.5, color="k", ls=":", lw=1)
    ax.set_ylabel("итог за год, % (сумма сделок)")
    ax.set_title(f"H90: «отстающий сосед», порог {sel['порог %']}%, вход через {sel['delay']} мин, "
                 f"издержки 0.1% — по годам (число сделок над столбцом)")
    fig.tight_layout()
    fig.savefig(OUT / "years.png", dpi=120)
    plt.close(fig)


def pick_examples(best: pd.DataFrame) -> pd.DataFrame:
    # типичные толчки (до 4%), без обвала 24.02.2022 и остановки торгов в марте 2022
    ev = best[(best["t"].dt.year >= 2021) & (best["shock"] < 4) & best["net"].notna()
              & ~best["t"].between("2022-02-20", "2022-04-01")].drop_duplicates("t")
    return ev.sample(6, random_state=1).sort_values("t")


def plot_examples(m: pd.DataFrame, best: pd.DataFrame) -> None:
    ev = best if len(best) <= 6 else pick_examples(best)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (_, r) in zip(axes.ravel(), ev.iterrows()):
        w = m.loc[r["t"] - pd.Timedelta(minutes=40): r["t"] + pd.Timedelta(minutes=75)]
        w = w[w.index.normalize() == r["t"].normalize()]
        base = w.iloc[0]
        for tk in TK:
            lw = 2.5 if tk in (r["src"], r["nb"]) else 0.8
            ax.plot(w.index, (w[tk] / base[tk] - 1) * 100, lw=lw, label=tk)
        ax.axvline(r["t"], color="k", lw=0.8)
        ax.axvline(r["t"] + pd.Timedelta(minutes=int(r["delay"])), color="k", lw=0.8, ls=":")
        ax.set_title(f"{r['t']:%d.%m.%Y %H:%M} толчок {r['src']} {'+' if r['dir'] > 0 else '−'}{r['shock']:.1f}%, "
                     f"вход {r['nb']}: {r['net']:+.2f}%", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("H90: шесть случайных толчков 2–4% на проверке (жирные — бумага с толчком и купленный сосед)")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=110)
    plt.close(fig)


def part2(m: pd.DataFrame) -> list[str]:
    dc = m.groupby(m.index.normalize()).last().dropna()
    r = np.log(dc).diff().dropna() * 100
    sec = r.mean(axis=1)
    rows = []
    for tk in TK:
        rel = r[tk] - sec
        spread = rel.cumsum()
        z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        corr = r[tk].rolling(20).corr(sec)
        weak = corr < corr.rolling(250).median() - 0.2
        fut = {h: rel[::-1].rolling(h).sum()[::-1].shift(-1) for h in (1, 3, 5)}
        fr = r[tk][::-1].rolling(5).sum()[::-1].shift(-1)
        ev = (z.abs() >= 2) & weak
        last = None
        for t in z.index[ev.fillna(False)]:
            if last is not None and (t - last).days < 5:
                continue
            last = t
            s = -np.sign(z[t])
            rows.append({"t": t, "tk": tk, "z": z[t], "dir": s, **{f"rel{h}": s * fut[h][t] for h in (1, 3, 5)},
                         "gross": s * fr[t]})
    d = pd.DataFrame(rows).dropna()
    d["per"] = period(d["t"])
    d["net"] = d["gross"] - COST
    lines = ["## Часть 2 — связь бумаги с сектором разошлась", ""]
    tab = d.groupby("per").agg(событий=("t", "size"), rel1=("rel1", "mean"), rel3=("rel3", "mean"),
                               rel5=("rel5", "mean")).round(3)
    lines += ["Ход бумаги против сектора в сторону схождения, % (1 / 3 / 5 дней после события):", "",
              md(tab, True), ""]
    rows2 = [{"период": p, **trade_stats(t)} for p, t in d.groupby("per")]
    rows2 += [{"период": p + " (лонг)", **trade_stats(t)} for p, t in d[d["dir"] > 0].groupby("per")]
    lines += ["Торговля бумагой в сторону схождения 5 дней, издержки 0.1%:", "",
              md(pd.DataFrame(rows2), False), ""]
    d.to_csv(OUT / "corr_events.csv", index=False)
    return lines


def main() -> None:
    m = load_minutes()
    lines = part1(m) + part2(m)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__" and "--examples" not in __import__("sys").argv:
    main()


def examples_only() -> None:
    """Перерисовать examples.png по готовому best_trades.csv, читая минутки кусками только за дни примеров."""
    best = pd.read_csv(OUT / "best_trades.csv", parse_dates=["t"])
    ev = pick_examples(best)
    days = set(ev["t"].dt.strftime("%Y-%m-%d"))
    cols = {}
    for tk in TK:
        src = SRC.get(tk, RAW / "moex" / f"{tk}_1m.csv")
        parts = []
        for ch in pd.read_csv(src, usecols=["close", "begin"], chunksize=300_000):
            ch = ch[ch["begin"].str[:10].isin(days)]
            if len(ch):
                parts.append(ch)
        d = pd.concat(parts)
        d["begin"] = pd.to_datetime(d["begin"])
        cols[tk] = d.drop_duplicates("begin").set_index("begin")["close"]
    m = pd.DataFrame(cols).sort_index().ffill(limit=5)
    plot_examples(m, ev)


if __name__ == "__main__" and "--examples" in __import__("sys").argv:
    examples_only()
