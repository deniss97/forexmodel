"""H89: экономический календарь — первый заход, только новости по нефти (план подтверждён в банке 15.11).

Ответы автора на вопросы плана: календарь — история ru.investing.com «вполне подойдёт»; в этом заходе — только новости
по нефти, и смотреть влияние «через сам товар», акции — тоже интересно; интересны оба варианта — фильтр «не входить
перед событием» и торговля самого движения.

Событие — еженедельный отчёт EIA о запасах нефти в США (Weekly Petroleum Status Report): главное регулярное событие
по нефти, 10:30 по Нью-Йорку в среду (в неделю с праздником в понедельник–среду — на день позже, 11:00). Дата и запасы —
открытая таблица EIA (WCESTUS1, запасы без стратегического резерва, тыс. баррелей). Время по Москве — с переходом США
на летнее время: 17:30 летом и 18:30 зимой. ОПЕК+ в этот заход не вошёл: у его встреч нет расписания, нужна история
календаря (investing.com) — следующий шаг.

1. Размах вокруг события: средний |ход за минуту| в окне −60…+120 минут от выхода отчёта, по 5-минутным корзинам,
   против тех же минут суток в обычные дни (вторник и четверг без отчёта). Бумаги: нефтегаз (LKOH, ROSN, TATN, SNGSP,
   GAZP, NVTK) и для сравнения SBER, VTBR. Нефть — фьючерс BR из ленты ALGOPACK (2022–2026), если выгружен.
2. Направление: знак изменения запасов (рост запасов — плохо для нефти) против хода нефтянки за 30 минут после
   отчёта; и «продолжение»: ход первых 5 минут после отчёта против следующих 60 минут.
3. Фильтр: сделки S42/S46/S48 (журнал, как в H40), открытые за 24 часа до отчёта, против остальных — PF по половинам.
   Критерий плана: PF с фильтром выше в 2016–2020 и в 2021–2026, убирается не больше 20% сделок.
    python scripts/h89_oil_calendar.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.tseries.holiday import USFederalHolidayCalendar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CAL = ROOT / "data" / "calendar"
OUT = ROOT / "docs" / "results" / "h89_oil"
OUT.mkdir(parents=True, exist_ok=True)
OIL = ["LKOH", "ROSN", "TATN", "SNGSP", "GAZP", "NVTK"]
CTRL = ["SBER", "VTBR"]
SRC = {"GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv", "SBER": RAW / "SBER_1m_15-26.csv"}
BR = ROOT / "data" / "derived" / "h89_br_1m.parquet"
W0, W1 = -60, 120
COST = 0.10
C1, C2, C3 = "#2a78d6", "#eb6834", "#888888"


# ---------------------------------------------------------------- календарь EIA

def eia_calendar(html: Path) -> pd.DataFrame:
    """Даты недель и запасы из HTML-таблицы EIA → дата и время выхода отчёта по Москве."""
    s = re.sub(r"\s+", " ", html.read_text(errors="ignore"))
    rows = []
    for m in re.finditer(r"(\d{4})-(\w{3})</td>(.*?)</tr>", s):
        year = int(m.group(1))
        for md, val in re.findall(r"class='B5'>(\d\d/\d\d)&nbsp;</td> <td class='B3'>([\d,]+)&nbsp;", m.group(3)):
            mm, dd = md.split("/")
            # неделя кончается в пятницу; у недели в конце декабря дата может быть в январе следующего года и наоборот
            d = pd.Timestamp(year=year, month=int(mm), day=int(dd))
            if m.group(2) == "Jan" and int(mm) == 12:
                d = d.replace(year=year - 1)
            if m.group(2) == "Dec" and int(mm) == 1:
                d = d.replace(year=year + 1)
            rows.append((d, float(val.replace(",", ""))))
    w = pd.DataFrame(rows, columns=["week_end", "stocks"]).drop_duplicates("week_end").sort_values("week_end")
    w["change"] = w["stocks"].diff()
    w = w[w["week_end"] >= "2015-12-01"].copy()
    hol = set(USFederalHolidayCalendar().holidays("2015-01-01", "2027-12-31"))
    rel = []
    for we in w["week_end"]:
        wed = we + pd.Timedelta(days=5)
        mon = wed - pd.Timedelta(days=2)
        shift = any((mon + pd.Timedelta(days=k)) in hol for k in range(3))
        day = wed + pd.Timedelta(days=1) if shift else wed
        t = pd.Timestamp(f"{day:%Y-%m-%d} {'11:00' if shift else '10:30'}", tz="America/New_York")
        rel.append(t.tz_convert("Europe/Moscow").tz_localize(None))
    w["release"] = rel
    w["shifted"] = [r.weekday() != 2 for r in rel]
    z = (w["change"] - w["change"].rolling(52, min_periods=20).mean()) / w["change"].rolling(52, min_periods=20).std()
    w["z"] = z
    w = w[(w["release"] >= "2016-01-01") & (w["release"] <= pd.Timestamp.now())]
    return w.reset_index(drop=True)


# ---------------------------------------------------------------- минутки

def minutes(tk: str, days: set[str]) -> pd.Series:
    src = SRC.get(tk, RAW / "moex" / f"{tk}_1m.csv")
    parts = []
    for ch in pd.read_csv(src, usecols=["close", "begin"], chunksize=400_000, dtype={"close": "float32"}):
        ch = ch[ch["begin"].str[:10].isin(days)]
        if len(ch):
            parts.append(ch)
    d = pd.concat(parts)
    d["begin"] = pd.to_datetime(d["begin"])
    return d.drop_duplicates("begin").set_index("begin")["close"].sort_index()


def window_abs(px: pd.Series, anchors: list[pd.Timestamp]) -> np.ndarray:
    """|лог-ход за минуту|, % — матрица событий × минут окна (NaN, где минуты нет)."""
    lp = np.log(px.astype("float64"))
    r = (lp.diff() * 100).abs()
    r[lp.index.to_series().diff() > pd.Timedelta(minutes=5)] = np.nan  # первый бар после перерыва
    out = np.full((len(anchors), W1 - W0), np.nan)
    for i, a in enumerate(anchors):
        idx = pd.date_range(a + pd.Timedelta(minutes=W0), periods=W1 - W0, freq="min")
        out[i] = r.reindex(idx).to_numpy()
    return out


def move(px: pd.Series, a: pd.Timestamp, m0: int, m1: int) -> float:
    """Лог-ход, %, от закрытия минуты a+m0−1 до закрытия минуты a+m1−1 (последняя цена не позже)."""
    s = px.loc[a + pd.Timedelta(minutes=W0): a + pd.Timedelta(minutes=W1)]
    if s.empty:
        return np.nan
    p0 = s.loc[: a + pd.Timedelta(minutes=m0 - 1)]
    p1 = s.loc[: a + pd.Timedelta(minutes=m1 - 1)]
    if p0.empty or p1.empty or p1.index[-1] <= a + pd.Timedelta(minutes=m0 - 1):
        return np.nan
    return float(np.log(p1.iloc[-1] / p0.iloc[-1]) * 100)


def half(t) -> str:
    return "2016–2020" if pd.Timestamp(t).year <= 2020 else "2021–2026"


# ---------------------------------------------------------------- расчёт

def main() -> None:
    cal = eia_calendar(Path(sys.argv[1]) if len(sys.argv) > 1 else CAL / "eia_wpsr.html")
    cal.to_csv(CAL / "eia_wpsr_releases.csv", index=False)
    print("отчётов EIA:", len(cal), cal["release"].min(), cal["release"].max(), "сдвинуто:", cal["shifted"].sum())
    ev_days = set(cal["release"].dt.strftime("%Y-%m-%d"))
    # обычные дни для сравнения — вторники и четверги без отчёта, то же время суток (летнее/зимнее время США)
    base = []
    for _, r in cal.iterrows():
        for k in (-1, 1):
            d = r["release"] + pd.Timedelta(days=k)
            if d.strftime("%Y-%m-%d") not in ev_days and d.weekday() < 5:
                base.append(d)
    base_days = {d.strftime("%Y-%m-%d") for d in base}
    lines = ["## Числа", ""]
    prof, rows, dirrows = {}, [], []
    for tk in OIL + CTRL:
        px = minutes(tk, ev_days | base_days)
        e = window_abs(px, list(cal["release"]))
        b = window_abs(px, base)
        prof[tk] = (e, b)
        for h in ("2016–2020", "2021–2026"):
            me = np.array([half(t) == h for t in cal["release"]])
            mb = np.array([half(t) == h for t in base])
            # 30 минут после отчёта против тех же минут обычных дней
            e30 = np.nanmean(e[me][:, -W0:-W0 + 30])
            b30 = np.nanmean(b[mb][:, -W0:-W0 + 30])
            pre = np.nanmean(e[me][:, :-W0]) / np.nanmean(b[mb][:, :-W0])
            n_ok = int(np.sum(np.isfinite(e[me][:, -W0:-W0 + 30]).any(axis=1)))
            rows.append({"бумага": tk, "половина": h, "событий с данными": n_ok, "размах 30 мин после, событие %":
                         round(e30, 4), "обычный день %": round(b30, 4), "отношение": round(e30 / b30, 2),
                         "отношение за час до": round(pre, 2)})
        for _, r in cal.iterrows():
            a = r["release"]
            dirrows.append({"tk": tk, "t": a, "half": half(a), "z": r["z"], "change": r["change"],
                            "m5": move(px, a, 0, 5), "m5_65": move(px, a, 5, 65), "m30": move(px, a, 0, 30)})
        print(tk, "готово", flush=True)
    vol = pd.DataFrame(rows)
    vol.to_csv(OUT / "volatility.csv", index=False)
    lines += ["Размах (средний |ход за минуту|) за 30 минут после отчёта EIA против тех же минут вторника и четверга:",
              "", md(vol), ""]
    dr = pd.DataFrame(dirrows)
    dr.to_csv(OUT / "direction.csv", index=False)
    # направление: запасы против хода нефтянки
    rows = []
    for (tk, h), g in dr.groupby(["tk", "half"]):
        g = g.dropna(subset=["z", "m30"])
        g2 = dr[(dr["tk"] == tk) & (dr["half"] == h)].dropna(subset=["m5", "m5_65"])
        g2 = g2[g2["m5"].abs() > 0.05]
        tr = np.sign(g2["m5"]) * g2["m5_65"] - COST
        rows.append({"бумага": tk, "половина": h, "событий": len(g),
                     "корр. (запасы, ход 30 мин)": round(g["z"].corr(g["m30"]), 3),
                     "сделок «за первыми 5 мин»": len(tr), "средняя %": round(tr.mean(), 3),
                     "PF": round(pf(tr), 2), "доля плюсовых": round((tr > 0).mean(), 2)})
    dt = pd.DataFrame(rows)
    dt.to_csv(OUT / "direction_summary.csv", index=False)
    lines += ["Направление: корреляция неожиданности запасов (z изменения к 52 неделям) с ходом за 30 минут; торговля «за "
              "первыми 5 минутами» (вход на 5-й минуте в сторону хода, выход через 60 минут, издержки 0.1%):", "",
              md(dt), ""]
    # нефть сама (BR из ленты), если выгружена
    if BR.exists():
        br = pd.read_parquet(BR)["close"]
        e = window_abs(br, list(cal["release"]))
        b = window_abs(br, base)
        prof["BR (нефть)"] = (e, b)
        ok = np.isfinite(e[:, -W0:-W0 + 30]).any(axis=1)
        lines += [f"Нефть BR (лента ALGOPACK, {ok.sum()} отчётов): размах 30 мин после "
                  f"{np.nanmean(e[:, -W0:-W0 + 30]):.4f}% против {np.nanmean(b[:, -W0:-W0 + 30]):.4f}% — "
                  f"отношение {np.nanmean(e[:, -W0:-W0 + 30]) / np.nanmean(b[:, -W0:-W0 + 30]):.2f}", ""]
        for _, r in cal.iterrows():
            pass
    chart_profile(prof, cal, base)
    lines += filt(cal)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def filt(cal: pd.DataFrame) -> list[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from h40_trades import trades  # noqa: E402
    t = trades()
    t = t[t["strategy"] != "модель LKOH"].copy()
    rel = np.sort(cal["release"].to_numpy())
    i = np.searchsorted(rel, t["open_dt"].to_numpy())
    nxt = pd.to_datetime(np.where(i < len(rel), rel[np.minimum(i, len(rel) - 1)], np.datetime64("NaT")))
    t["до отчёта ч"] = (nxt - t["open_dt"]).dt.total_seconds() / 3600
    t["перед"] = t["до отчёта ч"] <= 24
    t["oil"] = t["tk"].isin(OIL)
    t["half"] = t["open_dt"].map(half)
    rows = []
    for (s, h), g in t.groupby(["strategy", "half"]):
        for scope, gg in (("все бумаги", g), ("нефтегаз", g[g["oil"]])):
            rows.append({"стратегия": s, "половина": h, "бумаги": scope, "сделок": len(gg),
                         "убирается %": round(gg["перед"].mean() * 100, 1),
                         "PF без фильтра": round(pf(gg["profit_pct"]), 2),
                         "PF с фильтром": round(pf(gg.loc[~gg["перед"], "profit_pct"]), 2),
                         "PF убранных": round(pf(gg.loc[gg["перед"], "profit_pct"]), 2)})
    ft = pd.DataFrame(rows)
    ft.to_csv(OUT / "filter.csv", index=False)
    chart_filter(ft)
    return ["Фильтр «не входить за 24 часа до отчёта EIA» (сделки журнала, как в H40):", "", md(ft), ""]


# ---------------------------------------------------------------- графики

def style(ax):
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def chart_profile(prof: dict, cal: pd.DataFrame, base: list) -> None:
    keys = list(prof)
    n = len(keys)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows), sharex=True)
    x = np.arange(W0, W1)
    xb = x.reshape(-1, 5).mean(axis=1)
    for ax, tk in zip(axes.ravel(), keys):
        e, b = prof[tk]
        for h, col in (("2016–2020", C3), ("2021–2026", C1)):
            me = np.array([half(t) == h for t in cal["release"]])
            mb = np.array([half(t) == h for t in base])
            if tk.startswith("BR"):
                me = np.ones(len(cal), bool)
                mb = np.ones(len(base), bool)
            ev = np.nanmean(e[me], axis=0).reshape(-1, 5).mean(axis=1)
            bs = np.nanmean(b[mb], axis=0).reshape(-1, 5).mean(axis=1)
            ax.plot(xb, ev / np.nanmean(bs), color=col, lw=2, label=f"отчёт EIA, {h}")
            ax.plot(xb, bs / np.nanmean(bs), color=col, lw=1, ls="--", label=f"обычный день, {h}")
            if tk.startswith("BR"):
                break
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(tk, fontsize=10)
        style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.supxlabel("минуты от выхода отчёта EIA (10:30 Нью-Йорк = 17:30 МСК летом, 18:30 зимой)")
    fig.supylabel("размах за минуту / средний размах обычного дня")
    fig.suptitle("H89: размах акций вокруг отчёта EIA о запасах нефти — сплошные: дни отчёта, пунктир: те же минуты"
                 " вторника и четверга")
    fig.tight_layout()
    fig.savefig(OUT / "profile.png", dpi=110)
    plt.close(fig)


def chart_filter(ft: pd.DataFrame) -> None:
    f = ft[ft["бумаги"] == "все бумаги"]
    fig, ax = plt.subplots(figsize=(10, 4))
    lab = [f"{s.split()[0]}\n{h}" for s, h in zip(f["стратегия"], f["половина"])]
    x = np.arange(len(f))
    ax.bar(x - 0.2, f["PF без фильтра"], 0.4, color=C3, label="без фильтра")
    ax.bar(x + 0.2, f["PF с фильтром"], 0.4, color=C1, label="без входов за 24 ч до отчёта EIA")
    for i, (u, p) in enumerate(zip(f["убирается %"], f["PF убранных"])):
        ax.text(i, max(f["PF без фильтра"].iloc[i], f["PF с фильтром"].iloc[i]) + 0.03,
                f"убрано {u:.0f}%\nPF убранных {p}", ha="center", fontsize=7)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("PF")
    ax.set_title("H89: фильтр «не входить за сутки до отчёта EIA» для стратегий S42/S46/S48")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.95)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "filter.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- мелочи

def pf(x) -> float:
    x = pd.Series(x).dropna()
    g, l_ = x[x > 0].sum(), -x[x < 0].sum()
    return float(g / l_) if l_ > 0 else np.nan


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    main()
