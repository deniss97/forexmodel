"""Образцы визуализации для гипотез без сделок (документ Тимура «Порядок проверки гипотез 2.0», 13.11).

1. «Событие» — что в среднем делают акции вокруг события (здесь: плановые решения ЦБ по ставке): путь цены и сила
   колебаний по часам до и после, отдельно до 2022 и с 2022 года. Подходит для H79, H89 (календарь), H90 (соседи).
2. «Признак модели» — как будущая доходность зависит от значения признака (10 корзин от меньшего к большему), отдельно по
   периодам: видно, есть ли связь и держится ли она во времени. Подходит для любых фич (H40, H41, H42 и т. п.).
Сделки на свечах — третий вид — уже есть в журнале экспериментов.

Попутно: часовые свечи 19 акций из минуток → data/derived/hourly/<бумага>.parquet (кеш, пересобирается, если нет файла).

    python scripts/viz_examples.py          # картинки в docs/viz/
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
HOURLY = ROOT / "data" / "derived" / "hourly"
OUT = ROOT / "docs" / "viz"
TICKERS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
           "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS"]
SRC = {"SBER": RAW / "SBER_1m_15-26.csv", "GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv"}

# палитра (dataviz: слоты 1–2 проверены валидатором — синий / оранжевый; текст и сетка — нейтральные)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"


def hourly(t: str) -> pd.DataFrame:
    f = HOURLY / f"{t}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    src = SRC.get(t, RAW / "moex" / f"{t}_1m.csv")
    parts = []
    for ch in pd.read_csv(src, usecols=["open", "close", "high", "low", "volume", "begin"], chunksize=500_000):
        ch["h"] = pd.to_datetime(ch["begin"]).dt.floor("h")
        parts.append(ch.groupby("h").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                         close=("close", "last"), volume=("volume", "sum")))
    d = pd.concat(parts)
    d = d.groupby(level=0).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                               volume=("volume", "sum"))
    HOURLY.mkdir(parents=True, exist_ok=True)
    d.to_parquet(f)
    return d


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def event_study(data: dict[str, pd.DataFrame]) -> Path:
    cal = pd.read_csv(ROOT / "data" / "calendar" / "cbr_key_rate_meetings.csv", comment="#")
    cal = cal[cal["scheduled"] == 1]
    ev = pd.to_datetime(cal["date"]) + pd.Timedelta(hours=13)            # последний час до решения 13:30 — 12:00–12:59
    W = 24
    rows = []
    for t, d in data.items():
        r = np.log(d["close"]).diff()
        # обычная сила колебания для того же часа дня (первый час торгов всегда бурный) — медиана за прошлый год
        typ = r.abs().groupby(d.index.hour).transform(lambda x: x.rolling(250, min_periods=50).median().shift())
        idx = d.index
        for e in ev:
            k = idx.searchsorted(e)                                         # первый час после 13:00 (13:00–13:59)
            if k - W < 1 or k + W >= len(idx):
                continue
            seg = r.iloc[k - W:k + W].to_numpy()
            norm = typ.iloc[k - W:k + W].to_numpy()
            if not np.all(np.isfinite(norm)) or np.any(norm == 0):
                continue
            path = np.concatenate([[0.0], np.cumsum(seg)])[1:] - np.sum(seg[:W])   # 0 — в момент перед решением
            rows.append({"t": t, "e": e, "path": path * 1e4, "vol": np.abs(seg) / norm})
    df = pd.DataFrame(rows)
    df["era"] = np.where(df["e"] < "2022-01-01", "2016–2021", "2022–2026")
    x = np.arange(-W, W) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    for era, c in (("2016–2021", C1), ("2022–2026", C2)):
        g = df[df["era"] == era]
        n_ev = g["e"].nunique()
        for ax, key, ylab in ((axes[0], "path", "средний путь цены, б.п. (0 — перед решением)"),
                              (axes[1], "vol", "сила колебания, × к обычной для этого часа дня")):
            m = np.vstack(g.groupby("e")[key].apply(lambda s: np.mean(np.vstack(s), axis=0)).to_numpy())
            mu, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(len(m))
            ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.15, lw=0)
            ax.plot(x, mu, color=c, lw=2, label=f"{era} ({n_ev} решений)")
            ax.set_ylabel(ylab, color=INK2, fontsize=9)
    for ax in axes:
        style(ax)
        ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xlabel("часы торгов от решения ЦБ (13:30 МСК)", color=INK2, fontsize=9)
    axes[1].axhline(1, color=INK2, lw=0.8)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Образец «событие»: 19 акций вокруг плановых решений ЦБ по ставке (среднее ± ошибка среднего)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "example_event_cbr.png"
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p


def feature_deciles(data: dict[str, pd.DataFrame]) -> Path:
    parts = []
    for t, d in data.items():
        c = d["close"]
        tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(1)
        atr = tr.rolling(80, min_periods=40).mean()
        z = (c - c.rolling(80, min_periods=40).mean()) / atr
        fut = np.log(c.shift(-16) / c) * 1e4
        parts.append(pd.DataFrame({"z": z, "fut": fut, "y": d.index.year}).dropna())
    df = pd.concat(parts)
    periods = [("2016–2020", 2016, 2020), ("2021–2023", 2021, 2023), ("2024–2026", 2024, 2026)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, facecolor=SURF)
    for ax, (lab, a, b) in zip(axes, periods):
        g = df[(df["y"] >= a) & (df["y"] <= b)].copy()
        g["q"] = pd.qcut(g["z"], 10, labels=False) + 1
        s = g.groupby("q")["fut"].agg(["mean", "std", "count"])
        se = s["std"] / np.sqrt(s["count"] / 16)                           # перекрытие окон 16 ч — грубая поправка
        ax.bar(s.index, s["mean"], width=0.8, color=C1)
        ax.errorbar(s.index, s["mean"], yerr=se, fmt="none", ecolor=INK2, lw=1, capsize=0)
        ax.axhline(0, color=INK2, lw=0.8)
        style(ax)
        ax.set_xticks([1, 5, 10], ["1\nниже среднего", "5", "10\nвыше среднего"])
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("доходность за 16 ч вперёд, б.п.", color=INK2, fontsize=9)
    fig.suptitle("Образец «признак модели»: отклонение цены от средней за 80 ч (в ATR), 10 корзин; 19 акций",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = OUT / "example_feature_deciles.png"
    fig.savefig(p, dpi=150, facecolor=SURF)
    plt.close(fig)
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for t in TICKERS:
        d = hourly(t)
        data[t] = d[(d.index.hour >= 10) & (d.index.hour <= 18)]           # основная сессия
        print(t, len(d), flush=True)
    print(event_study(data))
    print(feature_deciles(data))


if __name__ == "__main__":
    main()
