"""Обезличенные сделки по-новому (гипотезы Тимура 05.10, банк H13 / H14): дельта за несколько часов
и «поглощение».

Данные — секундные агрегаты ALGOPACK 2024–2026: LKOH (TQBR) и фьючерсы на серебро (SPBFUT, все
контракты). Секунды сворачиваются в часы по частям (row group за row group), у фьючерсов цена — по
самому активному в этот час контракту, дельта — сумма по всем контрактам. Дельта = объём покупок
агрессором − объём продаж.

H13. Дельта, просуммированная за k часов (k = 1, 3, 6, 12, 24), в z-оценке к своему разбросу за
     прошлые 20 дней, против хода цены дальше на h часов (h = 1, 6, 24, 48): ранговая корреляция (IC)
     и средний ход после крупной дельты (|z| ≥ 2) в её сторону. Если суммирование за окно помогает,
     IC растёт с k.
H14. «Поглощение»: крупная дельта за 6 ч (|z| ≥ 2), а цена почти не сдвинулась (|ход за 6 ч| ≤ 0.5
     ATR) — что дальше: продолжение в сторону дельты или разворот? Для сравнения — та же дельта,
     когда цена пошла с ней (≥ 1 ATR в её сторону) и против неё.
Каждый вывод — отдельно по 2024 и 2025–2026: интересно только то, что держит знак в обоих.

    python scripts/orderflow_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "orderflow"
DOC = ROOT / "docs" / "results" / "orderflow"
SOURCES = {"lkoh": ("LKOH (акции, TQBR)", ROOT / "data" / "raw" / "orderflow_lkoh_1s.parquet"),
           "silver_fut": ("серебро (фьючерсы SPBFUT)", ROOT / "data" / "raw" / "orderflow_spbfut_1s.parquet")}
WINDOWS, HORIZONS = (1, 3, 6, 12, 24), (1, 6, 24, 48)
COLS = ["TICKER_CC", "time", "price", "buy_volume", "sell_volume", "volume"]

BG, INK, INK2, GRID = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc"
BLUE, ORANGE, GRAY = "#2459b3", "#c8641e", "#9aa3ae"
FS = 14


def hourly(path: Path) -> pd.DataFrame:
    """Часовые бары из секунд: по контракту и часу — first/last/max/min цены, покупки, продажи, объём."""
    cache = OUT / f"{path.stem}_hourly.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=COLS).to_pandas()
        t = pd.to_datetime(d["time"])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(None)
        d["time"] = t + pd.Timedelta(hours=3)                      # МСК
        d["hour"] = d["time"].dt.floor("h")
        g = d.sort_values("time").groupby(["TICKER_CC", "hour"], observed=True)
        parts.append(g.agg(t0=("time", "first"), t1=("time", "last"), open=("price", "first"), close=("price", "last"),
                           high=("price", "max"), low=("price", "min"), buy=("buy_volume", "sum"),
                           sell=("sell_volume", "sum"), volume=("volume", "sum")).reset_index())
        del d
    a = pd.concat(parts).sort_values("t0")
    g = a.groupby(["TICKER_CC", "hour"], observed=True)                # склейка часов на стыках row group
    h = g.agg(open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"),
              buy=("buy", "sum"), sell=("sell", "sum"), volume=("volume", "sum")).reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_pickle(cache)
    return h


def series(h: pd.DataFrame) -> pd.DataFrame:
    """Один ряд на час: цена активного контракта, дельта и объём — по всем контрактам."""
    h = h.assign(delta=h["buy"] - h["sell"])
    tot = h.groupby("hour")[["delta", "volume"]].sum()
    front = h.sort_values("volume").drop_duplicates("hour", keep="last").set_index("hour").sort_index()
    s = front[["TICKER_CC", "open", "high", "low", "close"]].join(tot)
    pc = s["close"].shift(1).where(s["TICKER_CC"] == s["TICKER_CC"].shift(1))
    tr = np.maximum(s["high"] - s["low"], np.maximum((s["high"] - pc).abs(), (s["low"] - pc).abs()))
    s["atr"] = tr.rolling(14, min_periods=10).mean()
    tick = s["TICKER_CC"].astype(str)
    for k in WINDOWS:
        dk = s["delta"].rolling(k).sum()
        sd = dk.rolling(20 * 14, min_periods=100).std()               # ~20 торговых дней по 14 часов
        s[f"z{k}"] = dk / sd
        same = tick == tick.shift(k)
        s[f"move{k}"] = ((s["close"] - s["close"].shift(k)) / s["atr"]).where(same)
    s["period"] = np.where(s.index.year <= 2024, "2024", "2025–26")
    for hz in HORIZONS:
        same = tick == tick.shift(-hz)
        f = ((s["close"].shift(-hz) / s["close"] - 1) * 100).where(same)
        # хвосты 1% обрезаются по периоду: обвал серебра 30.01.2026 (−26% за день) иначе решает всё среднее
        lo = f.groupby(s["period"]).transform(lambda x: x.quantile(0.01))
        hi = f.groupby(s["period"]).transform(lambda x: x.quantile(0.99))
        s[f"fwd{hz}"] = f.clip(lo, hi)
    return s


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def analyse(name: str, s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    lines = [f"\n=== {SOURCES[name][0]}: {len(s)} часов, {s.index.min():%Y-%m-%d} .. {s.index.max():%Y-%m-%d}"]
    ic = []
    for per in ("2024", "2025–26"):
        p = s[s["period"] == per]
        for k in WINDOWS:
            for hz in HORIZONS:
                x = p[[f"z{k}", f"fwd{hz}"]].dropna()
                big = x[x[f"z{k}"].abs() >= 2]
                aligned = np.sign(big[f"z{k}"]) * big[f"fwd{hz}"]
                # t по неперекрывающимся точкам: каждая hz-я крупная дельта
                ic.append({"инструмент": name, "период": per, "окно_ч": k, "вперёд_ч": hz,
                           "IC": x[f"z{k}"].corr(x[f"fwd{hz}"], method="spearman"), "n": len(x),
                           "крупных": len(big), "ход_в_сторону_дельты_%": aligned.mean(),
                           "t": tstat(aligned.iloc[::max(1, hz // max(1, k))])})
    IC = pd.DataFrame(ic)
    lines.append("H13 · ранговая корреляция дельты за окно с ходом цены дальше (IC) и средний ход после |z| ≥ 2 "
                 "в сторону дельты, %:")
    lines.append(IC.pivot_table(index=["окно_ч"], columns=["период", "вперёд_ч"], values="IC").round(3).to_string())
    lines.append(IC.pivot_table(index=["окно_ч"], columns=["период", "вперёд_ч"], values="ход_в_сторону_дельты_%").round(3).to_string())
    ab = []
    for per in ("2024", "2025–26"):
        p = s[(s["period"] == per) & (s["z6"].abs() >= 2)]
        sign = np.sign(p["z6"])
        rel = sign * p["move6"]                                       # ход за 6 ч в сторону дельты, ATR
        groups = {"поглощение: цена стоит (|ход| ≤ 0.5 ATR)": rel.abs() <= 0.5,
                  "цена пошла с дельтой (≥ 1 ATR)": rel >= 1,
                  "цена пошла против дельты (≥ 1 ATR)": rel <= -1}
        for gname, m in groups.items():
            for hz in (6, 24, 48):
                v = sign[m] * p.loc[m, f"fwd{hz}"]
                ab.append({"инструмент": name, "период": per, "случай": gname, "вперёд_ч": hz, "событий": int(m.sum()),
                           "ход_в_сторону_дельты_%": v.mean(), "медиана_%": v.median(),
                           "t": tstat(v.iloc[::max(1, hz // 6)])})
    AB = pd.DataFrame(ab)
    lines.append("H14 · крупная дельта за 6 ч (|z| ≥ 2): средний ход цены дальше в сторону дельты, % (минус — разворот):")
    lines.append(AB.pivot_table(index=["случай"], columns=["период", "вперёд_ч"], values="ход_в_сторону_дельты_%",
                                sort=False).round(3).to_string())
    lines.append("  медиана, %:\n" + AB.pivot_table(index=["случай"], columns=["период", "вперёд_ч"], values="медиана_%",
                                                    sort=False).round(3).to_string())
    lines.append("  t (по неперекрывающимся событиям):\n" + AB.pivot_table(index=["случай"], columns=["период", "вперёд_ч"],
                                                                         values="t", sort=False).round(2).to_string())
    lines.append("  событий:\n" + AB.pivot_table(index=["случай"], columns=["период", "вперёд_ч"], values="событий",
                                                sort=False).to_string())
    return IC, AB, lines


def plot_ic(IC: pd.DataFrame) -> None:
    names = list(IC["инструмент"].unique())
    f, axes = plt.subplots(len(names), 2, figsize=(13, 4.2 * len(names)), facecolor=BG, squeeze=False)
    for r, name in enumerate(names):
        for c, per in enumerate(("2024", "2025–26")):
            ax = axes[r][c]
            p = IC[(IC["инструмент"] == name) & (IC["период"] == per)].pivot_table(index="окно_ч", columns="вперёд_ч", values="IC")
            ax.imshow(p.to_numpy(), cmap="RdBu", vmin=-0.08, vmax=0.08, aspect="auto")
            for (i, j), v in np.ndenumerate(p.to_numpy()):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=FS - 1, color=BG if abs(v) > 0.05 else INK)
            ax.set_xticks(range(p.shape[1]))
            ax.set_xticklabels([f"{h} ч" for h in p.columns], fontsize=FS - 2, color=INK2)
            ax.set_yticks(range(p.shape[0]))
            ax.set_yticklabels([f"{k} ч" for k in p.index], fontsize=FS - 2, color=INK2)
            ax.set_title(f"{SOURCES[name][0]} · {per}", fontsize=FS, color=INK, loc="left")
            ax.set_xlabel("ход цены дальше, часов", fontsize=FS - 2, color=INK2)
            if c == 0:
                ax.set_ylabel("дельта за окно", fontsize=FS - 2, color=INK2)
            for sp in ax.spines.values():
                sp.set_visible(False)
    f.tight_layout()
    f.savefig(DOC / "delta_windows_ic.png", dpi=150, facecolor=BG)
    plt.close(f)


def plot_absorption(AB: pd.DataFrame) -> None:
    names = list(AB["инструмент"].unique())
    f, axes = plt.subplots(1, len(names), figsize=(13, 5), facecolor=BG, squeeze=False)
    cases = list(AB["случай"].unique())
    colors = [BLUE, GRAY, ORANGE]
    for c, name in enumerate(names):
        ax = axes[0][c]
        ax.set_facecolor(BG)
        p = AB[(AB["инструмент"] == name) & (AB["вперёд_ч"] == 24)]
        x = np.arange(2)
        for j, case in enumerate(cases):
            v = [p[(p["случай"] == case) & (p["период"] == per)]["ход_в_сторону_дельты_%"].mean() for per in ("2024", "2025–26")]
            ax.bar(x + (j - 1) * 0.26, v, 0.26, color=colors[j], label=case)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["2024", "2025–26"], fontsize=FS, color=INK)
        ax.set_title(SOURCES[name][0], fontsize=FS, color=INK, loc="left")
        ax.set_ylabel("ход за 24 ч в сторону дельты, %", fontsize=FS - 2, color=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=GRID)
    h, lab = axes[0][0].get_legend_handles_labels()
    f.legend(h, lab, frameon=False, fontsize=FS - 2, labelcolor=INK, loc="lower center", ncol=3)
    f.tight_layout(rect=(0, 0.08, 1, 1))
    f.savefig(DOC / "absorption.png", dpi=150, facecolor=BG)
    plt.close(f)


def plot_episode(s: pd.DataFrame, name: str) -> str:
    """Самый крупный часовой сброс по дельте в январе 2026 (пример Тимура по серебру): цена и накопленная дельта."""
    jan = s[(s.index >= "2026-01-01") & (s.index < "2026-02-01")]
    if jan.empty:
        return ""
    t0 = jan["delta"].idxmin()
    w = s[(s.index >= t0 - pd.Timedelta("5D")) & (s.index <= t0 + pd.Timedelta("7D"))]
    f, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.set_facecolor(BG)
    x = np.arange(len(w))
    ax.plot(x, w["close"], color=INK, linewidth=2, label="цена (активный контракт)")
    ax2 = ax.twinx()
    ax2.plot(x, w["delta"].cumsum(), color=ORANGE, linewidth=2, label="накопленная дельта, контрактов")
    i0 = int(np.flatnonzero(w.index == t0)[0])
    ax.axvline(i0, color=BLUE, linewidth=2, linestyle="--")
    ax.text(i0 + 1, ax.get_ylim()[1], f" дельта часа {int(w.loc[t0, 'delta']):+,}".replace(",", " "), color=BLUE,
            fontsize=FS - 1, va="top")
    step = max(1, len(w) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%d.%m %H:%M") for t in w.index[::step]], fontsize=FS - 3, color=INK2)
    for a in (ax, ax2):
        a.tick_params(colors=INK2, labelsize=FS - 3)
        for sp in ("top",):
            a.spines[sp].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=FS - 2, labelcolor=INK, loc="lower left")
    f.tight_layout()
    f.savefig(DOC / f"episode_{name}.png", dpi=150, facecolor=BG)
    plt.close(f)
    after = {hz: s.loc[t0, f"fwd{hz}"] for hz in HORIZONS}
    return (f"{SOURCES[name][0]}: крупнейшая часовая дельта января 2026 — {t0:%Y-%m-%d %H:%M}, "
            f"{int(s.loc[t0, 'delta']):+d} контрактов (z6 = {s.loc[t0, 'z6']:+.1f}); ход цены дальше: "
            + ", ".join(f"{h} ч {v:+.2f}%" for h, v in after.items()))


def main() -> int:
    pd.set_option("display.width", 250)
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    ICs, ABs, lines = [], [], []
    for name, (_, path) in SOURCES.items():
        s = series(hourly(path))
        IC, AB, ln = analyse(name, s)
        ICs.append(IC)
        ABs.append(AB)
        lines += ln
        if name == "silver_fut":
            lines.append(plot_episode(s, name))
    IC, AB = pd.concat(ICs), pd.concat(ABs)
    IC.to_csv(OUT / "delta_windows_ic.csv", index=False)
    AB.to_csv(OUT / "absorption.csv", index=False)
    plot_ic(IC)
    plot_absorption(AB)
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
