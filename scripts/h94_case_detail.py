"""H94, отдельные случаи по просьбе Тимура (созвон 23.11 вечером): ходы LKOH на 3 часовых ATR № 153, 159, 160 со
страницы docs/results/h94/LKOH_3ATR_5m.html — «какие объёмы зашли, что зашло, как выглядел стакан в это время».

Данные:
* минутные свечи — горячий файл контура data/raw/live/LKOH_1m_live.csv;
* лента сделок Т-Инвестиций с направлением (покупка/продажа по инициатору) — Яндекс Диск,
  «forexmodel/лента Т-Инвестиций/LKOH/<год>/<дата>.csv.gz» (копится с 07.04.2026);
* стакан пишем только с 13.11.2026 — для августовских случаев его нет.

Для каждого случая: окно −2 часа до начала хода … +2 часа после пика; объём покупок и продаж по минутам против обычного
объёма этой минуты дня (медиана 20 прошлых торговых дней), накопленная разница покупок и продаж; крупные сделки
(от 5 млн ₽). В таблице: объём за 30 минут до хода и во время хода в разах к обычному, доля покупок, крупные сделки,
и с какой минуты объём за 10 минут держится выше обычного в 2+ раза — раньше или позже, чем цена начала ход.

    python scripts/h94_case_detail.py [153 159 160]    → docs/results/h94/case_<№>.png, case_detail.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "h94"
PAGE = OUT / "LKOH_3ATR_5m.html"
MIN1 = ROOT / "data" / "raw" / "live" / "LKOH_1m_live.csv"
TAPE = Path.home() / "yadisk" / "forexmodel" / "лента Т-Инвестиций" / "LKOH"
BIG = 5e6                      # крупная сделка, ₽
UP, DOWN, INK2 = "#16a34a", "#dc2626", "#667085"


def cases() -> list[dict]:
    s = PAGE.read_text(encoding="utf-8")
    i = s.index("const D = ") + len("const D = ")
    return json.loads(s[i:s.index(";\nconst css", i)])["cases"]


def ts(x: int) -> pd.Timestamp:
    return pd.Timestamp(x, unit="s")                     # время МСК, записанное как UTC (так на странице)


def minutes(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(MIN1, usecols=["begin", "open", "high", "low", "close", "volume"], chunksize=400_000):
        ch["begin"] = pd.to_datetime(ch["begin"])
        ch = ch[(ch["begin"] >= t0) & (ch["begin"] <= t1)]
        if len(ch):
            parts.append(ch)
    return pd.concat(parts).drop_duplicates("begin").set_index("begin").sort_index()


def tape(day: pd.Timestamp) -> pd.DataFrame | None:
    f = TAPE / f"{day:%Y}" / f"{day:%Y-%m-%d}.csv.gz"
    if not f.exists():
        return None
    t = pd.read_csv(f, usecols=["TRADE_TS", "DIRECTION", "PRICE", "QUANTITY"])
    t["t"] = pd.to_datetime(t["TRADE_TS"], utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    t["rub"] = t["PRICE"] * t["QUANTITY"]
    return t


def detail(n: int, c: dict) -> dict:
    base, peak, dirn = ts(c["base"]), ts(c["peak_t"]), c["dir"]
    w0, w1 = base - pd.Timedelta(hours=2), peak + pd.Timedelta(hours=2)
    m = minutes(base.normalize() - pd.Timedelta(days=35), w1)
    hist = m[m.index < base.normalize()]
    days = sorted(set(hist.index.normalize()))[-20:]
    hist = hist[hist.index.normalize().isin(days)]
    typ = hist.groupby(hist.index.strftime("%H:%M"))["volume"].median()
    win = m[(m.index >= w0) & (m.index <= w1)].copy()
    win["typ"] = typ.reindex(win.index.strftime("%H:%M")).to_numpy()
    t = tape(base.normalize())
    if t is not None:
        t = t[(t["t"] >= w0) & (t["t"] <= w1 + pd.Timedelta(minutes=1))]
        g = t.groupby([t["t"].dt.floor("min"), "DIRECTION"])["QUANTITY"].sum().unstack(fill_value=0)
        win["buy"] = g.get("BUY", pd.Series(dtype=float)).reindex(win.index).fillna(0).to_numpy()
        win["sell"] = g.get("SELL", pd.Series(dtype=float)).reindex(win.index).fillna(0).to_numpy()
        big = t[t["rub"] >= BIG]
    else:
        big = None
    pre = win[(win.index >= base - pd.Timedelta(minutes=30)) & (win.index < base)]
    mv = win[(win.index >= base) & (win.index <= peak)]
    ratio = win["volume"].rolling(10).sum() / win["typ"].rolling(10).sum()     # объём 10 минут к обычному
    first = None                                    # с какой минуты объём держится выше обычного в 2+ раза до начала хода
    if ratio.asof(base) >= 2:
        before = ratio[ratio.index < base]
        calm = before[before < 2]
        first = calm.index.max() + pd.Timedelta(minutes=1) if len(calm) else before.index.min()
    else:
        during = ratio[(ratio.index >= base) & (ratio >= 2)]
        first = during.index.min() if len(during) else None
    r = {"№": n, "дата": f"{base:%d.%m.%Y}", "ход": f"{base:%H:%M}–{peak:%H:%M}", "напр.": "рост" if dirn > 0 else "падение",
         "ход %": c["move_pct"], "ATR": c["peak"],
         "объём до (30 мин), × обычного": round(pre["volume"].sum() / max(pre["typ"].sum(), 1), 1),
         "объём хода, × обычного": round(mv["volume"].sum() / max(mv["typ"].sum(), 1), 1),
         "объём 10 мин ≥ 2× обычного с": "нет" if first is None else (f"{first:%H:%M} — за {int((base - first).total_seconds() // 60)} мин до хода"
                                                          if first < base else f"{first:%H:%M} — через {int((first - base).total_seconds() // 60)} мин после начала"),
         }
    if "buy" in win:
        b, s = mv["buy"].sum(), mv["sell"].sum()
        r["покупки во время хода, %"] = round(100 * b / max(b + s, 1))
        bb = big[(big["t"] >= base) & (big["t"] <= peak)]
        r["крупные сделки (≥5 млн ₽) за ход"] = (f"{len(bb)}: покупки {int((bb['DIRECTION'] == 'BUY').sum())}, "
                                                 f"продажи {int((bb['DIRECTION'] == 'SELL').sum())}, {bb['rub'].sum() / 1e6:.0f} млн ₽") if len(bb) else "0"
    plot(n, c, win, base, peak, big)
    return r


def plot(n, c, win, base, peak, big) -> None:
    rows = 3 if "buy" in win else 2
    fig, ax = plt.subplots(rows, 1, figsize=(12, 3.0 * rows + 1), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1][:rows]})
    x = np.arange(len(win))
    for i, (_, r) in enumerate(win.iterrows()):
        col = UP if r["close"] >= r["open"] else DOWN
        ax[0].vlines(i, r["low"], r["high"], color=col, lw=0.7)
        ax[0].vlines(i, min(r["open"], r["close"]), max(r["open"], r["close"]), color=col, lw=2.2)
    i0, i1 = win.index.searchsorted(base), win.index.searchsorted(peak)
    for a in ax:
        a.axvspan(i0, i1, color="#2a6bd8", alpha=0.08, lw=0)
        a.grid(alpha=0.3)
    if big is not None and len(big):
        bb = big.assign(i=win.index.searchsorted(big["t"].dt.floor("min")))
        for d, col, mk in (("BUY", UP, "^"), ("SELL", DOWN, "v")):
            s = bb[bb["DIRECTION"] == d]
            ax[0].scatter(s["i"], s["PRICE"], s=12 + s["rub"] / 1e6 * 4, marker=mk, color=col, alpha=0.6, edgecolors="none",
                          label=f"крупные {'покупки' if d == 'BUY' else 'продажи'} (≥5 млн ₽)")
        ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"LKOH, случай № {n}: {'рост' if c['dir'] > 0 else 'падение'} {c['move_pct']:+.2f}% = {c['peak']:.1f} ATR за 60 минут, "
                    f"{base:%d.%m.%Y %H:%M}–{peak:%H:%M} (полоса)", fontsize=11)
    if "buy" in win:
        ax[1].bar(x, win["buy"], color=UP, width=0.9, label="покупки (по ленте)")
        ax[1].bar(x, -win["sell"], color=DOWN, width=0.9, label="продажи")
        ax[1].plot(x, win["typ"], color=INK2, lw=1, label="обычный объём минуты (медиана 20 дней)")
        ax[1].plot(x, -win["typ"], color=INK2, lw=1)
        ax[1].set_ylabel("шт. за минуту")
        ax[1].legend(fontsize=8, loc="upper left", ncol=3)
        ax[2].plot(x, (win["buy"] - win["sell"]).cumsum(), color="#2a6bd8", lw=1.3)
        ax[2].axhline(0, color=INK2, lw=0.6)
        ax[2].set_ylabel("покупки − продажи,\nнакопленно, шт.")
    else:
        ax[1].bar(x, win["volume"], color="#94a3b8", width=0.9, label="объём")
        ax[1].plot(x, win["typ"], color=INK2, lw=1, label="обычный объём минуты")
        ax[1].legend(fontsize=8)
    ticks = list(range(0, len(win), 30))
    ax[-1].set_xticks(ticks)
    ax[-1].set_xticklabels([win.index[i].strftime("%H:%M") for i in ticks], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"case_{n}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    nums = [int(a) for a in sys.argv[1:]] or [153, 159, 160]
    cs = cases()
    rows = [detail(n, cs[n - 1]) for n in nums]
    t = pd.DataFrame(rows)
    md = ["# H94: случаи LKOH № " + ", ".join(map(str, nums)) + " — объёмы и лента (просьба Тимура, созвон 23.11)", "",
          "`python scripts/h94_case_detail.py " + " ".join(map(str, nums)) + "` · случаи — страница `LKOH_3ATR_5m.html` · "
          "лента — Т-Инвестиции (направление по инициатору сделки), с 07.04.2026 · стакан пишем только с 13.11.2026 — для "
          "этих случаев его нет.", "",
          "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    md += ["", "Картинки: " + ", ".join(f"`case_{n}.png`" for n in nums) + " — свечи по минутам (треугольники — крупные сделки, "
           "размер — сумма), объём покупок вверх и продаж вниз против обычного объёма этой минуты, накопленная разница."]
    (OUT / "case_detail.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
