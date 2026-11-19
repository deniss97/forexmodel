"""H94, картинки случаев (договорённость 17.11: не только средние, но и сами случаи).

Резкие падения и взлёты на 2 дневных ATR за 60 минут (не на открытии сессии), 2023–2026: 5-минутные свечи от двух
часов до случая до конца дня, окно хода закрашено, точка случая — чёрная линия, внизу объём.
    python scripts/h94_cases.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import h94_analysis as A  # noqa: E402

OUT = A.OUT


def candles(ax, b: pd.DataFrame) -> None:
    x = np.arange(len(b))
    up = b["close"] >= b["open"]
    col = np.where(up, "#1d7a55", "#b4462a")
    ax.vlines(x, b["low"], b["high"], color=col, lw=0.8)
    ax.bar(x, (b["close"] - b["open"]).abs().clip(lower=1e-9), bottom=np.minimum(b["open"], b["close"]), color=col, width=0.7)


def main() -> None:
    ev = pd.read_parquet(A.DER.parent / "h94_cases.parquet")
    x = ev[(ev["kind"] == "день") & (ev["k"] == 2.0) & (ev["w"] == 60) & (~ev["opening"]) & (ev["year"] >= 2023)]
    x = x.assign(day=x["t"].dt.normalize()).drop_duplicates(["day", "dir"]).sort_values("t")
    down, up = x[x["dir"] < 0].tail(6), x[x["dir"] > 0].tail(2)
    pick = pd.concat([down, up]).sort_values("t")
    cache = {}
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.35)
    for n, (_, r) in enumerate(pick.iterrows()):
        if r["tk"] not in cache:
            cache[r["tk"]] = A.load(r["tk"]).set_index("time")
        d = cache[r["tk"]]
        t = r["t"]
        s = d.loc[t - pd.Timedelta(minutes=120): t.normalize() + pd.Timedelta(days=1)]
        s = s[s.index.normalize() == t.normalize()] if len(s) else s
        b = s.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        row, col = divmod(n, 4)
        ax = fig.add_subplot(gs[row * 2, col])
        av = fig.add_subplot(gs[row * 2 + 1, col], sharex=ax)
        candles(ax, b)
        i_ev = b.index.searchsorted(t.floor("5min"))
        i_0 = b.index.searchsorted((t - pd.Timedelta(minutes=60)).floor("5min"))
        ax.axvspan(i_0 - 0.5, i_ev + 0.5, color="#c8641e", alpha=0.12)
        ax.axvline(i_ev, color="k", lw=0.9)
        av.bar(np.arange(len(b)), b["volume"], color="#9aa3ae", width=0.8)
        ticks = np.arange(0, len(b), max(1, len(b) // 5))
        av.set_xticks(ticks)
        av.set_xticklabels([b.index[i].strftime("%H:%M") for i in ticks], fontsize=7)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(labelsize=7)
        av.tick_params(labelsize=6)
        cont = r["eod"] * r["dir"] > 0                      # после случая цена пошла в ту же сторону
        what = "падение" if r["dir"] < 0 else "взлёт"
        ax.set_title(f"{r['tk']} {t:%d.%m.%Y %H:%M} · {what} {r['move_pct']:+.1f}% за час\n"
                     f"до закрытия дня: {r['eod']:+.1f}% ({'ход продолжился' if cont else 'отскок'})", fontsize=8.5)
        for a in (ax, av):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.suptitle("H94: резкие ходы на 2 дневных ATR за час (2023–2026), 5-минутные свечи; закрашено — окно хода, "
                 "чёрная линия — момент случая, внизу объём", fontsize=11)
    fig.savefig(OUT / "cases_2023_2026.png", dpi=105, bbox_inches="tight")
    print(OUT / "cases_2023_2026.png", len(pick))


if __name__ == "__main__":
    main()
