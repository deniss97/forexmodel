"""H89, картинка случаев (договорённость 17.11: не только средние, но и сами случаи): 8 последних отчётов EIA —
нефть BR и LKOH минута к минуте, −60…+120 минут от выхода отчёта, ход в % от цены за минуту до отчёта.
    python scripts/h89_cases.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "results" / "h89_oil"
SRC = {"BR": RAW / "moex" / "BR_1m_cont.csv", "LKOH": RAW / "live" / "LKOH_1m_live.csv"}  # живой файл, дописывается каждый час
N = 8


def minutes(src: Path, days: set) -> pd.Series:
    parts = []
    for ch in pd.read_csv(src, usecols=["close", "begin"], chunksize=400_000):
        ch = ch[ch["begin"].str[:10].isin(days)]
        if len(ch):
            parts.append(ch)
    d = pd.concat(parts)
    d["begin"] = pd.to_datetime(d["begin"])
    return d.drop_duplicates("begin").set_index("begin")["close"].sort_index()


def main() -> None:
    cal = pd.read_csv(ROOT / "data" / "calendar" / "eia_wpsr_releases.csv", parse_dates=["release"])
    cal = cal.sort_values("release").tail(N)
    days = set(cal["release"].dt.strftime("%Y-%m-%d"))
    px = {k: minutes(v, days) for k, v in SRC.items()}
    fig, axes = plt.subplots(2, N // 2, figsize=(16, 7), sharex=True)
    for ax, (_, r) in zip(axes.ravel(), cal.iterrows()):
        a = r["release"]
        for k, col in (("BR", "#c8641e"), ("LKOH", "#2459b3")):
            s = px[k].loc[a - pd.Timedelta(minutes=60): a + pd.Timedelta(minutes=120)]
            ref = px[k].loc[: a - pd.Timedelta(minutes=1)]
            if s.empty or ref.empty:
                continue
            x = (s.index - a) / pd.Timedelta(minutes=1)
            ax.plot(x, (s / ref.iloc[-1] - 1) * 100, color=col, lw=1.4, label="нефть BR" if k == "BR" else "LKOH")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="#999", lw=0.6)
        z = r.get("z")
        ax.set_title(f"{a:%d.%m.%Y %H:%M} МСК · запасы {r['change']:+,.0f} тыс. барр. (z {z:+.1f})".replace(",", " "), fontsize=9)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes.ravel()[0].legend(fontsize=8)
    fig.supxlabel("минуты от выхода отчёта EIA")
    fig.supylabel("ход от цены за минуту до отчёта, %")
    fig.suptitle("H89: 8 последних отчётов EIA — нефть вздрагивает в первые минуты, LKOH почти не замечает; "
                 "направление первых минут дальше не держится")
    fig.tight_layout()
    fig.savefig(OUT / "cases_recent.png", dpi=110)
    print(OUT / "cases_recent.png")


if __name__ == "__main__":
    main()
