"""H77 (созвон 05.11, Тимур): импульс S42 на 15-минутных свечах — отчёт docs/results/h77_m15.md.

Прогоны — scripts/queues/h77_m15.sh (штатный walk-forward, минутный симулятор, издержки биржи, дивиденды). Сетка из банка:
ход за 12 или 24 свечи (3 или 6 ч) ≥ 2, 3 или 4 ATR(15 мин) × стоп и трейлинг 6 или 12 ATR(15 мин). Выбор — только по
2016–2020 (наибольший PF), проверка 2021–2026 — один раз. Критерий: не хуже S42 на акциях по PF (1.37) и итогу к просадке,
без 2 лучших > 0.

    python scripts/h77_m15.py
"""
from __future__ import annotations

import itertools

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import TEST_START, metrics  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, load_catalog, load_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "h77_m15"
DOC = ROOT / "docs" / "results" / "h77_m15.md"
STOCKS = ["gazp", "sber", "lkoh"]
GRIDS = list(itertools.product((6, 12), (12, 24), (2, 3, 4)))
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def exp(sl, n, x) -> dict:
    return {"id": f"m15 sl{sl} {n}/{x}", "sources": {s: [f"{s}_m15_sl{sl}__impulse_{n}_{x}_long"] for s in STOCKS},
            "costs": {"profile": "exchange"}, "dividends": True}


def ratio(m):
    return m["итог"] / abs(m["просадка"]) if m.get("просадка") else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, exps = load_catalog()
    base = load_trades(exps["S42"])
    base = base[base["instrument"].isin(STOCKS)]
    bm = {p: metrics(base, STOCKS, p) for p in ("выбор", "проверка")}
    rows, tr = [], {}
    for sl, n, x in GRIDS:
        t = load_trades(exp(sl, n, x))
        tr[(sl, n, x)] = t
        r = {"стоп/трейлинг, ATR15": sl, "ход за": f"{n // 4} ч", "порог, ATR15": x}
        for p in ("выбор", "проверка"):
            m = metrics(t, STOCKS, p)
            r.update({f"{p}: {k}": m[k] for k in ("сделок", "итог", "PF", "без_2_лучших", "просадка", "t")})
            r[f"{p}: итог/просадка"] = ratio(m)
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "grid.csv", index=False)
    best_i = int(R["выбор: PF"].idxmax())
    best = R.iloc[best_i]
    key = GRIDS[best_i]
    ok = (best["проверка: PF"] >= bm["проверка"]["PF"] and best["проверка: итог/просадка"] >= ratio(bm["проверка"])
          and best["проверка: без_2_лучших"] > 0)
    print(R.round(2).to_string(), "\nвыбран", key, "критерий:", ok)

    f, ax = plt.subplots(1, 2, figsize=(16, 5.6), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(True, color=GRID)
    ax[0].scatter(R["выбор: PF"], R["проверка: PF"], s=70, color=GRAY, label="12 вариантов на 15 мин")
    ax[0].scatter([best["выбор: PF"]], [best["проверка: PF"]], s=140, color=BLUE, label="выбран по 2016–2020", zorder=3)
    ax[0].scatter([bm["выбор"]["PF"]], [bm["проверка"]["PF"]], s=160, marker="*", color="#c8641e", label="S42 (часы)", zorder=3)
    ax[0].axhline(1, color=INK2, linestyle=":")
    ax[0].axvline(1, color=INK2, linestyle=":")
    ax[0].set_xlabel("PF на выборе 2016–2020", color=INK2)
    ax[0].set_ylabel("PF на проверке 2021–2026", color=INK2)
    ax[0].legend(frameon=False, labelcolor=INK)
    for lab, t, col in (("S42 (часовые свечи)", base, GRAY), (f"15 мин: ход за {key[1] // 4} ч ≥ {key[2]} ATR, стоп {key[0]} ATR", tr[key], BLUE)):
        x = t.sort_values("close_dt")
        ax[1].plot(x["close_dt"], (x["profit_pct"] / len(STOCKS)).cumsum(), color=col, linewidth=2.2, label=lab)
    ax[1].axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax[1].axhline(0, color=INK2, linewidth=0.8)
    ax[1].set_ylabel("накопленный итог, % (акции поровну)", color=INK2)
    ax[1].legend(frameon=False, labelcolor=INK, loc="upper left")
    f.tight_layout()
    f.savefig(OUT / "pf_and_equity.png", dpi=120, facecolor=BG)
    plt.close(f)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Импульс S42 на 15-минутных свечах (H77)", "",
          "Тимур (созвон 05.11): «стоит на пятнадцатиминутках посмотреть». Тот же импульс, что у S42 (ход за N свечей ≥ X ATR при "
          "тренд-гейте вверх, стоп и трейлинг в ATR, выход не позже 240 ч), но на 15-минутных свечах: GAZP, SBER, LKOH в лонг, "
          "издержки биржи, дивиденды, штатный walk-forward. Сетка (банк, до расчёта): ход за 3 или 6 ч ≥ 2, 3 или 4 ATR(15 мин) × "
          "стоп и трейлинг 6 или 12 ATR(15 мин). Выбор — только по 2016–2020, проверка — один раз. Расчёт — "
          "`scripts/queues/h77_m15.sh`, `scripts/h77_m15.py`.", "",
          f"**Критерий:** выбранный на проверке не хуже S42 на акциях по PF ({bm['проверка']['PF']:.2f}) и итогу к просадке "
          f"({ratio(bm['проверка']):.2f}), без 2 лучших > 0.", "", verdict, "", "![PF и итог](h77_m15/pf_and_equity.png)", "",
          "| вариант | выбор: сделок | PF | итог % | проверка: сделок | PF | итог % | без 2 лучших | просадка | итог / просадка |",
          "|---|---|---|---|---|---|---|---|---|---|",
          f"| **S42, часы** (ход за 6 ч ≥ 3 ATR, стоп 6 ATR) | {bm['выбор']['сделок']} | {bm['выбор']['PF']:.2f} | {bm['выбор']['итог']:+.1f} | "
          f"{bm['проверка']['сделок']} | {bm['проверка']['PF']:.2f} | {bm['проверка']['итог']:+.1f} | {bm['проверка']['без_2_лучших']:+.1f} | "
          f"{bm['проверка']['просадка']:.1f} | {ratio(bm['проверка']):.2f} |"]
    for i, r in R.sort_values("выбор: PF", ascending=False).iterrows():
        name = f"15 мин: ход за {r['ход за']} ≥ {r['порог, ATR15']} ATR, стоп {r['стоп/трейлинг, ATR15']} ATR"
        name = f"**{name} — выбран**" if i == best_i else name
        md.append(f"| {name} | {r['выбор: сделок']} | {r['выбор: PF']:.2f} | {r['выбор: итог']:+.1f} | {r['проверка: сделок']} | "
                  f"{r['проверка: PF']:.2f} | {r['проверка: итог']:+.1f} | {r['проверка: без_2_лучших']:+.1f} | {r['проверка: просадка']:.1f} | "
                  f"{r['проверка: итог/просадка']:.2f} |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
