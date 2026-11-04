"""Гипотеза H71 (T46): потолок убытка сделки 3 / 4 / 5% у S42 на акциях — отчёт docs/results/h71_stop_caps.md.

Прогоны — scripts/queues/h71_stop_caps.sh (walk-forward, штатный минутный симулятор, simulation.max_loss_pct):
стоп = ближний из «6 ATR» и «X% от входа», всё остальное как у S42. Метрики — кодом журнала экспериментов
(акции поровну, издержки биржи). Плюс по минуткам сделок S42 без потолка: как часто сделка, ушедшая в −X%, потом
вернулась в ноль и закрылась в плюс (вопрос Тимура «выйдет ли сделка в глубоком убытке хотя бы в 0»).

    python scripts/h71_stop_caps.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import TEST_START, metrics  # noqa: E402

from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT, STOCK_FILES, load_trades, raw_prices  # noqa: E402

OUT = ROOT / "docs" / "results" / "h71_stop_caps"
DOC = ROOT / "docs" / "results" / "h71_stop_caps.md"
STOCKS = ["gazp", "sber", "lkoh"]
CAPS = [None, 5, 4, 3]
COLORS = {None: GRAY, 5: BLUE, 4: "#8a5cc4", 3: "#c8641e"}
KEYS = ("сделок", "итог", "PF", "без_2_лучших", "просадка", "t", "лет_в_плюсе")


def label(cap) -> str:
    return "S42 (стоп 6 ATR)" if cap is None else f"потолок {cap}%"


def exp(cap) -> dict:
    tag = "lo" if cap is None else f"lo_cap{cap}"
    return {"id": label(cap), "sources": {s: [f"{s}_{tag}__impulse_6_3_long"] for s in STOCKS},
            "costs": {"profile": "exchange"}, "dividends": True}


def mae(t: pd.DataFrame) -> pd.DataFrame:
    """Худшая цена внутри сделки (минимум минут от входа до выхода), % от входа — по сырым минуткам."""
    t = raw_prices(t)
    out = []
    for inst, g in t.groupby("instrument"):
        m = pd.read_pickle(ROOT / "reports" / "_cache" / f"{STOCK_FILES[inst]}_1m.pkl").sort_values("time")
        tt, lo = m["time"].to_numpy(), m["low"].to_numpy(float)
        cm = pd.Series(lo)
        for _, r in g.iterrows():
            a, b = np.searchsorted(tt, r["open_dt"], "left"), np.searchsorted(tt, r["close_dt"], "right")
            w = cm.iloc[a:b]
            out.append({**r.to_dict(), "mae_pct": (w.min() / r["open_price"] - 1) * 100 if len(w) else np.nan})
    return pd.DataFrame(out)


def fmt(v, d=1, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = {c: load_trades(exp(c)) for c in CAPS}
    rows = []
    for c in CAPS:
        for per in ("выбор", "проверка"):
            m = metrics(tr[c], STOCKS, per)
            x = tr[c][(tr[c]["open_dt"] >= TEST_START) == (per == "проверка")]
            rows.append({"вариант": label(c), "период": per, **{k: m.get(k) for k in KEYS},
                         "худшая": x["profit_pct"].min(), "выход по стопу, %": (x["exit_reason"] == "stop_loss").mean() * 100})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "metrics.csv", index=False)

    base = mae(tr[None])
    base.to_csv(OUT / "s42_mae.csv", index=False)
    rec = []
    for x in (3, 4, 5):
        deep = base[base["mae_pct"] <= -x]
        rec.append({"ушла в": f"−{x}%", "сделок": len(deep), "доля всех": len(deep) / len(base) * 100,
                    "закрылась ≥ 0": (deep["gross_pct"] >= 0).mean() * 100 if len(deep) else np.nan,
                    "закрылась в плюс после издержек": (deep["profit_pct"] > 0).mean() * 100 if len(deep) else np.nan,
                    "средний итог": deep["profit_pct"].mean() if len(deep) else np.nan})
    Q = pd.DataFrame(rec)

    # график: накопленный итог трёх акций поровну
    f, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    for c in CAPS:
        x = tr[c][tr[c]["instrument"].isin(STOCKS)].sort_values("close_dt")
        ax.plot(x["close_dt"], (x["profit_pct"] / len(STOCKS)).cumsum(), color=COLORS[c], lw=2.4 if c is None else 1.6, label=label(c))
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=BLUE, fontsize=9)
    ax.set_title("S42 на GAZP, SBER, LKOH: накопленный итог, % (бумаги поровну)", color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    f.tight_layout()
    f.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(f)

    md = ["# Потолок убытка сделки 3 / 4 / 5% у S42 (H71, T46)", "",
          "Вопрос Тимура: в живой торговле, особенно с плечом, глубокая сделка опасна — как меняется импульсная стратегия, если "
          "ограничить убыток сделки 3, 4 или 5%, и как часто сделка в глубоком минусе потом выходит хотя бы в ноль.", "",
          "Как считали: S42 на акциях (GAZP, SBER, LKOH на бирже в лонг, дивиденды, издержки биржи) — штатный walk-forward, "
          "минутный симулятор; стоп = ближний из «6 ATR от входа» и «X% от входа» (`simulation.max_loss_pct`), трейлинг, гейт и "
          "выход 240 ч — как у S42. Бумаги поровну, метрики — кодом журнала. Расчёт — `scripts/queues/h71_stop_caps.sh`, "
          "`scripts/h71_stop_caps.py`.", "", "<!-- вывод -->", "<!-- /вывод -->", "", "## Метрики", "",
          "| вариант | период | сделок | итог, % | PF | без 2 лучших | просадка | t | лет в плюсе | худшая сделка | по стопу |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {fmt(r['итог'])} | {fmt(r['PF'], 2, False)} | "
                  f"{fmt(r['без_2_лучших'])} | {fmt(r['просадка'])} | {fmt(r['t'], 2)} | {r['лет_в_плюсе']} | "
                  f"{fmt(r['худшая'])}% | {r['выход по стопу, %']:.0f}% |")
    md += ["", "![накопленный итог](h71_stop_caps/equity.png)", "",
           "## Возвращается ли сделка из глубокого минуса", "",
           f"Сделки S42 без потолка ({len(base)}, 2016 — июнь 2026), худшая цена внутри сделки по минуткам:", "",
           "| ушла в | сделок | доля всех | закрылась не в минус (до издержек) | в плюс после издержек | средний итог сделки |",
           "|---|---|---|---|---|---|"]
    for _, r in Q.iterrows():
        md.append(f"| {r['ушла в']} | {r['сделок']} | {r['доля всех']:.1f}% | {fmt(r['закрылась ≥ 0'], 0, False)}% | "
                  f"{fmt(r['закрылась в плюс после издержек'], 0, False)}% | {fmt(r['средний итог'], 2)}% |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(R.to_string())
    print(Q.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
