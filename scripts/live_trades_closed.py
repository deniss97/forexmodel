"""Закрытые реальные сделки (счёт Т-Инвестиций) за период: часовые свечи с входом, стопом, уровнем трейлинга и выходом;
таблица «вход → выход → результат». Для еженедельного отчёта.
    python scripts/live_trades_closed.py --out docs/results/weekly/2026-11-23 --since 2026-11-10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from live_trades_week import LIVE, ROOT, tail_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="2026-11-10")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / "reports" / "live" / "account_trades.csv")
    t = t[(t["исполнение"] == "API") & (t["вход_в"] >= a.since) & t["выход"].notna()]
    fig, axes = plt.subplots(len(t), 1, figsize=(11, 3.8 * len(t)))
    axes = [axes] if len(t) == 1 else list(axes)
    rows = []
    for ax, (_, p) in zip(axes, t.iterrows()):
        tk = p["бумага"]
        d = tail_csv(LIVE / f"{tk}_1m_live.csv", 20000)
        t0 = pd.Timestamp(p["исполнено_в"]) - pd.Timedelta(days=2)
        t1 = pd.Timestamp(p["выход"]) + pd.Timedelta(days=1)
        d = d[(d.index >= t0) & (d.index <= t1)] if isinstance(d.index, pd.DatetimeIndex) else d[(d["begin"] >= t0) & (d["begin"] <= t1)].set_index("begin")
        h = d.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        for i, (_, r) in enumerate(h.iterrows()):
            c = "#2f9e44" if r["close"] >= r["open"] else "#e03131"
            ax.vlines(i, r["low"], r["high"], color=c, lw=0.8)
            ax.vlines(i, min(r["open"], r["close"]), max(r["open"], r["close"]), color=c, lw=3)
        i_in = h.index.get_indexer([pd.Timestamp(p["исполнено_в"]).floor("h")], method="nearest")[0]
        i_out = h.index.get_indexer([pd.Timestamp(p["выход"]).floor("h")], method="nearest")[0]
        res = (p["цена выхода"] / p["цена_входа"] - 1) * 100
        rub = (p["цена выхода"] - p["цена_входа"]) * p["штук"]
        ax.scatter([i_in], [p["цена_входа"]], marker="^", s=120, color="#1c7ed6", zorder=5,
                   label=f"покупка {str(p['исполнено_в'])[5:16]} по {p['цена_входа']:.2f}")
        ax.scatter([i_out], [p["цена выхода"]], marker="v", s=120, color="#e03131", zorder=5,
                   label=f"выход {str(p['выход'])[5:16]} по {p['цена выхода']:.2f}")
        ax.axhline(p["стоп стратегии"], color="#e03131", ls="--", lw=1, label=f"стоп стратегии {p['стоп стратегии']:.2f}")
        if abs(p["стоп"] - p["стоп стратегии"]) > 1e-6:
            ax.axhline(p["стоп"], color="#f08c00", ls="--", lw=1, label=f"стоп-заявка у брокера {p['стоп']:.2f}")
        ax.axhline(p["активация"], color="#2f9e44", ls=":", lw=1, label=f"трейлинг включился бы с {p['активация']:.2f}")
        ax.set_title(f"{tk} ({p['стратегия']}): {p['штук']} шт. на {p['сумма']:.0f} ₽ → {res:+.2f}% ({rub:+.0f} ₽)", fontsize=11)
        ticks = list(range(0, len(h), max(1, len(h) // 10)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([h.index[i].strftime("%d.%m %H:%M") for i in ticks], fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        rows.append({"бумага": tk, "стратегия": p["стратегия"], "вход": str(p["исполнено_в"])[5:16], "цена входа": p["цена_входа"],
                     "выход": str(p["выход"])[5:16], "цена выхода": round(p["цена выхода"], 2), "причина": p["причина"],
                     "штук": p["штук"], "результат %": round(res, 2), "результат ₽": round(rub)})
    fig.suptitle("Реальные сделки S42 за 11–20.11: обе закрыты в минус")
    fig.tight_layout()
    fig.savefig(out / "live_closed.png", dpi=115)
    r = pd.DataFrame(rows)
    r.to_csv(out / "live_closed.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
