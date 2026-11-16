"""Реальные сделки недели (счёт Т-Инвестиций, режим auto с 11.11): часовые свечи SBER и GAZP с точкой входа,
стопом и уровнем включения трейлинга; таблица «сигнал → заявка → исполнение».
    python scripts/live_trades_week.py [--out docs/results/weekly/2026-11-16]
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "raw" / "live"


def tail_csv(path: Path, n: int) -> pd.DataFrame:
    with open(path) as f:
        head = f.readline().strip().split(",")
        rows = deque((ln.rstrip("\n").split(",") for ln in f), maxlen=n)
    d = pd.DataFrame(list(rows), columns=head)
    for c in ("open", "high", "low", "close"):
        d[c] = d[c].astype(float)
    d["begin"] = pd.to_datetime(d["begin"])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "weekly" / "2026-11-16"))
    ap.add_argument("--since", default="2026-11-09")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = json.loads((ROOT / "reports" / "live" / "account_state.json").read_text())
    pos = [p for p in acc["позиции"] if p.get("исполнение") == "API"]
    fig, axes = plt.subplots(len(pos), 1, figsize=(11, 4 * len(pos)))
    axes = [axes] if len(pos) == 1 else axes
    rows = []
    for ax, p in zip(axes, pos):
        tk = p["бумага"]
        d = tail_csv(LIVE / f"{tk}_1m_live.csv", 12000)
        d = d[d["begin"] >= a.since].set_index("begin")
        h = d.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        x = range(len(h))
        for i, (_, r) in enumerate(h.iterrows()):
            c = "#2f9e44" if r["close"] >= r["open"] else "#e03131"
            ax.vlines(i, r["low"], r["high"], color=c, lw=0.8)
            ax.vlines(i, min(r["open"], r["close"]), max(r["open"], r["close"]), color=c, lw=3)
        t_in = pd.Timestamp(p["исполнено_в"]).floor("h")
        i_in = h.index.get_indexer([t_in], method="nearest")[0]
        ax.scatter([i_in], [p["цена_входа"]], marker="^", s=120, color="#1c7ed6", zorder=5,
                   label=f"покупка {p['исполнено_в'][5:16]} по {p['цена_входа']}")
        ax.axhline(p["стоп"], color="#e03131", ls="--", lw=1, label=f"стоп {p['стоп']:.1f}")
        ax.axhline(p["активация"], color="#2f9e44", ls=":", lw=1, label=f"трейлинг с {p['активация']:.1f}")
        last = h["close"].iloc[-1]
        ch = (last / p["цена_входа"] - 1) * 100
        ax.set_title(f"{tk} (S42): {p['штук']} шт. на {p['сумма']:.0f} ₽, сейчас {last:.2f} ({ch:+.2f}%), "
                     f"закрыть не позже {p['срок'][:10]}")
        ticks = list(range(0, len(h), max(1, len(h) // 10)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([h.index[i].strftime("%d.%m %H:%M") for i in ticks], fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        rows.append({"бумага": tk, "сигнал (час)": p["назначено"][5:16], "исполнено": p["исполнено_в"][5:19],
                     "цена контура": p["цена"], "цена исполнения": p["цена_входа"],
                     "проскальзывание %": round((p["цена_входа"] / p["цена"] - 1) * 100, 3),
                     "сумма ₽": round(p["сумма"]), "сейчас": last, "результат %": round(ch, 2),
                     "результат ₽": round((last - p["цена_входа"]) * p["штук"], 1)})
    fig.suptitle("Первые сделки на реальные деньги: покупка через API сразу по сигналу S42")
    fig.tight_layout()
    fig.savefig(out / "live_trades.png", dpi=120)
    t = pd.DataFrame(rows)
    t.to_csv(out / "live_trades.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
