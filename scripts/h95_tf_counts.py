"""Фигуры кодом Тимура (docs/timur/2026-11-15/chart_patterns.py) на младших таймфреймах — сколько примеров даёт разметка.

Созвон 23.11 (Тимур): на дневках по фигуре ~50 примеров — «ни о чём»; фигуры — фрактал, их можно размечать на часе и
15 минутах, нужно 200–1000 примеров. Здесь только подсчёт (подготовка к H95, план которой ждёт ответов): сколько фигур
каждого типа, сколько с пробоем, у какой доли пробой раньше, чем фигура видна (known_at — заглядывание вперёд, как в H74).
Настройки автора по умолчанию (Config()), ATR — в барах своего таймфрейма.

    python scripts/h95_tf_counts.py          # день и час — 19 акций, 15 минут — SBER, GAZP, LKOH
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "docs" / "timur" / "2026-11-15"))
import chart_patterns as cp  # noqa: E402
import h74_patterns as mine  # noqa: E402

OUT = ROOT / "docs" / "results" / "h95"
M15 = ["SBER", "GAZP", "LKOH"]
SRC = {"SBER": ROOT / "data/raw/SBER_1m_15-26.csv", "GAZP": ROOT / "data/raw/GAZP_1m_15-26_upd.csv",
       "LKOH": ROOT / "data/raw/LKOH_1m_15-26.csv"}


def m15(tk: str) -> pd.DataFrame:
    d = pd.read_csv(SRC[tk], usecols=["open", "high", "low", "close", "volume", "begin"], parse_dates=["begin"]).set_index("begin")
    return d.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def run(tk: str, tf: str, bars: pd.DataFrame) -> list[dict]:
    pats = cp.detect_patterns(bars, cp.Config())
    out = []
    for p in pats:
        bo, ka = p.breakout_idx, p.known_at
        out.append({"бумага": tk, "таймфрейм": tf, "фигура": p.name_ru, "pattern": p.name, "статус": p.status,
                    "пробой раньше known_at": bo is not None and (ka is None or bo < ka),
                    "год": bars.index[p.start_idx].year})
    print(tk, tf, len(bars), "баров,", len(pats), "фигур", flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tk in mine.TICKERS:
        h = mine.hourly(tk)
        rows += run(tk, "день", mine.to_daily(h))
        rows += run(tk, "час", h)
        if tk in M15:
            rows += run(tk, "15 мин", m15(tk))
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "patterns_by_tf.csv", index=False)
    c = a[a["статус"] == "confirmed"]
    t = c.pivot_table(index="фигура", columns="таймфрейм", values="бумага", aggfunc="size", fill_value=0)
    # на 15 минутах — только 3 бумаги: для сравнения ещё день и час на тех же трёх
    c3 = c[c["бумага"].isin(M15)]
    t3 = c3.pivot_table(index="фигура", columns="таймфрейм", values="бумага", aggfunc="size", fill_value=0).add_prefix("SBER+GAZP+LKOH: ")
    peek = c.groupby(["фигура", "таймфрейм"])["пробой раньше known_at"].mean().mul(100).round(0).unstack().add_prefix("заглядывание %, ")
    res = t.join(t3).join(peek).sort_values("час", ascending=False)
    res.to_csv(OUT / "counts_by_tf.csv")
    print(res.to_string())


if __name__ == "__main__":
    main()
