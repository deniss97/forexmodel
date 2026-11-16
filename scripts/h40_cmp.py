"""H40: сходятся ли уровни, посчитанные по ленте сделок и по минуткам.

План гипотезы называл источником уровней ленту обезличенных сделок, но её выгрузка с Диска идёт часами, а сделки
стратегий начинаются в 2016-м. Поэтому основной расчёт сделан по минуткам (объём минуты в уровень её средней цены),
а здесь проверяется, что на днях, где лента уже выгружена, оба источника дают одни и те же уровни:
  * доля сделок, где совпал признак «большой уровень по ходу ближе 1 ATR»;
  * связь расстояний до уровня (корреляция Спирмена) и разница в PF по группам.

    python scripts/h40_cmp.py min_main tape_main
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "h40_trades"
KEY = ["tk", "open_dt", "strategy"]


def pf(v: np.ndarray) -> float:
    pos, neg = v[v > 0].sum(), -v[v < 0].sum()
    return float(pos / neg) if neg > 0 else np.nan


def main(argv: list[str]) -> int:
    a, b = (argv + ["min_main", "tape_main"])[:2]
    A = pd.read_csv(REP / f"trades_{a}.csv", parse_dates=["open_dt"])
    B = pd.read_csv(REP / f"trades_{b}.csv", parse_dates=["open_dt"])
    m = A.merge(B, on=KEY, suffixes=("_min", "_tape"))
    if m.empty:
        print("нет общих сделок")
        return 1
    na = m["d_block_min"].fillna(np.inf) <= 1
    nb = m["d_block_tape"].fillna(np.inf) <= 1
    both = m["d_block_min"].notna() & m["d_block_tape"].notna()
    rho = spearmanr(m.loc[both, "d_block_min"], m.loc[both, "d_block_tape"]).statistic if both.sum() > 10 else np.nan
    rows = [{
        "общих сделок": len(m),
        "период": f"{m['open_dt'].min():%Y-%m-%d} — {m['open_dt'].max():%Y-%m-%d}",
        "есть уровень (минутки), %": round(100 * float(m["d_block_min"].notna().mean()), 1),
        "есть уровень (лента), %": round(100 * float(m["d_block_tape"].notna().mean()), 1),
        "совпал признак «ближе 1 ATR», %": round(100 * float((na == nb).mean()), 1),
        "корреляция расстояний": round(float(rho), 3),
        "PF рядом (минутки)": round(pf(m.loc[na, "profit_pct_min"].to_numpy()), 3),
        "PF рядом (лента)": round(pf(m.loc[nb, "profit_pct_min"].to_numpy()), 3),
        "PF остальные (минутки)": round(pf(m.loc[~na, "profit_pct_min"].to_numpy()), 3),
        "PF остальные (лента)": round(pf(m.loc[~nb, "profit_pct_min"].to_numpy()), 3),
    }]
    t = pd.DataFrame(rows).T
    t.to_csv(REP / f"cmp_{a}_{b}.csv", header=False)
    print(t.to_string(header=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
