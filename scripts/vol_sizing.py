"""Размер позиции по волатильности на готовых сделках (гипотеза H07).

Стоп импульса стоит в 6 ATR, поэтому при одинаковом размере риск сделки растёт с волатильностью. Здесь
размер сделки = медиана ATR% инструмента по сделкам выбора (до 2021 года) / ATR% на входе, в пределах
0.5–2: одинаковый риск на сделку, на выборе средний размер около 1. Путь сделки от размера не зависит,
поэтому пересчитываются только колонки результата; издержки журнал пересчитывает с размером сам (recost).

    python scripts/vol_sizing.py silver_costs__impulse_6_3_h240_tr6 lkoh_lo__impulse_6_3_long ...

Пишет reports/walk_forward/<прогон>__vs_trades.csv — источник для catalog.yaml.
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START

from experiments import WF

LO, HI = 0.5, 2.0
SCALED = ("gross_pct", "commission_pct", "swap_pct", "profit_pct", "profit_points")


def resize(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    atr_pct = t["atr_at_entry"] / t["open_price"]
    ref = atr_pct[t["open_dt"] < TEST_START].median()
    old = t["size"] if "size" in t else pd.Series(1.0, index=t.index)
    new = np.clip(ref / atr_pct, LO, HI)
    if "gross_unit_pct" not in t:
        t["gross_unit_pct"] = t["gross_pct"] / old
    for c in SCALED:
        if c in t:
            t[c] = t[c] / old * new
    t["size"] = new
    return t


def main(runs: list[str]) -> int:
    for run in runs:
        t = pd.read_csv(WF / f"{run}_trades.csv", parse_dates=["open_dt", "close_dt"])
        r = resize(t)
        r.to_csv(WF / f"{run}__vs_trades.csv", index=False)
        sel = r["open_dt"] < TEST_START
        print(f"{run}: средний размер — выбор {r.loc[sel, 'size'].mean():.2f}, проверка {r.loc[~sel, 'size'].mean():.2f}; "
              f"на границах 0.5 / 2: {(r['size'] <= LO).mean():.0%} / {(r['size'] >= HI).mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
