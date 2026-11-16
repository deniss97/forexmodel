"""H40: как часто максимум цены в сделке действительно останавливается у большого уровня (и не случайность ли это).

Читает признаки сделок, посчитанные scripts/h40_trades.py (reports/h40_trades/trades_<тег>.csv):
  * «остановился у уровня» — лучший ход сделки (MFE, в ATR) отличается от расстояния до уровня не больше чем на 0.2 ATR;
  * пустышка — то же, но вместо уровня берётся цена на 0.5 и 1.0 ATR дальше (уровня там нет).
Если доля остановок у уровня не выше, чем у пустышки, «цена упирается в уровень» на наших сделках не видно.

    python scripts/h40_stall.py min_main
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "h40_trades"


def main(argv: list[str]) -> int:
    tag = argv[0] if argv else "min_main"
    f = pd.read_csv(REP / f"trades_{tag}.csv")
    out = []
    for thr in (1.0, 2.0, 99.0):
        g = f[np.isfinite(f["d_block"]) & (f["d_block"] <= thr)]
        if g.empty:
            continue
        row = {"уровень ближе ATR": thr, "сделок": len(g)}
        for lab, shift in (("у уровня", 0.0), ("+0.5 ATR", 0.5), ("+1 ATR", 1.0)):
            row[lab] = 100 * float((np.abs(g["mfe_atr"] - (g["d_block"] + shift)) <= 0.2).mean())
        row["дошёл до уровня"] = 100 * float((g["mfe_atr"] >= g["d_block"]).mean())
        out.append(row)
    t = pd.DataFrame(out).round(1)
    t.to_csv(REP / f"stall_{tag}.csv", index=False)
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
