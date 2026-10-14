"""Входы по паттернам дневных свечей на часовых закрытиях (ТЗ 15.10, гипотезы H46, H47).

Дневная свеча — календарный день данных; её максимум и минимум известны только после окончания дня,
поэтому часовой бар дня D видит лишь дни до D. Сигнал — на закрытии часа (вход, как у импульса, с
open_delay_minutes), не чаще одного на день и сторону.

  nr7 / nr4   день с самым узким диапазоном (high − low) из последних 7 / 4 дней (Крабел): на следующий
              день первое закрытие часа выше его максимума — лонг, ниже минимума — шорт. На часах NR7
              ловил бы азиатские часы, поэтому только дневки
  ib          inside bar: максимум дня ниже прошлого, минимум выше — пробой его границ на следующий день
  nr7ib       NR7 и inside bar одновременно (сильнее сжатие)
  donch:N[:b] пробой максимума / минимума N прошлых дней (Дончиан; N = 1 — вчерашний день): первое
              закрытие часа за уровнем + b·ATR (зона вокруг уровня, по умолчанию 0)
  soup:N      ложный пробой («Turtle Soup»): час вышел за N-дневный максимум, но закрылся под ним —
              шорт; зеркально — лонг
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

LONG, SHORT = 2, 0
PATTERNS = ("nr7", "nr4", "ib", "nr7ib", "donch", "soup")


def _daily(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    day = pd.to_datetime(df["time"]).dt.floor("D")
    d = df.groupby(day.to_numpy()).agg(high=("high", "max"), low=("low", "min"))
    return d, day.to_numpy()


def _first_per_day(day: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """Только первое срабатывание за день."""
    s = pd.Series(hit.astype(int))
    return hit & (s.groupby(day).cumsum().to_numpy() == 1)


def pattern_signal(df: pd.DataFrame, spec: str, atr_col: str) -> pd.Series:
    name, *p = spec.split(":")
    if name not in PATTERNS:
        raise ValueError(f"simulation.pattern: неизвестный паттерн {spec!r} (ожидалось одно из {PATTERNS})")
    d, day = _daily(df)
    c = df["close"].to_numpy(dtype=float)
    hi_h, lo_h = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    rng = d["high"] - d["low"]
    if name in ("nr7", "nr4", "ib", "nr7ib"):
        nr = rng <= rng.rolling(7 if name in ("nr7", "nr7ib") else 4).min()
        ib = (d["high"] < d["high"].shift(1)) & (d["low"] > d["low"].shift(1))
        flag = {"nr7": nr, "nr4": nr, "ib": ib, "nr7ib": nr & ib}[name]
        # уровень дня-паттерна действует на следующий торговый день
        lvl = pd.DataFrame({"up": d["high"].where(flag), "dn": d["low"].where(flag)}).shift(1)
        up, dn = lvl["up"].reindex(day).to_numpy(), lvl["dn"].reindex(day).to_numpy()
        buf = 0.0
    else:
        n = int(p[0])
        up = d["high"].rolling(n).max().shift(1).reindex(day).to_numpy()
        dn = d["low"].rolling(n).min().shift(1).reindex(day).to_numpy()
        buf = float(p[1]) if len(p) > 1 else 0.0
    atr = df[atr_col].to_numpy(dtype=float) if atr_col in df.columns else np.zeros(len(df))
    if name == "soup":
        go_short = (hi_h > up) & (c < up)
        go_long = (lo_h < dn) & (c > dn)
    else:
        go_long = c > up + buf * np.nan_to_num(atr)
        go_short = c < dn - buf * np.nan_to_num(atr)
    go_long = _first_per_day(day, np.nan_to_num(go_long).astype(bool))
    go_short = _first_per_day(day, np.nan_to_num(go_short).astype(bool))
    both = go_long & go_short
    signal = pd.Series(np.nan, index=df.index, dtype=float)
    signal[go_long & ~both] = LONG
    signal[go_short & ~both] = SHORT
    log.info("Паттерн %s: лонгов %d, шортов %d из %d баров", spec, int((signal == LONG).sum()),
             int((signal == SHORT).sum()), len(df))
    return signal
