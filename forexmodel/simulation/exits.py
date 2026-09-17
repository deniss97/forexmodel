"""Правила выхода из позиции.

Все функции работают на numpy-срезах минутных баров и возвращают
(относительный индекс бара выхода, цена выхода, причина).

Разрешение неоднозначности внутри бара: если в одном минутном баре задеты и
TP, и SL, считается сработавшим СТОП. В ноутбуке приоритет был у TP, что
систематически завышало результат.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["fixed_barrier_exit", "trailing_exit", "first_index"]

ExitResult = Tuple[int, float, str]


def first_index(mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(idx[0]) if idx.size else np.inf


def fixed_barrier_exit(
    high: np.ndarray,
    low: np.ndarray,
    side: str,
    tp_price: Optional[float],
    sl_price: Optional[float],
) -> Optional[ExitResult]:
    """Первое касание TP/SL. None — ни один барьер не задет за окно."""
    if side == "buy":
        tp_hit = first_index(high >= tp_price) if tp_price is not None else np.inf
        sl_hit = first_index(low <= sl_price) if sl_price is not None else np.inf
    else:
        tp_hit = first_index(low <= tp_price) if tp_price is not None else np.inf
        sl_hit = first_index(high >= sl_price) if sl_price is not None else np.inf

    if np.isinf(tp_hit) and np.isinf(sl_hit):
        return None
    if sl_hit <= tp_hit:  # ничья внутри бара трактуется в пользу стопа
        return int(sl_hit), float(sl_price), "stop_loss"
    return int(tp_hit), float(tp_price), "take_profit"


def trailing_exit(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side: str,
    entry_price: float,
    atr: float,
    sl_atr: float = 1.0,
    trail_atr: float = 1.5,
    activate_atr: float = 1.0,
) -> ExitResult:
    """Трейлинг вместо фиксированного TP.

    Стоп = entry ∓ sl_atr*ATR; после хода в нашу сторону на activate_atr*ATR
    стоп подтягивается на trail_atr*ATR от лучшей цены. Фиксированного TP нет:
    ложный пробой режется стопом, а большое движение забирается целиком.

    Это ответ на вторую половину проблемы «не пропускать большие движения»:
    симметричные TP=SL=0.5% режут прибыль там, где движение большое, и полностью
    отдают убыток там, где пробой ложный.
    """
    sgn = 1.0 if side == "buy" else -1.0
    stop = entry_price - sgn * sl_atr * atr
    best = entry_price
    activated = False

    for j in range(len(high)):
        worst = low[j] if sgn > 0 else high[j]
        if (worst - stop) * sgn <= 0:
            return j, float(stop), "trailing_stop" if activated else "stop_loss"

        ext = high[j] if sgn > 0 else low[j]
        if (ext - best) * sgn > 0:
            best = ext
            if (best - entry_price) * sgn >= activate_atr * atr:
                activated = True
                new_stop = best - sgn * trail_atr * atr
                stop = max(stop, new_stop) if sgn > 0 else min(stop, new_stop)

    last = len(close) - 1
    return last, float(close[last]), "timeout"
