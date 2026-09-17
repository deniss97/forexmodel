"""Triple-barrier разметка с АСИММЕТРИЧНЫМИ барьерами в ATR.

Класс 2: цена дошла до +tp_atr*ATR раньше, чем до -sl_atr*ATR.
Класс 0: симметрично вниз.
Класс 1: ни одна из «прибыльных» конфигураций не сложилась.

Зачем это нужно (главный вывод разбора пайплайна):

  при tp_atr > sl_atr бар «на излёте» движения почти никогда не успевает дойти
  до цели раньше стопа, поэтому получает класс 1 или противоположный — и модель
  ВЫНУЖДЕНА научиться отличать раннюю фазу движения от поздней. При симметричных
  барьерах (±0.5%) старт и излёт для модели неотличимы: до цели цена добегает и
  там, и там, а значит извлечь эту информацию из данных невозможно в принципе.

Барьеры в ATR дополнительно снимают зависимость от режима волатильности —
фиксированные 0.5% на периоде 2016-2026 означают совершенно разные задачи в
спокойный и в турбулентный год.

ВАЖНО: эти же tp_atr/sl_atr должны использоваться в симуляции
(simulation.exit_mode: atr), иначе модель учится на одних уровнях, а торгует
на других — ровно та ошибка, что была в ноутбуке.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LabelingConfig
from ..logging_utils import get_logger
from .core import MinuteBook, first_index

log = get_logger(__name__)

__all__ = ["generate_labels_atr_asym"]


def generate_labels_atr_asym(df_main: pd.DataFrame, df_fine: pd.DataFrame, cfg: LabelingConfig) -> pd.DataFrame:
    main = df_main.sort_values("time").reset_index(drop=True)
    if cfg.atr_col not in main.columns:
        raise KeyError(f"Нет колонки {cfg.atr_col!r}: разметка в ATR требует посчитанных признаков")

    book = MinuteBook.from_frame(df_fine)

    n = len(main)
    labels = np.full(n, np.nan)
    t1_idx = np.full(n, -1, dtype=int)

    times = pd.to_datetime(main["time"])
    opens = main["open"].to_numpy(dtype=float)
    atr_values = main[cfg.atr_col].to_numpy(dtype=float)
    delay = pd.Timedelta(minutes=cfg.min_minutes_after)

    for i in range(n):
        j_end = i + 1 + cfg.horizon
        if j_end >= n:
            continue

        atr = atr_values[i]
        if not np.isfinite(atr) or atr <= 0:
            continue

        base = opens[i + 1] if cfg.use_next_open else float(main["close"].iloc[i])
        long_tp, long_sl = base + cfg.tp_atr * atr, base - cfg.sl_atr * atr
        short_tp, short_sl = base - cfg.tp_atr * atr, base + cfg.sl_atr * atr

        win = book.window(times.iloc[i + 1] + delay, times.iloc[j_end])
        if win.stop <= win.start:
            continue

        highs = book.high[win]
        lows = book.low[win]

        long_win = first_index(highs >= long_tp) < first_index(lows <= long_sl)
        short_win = first_index(lows <= short_tp) < first_index(highs >= short_sl)

        if long_win and not short_win:
            labels[i] = 2
        elif short_win and not long_win:
            labels[i] = 0
        else:
            labels[i] = 1

        t1_idx[i] = j_end

    out = pd.DataFrame({"label": labels, "label_t1": t1_idx}, index=main.index)
    dist = out["label"].value_counts(normalize=True, dropna=True).sort_index()
    log.info(
        "Разметка atr_asym (tp=%.2f ATR, sl=%.2f ATR, horizon=%d): %s",
        cfg.tp_atr,
        cfg.sl_atr,
        cfg.horizon,
        {int(k): round(float(v), 3) for k, v in dist.items()},
    )
    return out
