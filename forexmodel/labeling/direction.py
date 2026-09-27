"""Направленная разметка: знак хода за горизонт, без барьеров.

Зачем ещё один режим. Triple-barrier с целью 1.5 ATR и стопом 0.75 ATR за 10 баров
спрашивает «дойдёт ли цена до барьера раньше стопа» — это вопрос про
волатильность, и модель на нём учится волатильности: у primary на серебре треть
важности — час суток, ещё 16% — `atr_ratio`, а вероятности лонга и шорта почти
равны (docs/results/silver_1h_cb/trade_review.md). Здесь вопрос другой: «куда
уйдёт цена за horizon баров», в единицах ATR бара сигнала:

    move = (close[i + 1 + horizon] - open[i + 1]) / atr[i]
    label = 2, если move >= +dir_atr; 0, если move <= -dir_atr; иначе 1.

Вход — по открытию следующего бара, как в симуляции (`use_next_open`). Минутные
данные не нужны: ход измеряется по часовым закрытиям. `label_t1` — индекс бара
конца горизонта, он нужен весам уникальности и embargo, как и у барьеров.

Ограничение: метка ничего не знает о пути цены внутри горизонта, поэтому стоп в
симуляции может выбить сделку, которая по метке «правильная». Это цена за то,
чтобы модель училась направлению, а не волатильности.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LabelingConfig
from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["generate_labels_direction"]


def generate_labels_direction(df_main: pd.DataFrame, cfg: LabelingConfig) -> pd.DataFrame:
    main = df_main.sort_values("time").reset_index(drop=True)
    if cfg.atr_col not in main.columns:
        raise KeyError(f"Нет колонки {cfg.atr_col!r}: разметка в ATR требует посчитанных признаков")

    n = len(main)
    h = cfg.horizon
    opens = main["open"].to_numpy(dtype=float)
    closes = main["close"].to_numpy(dtype=float)
    atr = main[cfg.atr_col].to_numpy(dtype=float)

    labels = np.full(n, np.nan)
    moves = np.full(n, np.nan)
    t1_idx = np.full(n, -1, dtype=int)
    last = n - 1 - h  # последний бар, у которого горизонт целиком в данных
    if last > 0:
        i = np.arange(last)
        move = (closes[i + 1 + h] - opens[i + 1]) / atr[i]
        ok = np.isfinite(move) & (atr[i] > 0)
        lab = np.where(move >= cfg.dir_atr, 2.0, np.where(move <= -cfg.dir_atr, 0.0, 1.0))
        labels[i[ok]] = lab[ok]
        moves[i[ok]] = move[ok]
        t1_idx[i[ok]] = i[ok] + 1 + h

    # move — сам ход в ATR: цель для регрессии (catboost.objective: regression)
    out = pd.DataFrame({"label": labels, "label_t1": t1_idx, "move": moves}, index=main.index)
    valid = out["label"].dropna()
    if len(valid):
        dist = valid.value_counts(normalize=True).sort_index().round(3).to_dict()
        log.info("Разметка direction (порог %.2f ATR, горизонт %d): %d баров, классы %s", cfg.dir_atr, h, len(valid), dist)
    return out
