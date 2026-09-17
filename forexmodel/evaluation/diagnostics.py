"""Диагностика «входов на излёте» и санитарные проверки пайплайна.

`late_entry_diagnostic` — то, с чего стоит начинать любую итерацию: она
показывает PnL в разрезе того, сколько ATR цена уже прошла от противоположного
экстремума на момент входа. Если winrate падает с ростом растяжения, гипотеза
«входим поздно» подтверждена, и можно даже не переобучаться сразу, а поставить
грубое правило `simulation.max_extension_atr = 1.5`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["late_entry_diagnostic", "exit_reason_breakdown", "check_split_sanity"]


def late_entry_diagnostic(
    trades_df: pd.DataFrame,
    df_hour: pd.DataFrame,
    atr_col: str = "atr_14",
    lookback: int = 24,
    bins: Sequence[float] = (0, 0.5, 1, 1.5, 2, 3, 99),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """extension_atr = сколько ATR пройдено от противоположного экстремума до входа."""
    if trades_df.empty:
        log.warning("Нет сделок для диагностики")
        return trades_df, pd.DataFrame()

    h = df_hour.sort_values("time").reset_index(drop=True).copy()
    h["roll_low"] = h["low"].rolling(lookback).min().shift(1)   # без текущего бара
    h["roll_high"] = h["high"].rolling(lookback).max().shift(1)
    h["atr_prev"] = h[atr_col].shift(1)
    lookup = h[["time", "roll_low", "roll_high", "atr_prev"]].rename(columns={"time": "signal_dt"})

    t = trades_df.copy()
    t["signal_dt"] = pd.to_datetime(t["signal_dt"])
    t = t.merge(lookup, on="signal_dt", how="left")

    long_mask = t["side"].eq("buy")
    atr_prev = t["atr_prev"].replace(0, np.nan)
    t["extension_atr"] = np.where(
        long_mask,
        (t["open_price"] - t["roll_low"]) / atr_prev,
        (t["roll_high"] - t["open_price"]) / atr_prev,
    )
    t["ext_bin"] = pd.cut(t["extension_atr"], bins=list(bins))

    summary = (
        t.groupby("ext_bin", observed=True)
        .agg(
            n=("profit_pct", "size"),
            winrate=("profit_pct", lambda s: (s > 0).mean() * 100),
            avg_pnl=("profit_pct", "mean"),
            total_pnl=("profit_pct", "sum"),
            tp=("exit_reason", lambda s: (s == "take_profit").mean() * 100),
            sl=("exit_reason", lambda s: (s == "stop_loss").mean() * 100),
        )
        .round(2)
    )
    log.info("Диагностика поздних входов (PnL по растяжению в ATR):\n%s", summary)
    return t, summary


def exit_reason_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    table = (
        trades_df.groupby("exit_reason")
        .agg(n=("profit_pct", "size"), avg_pnl=("profit_pct", "mean"), total_pnl=("profit_pct", "sum"))
        .round(3)
        .sort_values("total_pnl", ascending=False)
    )
    log.info("Разбивка по причинам выхода:\n%s", table)
    return table


def check_split_sanity(splits: dict[str, pd.DataFrame], time_col: str = "time") -> None:
    """Печатает границы выборок и падает, если они пересекаются.

    Именно эта проверка ловит ошибку исходного ноутбука, где df_min_test
    целиком лежал внутри df_min_train.
    """
    bounds = {}
    for name, df in splits.items():
        if df.empty:
            log.warning("Выборка %s пуста", name)
            continue
        bounds[name] = (df[time_col].iloc[0], df[time_col].iloc[-1])
        log.info("%-6s: %s .. %s (%d баров)", name, bounds[name][0], bounds[name][1], len(df))

    names = list(bounds)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            a_start, a_end = bounds[a]
            b_start, b_end = bounds[b]
            if a_start <= b_end and b_start <= a_end:
                raise ValueError(f"Выборки {a} и {b} пересекаются по времени: {bounds[a]} vs {bounds[b]}")
