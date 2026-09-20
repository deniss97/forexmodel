"""Минутная симуляция сделок.

Главное исправление относительно ноутбука — согласование барьеров с разметкой.
Было:

    open_price = raw_open * (1 + commission)      # комиссия зашита в цену входа
    tp_price   = open_price * (1 + tp/100)        # барьеры считаются ОТ НЕЁ
    exit_net   = exit_price * (1 - 0)             # комиссия на выходе не берётся

То есть при комиссии 0.15% и TP/SL 0.5% реальный TP был +0.65% от сырой цены,
а SL -0.35%, тогда как модель обучалась на симметричных ±0.5%. «Правильная»
метка и «сработавший TP» — разные события, и результаты симуляции с метриками
модели сравнивать было нельзя.

Стало: барьеры считаются от СЫРОЙ цены входа, ровно как в разметке, а комиссия
за круг вычитается из PnL отдельно.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from ..progress import Progress
from .exits import fixed_barrier_exit, trailing_exit
from .report import build_report

log = get_logger(__name__)

__all__ = ["simulate_trades"]

LONG, SHORT = 2, 0


def simulate_trades(
    df_signals: pd.DataFrame,
    df_prices_1m: pd.DataFrame,
    cfg: Config,
    signal_col: str = "final_class",
    time_col: str = "time",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sim = cfg.simulation
    horizon_minutes = cfg.horizon_minutes

    sig = df_signals.copy()
    sig[time_col] = pd.to_datetime(sig[time_col])
    sig = sig.sort_values(time_col).reset_index(drop=True)

    px = df_prices_1m.copy()
    px[time_col] = pd.to_datetime(px[time_col])
    px = px.sort_values(time_col).reset_index(drop=True)

    if signal_col not in sig.columns:
        raise KeyError(f"Нет колонки сигнала {signal_col!r} — вызовите signals.build_signal_column")

    trend_minutes = _trend_on_minutes(sig, px, sim.trend_col, time_col) if sim.close_on_trend_flip else None

    p_time = px[time_col].to_numpy()
    p_open = px["open"].to_numpy(dtype=float)
    p_high = px["high"].to_numpy(dtype=float)
    p_low = px["low"].to_numpy(dtype=float)
    p_close = px["close"].to_numpy(dtype=float)

    active = sig[pd.to_numeric(sig[signal_col], errors="coerce").isin([LONG, SHORT])].reset_index(drop=True)
    if active.empty:
        log.warning("Нет сигналов после фильтрации")
        return pd.DataFrame(), build_report(pd.DataFrame())

    log.info(
        "Симуляция: %d сигналов, exit_mode=%s, horizon=%d мин, комиссия %.3f%% за круг",
        len(active),
        sim.exit_mode,
        horizon_minutes,
        sim.commission_pct,
    )

    trades: List[dict] = []
    last_exit_time: Optional[pd.Timestamp] = None
    skipped = {"overlap": 0, "trend": 0, "no_prices": 0, "no_atr": 0}

    # колонки вытаскиваем в массивы заранее: itertuples ломается на «неудобных»
    # именах колонок, а .loc[i, col] в цикле — это основной тормоз старой версии
    sig_values = pd.to_numeric(active[signal_col], errors="coerce").to_numpy(dtype=float)
    sig_times = active[time_col]
    sig_trend = _column_or_nan(active, sim.trend_col)
    sig_atr = _column_or_nan(active, sim.atr_col)

    # на бэктесте сигналов сотни и цикл мгновенный, а при сборе мета-меток их
    # десятки тысяч — min_interval гарантирует, что в первом случае в лог не
    # попадёт ничего лишнего, а во втором прогресс будет виден
    bar = Progress(
        len(active), label="Симуляция сигналов", unit="сиг", logger=log, min_interval=15.0, min_total=2000
    )

    for i in range(len(active)):
        bar.update()
        signal = int(sig_values[i])
        signal_dt = sig_times.iloc[i]

        if not sim.allow_overlapping_positions and last_exit_time is not None and signal_dt <= last_exit_time:
            skipped["overlap"] += 1
            continue

        if sim.use_trend_filter:
            trend = sig_trend[i]
            if np.isfinite(trend) and ((signal == LONG and trend <= 0) or (signal == SHORT and trend >= 0)):
                skipped["trend"] += 1
                continue

        open_time = signal_dt + pd.Timedelta(minutes=sim.open_delay_minutes)
        open_idx = int(np.searchsorted(p_time, open_time.to_datetime64(), side="left"))
        if open_idx >= len(px):
            skipped["no_prices"] += 1
            continue

        entry_price = float(p_open[open_idx])
        side = "buy" if signal == LONG else "sell"

        horizon_end = (open_time + pd.Timedelta(minutes=horizon_minutes)).to_datetime64()
        last_idx = int(np.searchsorted(p_time, horizon_end, side="right")) - 1
        last_idx = min(max(last_idx, open_idx), len(px) - 1)

        # если внутри горизонта есть только сам бар входа, сделке негде
        # разыграться: она закрывается той же минутой по той же цене и даёт
        # ровно минус комиссию. Так было на открытии рынка серебра после
        # выходных, где за первой минутой в данных идёт разрыв
        if last_idx <= open_idx:
            skipped["no_prices"] += 1
            continue
        win = slice(open_idx, last_idx + 1)

        atr = float(sig_atr[i])
        if sim.exit_mode != "fixed_pct" and (not np.isfinite(atr) or atr <= 0):
            skipped["no_atr"] += 1
            continue

        exit_rel, exit_price, exit_reason = _resolve_exit(
            sim, side, entry_price, atr, p_high[win], p_low[win], p_close[win]
        )

        if trend_minutes is not None:
            flip = _trend_flip_index(trend_minutes[win], side, sim.trend_flip_mode)
            if flip is not None and flip < exit_rel:
                exit_rel, exit_price, exit_reason = flip, float(p_close[win][flip]), "trend_flip"

        exit_idx = open_idx + exit_rel
        exit_dt = px[time_col].iloc[exit_idx]
        last_exit_time = exit_dt

        direction = 1.0 if side == "buy" else -1.0
        gross_pct = direction * (exit_price - entry_price) / entry_price * 100.0
        profit_pct = gross_pct - sim.commission_pct  # комиссия за круг, отдельно от барьеров

        trades.append(
            {
                "signal_dt": signal_dt,
                "open_dt": px[time_col].iloc[open_idx],
                "close_dt": exit_dt,
                "side": side,
                "predicted_class": signal,
                "open_price": entry_price,
                "exit_price": exit_price,
                "gross_pct": gross_pct,
                "commission_pct": sim.commission_pct,
                "profit_pct": profit_pct,
                "profit_points": direction * (exit_price - entry_price),
                "exit_reason": exit_reason,
                "atr_at_entry": atr,
                "minutes_in_trade": int((exit_dt - px[time_col].iloc[open_idx]).total_seconds() / 60),
            }
        )

    trades_df = pd.DataFrame(trades)
    log.info(
        "Сделок: %d | пропущено: перекрытие=%d, тренд=%d, нет цен=%d, нет ATR=%d",
        len(trades_df),
        skipped["overlap"],
        skipped["trend"],
        skipped["no_prices"],
        skipped["no_atr"],
    )

    report = build_report(trades_df)
    # причины, по которым сигнал не стал сделкой, нужны в отчёте: без них
    # «сигналов 9, сделок 4» выглядит как ошибка, а не как работа фильтров
    report.update({f"skipped_{reason}": count for reason, count in skipped.items()})
    report["n_signals_in"] = len(active)
    return trades_df, report


def _column_or_nan(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _resolve_exit(sim, side, entry_price, atr, high, low, close):
    """Выбор правила выхода. Барьеры — ВСЕГДА от сырой цены входа."""
    if sim.exit_mode == "trailing":
        return trailing_exit(
            high, low, close, side, entry_price, atr, sim.sl_atr, sim.trail_atr, sim.activate_atr
        )

    if sim.exit_mode == "atr":
        sgn = 1.0 if side == "buy" else -1.0
        tp_price = entry_price + sgn * sim.tp_atr * atr
        sl_price = entry_price - sgn * sim.sl_atr * atr
    else:  # fixed_pct
        sgn = 1.0 if side == "buy" else -1.0
        tp_price = entry_price * (1 + sgn * sim.take_profit_pct / 100.0)
        sl_price = entry_price * (1 - sgn * sim.stop_loss_pct / 100.0)

    hit = fixed_barrier_exit(high, low, side, tp_price, sl_price)
    if hit is not None:
        return hit
    last = len(close) - 1
    return last, float(close[last]), "timeout"


def _trend_on_minutes(sig: pd.DataFrame, px: pd.DataFrame, trend_col: str, time_col: str) -> Optional[np.ndarray]:
    """Растягивает тренд рабочего ТФ на минутную сетку (последнее известное значение)."""
    if trend_col not in sig.columns:
        log.warning("close_on_trend_flip=True, но колонки %r нет — выход по флипу отключён", trend_col)
        return None

    lookup = (
        sig[[time_col, trend_col]]
        .dropna(subset=[trend_col])
        .drop_duplicates(subset=[time_col])
        .sort_values(time_col)
    )
    if lookup.empty:
        log.warning("Нет валидных значений тренда — выход по флипу отключён")
        return None

    merged = pd.merge_asof(px[[time_col]], lookup, on=time_col, direction="backward")
    return merged[trend_col].to_numpy(dtype=float)


def _trend_flip_index(trend_window: np.ndarray, side: str, mode: str) -> Optional[int]:
    """Первый бар ПОСЛЕ входа, на котором тренд развернулся."""
    if len(trend_window) <= 1:
        return None
    t = trend_window[1:]
    if mode == "opposite":
        flipped = (t < 0) if side == "buy" else (t > 0)
    else:  # not_aligned — нейтраль тоже считается поводом выйти
        flipped = (t <= 0) if side == "buy" else (t >= 0)
    flipped = flipped & np.isfinite(t)
    idx = np.flatnonzero(flipped)
    return int(idx[0]) + 1 if idx.size else None
