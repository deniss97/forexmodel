"""Тренд-фильтр старшего ТФ (4H) — только каузальные реализации.

Из ноутбука сюда НЕ перенесена `add_trend_filter_enhanced`: она мёрджила
4h/daily бары по времени НАЧАЛА бина, то есть каждый часовой бар внутри бина
получал тренд, посчитанный по close этого же бина — до 3 часов (а для daily
до 23 часов) будущего. Это мина замедленного действия, и держать такой код
рядом с рабочим нельзя.

Обе оставшиеся версии мёрджатся через `merge_htf_causal`: значение старшего ТФ
становится видно только с момента фактического ЗАКРЫТИЯ бина.

  * `confirm` — подтверждающий фильтр (EMA + ADX выше порога). Появляется,
    когда тренд уже отработал заметную часть пути.
  * `early`   — опережающий: наклон регрессии в ATR + ADX В КОРИДОРЕ
    [adx_min, adx_max] и растущий. Верхняя граница по ADX — прямой запрет
    входов в перегретый тренд (поздняя фаза), нижняя с условием роста ловит
    разгон.

Два общих режима (`cfg.update_every_bar`, `cfg.hold_bars`) — ответ на
разбор сделок docs/results/silver_1h_cb/trade_review.md:

  * при одной сетке 00/04/08.. значение обновляется раз в 4 часа, и обвал,
    начавшийся в 00:00, фильтр видит только в 04:00. С `update_every_bar`
    4-часовые бины строятся на каждой сетке со сдвигом на бар рабочего ТФ
    (00/04.., 01/05.., 02/06.., 03/07..), и на каждом часе берётся окно,
    закрывшееся ровно сейчас. Каузальность та же: значение бина видно только
    после его закрытия;
  * соседние сетки иногда расходятся, и тренд мигает (включается на час).
    `hold_bars` не даёт короткой нейтрали сбросить направление.

Слои поверх фильтра (по умолчанию выключены, порядок применения такой):

  * `hysteresis` (только confirm) — гистерезис из ноутбука S_TrendFiter_dev
    (add_trend_filter_v3): закрытие бина по другую сторону EMA без подтверждения
    ADX+DI — нейтраль, а не разворот;
  * шок-слой (`shock_atr` > 0) — быстрый слой ноутбука (apply_shock_override):
    резкий ход за несколько баров рабочего ТФ сразу задаёт направление в нейтрали
    и гасит встречный тренд;
  * сброс по EMA (`ema_reset_period` > 0) — идея 2026-10-03: тренд не может быть
    +1, пока свеча закрывается ниже EMA, и −1 — пока выше. Применяется последним,
    поэтому правило выполняется на каждой свече рабочего ТФ.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd

from ..config import TrendConfig
from ..data.loader import resample_ohlcv
from ..logging_utils import get_logger
from . import indicators as ind

log = get_logger(__name__)

__all__ = [
    "merge_htf_causal",
    "add_trend_filter",
    "add_trend_filter_early",
    "add_trend_filter_confirm",
    "trend_gate_values",
    "hold_direction",
    "ema_reset",
    "reset_ema",
    "shock_direction",
    "TREND_COLUMNS",
]

TREND_COLUMNS = ["trend_4h", "adx_4h", "adx_slope_4h", "slope_atr_4h", "trend_age_4h"]


def merge_htf_causal(df: pd.DataFrame, df_htf: pd.DataFrame, timeframe: str, cols: List[str]) -> pd.DataFrame:
    """merge_asof со сдвигом на длину бина: значение доступно только после закрытия."""
    shifted = df_htf.copy()
    shifted["time"] = shifted["time"] + pd.Timedelta(timeframe)
    return pd.merge_asof(
        df.sort_values("time"),
        shifted[["time"] + cols].sort_values("time"),
        on="time",
        direction="backward",
    )


def hold_direction(trend: pd.Series, hold_bars: int) -> pd.Series:
    """Нейтраль не длиннее hold_bars баров подряд не сбрасывает направление.

    Противоположный знак срабатывает сразу, NaN (до первого закрытого бина)
    остаётся NaN. Решение на баре i зависит только от баров <= i.
    """
    if hold_bars <= 0:
        return trend
    values = trend.to_numpy(dtype=float)
    out = values.copy()
    current, gap = 0.0, 0
    for i, v in enumerate(values):
        if np.isnan(v):
            current, gap = 0.0, 0
        elif v != 0:
            current, gap = v, 0
        elif current != 0:
            gap += 1
            if gap <= hold_bars:
                out[i] = current
            else:
                current = 0.0
    return pd.Series(out, index=trend.index, name=trend.name)


def _trend_age(trend: pd.Series) -> pd.Series:
    return trend.groupby((trend != trend.shift()).cumsum()).cumcount().astype(float)


def _htf_grids(df: pd.DataFrame, cfg: TrendConfig) -> List[pd.DataFrame]:
    """Бины старшего ТФ: одна сетка или по сетке на каждый сдвиг в бар рабочего ТФ."""
    if not cfg.update_every_bar:
        return [resample_ohlcv(df, cfg.timeframe)]
    diffs = df["time"].diff()
    step = diffs[diffs > pd.Timedelta(0)].min()  # рабочий ТФ: разрывы (ночь, выходные) только длиннее
    n = int(pd.Timedelta(cfg.timeframe) / step)
    if n < 2:
        return [resample_ohlcv(df, cfg.timeframe)]
    return [resample_ohlcv(df, cfg.timeframe, offset=i * step) for i in range(n)]


def _apply_on_grids(
    df: pd.DataFrame, cfg: TrendConfig, compute: Callable[[pd.DataFrame], pd.DataFrame], cols: List[str]
) -> pd.DataFrame:
    """Считает compute на каждой сетке и мёрджит самое свежее ЗАКРЫТОЕ окно."""
    frames = [compute(htf)[["time"] + cols] for htf in _htf_grids(df, cfg)]
    out = merge_htf_causal(df, pd.concat(frames, ignore_index=True), cfg.timeframe, cols)
    if cfg.hold_bars:
        out["trend_4h"] = hold_direction(out["trend_4h"], cfg.hold_bars)
    return out


def add_trend_filter(df: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    """Диспетчер: ставит колонки TREND_COLUMNS согласно cfg.mode."""
    if not cfg.enabled:
        out = df.copy()
        for c in TREND_COLUMNS:
            out[c] = np.nan
        out["trend_4h"] = 0.0
        return out

    out = add_trend_filter_early(df, cfg) if cfg.mode == "early" else add_trend_filter_confirm(df, cfg)
    if cfg.shock_atr > 0:
        out["trend_4h"] = _apply_shock(out, cfg)
    if cfg.ema_reset_period > 0:
        ema = reset_ema(out, cfg)
        side = np.sign(out["close"].to_numpy(dtype=float) - ema)
        out["trend_4h"] = ema_reset(out["trend_4h"].to_numpy(dtype=float), side, cfg.ema_reset_confirm_bars)
    return out


def reset_ema(df: pd.DataFrame, cfg: TrendConfig) -> np.ndarray:
    """EMA для сброса тренда — на каждой свече рабочего ТФ.

    base: EMA закрытий рабочего ТФ (включая текущую свечу — её close известен на закрытии);
    htf:  EMA закрытий старшего ТФ по последнему ЗАКРЫТОМУ бину (при update_every_bar —
          самое свежее окно из всех сеток), как линия ema_4h на графике ноутбука.
    """
    if cfg.ema_reset_tf == "base":
        return ind.ema(df["close"].astype(float), cfg.ema_reset_period).to_numpy(dtype=float)
    frames = []
    for htf in _htf_grids(df[["time", "open", "high", "low", "close"]], cfg):
        frames.append(pd.DataFrame({"time": htf["time"], "_ema": ind.ema(htf["close"], cfg.ema_reset_period)}))
    merged = merge_htf_causal(df[["time"]], pd.concat(frames, ignore_index=True), cfg.timeframe, ["_ema"])
    return merged["_ema"].to_numpy(dtype=float)


def ema_reset(trend: np.ndarray, side: np.ndarray, confirm_bars: int = 1) -> np.ndarray:
    """Сброс тренда по стороне EMA: +1 при закрытии ниже EMA и −1 при закрытии выше → 0.

    side — знак (close − EMA) на каждой свече (NaN — EMA ещё нет, сброса нет). После сброса
    направление возвращается, когда confirm_bars свечей подряд закрылись на «своей» стороне
    (1 — сразу же на первой). Новый эпизод тренда (смена знака или выход из нейтрали)
    начинается без памяти о сбросах прошлого. Решение на свече i зависит только от свечей <= i.
    """
    out = trend.astype(float).copy()
    cur, latched, run = 0.0, False, 0
    for i in range(len(trend)):
        t = trend[i]
        if np.isnan(t) or t == 0:
            cur, latched, run = 0.0, False, 0
            continue
        if t != cur:
            cur, latched, run = t, False, 0
        s = side[i]
        if s == -t:
            out[i], latched, run = 0.0, True, 0
        elif latched:
            run += 1
            if run >= confirm_bars:
                latched = False
            else:
                out[i] = 0.0
    return out


def shock_direction(df: pd.DataFrame, window: int, atr_mult: float, confirm_bars: int = 1) -> np.ndarray:
    """Направление резкого хода: |close − close[−window]| >= atr_mult · ATR(14) и последние
    confirm_bars баров в ту же сторону. В ноутбуке направление бралось по последним барам,
    даже если сам ход был в другую сторону; здесь они обязаны совпадать."""
    close = df["close"].astype(float)
    atr = ind.atr(df, 14).replace(0, np.nan)
    move = (close - close.shift(window)) / atr
    move_dir = np.sign(move)
    bar_dir = np.sign(close.diff())
    run = bar_dir.rolling(confirm_bars).sum()                   # ±confirm_bars — все бары в одну сторону
    same = np.where(run == confirm_bars, 1.0, np.where(run == -confirm_bars, -1.0, 0.0))
    shock = np.where((move.abs() >= atr_mult) & (same == move_dir), move_dir, 0.0)
    return np.nan_to_num(shock)


def _apply_shock(out: pd.DataFrame, cfg: TrendConfig) -> np.ndarray:
    trend = out["trend_4h"].to_numpy(dtype=float).copy()
    shock = shock_direction(out, cfg.shock_window, cfg.shock_atr, cfg.shock_confirm_bars)
    decided = ~np.isnan(trend)
    conflict = decided & (shock != 0) & (trend != 0) & (np.sign(trend) != shock)
    neutral = decided & (shock != 0) & (trend == 0)
    trend[conflict] = shock[conflict] if cfg.shock_mode == "flip" else 0.0
    trend[neutral] = shock[neutral]
    return trend


def _hysteresis(close: np.ndarray, ema: np.ndarray, adx_dir: np.ndarray, slope_dir) -> np.ndarray:
    """Гистерезис add_trend_filter_v3 из ноутбука, на барах старшего ТФ.

    Пока тренд ±1, закрытие по другую сторону EMA без подтверждения (ADX+DI, при
    require_slope_agreement — и наклон, в ту же сторону, что EMA) — нейтраль, а не разворот;
    из нейтрали — только при подтверждении; с подтверждением — разворот сразу.
    """
    out = np.zeros(len(close))
    state = 0.0
    for i in range(len(close)):
        if np.isnan(close[i]) or np.isnan(ema[i]):
            out[i] = state
            continue
        side = 1.0 if close[i] > ema[i] else (-1.0 if close[i] < ema[i] else 0.0)
        ok = side != 0 and adx_dir[i] == side
        if slope_dir is not None:
            ok = ok and slope_dir[i] == side
        if state == 0:
            if ok:
                state = side
        elif side != state:
            state = side if ok else 0.0
        out[i] = state
    return out


def trend_gate_values(df: pd.DataFrame, cfg: TrendConfig) -> np.ndarray:
    """Направление тренда по отдельным настройкам — для колонки `trend_gate`.

    Считается на тех же барах, что и признаки, но не трогает колонки
    TREND_COLUMNS: они остаются признаками модели в том виде, в котором
    модель на них училась.
    """
    base = df[["time", "open", "high", "low", "close"]].copy()
    base["time"] = pd.to_datetime(base["time"])
    out = add_trend_filter(base, cfg)
    if not base["time"].reset_index(drop=True).equals(out["time"].reset_index(drop=True)):
        raise ValueError("trend_gate: ряд должен быть отсортирован по времени без дублей")
    return out["trend_4h"].to_numpy(dtype=float)


def add_trend_filter_early(df: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    """Опережающий тренд-фильтр.

    Тренд активен, если:
      * наклон регрессии за slope_period бинов (в ATR) задаёт направление,
      * это направление подтверждено знаком DI,
      * ADX в коридоре (adx_min, adx_max) и (опц.) растёт.

    `trend_age_4h` — сколько бинов тренд уже активен — отдаётся модели как
    признак: это ещё один способ отличить разгон от излёта.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    def compute(htf: pd.DataFrame) -> pd.DataFrame:
        atr = ind.atr(htf, cfg.adx_period, method="wilder").replace(0, np.nan)
        htf["slope_atr_4h"] = ind.linreg_slope(htf["close"], cfg.slope_period) / atr

        adx_df = ind.adx(htf, cfg.adx_period)
        htf["adx_4h"] = adx_df["adx"]
        htf["adx_slope_4h"] = adx_df["adx"].diff()
        di_dir = np.sign(adx_df["dmp"] - adx_df["dmn"])

        ok = (htf["adx_4h"] > cfg.adx_min) & (htf["adx_4h"] < cfg.adx_max)
        if cfg.require_adx_rising:
            ok &= htf["adx_slope_4h"] > 0

        direction = np.sign(htf["slope_atr_4h"])
        htf["trend_4h"] = np.where(ok & (direction == di_dir), direction, 0.0).astype(float)
        htf["trend_age_4h"] = _trend_age(htf["trend_4h"])
        return htf

    out = _apply_on_grids(df, cfg, compute, TREND_COLUMNS)
    log.info("Тренд-фильтр (early, %s): доля баров с трендом %.1f%%", cfg.timeframe, 100 * (out["trend_4h"] != 0).mean())
    return out


def add_trend_filter_confirm(df: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    """Подтверждающий фильтр (переработанная `add_trend_filter` из ноутбука).

    Сохранены оба исправления оригинала: ограниченный ffill (`max_ffill_bars`,
    после лимита — явный NEUTRAL, а не унаследованное чужое направление) и
    согласование EMA-направления со знаком наклона регрессии.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    def compute(htf: pd.DataFrame) -> pd.DataFrame:
        htf["ema"] = ind.ema(htf["close"], cfg.ema_period)
        trend_ema = np.where(htf["close"] > htf["ema"], 1.0, -1.0)

        adx_df = ind.adx(htf, cfg.adx_period)
        htf["adx_4h"] = adx_df["adx"]
        htf["adx_slope_4h"] = adx_df["adx"].diff()

        strong = htf["adx_4h"] > cfg.adx_threshold
        di_dir = np.sign(adx_df["dmp"] - adx_df["dmn"])
        trend_adx = np.where(strong, di_dir, 0.0)

        atr = ind.atr(htf, cfg.adx_period, method="wilder").replace(0, np.nan)
        htf["slope_atr_4h"] = ind.linreg_slope(htf["close"], cfg.slope_period) / atr

        if cfg.hysteresis:
            slope_dir = np.sign(htf["slope_atr_4h"].to_numpy()) if cfg.require_slope_agreement else None
            htf["trend_4h"] = _hysteresis(htf["close"].to_numpy(dtype=float), htf["ema"].to_numpy(dtype=float),
                                          trend_adx, slope_dir)
        else:
            raw = pd.Series(trend_ema * (trend_adx != 0), index=htf.index, dtype=float)
            if cfg.require_slope_agreement:
                raw = raw * (np.sign(raw) == np.sign(htf["slope_atr_4h"]))
            raw = raw.replace(0, np.nan)
            if cfg.max_ffill_bars > 0:
                raw = raw.ffill(limit=cfg.max_ffill_bars)
            htf["trend_4h"] = raw.fillna(0.0)
        htf["trend_age_4h"] = _trend_age(htf["trend_4h"])
        return htf

    out = _apply_on_grids(df, cfg, compute, TREND_COLUMNS + ["ema"])
    out = out.rename(columns={"ema": "ema_4h"})

    atr_cols = [c for c in out.columns if c.startswith("atr_")]
    if atr_cols:
        atr_main = out[atr_cols[0]].replace(0, np.nan)
        out["pullback_atr"] = (out["close"] - out["ema_4h"]) / atr_main
        out["pullback_filter"] = np.where(
            out["trend_4h"] == 1,
            out["pullback_atr"] > -cfg.pullback_atr_threshold,
            out["pullback_atr"] < cfg.pullback_atr_threshold,
        ).astype(float)

    out["trend_strength"] = out["adx_4h"] * out["trend_4h"].abs()
    log.info(
        "Тренд-фильтр (confirm, %s): доля баров с трендом %.1f%%", cfg.timeframe, 100 * (out["trend_4h"] != 0).mean()
    )
    return out
