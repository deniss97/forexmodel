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
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import TrendConfig
from ..data.loader import resample_ohlcv
from ..logging_utils import get_logger
from . import indicators as ind

log = get_logger(__name__)

__all__ = ["merge_htf_causal", "add_trend_filter", "add_trend_filter_early", "add_trend_filter_confirm", "TREND_COLUMNS"]

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


def add_trend_filter(df: pd.DataFrame, cfg: TrendConfig) -> pd.DataFrame:
    """Диспетчер: ставит колонки TREND_COLUMNS согласно cfg.mode."""
    if not cfg.enabled:
        out = df.copy()
        for c in TREND_COLUMNS:
            out[c] = np.nan
        out["trend_4h"] = 0.0
        return out

    if cfg.mode == "early":
        return add_trend_filter_early(df, cfg)
    return add_trend_filter_confirm(df, cfg)


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
    htf = resample_ohlcv(df, cfg.timeframe)

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
    htf["trend_age_4h"] = (
        htf["trend_4h"].groupby((htf["trend_4h"] != htf["trend_4h"].shift()).cumsum()).cumcount().astype(float)
    )

    out = merge_htf_causal(df, htf, cfg.timeframe, TREND_COLUMNS)
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
    htf = resample_ohlcv(df, cfg.timeframe)

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

    raw = pd.Series(trend_ema * (trend_adx != 0), index=htf.index, dtype=float)
    if cfg.require_slope_agreement:
        raw = raw * (np.sign(raw) == np.sign(htf["slope_atr_4h"]))

    raw = raw.replace(0, np.nan)
    htf["trend_4h"] = raw.ffill(limit=cfg.max_ffill_bars).fillna(0.0)
    htf["trend_age_4h"] = (
        htf["trend_4h"].groupby((htf["trend_4h"] != htf["trend_4h"].shift()).cumsum()).cumcount().astype(float)
    )

    out = merge_htf_causal(df, htf, cfg.timeframe, TREND_COLUMNS + ["ema"])
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
