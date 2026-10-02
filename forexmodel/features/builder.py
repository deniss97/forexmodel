"""Сборка всех признаков в одном месте.

Важно: признаки считаются НА НЕПРЕРЫВНОМ ряде и только потом режутся на
train/test/sim. В ноутбуке каждая выборка обрабатывалась отдельно, поэтому
первые ~50-100 баров sim уходили в модель с NaN/warm-up значениями
(rolling(50), rolling(100), EMA-разогрев) — на коротком sim-периоде это
заметная доля данных.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .extension import add_extension_features, add_regime_features
from .technical import add_candle_features, add_technical_indicators
from .trend import add_trend_filter, trend_gate_values

log = get_logger(__name__)

__all__ = ["build_features"]


EMA_ZONE_COLUMNS = ["ema5_1h_dist_atr", "ema5_4h_dist_atr", "trend_ema5_1h_side", "trend_ema5_4h_side"]


def add_ema_zone_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Слабая зона тренда по EMA5 — признаки вместо жёсткого сброса тренда (см. config.use_ema_zone_features).

    EMA5 4ч — закрытия старшего ТФ по последнему закрытому бину (как ema_reset_tf=htf гейта), EMA5 часа
    включает текущую свечу (её close известен на закрытии). Только прошлое.
    """
    import dataclasses

    import numpy as np

    from .trend import reset_ema

    tc = cfg.trend_gate.resolve(cfg.trend) if cfg.trend_gate.enabled else cfg.trend
    close = df["close"].to_numpy(dtype=float)
    atr = df[cfg.atr_col].replace(0, np.nan).to_numpy(dtype=float)
    trend = (df["trend_gate"] if "trend_gate" in df.columns else df["trend_4h"]).fillna(0).to_numpy(dtype=float)
    ema1 = reset_ema(df, dataclasses.replace(tc, ema_reset_period=5, ema_reset_tf="base"))
    ema4 = reset_ema(df, dataclasses.replace(tc, ema_reset_period=5, ema_reset_tf="htf"))
    df = df.copy()
    df["ema5_1h_dist_atr"] = (close - ema1) / atr
    df["ema5_4h_dist_atr"] = (close - ema4) / atr
    df["trend_ema5_1h_side"] = trend * np.sign(close - ema1)
    df["trend_ema5_4h_side"] = trend * np.nan_to_num(np.sign(close - ema4))
    log.info("Слабая зона по EMA5 4ч: %.1f%% баров с трендом", 100 * (df["trend_ema5_4h_side"] < 0).sum()
             / max(1, (trend != 0).sum()))
    return df


def build_features(df_tf: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Полный набор признаков для рабочего ТФ (непрерывный ряд целиком)."""
    df = df_tf.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df = add_candle_features(df)
    df = add_technical_indicators(df, cfg.features)

    if cfg.features.use_extension_features:
        df = add_extension_features(df, cfg.features, atr_col=cfg.atr_col)
    if cfg.features.use_regime_features or (cfg.meta.enabled and cfg.meta.context_features):
        # нужны непрерывному ряду; в primary попадают только при use_regime_features
        df = add_regime_features(df, cfg.features)

    if cfg.features.use_orderflow_features:
        if not cfg.data.orderflow_path:
            raise ValueError("features.use_orderflow_features=true, но data.orderflow_path не задан")
        from .orderflow import add_orderflow_features, aggregate_orderflow, load_orderflow

        of = load_orderflow(cfg.data.orderflow_path, cfg.data.orderflow_tz_shift_hours, cfg.data.orderflow_tz)
        of_bars = aggregate_orderflow(of, cfg.features, cfg.data.base_timeframe)
        del of  # ~700 МБ секундных строк, дальше не нужны
        df = add_orderflow_features(df, of_bars, cfg.features, atr_col=cfg.atr_col)

    df = add_trend_filter(df, cfg.trend)
    if cfg.trend_gate.enabled:
        # гейт входа — отдельно от признака trend_4h, см. TrendGateConfig
        df["trend_gate"] = trend_gate_values(df, cfg.trend_gate.resolve(cfg.trend))
        log.info("Гейт входа trend_gate: доля баров с трендом %.1f%%", 100 * (df["trend_gate"] != 0).mean())
    if cfg.features.use_ema_zone_features:
        df = add_ema_zone_features(df, cfg)
    if cfg.trend_gate.enabled:
        for name in cfg.trend_gate.variants:
            col = f"trend_gate__{name}"
            df[col] = trend_gate_values(df, cfg.trend_gate.resolve_variant(cfg.trend, name))
            log.info("Вариант гейта %s: доля баров с трендом %.1f%%", col, 100 * (df[col] != 0).mean())

    log.info("Признаки построены: %d баров, %d колонок", len(df), df.shape[1])
    return df
