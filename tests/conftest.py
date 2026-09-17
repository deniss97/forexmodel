"""Синтетические данные для тестов (реальные котировки в репозиторий не кладём)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import Config


@pytest.fixture
def minute_df() -> pd.DataFrame:
    """60 дней минутных баров со случайным блужданием и заметным трендом."""
    rng = np.random.default_rng(42)
    n = 60 * 24 * 60
    times = pd.date_range("2024-01-01", periods=n, freq="1min")

    steps = rng.normal(0, 0.5, n) + np.sin(np.arange(n) / 5000) * 0.3
    close = 1000 + np.cumsum(steps)
    high = close + rng.uniform(0, 1.0, n)
    low = close - rng.uniform(0, 1.0, n)
    open_ = np.r_[close[0], close[:-1]]

    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(100, 10_000, n).astype(float),
        }
    )


@pytest.fixture
def hourly_df(minute_df: pd.DataFrame) -> pd.DataFrame:
    from forexmodel.data.loader import resample_ohlcv

    return resample_ohlcv(minute_df, "1h")


@pytest.fixture
def cfg() -> Config:
    """Конфиг под синтетику: короткие окна, выключенные тяжёлые модели."""
    from forexmodel.config import config_from_dict

    return config_from_dict(
        {
            "data": {
                "csv_path": "unused.csv",
                "splits": {
                    "train_end": "2024-02-01 00:00:00",
                    "test_start": "2024-02-01 00:00:00",
                    "test_end": "2024-02-15 00:00:00",
                    "sim_start": "2024-02-15 00:00:00",
                },
            },
            "features": {"rsi_z_window": 50, "extension_windows": [6, 12, 24]},
            "labeling": {"mode": "atr_asym", "horizon": 5, "tp_atr": 1.5, "sl_atr": 0.75},
            "nn": {"enabled": False},
            "meta": {"enabled": False},
            "simulation": {"signal_source": "cb", "exit_mode": "fixed_pct", "use_expected_value_filter": False},
        }
    )
