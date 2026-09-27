"""Эксперимент TokDiT: признаки и окна не видят будущего (scripts/tokdit_direction.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tokdit_direction as td  # noqa: E402


def _hourly(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"time": pd.date_range("2023-12-01", periods=n, freq="1h"),
                         "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def test_windows_are_zero_at_forecast_time_and_end_at_horizon_return():
    df = _hourly()
    L, H = 24, 5
    paths, t_pred, t_end = td.real_windows(df, L, H)
    assert np.allclose(paths[:, L - 1], 0.0)
    i = 10
    ret = 100 * np.log(df["close"].iloc[i + L - 1 + H] / df["close"].iloc[i + L - 1])
    assert paths[i, -1] == np.float32(ret)
    assert t_end[i] - t_pred[i] == np.timedelta64(H, "h")


def test_tabular_features_ignore_horizon():
    df = _hourly()
    L, H = 24, 5
    paths, _, _ = td.real_windows(df, L, H)
    changed = paths.copy()
    changed[:, L:] = 1e3                       # будущее меняется произвольно
    assert np.array_equal(td.tabular(paths, L), td.tabular(changed, L))


def test_trade_does_not_overlap_positions():
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.5, 0.9])
    ret = np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    out = td.trade(p, ret, H=3, margin=0.0)
    # входы на 0 и 3 (1, 2 заняты позицией; 4 — p = 0.5 не проходит порог); 5 занят
    assert out.tolist() == [1.0 - td.COMMISSION, 1.0 - td.COMMISSION]
