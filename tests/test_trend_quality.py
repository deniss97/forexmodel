"""Метрики качества тренд-фильтра (forexmodel/evaluation/trend_quality.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.evaluation.trend_quality import follow_pnl, score_trend_filter, trend_quality, zigzag_legs


def _bars(close, start="2019-01-01"):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"time": pd.date_range(start, periods=len(close), freq="1h"), "open": close,
                         "high": close + 0.5, "low": close - 0.5, "close": close})


def test_zigzag_finds_up_and_down_legs():
    close = np.r_[np.linspace(100, 130, 61), np.linspace(130, 100, 61)[1:], np.linspace(100, 125, 51)[1:]]
    atr = np.ones(len(close))
    legs = zigzag_legs(close, atr, k=6)
    assert legs["dir"][30] == 1 and legs["dir"][90] == -1 and legs["dir"][150] == 1
    assert legs["size_atr"][30] == pytest.approx(30, rel=0.05)


def test_follow_pnl_charges_commission_per_change():
    close = np.array([100.0, 101.0, 102.0, 101.0])
    atr = np.ones(4)
    gross, net = follow_pnl(close, atr, np.array([1.0, 1.0, -1.0, 0.0]), commission_pct=0.04)
    assert gross[:3].tolist() == [1.0, 1.0, 1.0]           # +1·(+1), +1·(+1), −1·(−1)
    # вход (1 единица), разворот (2 единицы), выход (1 единица): половина круга 0.02% цены за единицу
    assert net[0] == pytest.approx(1 - 0.0002 * 100) and net[2] == pytest.approx(1 - 2 * 0.0002 * 102)


def test_perfect_trend_label_beats_always_long_and_neutral():
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(np.r_[np.full(300, 0.3), np.full(300, -0.3), np.full(300, 0.3)] + rng.normal(0, 0.2, 900))
    df = _bars(close)
    legs = zigzag_legs(close, np.full(900, 0.5), k=6)
    perfect = np.nan_to_num(legs["dir"])
    q_perfect = trend_quality(df, perfect, legs=legs)
    q_long = trend_quality(df, np.ones(900), legs=legs)
    assert q_perfect["следование"] > q_long["следование"] > trend_quality(df, np.zeros(900), legs=legs)["следование"] - 1
    assert q_perfect["в_крупных_против"] == 0 and q_long["в_крупных_против"] > 0.2


def test_score_trend_filter_splits_by_year_and_checks_length():
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 1, 24 * 365 * 3))
    frames = {"x": _bars(close, "2019-01-01")}
    res = score_trend_filter(frames, lambda df: np.sign(df["close"] - df["close"].rolling(50).mean()), split_year=2021)
    assert {"sel_median", "val_median", "sel_pos", "val_в_крупных_против"} <= set(res)
    with pytest.raises(ValueError, match="на каждый бар"):
        score_trend_filter(frames, lambda df: np.zeros(10))
