"""Направленная разметка и режимные признаки: корректность и каузальность."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from forexmodel.config import config_from_dict
from forexmodel.features.extension import add_regime_features, regime_columns
from forexmodel.features.technical import add_candle_features, add_technical_indicators
from forexmodel.labeling import attach_labels
from forexmodel.labeling.direction import generate_labels_direction


def _featured(hourly_df, cfg):
    return add_technical_indicators(add_candle_features(hourly_df), cfg.features)


def test_direction_label_matches_manual_formula(hourly_df, cfg):
    df = _featured(hourly_df, cfg)
    lc = dataclasses.replace(cfg.labeling, mode="direction", horizon=5, dir_atr=0.5)
    out = generate_labels_direction(df, lc)

    i = 300
    move = (df["close"].iloc[i + 6] - df["open"].iloc[i + 1]) / df["atr_14"].iloc[i]
    expected = 2 if move >= 0.5 else (0 if move <= -0.5 else 1)
    assert out["label"].iloc[i] == expected
    assert out["label_t1"].iloc[i] == i + 6
    # последние horizon+1 баров без метки: горизонт не помещается в данные
    assert out["label"].iloc[-6:].isna().all()
    assert out["label"].dropna().isin([0, 1, 2]).all()


def test_direction_label_is_causal_given_future_bars_only(hourly_df, cfg):
    """Метка бара i зависит только от баров i..i+1+horizon: обрезка ряда дальше
    этого горизонта её не меняет."""
    df = _featured(hourly_df, cfg)
    lc = dataclasses.replace(cfg.labeling, mode="direction", horizon=5)
    full = generate_labels_direction(df, lc)
    cut = generate_labels_direction(df.iloc[:400].copy(), lc)
    a, b = full["label"].iloc[:394].to_numpy(), cut["label"].iloc[:394].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_attach_labels_direction_needs_no_minute_data(hourly_df, cfg):
    dcfg = config_from_dict({**cfg.to_dict(), "labeling": {**cfg.to_dict()["labeling"], "mode": "direction", "horizon": 5}})
    df = _featured(hourly_df, dcfg)
    out = attach_labels(df, pd.DataFrame(), dcfg)
    assert {"label", "label_t1", "sample_weight"} <= set(out.columns)
    assert out["label"].isin([0, 1, 2]).all()


def test_regime_features_are_causal_and_bounded(hourly_df, cfg):
    fc = dataclasses.replace(cfg.features, use_regime_features=True, regime_windows=[48, 120])
    df = _featured(hourly_df, cfg)
    full = add_regime_features(df.copy(), fc)
    prefix = add_regime_features(df.iloc[:400].copy(), fc)
    for col in regime_columns(fc):
        assert col in full.columns
        a, b = full[col].iloc[:400].to_numpy(float), prefix[col].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), f"{col} зависит от будущего"
    assert full["acf1_120"].dropna().between(-1, 1).all()
    assert (full["vr4_120"].dropna() > 0).all()
    # синтетика в conftest несёт синусоидальный тренд, поэтому там VR заметно больше 1;
    # на чистом случайном блуждании VR должен быть около 1
    rng = np.random.default_rng(1)
    walk = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3000, freq="1h"), "close": 100 + np.cumsum(rng.normal(0, 0.5, 3000))})
    vr = add_regime_features(walk, dataclasses.replace(fc, regime_windows=[720]))["vr4_720"].dropna()
    assert 0.7 < vr.median() < 1.3
    assert full["vr4_120"].dropna().median() > 1.3  # тренд в синтетике распознан как накопление


def test_regime_features_off_by_default(hourly_df, cfg):
    assert regime_columns(cfg.features) == []
