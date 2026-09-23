"""Тесты часового пересчёта тренд-фильтра, удержания направления и гейта входа."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.features.builder import build_features
from forexmodel.features.selection import select_feature_columns
from forexmodel.features.trend import add_trend_filter_confirm, add_trend_filter_early, hold_direction

PREFIX = 400


def _hourly_cfg(cfg, **kw):
    return dataclasses.replace(cfg.trend, update_every_bar=True, **kw)


@pytest.mark.parametrize("builder", [add_trend_filter_early, add_trend_filter_confirm])
@pytest.mark.parametrize("hold", [0, 2])
def test_hourly_trend_is_causal(hourly_df, cfg, builder, hold):
    tcfg = _hourly_cfg(cfg, hold_bars=hold, require_adx_rising=False)
    full = builder(hourly_df, tcfg)
    prefix = builder(hourly_df.iloc[:PREFIX].copy(), tcfg)
    for col in ("trend_4h", "adx_4h", "slope_atr_4h"):
        a = full[col].iloc[:PREFIX].to_numpy(dtype=float)
        b = prefix[col].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"{builder.__name__}: {col} зависит от будущих баров"


def test_hourly_trend_matches_default_grid_at_bin_close(hourly_df, cfg):
    """В 00:00, 04:00, ... только что закрылось окно основной сетки — значения
    обязаны совпасть с обычным фильтром; в остальные часы значение обновляется."""
    default = add_trend_filter_early(hourly_df, cfg.trend)
    hourly = add_trend_filter_early(hourly_df, _hourly_cfg(cfg))

    at_close = hourly["time"].dt.hour % 4 == 0
    for col in ("trend_4h", "adx_4h", "slope_atr_4h"):
        assert np.allclose(
            default.loc[at_close, col].to_numpy(dtype=float),
            hourly.loc[at_close, col].to_numpy(dtype=float),
            equal_nan=True,
        ), col

    moved = hourly["adx_4h"].ne(hourly["adx_4h"].shift()) & hourly["adx_4h"].notna()
    assert (moved & ~at_close).any(), "при update_every_bar значение должно меняться и внутри 4H-бина"


def test_hold_direction_bridges_short_neutral_only():
    s = pd.Series([np.nan, 1, 0, 0, 1, 0, 0, 0, -1, 0, 1], dtype=float)
    out = hold_direction(s, 2)
    assert np.isnan(out.iloc[0])
    # две нейтрали подряд держатся, третья — уже нет; противоположный знак сразу
    assert out.tolist()[1:] == [1, 1, 1, 1, 1, 1, 0, -1, -1, 1]
    assert hold_direction(s, 0).equals(s)


def test_hold_direction_is_causal():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.choice([-1.0, 0.0, 0.0, 1.0], 300))
    full = hold_direction(s, 3)
    assert full.iloc[:150].equals(hold_direction(s.iloc[:150], 3))


def _gate_cfg(**gate):
    raw = {
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
        "trend_gate": {"enabled": True, **gate},
        "simulation": {"signal_source": "cb", "use_expected_value_filter": False, "trend_col": "trend_gate"},
    }
    return config_from_dict(raw)


def test_gate_inherits_unset_parameters_from_trend():
    cfg = _gate_cfg(require_adx_rising=False, update_every_bar=True, hold_bars=2)
    resolved = cfg.trend_gate.resolve(cfg.trend)
    assert resolved.require_adx_rising is False and resolved.update_every_bar and resolved.hold_bars == 2
    assert resolved.adx_max == cfg.trend.adx_max and resolved.slope_period == cfg.trend.slope_period
    assert cfg.trend.require_adx_rising is True  # признак модели не тронут


def test_gate_column_requires_enabled_gate():
    with pytest.raises(ValueError, match="trend_gate"):
        config_from_dict({"simulation": {"trend_col": "trend_gate"}, "meta": {"enabled": False}})


def test_gate_leaves_model_features_untouched(hourly_df, cfg):
    gate_cfg = _gate_cfg(require_adx_rising=False, update_every_bar=True, hold_bars=2)
    without = build_features(hourly_df, cfg)
    with_gate = build_features(hourly_df, gate_cfg)

    assert "trend_gate" in with_gate.columns and "trend_gate" not in without.columns
    for col in ("trend_4h", "adx_4h", "trend_age_4h"):
        assert np.allclose(without[col], with_gate[col], equal_nan=True), col
    assert "trend_gate" not in select_feature_columns(with_gate, gate_cfg.features)
