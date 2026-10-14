"""Тесты фильтров из ТЗ 15.10: трёхслойный скор (H44) и классические индикаторы (H45)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.features.trend import add_trend_filter
from forexmodel.features.trend_score import classic_direction, hysteresis_state, trend_score_components

PREFIX = 700

RULES = ["ema_price:21", "ema_slope:50:5", "ema_cross:12:26", "macd:12:26:9:sig", "macd:5:35:5:hist",
         "supertrend:10:3", "donchian:20:10", "keltner:20:10:2", "bb:20:2", "adx:14:20", "kama:10"]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("tf", ["1h", "4h", "24h"])
def test_classic_is_causal(hourly_df, cfg, rule, tf):
    tc = dataclasses.replace(cfg.trend, mode="classic", rule=rule, timeframe=tf, update_every_bar=True, hold_bars=0)
    full = add_trend_filter(hourly_df, tc)["trend_4h"].to_numpy()
    prefix = add_trend_filter(hourly_df.iloc[:PREFIX].copy(), tc)["trend_4h"].to_numpy()
    assert np.array_equal(full[:PREFIX], prefix), f"{rule} на {tf} зависит от будущих баров"
    assert set(np.unique(full)) <= {-1.0, 0.0, 1.0}
    if tf != "24h":                       # на 60 днях синтетики дневные каналы могут не пробиться ни разу
        assert (full != 0).mean() > 0.2


@pytest.mark.parametrize("extra", [{}, {"score_regime_tf": "4h"}, {"score_fast": "kama"},
                                   {"score_vol_adapt": True, "score_min_hold": 6, "score_shock_donchian": 24}])
def test_score_is_causal(hourly_df, cfg, extra):
    tc = dataclasses.replace(cfg.trend, mode="score", score_regime_ema=50, **extra)
    full = add_trend_filter(hourly_df, tc)["trend_4h"].to_numpy()
    prefix = add_trend_filter(hourly_df.iloc[:PREFIX].copy(), tc)["trend_4h"].to_numpy()
    assert np.array_equal(full[:PREFIX], prefix)
    comp = trend_score_components(hourly_df, tc)
    assert comp["ts_score"].abs().max() <= 1.0 + 1e-9


def test_hysteresis_enter_exit_and_hold():
    s = np.array([0.1, 0.35, 0.2, 0.05, -0.01, -0.2, -0.4, -0.1, 0.0])
    z = np.zeros(len(s))
    # вход > 0.3, держится, пока скор >= 0; при < 0 — нейтраль; < −0.3 — вниз
    assert hysteresis_state(s, z, 0.3, 0.0, 0).tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1]
    # минимальное удержание 5 баров: выход на баре 4 откладывается до бара 6 (−0.4 → сразу разворот)
    assert hysteresis_state(s, z, 0.3, 0.0, 5).tolist() == [0, 1, 1, 1, 1, 1, -1, -1, -1]
    # встречный шок разворачивает без ожидания удержания
    z2 = z.copy()
    z2[3] = -1
    assert hysteresis_state(s, z2, 0.3, 0.0, 5).tolist()[:4] == [0, 1, 1, -1]


def test_donchian_breakout_excludes_current_bar():
    import pandas as pd

    b = pd.DataFrame({"high": [10, 11, 12, 13, 14.0], "low": [9, 10, 11, 12, 13.0], "close": [9.5, 10.5, 11.5, 13.5, 13.2]})
    b["open"] = b["close"]
    # close 13.5 > максимума трёх прошлых баров (12) — пробой; на следующем баре держится до выхода
    assert classic_direction(b, "donchian:3:2").tolist() == [0, 0, 0, 1, 1]


def test_gate_variant_config_accepts_new_modes():
    raw = {"data": {"csv_path": "x.csv"}}
    raw["trend_gate"] = {"enabled": True, "variants": {
        "s": {"mode": "score", "score_min_hold": 4},
        "ec": {"mode": "classic", "rule": "ema_cross:50:200", "timeframe": "24h", "hold_bars": 0}}}
    cfg = config_from_dict(raw)
    assert cfg.trend_gate.resolve_variant(cfg.trend, "ec").rule == "ema_cross:50:200"
    raw["trend_gate"]["variants"]["bad"] = {"mode": "classic", "rule": "foo:1"}
    with pytest.raises(ValueError):
        config_from_dict(raw)


@pytest.mark.parametrize("spec", ["nr7", "nr4", "ib", "nr7ib", "donch:1", "donch:20:0.25", "soup:5"])
def test_pattern_signal_is_causal(hourly_df, spec):
    from forexmodel.features import indicators as ind
    from forexmodel.simulation.patterns import pattern_signal

    h = hourly_df.copy()
    h["atr_14"] = ind.atr(h, 14)
    full = pattern_signal(h, spec, "atr_14").to_numpy()
    cut = 1000
    prefix = pattern_signal(h.iloc[:cut].copy(), spec, "atr_14").to_numpy()
    assert np.array_equal(full[:cut], prefix, equal_nan=True), f"{spec} зависит от будущих баров"
    days = h["time"].dt.floor("D")
    sig = pd.Series(full).notna()
    assert sig.groupby(days.to_numpy()).sum().max() <= 2          # не больше одного сигнала на сторону в день


def test_donchian_day_breakout_uses_previous_days_only():
    from forexmodel.simulation.patterns import pattern_signal

    t = pd.date_range("2024-01-01", periods=72, freq="h")
    close = np.r_[np.full(24, 100.0), np.full(24, 101.0), np.full(10, 101.5), [103.0], np.full(13, 102.0)]
    h = pd.DataFrame({"time": t, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close})
    s = pattern_signal(h, "donch:1", "atr_14")
    # день 2: вчерашний максимум 100.5 — пробой на первом же часе; день 3: максимум дня 2 = 101.5 — пробой в 103
    assert s.notna().sum() == 2 and s.iloc[24] == 2 and s.iloc[58] == 2


def test_trend_score_features_are_causal(hourly_df, cfg):
    from forexmodel.features.builder import add_trend_score_features
    from forexmodel.features import indicators as ind

    c = dataclasses.replace(cfg, trend=dataclasses.replace(cfg.trend, score_regime_ema=50))
    h = hourly_df.copy()
    h[c.atr_col] = ind.atr(h, 14)
    full = add_trend_score_features(h, c)
    cut = 1100
    pre = add_trend_score_features(h.iloc[:cut].copy(), c)
    cols = [k for k in full.columns if k.startswith(("ts_", "lvl_"))]
    assert len(cols) == 14
    for k in cols:
        assert np.allclose(full[k].to_numpy()[:cut], pre[k].to_numpy(), equal_nan=True), k
