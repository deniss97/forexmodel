"""Прогноз волатильности и размер позиции: цель, размер, симулятор, пайплайн."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.models.volatility import position_sizes, volatility_target
from forexmodel.simulation.simulator import simulate_trades


def test_target_is_mean_true_range_of_next_bars():
    df = pd.DataFrame({"high": [11.0, 12.0, 14.0, 13.0], "low": [9.0, 10.0, 11.0, 12.0],
                       "close": [10.0, 11.0, 13.0, 12.5]})
    tgt = volatility_target(df, horizon=2)
    # TR[1] = max(2, |12-10|, |10-10|) = 2; TR[2] = max(3, |14-11|, |11-11|) = 3
    assert tgt.iloc[0] == pytest.approx((2 + 3) / 2 / 10 * 100)
    # у последних `horizon` баров горизонт не помещается в данные
    assert tgt.iloc[-2:].isna().all()


def test_target_of_bar_depends_only_on_its_horizon():
    """Цель бара i меняется только от баров i+1..i+h — дальше будущее не видно."""
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 1, 200).cumsum()
    df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1})
    h = 5
    full = volatility_target(df, h)
    cut = volatility_target(df.iloc[:100], h)
    assert np.allclose(full.iloc[: 100 - h], cut.iloc[: 100 - h])
    assert cut.iloc[100 - h:].isna().all()


def test_position_sizes():
    size = position_sizes(np.array([0.5, 0.9, 1.0, 2.0, 5.0, np.nan]), reference=1.0, min_ratio=0.9, power=1.0, cap=2.0)
    assert size.tolist() == pytest.approx([0.0, 0.9, 1.0, 2.0, 2.0, 0.0])
    only_filter = position_sizes(np.array([0.5, 1.5]), reference=1.0, min_ratio=1.0, power=0.0, cap=2.0)
    assert only_filter.tolist() == [0.0, 1.0]


def _sized_cfg(enabled=True):
    return config_from_dict({
        "labeling": {"horizon": 2},
        "simulation": {"exit_mode": "fixed_pct", "commission_pct": 0.1, "open_delay_minutes": 1,
                       "horizon_minutes": 100, "stop_loss_pct": 1.0, "take_profit_pct": 1.0,
                       "signal_source": "cb", "use_expected_value_filter": False,
                       "use_trend_filter": False, "close_on_trend_flip": False},
        "meta": {"enabled": False},
        "volatility": {"enabled": enabled},
    })


def test_simulator_scales_pnl_and_commission_by_size():
    path = [100.0] * 5 + [101.0] * 50 + [100.0] * 5 + [101.0] * 60
    px = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=len(path), freq="1min"),
                       "open": path, "high": path, "low": path, "close": path})
    signals = pd.DataFrame({"time": px["time"].iloc[[0, 30, 55]], "final_class": [2, 2, 2],
                            "atr_14": [1.0] * 3, "position_size": [2.0, 0.0, 0.5]})

    trades, _ = simulate_trades(signals, px, _sized_cfg())
    # второй сигнал нулевого размера: не открывается и не блокирует третий
    assert len(trades) == 2
    assert trades["size"].tolist() == [2.0, 0.5]
    assert trades["profit_pct"].tolist() == pytest.approx([2 * 0.9, 0.5 * 0.9])
    assert trades["commission_pct"].tolist() == pytest.approx([0.2, 0.05])
    assert trades["gross_unit_pct"].tolist() == pytest.approx([1.0, 1.0])

    # без volatility.enabled колонка размера игнорируется
    plain, _ = simulate_trades(signals, px, _sized_cfg(enabled=False))
    assert (plain["size"] == 1.0).all()


def test_training_and_backtest_with_volatility_sizing(tmp_path, minute_df):
    from forexmodel.pipelines.backtest_pipeline import run_backtest
    from forexmodel.pipelines.dataset import build_dataset
    from forexmodel.pipelines.train_pipeline import run_training

    path = tmp_path / "m.csv"
    minute_df.rename(columns={"time": "begin", "volume": "value"}).to_csv(path, index=False)
    cfg = config_from_dict({
        "data": {"csv_path": str(path), "minute_loader": "full", "splits": {
            "train_end": "2024-02-01 00:00:00", "test_start": "2024-02-01 00:00:00",
            "test_end": "2024-02-15 00:00:00", "sim_start": "2024-02-15 00:00:00"}},
        "features": {"rsi_z_window": 50},
        "labeling": {"mode": "atr_asym", "horizon": 5},
        "catboost": {"iterations": 40, "early_stopping_rounds": 10},
        "nn": {"enabled": False},
        "meta": {"enabled": False},
        "volatility": {"enabled": True, "iterations": 40, "ref_bars": 200, "min_ratio": 1.0, "power": 0.0},
        "simulation": {"signal_source": "cb", "exit_mode": "trailing", "use_trend_filter": False},
        "paths": {"artifacts_dir": str(tmp_path / "a"), "reports_dir": str(tmp_path / "r")},
    })
    ds = build_dataset(cfg)
    trained = run_training(cfg, dataset=ds, save=False)
    vm = trained.vol
    assert vm is not None and np.isfinite(vm.reference) and vm.reference > 0
    assert trained.metrics["volatility"]["trees"] > 0
    res = run_backtest(cfg, split="test", dataset=ds, primary=trained.primary, save=False, vol_model=vm)
    sig = res.signals
    assert sig["pred_vol_pct"].gt(0).all()
    # power 0: размер 1 выше порога, 0 ниже
    assert set(sig["position_size"].unique()) <= {0.0, 1.0}
    if len(res.trades):
        assert (res.trades["size"] == 1.0).all()
