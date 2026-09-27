"""Признаки мета-модели, которых нет у primary: знаковые, класс входа, исход последних сделок."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.models.meta_context import add_direction_context, add_recent_outcomes


def _frame():
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=6, freq="1h"),
        "pred": [2, 0, 1, 2, 0, 2],
        "ret_lag_2_atr": [0.5, 0.5, 0.5, -0.5, -1.5, 1.2],
        "log_return": [0.01, 0.01, 0.01, -0.01, -0.01, 0.0],
        "ext_from_low_24": [3.0] * 6,
        "ext_from_high_24": [1.0] * 6,
    })


def test_direction_context_signs_and_entry_class():
    out = add_direction_context(_frame(), "pred")
    assert out["mctx_s_ret2"].tolist()[:2] == [0.5, -0.5]           # лонг +, шорт −
    assert np.isnan(out["mctx_s_ret2"].iloc[2])                       # не полярный
    # классы: лонг +0.5 -> продолжение; шорт +0.5 против -> откат; лонг −0.5 -> откат; шорт −1.5 по ходу -> продолжение
    assert out["mctx_entry_class"].tolist()[:2] == [2.0, -1.0]
    assert out["mctx_entry_class"].iloc[3] == -1.0 and out["mctx_entry_class"].iloc[4] == 2.0
    # растяжение в сторону сделки и запас до противоположного экстремума
    assert out["mctx_ext_24"].iloc[0] == 3.0 and out["mctx_ext_24"].iloc[1] == 1.0
    assert out["mctx_room_24"].iloc[0] == 1.0 and out["mctx_room_24"].iloc[1] == 3.0


def test_recent_outcomes_exclude_trade_closing_exactly_at_decision():
    df = _frame()
    # решение по бару 01:00 принимается в 02:00: сделка, закрытая ровно в 02:00, ещё не видна
    outcomes = pd.DataFrame({
        "close_dt": pd.to_datetime(["2024-01-01 01:59", "2024-01-01 02:00"]),
        "profit_pct": [-1.0, 5.0],
        "side": ["sell", "buy"],
    })
    out = add_recent_outcomes(df, outcomes, "pred", windows=[5])
    assert out["mctx_recent_pnl_5"].iloc[1] == pytest.approx(-1.0)
    assert np.isnan(out["mctx_recent_pnl_5"].iloc[0])               # к 01:00 ничего не закрыто


def test_recent_outcomes_window_and_same_side():
    df = _frame()
    outcomes = pd.DataFrame({
        "close_dt": pd.to_datetime(["2024-01-01 00:30", "2024-01-01 01:30", "2024-01-01 02:30", "2024-01-01 03:30"]),
        "profit_pct": [1.0, -1.0, 3.0, -2.0],
        "side": ["buy", "sell", "buy", "sell"],
    })
    out = add_recent_outcomes(df, outcomes, "pred", windows=[2])
    # бар 03:00 (лонг), решение 04:00: видны все 4; последние 2 — +3 и −2
    assert out["mctx_recent_pnl_2"].iloc[3] == pytest.approx(0.5)
    assert out["mctx_recent_win_2"].iloc[3] == pytest.approx(0.5)
    # только лонги: +1 и +3
    assert out["mctx_recent_same_pnl_2"].iloc[3] == pytest.approx(2.0)
    assert out["mctx_recent_same_win_2"].iloc[3] == pytest.approx(1.0)
    # бар 01:00 (шорт), решение 02:00: видны +1 (лонг, 00:30) и −1 (шорт, 01:30)
    assert out["mctx_recent_pnl_2"].iloc[1] == pytest.approx(0.0)
    assert out["mctx_recent_same_pnl_2"].iloc[1] == pytest.approx(-1.0)
    # бар 00:00, решение 01:00: видна только 00:30
    assert out["mctx_recent_pnl_2"].iloc[0] == pytest.approx(1.0)
    # неполярный бар — NaN
    assert np.isnan(out["mctx_recent_pnl_2"].iloc[2])


def test_meta_with_context_trains_and_backtests(tmp_path, minute_df):
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
        "meta": {"enabled": True, "context_features": True, "n_splits": 3, "iterations": 30, "threshold": 0.3},
        "simulation": {"signal_source": "meta", "exit_mode": "trailing", "use_trend_filter": False},
        "paths": {"artifacts_dir": str(tmp_path / "a"), "reports_dir": str(tmp_path / "r")},
    })
    ds = build_dataset(cfg)
    assert not any(c.startswith(("vr4_", "acf1_", "mctx_")) for c in ds.features)   # режим — только мете
    trained = run_training(cfg, dataset=ds, save=False)
    assert trained.meta.context
    assert any(c.startswith("mctx_recent") for c in trained.meta.features)
    assert any(c.startswith("vr4_") for c in trained.meta.features)
    res = run_backtest(cfg, split="test", dataset=ds, primary=trained.primary, meta_model=trained.meta, save=False)
    assert "meta_proba" in res.signals.columns
    assert res.signals.loc[res.signals["y_pred_cb"].isin([0, 2]), "meta_proba"].notna().all()
