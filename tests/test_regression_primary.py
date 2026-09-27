"""Primary-модель как регрессия хода в ATR: обучение, предсказание, EV по предсказанному ходу."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.models.catboost_primary import predict_primary, train_primary
from forexmodel.pipelines.dataset import build_dataset
from forexmodel.simulation.signals import attach_model_predictions, build_signal_column


def _reg_cfg(tmp_path, minute_df, **catboost):
    path = tmp_path / "m.csv"
    minute_df.rename(columns={"time": "begin", "volume": "value"}).to_csv(path, index=False)
    return config_from_dict({
        "data": {"csv_path": str(path), "minute_loader": "full", "splits": {
            "train_end": "2024-02-01 00:00:00", "test_start": "2024-02-01 00:00:00",
            "test_end": "2024-02-15 00:00:00", "sim_start": "2024-02-15 00:00:00"}},
        "features": {"rsi_z_window": 50},
        "labeling": {"mode": "direction", "horizon": 4, "dir_atr": 0.3},
        "catboost": {"objective": "regression", "iterations": 60, "early_stopping_rounds": 20, "reg_signal_atr": 0.3, **catboost},
        "nn": {"enabled": False},
        "meta": {"enabled": False},
        "simulation": {"signal_source": "cb", "exit_mode": "trailing", "use_trend_filter": False, "commission_pct": 0.04},
    })


def test_regression_requires_direction_labels_and_no_meta():
    with pytest.raises(ValueError, match="direction"):
        config_from_dict({"catboost": {"objective": "regression"}, "meta": {"enabled": False}})
    with pytest.raises(ValueError, match="meta"):
        config_from_dict({"catboost": {"objective": "regression"}, "labeling": {"mode": "direction"}, "meta": {"enabled": True}})


def test_regression_primary_trains_predicts_and_feeds_ev(tmp_path, minute_df):
    cfg = _reg_cfg(tmp_path, minute_df)
    ds = build_dataset(cfg)
    assert "target_move" in ds.train.columns and "target_move" not in ds.features

    primary = train_primary(ds.train, ds.features, cfg.catboost, embargo=cfg.embargo_bars)
    assert primary.objective == "regression"
    assert "ic_spearman" in primary.metrics and "polar_edge" in primary.metrics

    preds = predict_primary(primary, ds.test)
    assert {"pred_move", "proba_0", "proba_1", "proba_2", "y_pred", "confidence"} <= set(preds.columns)
    assert preds["y_pred"].isin([0, 1, 2]).all()
    thr = primary.signal_threshold
    assert ((preds["pred_move"] >= thr) == (preds["y_pred"] == 2)).all()
    assert np.allclose(preds["proba_0"] + preds["proba_2"], 1.0)

    signals = build_signal_column(attach_model_predictions(ds.test, preds), cfg)
    # EV = знак · предсказанный ход · ATR% − комиссия; у лонга с pred_move > 0 она положительна при заметном ходе
    row = signals.dropna(subset=["expected_value"]).iloc[0]
    sgn = 1.0 if row["final_class"] == 2 else -1.0
    expected = sgn * row["pred_move_cb"] * row["atr_14"] / row["close"] * 100 - 0.04
    assert row["expected_value"] == pytest.approx(expected)
