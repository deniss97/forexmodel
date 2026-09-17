"""Тесты формирования сигнала, отбора признаков и нарезки выборок."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.data.splits import apply_embargo, build_splits
from forexmodel.features.selection import absolute_price_columns, select_feature_columns
from forexmodel.models.cv import chronological_split, purged_walk_forward_splits
from forexmodel.simulation.signals import build_signal_column


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="1h"),
            "close": [100.0] * 4,
            "atr_14": [1.0] * 4,
            "y_pred_cb": [2, 2, 0, 1],
            "confidence_cb": [0.9, 0.4, 0.8, 0.9],
            "proba_2_cb": [0.9, 0.4, 0.1, 0.2],
            "proba_0_cb": [0.05, 0.3, 0.8, 0.2],
            "proba_1_cb": [0.05, 0.3, 0.1, 0.6],
            "y_pred_nn": [2, 0, 0, 0],
            "confidence_nn": [0.8, 0.9, 0.7, 0.6],
        }
    )


def _cfg(**sim):
    base = {"signal_source": "ensemble", "use_expected_value_filter": False, "use_trend_filter": False}
    base.update(sim)
    return config_from_dict({"simulation": base, "meta": {"enabled": False}})


def test_agreement_rule_keeps_only_matching_signals():
    out = build_signal_column(_signal_frame(), _cfg(ensemble_rule="agreement"))
    assert out["final_class"].tolist()[:1] == [2]
    assert out["final_class"].notna().sum() == 2  # бары 0 (2/2) и 2 (0/0)


def test_confidence_threshold_drops_weak_signals():
    out = build_signal_column(_signal_frame(), _cfg(ensemble_rule="cb_priority", min_conf_cb=0.5))
    assert pd.isna(out["final_class"].iloc[1])  # confidence_cb = 0.4


def test_expected_value_filter_rejects_coinflip_trades():
    """p=0.4 при TP=SL и комиссии 0.15% — заведомо убыточная сделка,
    хотя «уверенность» модели могла бы пройти порог 0.5."""
    cfg = _cfg(
        signal_source="cb",
        use_expected_value_filter=True,
        exit_mode="fixed_pct",
        take_profit_pct=1.0,
        stop_loss_pct=1.0,
        commission_pct=0.15,
    )
    out = build_signal_column(_signal_frame(), cfg)
    assert out["final_class"].iloc[0] == 2      # p=0.9 -> EV > 0
    assert pd.isna(out["final_class"].iloc[1])  # p=0.4 -> EV < 0
    assert out["expected_value"].iloc[0] == pytest.approx(0.9 * 1 - 0.1 * 1 - 0.15)


def test_meta_source_requires_final_signal_column():
    cfg = config_from_dict({"simulation": {"signal_source": "meta"}, "meta": {"enabled": True}})
    with pytest.raises(KeyError):
        build_signal_column(_signal_frame(), cfg)


def test_extension_filter_drops_late_entries():
    df = _signal_frame()
    df["ext_from_low_24"] = [0.5, 0.5, 0.5, 0.5]
    df["ext_from_high_24"] = [0.5, 0.5, 3.0, 0.5]

    out = build_signal_column(df, _cfg(signal_source="cb", max_extension_atr=1.5))
    assert out["final_class"].iloc[0] == 2
    assert pd.isna(out["final_class"].iloc[2])  # шорт при растяжении 3 ATR от максимума


def test_absolute_price_columns_are_excluded_from_features():
    cfg = config_from_dict({})
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "close": [1.0, 2.0, 3.0],
            "sma_3": [1.0, 2.0, 3.0],
            "atr_14": [1.0, 2.0, 3.0],
            "macd": [1.0, 2.0, 3.0],
            "bb_pos": [0.1, 0.2, 0.3],
            "rsi_z": [0.1, 0.2, 0.3],
            "label": [0, 1, 2],
        }
    )
    features = select_feature_columns(df, cfg.features)

    assert set(features) == {"bb_pos", "rsi_z"}
    assert "sma_3" in absolute_price_columns(cfg.features)


def test_overlapping_splits_are_rejected():
    cfg = config_from_dict(
        {
            "data": {
                "splits": {
                    "train_end": "2025-12-30 21:00:00",
                    "test_start": "2024-12-30 21:00:00",   # внутри train — как в ноутбуке
                    "test_end": "2025-12-30 21:00:00",
                    "sim_start": "2025-12-30 21:00:00",
                }
            }
        }
    )
    with pytest.raises(ValueError, match="пересекаются"):
        build_splits(cfg)


def test_apply_embargo_drops_tail():
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=10, freq="1h"), "label": range(10)})
    out = apply_embargo(df, 3)
    assert len(out) == 7


def test_purged_cv_leaves_gap_between_train_and_val():
    splits = purged_walk_forward_splits(1000, n_splits=4, embargo=10)
    assert splits
    for train_idx, val_idx in splits:
        assert val_idx[0] - train_idx[-1] > 10


def test_chronological_split_has_embargo_gaps():
    train, val, hold = chronological_split(1000, test_size=0.2, val_size=0.1, embargo=15)
    assert val[0] - train[-1] > 15
    assert hold[0] - val[-1] > 15
    assert len(hold) == 200


def test_uniqueness_weights_do_not_change_scale():
    from forexmodel.labeling.uniqueness import average_uniqueness

    weights = average_uniqueness(np.arange(100) + 5, n_bars=120)
    assert weights[weights > 0].mean() == pytest.approx(1.0)
