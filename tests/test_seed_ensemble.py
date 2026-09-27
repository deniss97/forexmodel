"""Ансамбль primary-моделей из нескольких seed."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest

from forexmodel.config import config_from_dict
from forexmodel.models.catboost_primary import SeedEnsemble, predict_primary, train_primary
from forexmodel.pipelines.dataset import build_dataset


def test_seed_ensemble_averages_members(tmp_path, minute_df):
    path = tmp_path / "m.csv"
    minute_df.rename(columns={"time": "begin", "volume": "value"}).to_csv(path, index=False)
    cfg = config_from_dict({
        "data": {"csv_path": str(path), "minute_loader": "full", "splits": {
            "train_end": "2024-02-01 00:00:00", "test_start": "2024-02-01 00:00:00",
            "test_end": "2024-02-15 00:00:00", "sim_start": "2024-02-15 00:00:00"}},
        "features": {"rsi_z_window": 50},
        "labeling": {"mode": "atr_asym", "horizon": 5},
        "catboost": {"iterations": 40, "early_stopping_rounds": 10, "n_models": 3, "random_seed": 5},
        "nn": {"enabled": False},
        "meta": {"enabled": False},
        "simulation": {"signal_source": "cb"},
    })
    ds = build_dataset(cfg)
    bundle = train_primary(ds.train, ds.features, cfg.catboost, embargo=cfg.embargo_bars)

    assert isinstance(bundle.model, SeedEnsemble) and len(bundle.model.models) == 3
    assert bundle.metrics["ensemble_members"] == 3.0 and "polar_edge" in bundle.metrics
    X = ds.test[bundle.features]
    manual = np.mean([m.predict_proba(X) for m in bundle.model.models], axis=0)
    assert np.allclose(bundle.model.predict_proba(X), manual)
    # члены действительно разные (разные seed), иначе ансамбль бессмыслен
    p0, p1 = bundle.model.models[0].predict_proba(X), bundle.model.models[1].predict_proba(X)
    assert not np.allclose(p0, p1)

    preds = predict_primary(bundle, ds.test)
    assert np.allclose(preds[["proba_0", "proba_1", "proba_2"]].sum(axis=1), 1.0)
    assert isinstance(bundle.model.tree_count_, int)
    pickle.loads(pickle.dumps(bundle))   # артефакт сохраняется


def test_ensemble_rejects_regression():
    with pytest.raises(ValueError, match="n_models"):
        config_from_dict({"catboost": {"n_models": 3, "objective": "regression"},
                          "labeling": {"mode": "direction"}, "meta": {"enabled": False}})
