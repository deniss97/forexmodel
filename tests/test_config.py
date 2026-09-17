import pytest

from forexmodel.config import config_from_dict, load_config


def test_default_and_baseline_configs_load():
    for path in ("configs/default.yaml", "configs/baseline_notebook.yaml"):
        cfg = load_config(path)
        assert cfg.labeling.horizon > 0
        assert cfg.horizon_minutes == cfg.labeling.horizon * cfg.data.signal_tf_minutes


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="неизвестные параметры"):
        config_from_dict({"catboost": {"deth": 6}})


def test_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="Неизвестные секции"):
        config_from_dict({"catbost": {}})


def test_atr_col_must_match_atr_period():
    with pytest.raises(ValueError, match="atr_col"):
        config_from_dict({"features": {"atr_period": 20}, "labeling": {"atr_col": "atr_14"}})


def test_meta_source_requires_meta_model():
    with pytest.raises(ValueError, match="meta.enabled"):
        config_from_dict({"simulation": {"signal_source": "meta"}, "meta": {"enabled": False}})


def test_embargo_defaults_to_horizon_plus_one():
    cfg = config_from_dict({"labeling": {"horizon": 12}})
    assert cfg.embargo_bars == 13
    assert cfg.meta_embargo_bars == 24
