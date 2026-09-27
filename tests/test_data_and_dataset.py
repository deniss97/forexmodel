"""Тесты загрузки/ресемплинга и сборки датасета end-to-end (без обучения)."""

from __future__ import annotations

import pandas as pd

from forexmodel.data.loader import load_minute_csv, resample_ohlcv
from forexmodel.features.extension import extension_columns
from forexmodel.pipelines.dataset import build_dataset


def test_resample_keeps_volume(minute_df):
    hourly = resample_ohlcv(minute_df, "1h")
    assert "volume" in hourly.columns          # в ноутбуке объём терялся при ресемплинге
    assert hourly["volume"].iloc[0] > 0
    assert hourly["high"].iloc[0] >= hourly["open"].iloc[0]


def test_loader_reads_begin_and_value_columns(tmp_path, minute_df):
    raw = minute_df.rename(columns={"time": "begin", "volume": "value"})
    path = tmp_path / "quotes.csv"
    raw.to_csv(path, index=False)

    df = load_minute_csv(path)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(df) == len(minute_df)


def test_build_dataset_produces_disjoint_labeled_splits(tmp_path, minute_df, cfg):
    path = tmp_path / "quotes.csv"
    minute_df.rename(columns={"time": "begin", "volume": "value"}).to_csv(path, index=False)
    cfg.data.csv_path = str(path)
    cfg.features.rsi_z_window = 50

    ds = build_dataset(cfg)

    assert not ds.train.empty and not ds.sim.empty
    assert ds.train["time"].max() < ds.test["time"].min()
    assert ds.test["time"].max() < ds.sim["time"].min()
    assert ds.train["label"].notna().all()

    # признаки растяжения и объёма действительно доехали до отбора
    ext = set(extension_columns(cfg.features))
    assert ext & set(ds.features)
    assert "rel_vol_20" in ds.features

    # абсолютных уровней цены в признаках быть не должно
    assert not {"close", "sma_3", "atr_14", "macd", "bb_up"} & set(ds.features)


def test_minute_slice_covers_split_with_horizon_margin(tmp_path, minute_df, cfg):
    path = tmp_path / "quotes.csv"
    minute_df.rename(columns={"time": "begin", "volume": "value"}).to_csv(path, index=False)
    cfg.data.csv_path = str(path)

    ds = build_dataset(cfg)
    minutes = ds.minute_slice("sim")

    assert minutes["time"].min() <= ds.sim["time"].min()
    assert minutes["time"].max() >= ds.sim["time"].max()
    assert isinstance(minutes["time"].iloc[0], pd.Timestamp)


def test_resplit_matches_full_rebuild(tmp_path, minute_df, cfg):
    """Walk-forward перерезает один датасет вместо сборки заново — выборки обязаны совпасть."""
    import copy

    from forexmodel.pipelines.dataset import resplit_dataset

    path = tmp_path / "m.csv"
    minute_df.rename(columns={"time": "begin", "volume": "value"}).to_csv(path, index=False)
    cfg.data.csv_path = str(path)
    cfg.data.minute_loader = "full"
    ds = build_dataset(cfg)

    other = copy.deepcopy(cfg)
    other.data.splits.update({"train_end": "2024-01-25 00:00:00", "test_start": "2024-01-25 00:00:00",
                              "test_end": "2024-02-05 00:00:00", "sim_start": "2024-02-05 00:00:00"})
    fresh = build_dataset(other)
    resplit = resplit_dataset(ds, other)
    assert resplit.features == fresh.features
    for name in ("train", "test", "sim"):
        pd.testing.assert_frame_equal(resplit.splits[name], fresh.splits[name])


def test_walk_forward_quarter_windows():
    import importlib.util
    import pathlib
    import sys

    scripts = pathlib.Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))  # walk_forward импортирует _bootstrap из scripts/
    spec = importlib.util.spec_from_file_location("wf", scripts / "walk_forward.py")
    wf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wf)
    w = wf.windows(quarters=["2024Q4", "2025Q2"])
    assert [x[0] for x in w] == ["2024Q4", "2025Q1", "2025Q2"]
    assert w[0][1] == pd.Timestamp("2024-10-01") and w[0][2] == pd.Timestamp("2025-01-01")
    assert wf.windows(years=[2020, 2021])[1] == ("2021", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"))
