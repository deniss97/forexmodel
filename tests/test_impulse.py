"""Сигнал «импульс» (signal_source=impulse): направление и отсутствие заглядывания вперёд."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forexmodel.config import config_from_dict
from forexmodel.simulation.signals import LONG, SHORT, build_signal_column


def _cfg(**sim):
    base = {"signal_source": "impulse", "impulse_bars": 2, "impulse_atr": 3.0, "use_expected_value_filter": False}
    base.update(sim)
    return config_from_dict({"simulation": base, "meta": {"enabled": False}})


def test_impulse_direction_and_threshold():
    df = pd.DataFrame({"close": [100, 100, 104, 104, 100, 99.5], "atr_14": [1.0] * 6})
    sig = build_signal_column(df, _cfg())["final_class"]
    # ход за 2 бара: nan, nan, +4, +4, −4, −4.5
    assert np.isnan(sig.iloc[0]) and np.isnan(sig.iloc[1])
    assert sig.iloc[2] == LONG and sig.iloc[3] == LONG
    assert sig.iloc[4] == SHORT and sig.iloc[5] == SHORT


def test_impulse_uses_only_past_bars():
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 1, 300).cumsum()
    df = pd.DataFrame({"close": close, "atr_14": np.full(300, 1.0)})
    full = build_signal_column(df, _cfg(impulse_bars=6, impulse_atr=2.0))["final_class"]
    cut = build_signal_column(df.iloc[:150].copy(), _cfg(impulse_bars=6, impulse_atr=2.0))["final_class"]
    assert full.iloc[:150].equals(cut)
