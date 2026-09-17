"""Тесты симулятора: барьеры, комиссия, приоритеты выходов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.simulation.exits import fixed_barrier_exit, trailing_exit
from forexmodel.simulation.simulator import simulate_trades


def _minutes(path: list[float], start: str = "2024-01-01 00:00") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=len(path), freq="1min"),
            "open": path,
            "high": path,
            "low": path,
            "close": path,
        }
    )


def _cfg(**sim_overrides):
    sim = {
        "exit_mode": "fixed_pct",
        "commission_pct": 0.1,
        "open_delay_minutes": 1,
        "horizon_minutes": 100,
        "stop_loss_pct": 1.0,
        "take_profit_pct": 1.0,
        "signal_source": "cb",
        "use_expected_value_filter": False,
        "use_trend_filter": False,
        "close_on_trend_flip": False,
    }
    sim.update(sim_overrides)
    return config_from_dict({"labeling": {"horizon": 2}, "simulation": sim, "meta": {"enabled": False}})


def test_barriers_are_measured_from_raw_entry_price():
    """TP должен срабатывать ровно на +1% от цены входа, а не от «цены с
    зашитой комиссией» (в ноутбуке реальный TP уезжал на величину комиссии)."""
    path = [100.0] * 5 + [101.0] * 50
    px = _minutes(path)
    signals = pd.DataFrame({"time": [px["time"].iloc[0]], "final_class": [2], "atr_14": [1.0]})

    trades, report = simulate_trades(signals, px, _cfg())
    trade = trades.iloc[0]

    assert trade["open_price"] == pytest.approx(100.0)
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(101.0)
    assert trade["gross_pct"] == pytest.approx(1.0)
    assert trade["profit_pct"] == pytest.approx(0.9)  # комиссия 0.1% вычитается из PnL отдельно


def test_short_pnl_and_commission():
    path = [100.0] * 5 + [99.0] * 50
    px = _minutes(path)
    signals = pd.DataFrame({"time": [px["time"].iloc[0]], "final_class": [0], "atr_14": [1.0]})

    trades, _ = simulate_trades(signals, px, _cfg())
    trade = trades.iloc[0]
    assert trade["side"] == "sell"
    assert trade["exit_reason"] == "take_profit"
    assert trade["profit_pct"] == pytest.approx(0.9)


def test_timeout_exit_uses_last_close():
    path = [100.0] * 60
    px = _minutes(path)
    signals = pd.DataFrame({"time": [px["time"].iloc[0]], "final_class": [2], "atr_14": [1.0]})

    trades, _ = simulate_trades(signals, px, _cfg(horizon_minutes=10))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "timeout"
    assert trade["profit_pct"] == pytest.approx(-0.1)  # только комиссия


def test_stop_wins_on_intrabar_tie():
    """Если в одном баре задеты и TP, и SL, считается стоп (консервативно)."""
    high = np.array([102.0])
    low = np.array([98.0])
    hit = fixed_barrier_exit(high, low, "buy", tp_price=101.0, sl_price=99.0)
    assert hit[2] == "stop_loss"


def test_trailing_exit_locks_profit_after_activation():
    """Цена уходит на +3 ATR, затем откатывается: выходим по трейлингу, а не в минус."""
    high = np.array([101.0, 102.0, 103.0, 103.0, 100.0])
    low = np.array([100.0, 101.0, 102.0, 101.0, 100.0])
    close = np.array([101.0, 102.0, 103.0, 101.5, 100.0])

    idx, price, reason = trailing_exit(
        high, low, close, side="buy", entry_price=100.0, atr=1.0, sl_atr=1.0, trail_atr=1.5, activate_atr=1.0
    )
    assert reason == "trailing_stop"
    assert price == pytest.approx(101.5)  # 103 - 1.5 ATR
    assert idx == 3


def test_trailing_exit_cuts_false_breakout():
    high = np.array([100.5, 100.2, 99.5])
    low = np.array([100.0, 99.5, 98.5])
    close = np.array([100.2, 99.6, 98.6])

    idx, price, reason = trailing_exit(
        high, low, close, side="buy", entry_price=100.0, atr=1.0, sl_atr=1.0, trail_atr=1.5, activate_atr=1.0
    )
    assert reason == "stop_loss"
    assert price == pytest.approx(99.0)
    assert idx == 2


def test_overlapping_positions_are_skipped_by_default():
    path = [100.0] * 200
    px = _minutes(path)
    times = [px["time"].iloc[0], px["time"].iloc[5], px["time"].iloc[120]]
    signals = pd.DataFrame({"time": times, "final_class": [2, 2, 2], "atr_14": [1.0] * 3})

    trades, _ = simulate_trades(signals, px, _cfg(horizon_minutes=60))
    assert len(trades) == 2  # второй сигнал попадает внутрь первой позиции


def test_trend_filter_blocks_counter_trend_entries():
    path = [100.0] * 60
    px = _minutes(path)
    signals = pd.DataFrame(
        {"time": [px["time"].iloc[0]], "final_class": [2], "atr_14": [1.0], "trend_4h": [-1.0]}
    )

    trades, _ = simulate_trades(signals, px, _cfg(use_trend_filter=True))
    assert trades.empty


def test_atr_exit_mode_uses_atr_distances():
    path = [100.0] * 5 + [102.0] * 50
    px = _minutes(path)
    signals = pd.DataFrame({"time": [px["time"].iloc[0]], "final_class": [2], "atr_14": [1.0]})

    trades, _ = simulate_trades(signals, px, _cfg(exit_mode="atr", tp_atr=2.0, sl_atr=1.0))
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(102.0)
