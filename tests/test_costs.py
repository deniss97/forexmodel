"""Издержки брокера (перенос позиции, спред) и корректировка цен акций на дивиденды."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexmodel.config import config_from_dict
from forexmodel.data.dividends import adjust_for_dividends
from forexmodel.simulation.costs import profile_entry, recost, rollover_count
from forexmodel.simulation.simulator import simulate_trades

# 2024-01-01 — понедельник


def test_rollovers_stocks_triple_on_thursday():
    # акции РФ: перенос в 00:00 МСК после каждого торгового дня, с четверга — тройной
    assert rollover_count("2024-01-01 10:00", "2024-01-02 10:00", 0, 3) == 1        # пн → вт
    assert rollover_count("2024-01-04 10:00", "2024-01-05 10:00", 0, 3) == 3        # чт → пт
    assert rollover_count("2024-01-05 10:00", "2024-01-08 10:00", 0, 3) == 1        # пт → пн
    assert rollover_count("2024-01-03 10:00", "2024-01-10 10:00", 0, 3) == 7        # неделя = 7 дней
    assert rollover_count("2024-01-01 10:00", "2024-01-01 22:00", 0, 3) == 0        # внутри дня


def test_rollovers_forex_triple_on_wednesday_evening():
    # форекс-фид в нью-йоркском времени, перенос в 17:00; со среды — тройной
    assert rollover_count("2024-01-03 16:00", "2024-01-03 18:00", 17, 2) == 3
    assert rollover_count("2024-01-05 16:00", "2024-01-07 20:00", 17, 2) == 1        # пт → вс вечер
    assert rollover_count("2024-01-01 18:00", "2024-01-08 18:00", 17, 2) == 7


def test_rollovers_calendar_mode():
    assert rollover_count("2024-01-01 10:00", "2024-01-04 10:00", 0, None) == 3
    assert rollover_count("2024-01-05 10:00", "2024-01-08 10:00", 0, None) == 3


def test_alfaforex_profile_points_to_percent():
    e = profile_entry("configs/costs/alfaforex.yaml", "silver")
    assert e["spread_pct"] == pytest.approx(90 * 0.001 / 75.238 * 100)
    assert e["swap_long_pct"] == pytest.approx(-55 * 0.001 / 75.238 * 100)
    s = profile_entry("configs/costs/alfaforex.yaml", "sber")
    assert s["spread_pct"] == pytest.approx(0.15) and s["swap_short_pct"] == pytest.approx(-0.068)
    assert s["triple_weekday"] == 3


def test_recost_matches_simulator_swap():
    # одна длинная сделка с пн 10:00 до чт 10:00: три переноса, своп −0.1% за перенос
    t = pd.date_range("2024-01-01 10:00", "2024-01-04 10:30", freq="1min")
    px = pd.DataFrame({"time": t, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    sig = pd.DataFrame({"time": [t[0]], "final_class": [2], "atr_14": [1.0]})
    cfg = config_from_dict({"labeling": {"horizon": 2}, "meta": {"enabled": False}, "simulation": {
        "exit_mode": "fixed_pct", "commission_pct": 0.05, "open_delay_minutes": 1,
        "horizon_minutes": 3 * 24 * 60, "stop_loss_pct": 5.0, "take_profit_pct": 5.0, "signal_source": "cb",
        "use_expected_value_filter": False, "use_trend_filter": False,
        "swap_long_pct": -0.1, "swap_short_pct": -0.02, "swap_rollover_hour": 0, "swap_triple_weekday": 3}})
    trades, _ = simulate_trades(sig, px, cfg)
    tr = trades.iloc[0]
    assert tr["rollovers"] == 3
    assert tr["profit_pct"] == pytest.approx(-0.05 - 0.3)
    again = recost(trades, {"spread_pct": 0.05, "swap_long_pct": -0.1, "swap_short_pct": -0.02,
                            "rollover_hour": 0, "triple_weekday": 3})
    assert again["profit_pct"].iloc[0] == pytest.approx(tr["profit_pct"])


def test_dividend_adjustment_removes_gap_and_keeps_total_return():
    # 100 до отсечки, дивиденд 10, после отсечки цена 90: в скорректированных ценах гэпа нет
    t = pd.date_range("2024-01-01 10:00", periods=6, freq="1h")
    p = [100.0, 100.0, 100.0, 90.0, 90.0, 99.0]
    m = pd.DataFrame({"time": t, "open": p, "high": p, "low": p, "close": p})
    divs = pd.DataFrame({"ex_time": [t[3]], "amount": [10.0]})
    a = adjust_for_dividends(m, divs)
    assert np.allclose(a["close"].iloc[:3], 90.0)                 # до отсечки × (100 − 10) / 100
    assert np.allclose(a["close"].iloc[3:], m["close"].iloc[3:])  # после — без изменений
    # лонг с 100 до 99 по сырым ценам: −1%; с дивидендом 10 — полная доходность +10% (дивиденд
    # реинвестирован по цене отсечки 90; без реинвестирования было бы +9% — разница второго порядка)
    assert a["close"].iloc[-1] / a["close"].iloc[0] - 1 == pytest.approx(99 / 90 - 1)
