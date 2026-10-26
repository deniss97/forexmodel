"""Правило сверки живой торговли с симуляцией (scripts/live_deviation.py: judge, losing_streak)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_deviation import judge, losing_streak  # noqa: E402

REF = np.random.default_rng(1).normal(0.3, 2.0, 2000)        # бэктест: средняя сделка +0.3%, разброс 2%


def test_losing_streak():
    assert losing_streak(np.array([1, -1, -2, 0, 3, -1])) == 3
    assert losing_streak(np.array([])) == 0


def test_few_trades_not_judged():
    assert judge(np.array([-5.0, -5.0]), REF, np.random.default_rng(0))["статус"] == "мало сделок"


def test_like_backtest_is_normal():
    live = np.random.default_rng(2).normal(0.3, 2.0, 20)
    assert judge(live, REF, np.random.default_rng(0))["статус"] == "в норме"


def test_much_worse_than_backtest_stops():
    live = np.full(15, -1.8)                                  # 15 сделок подряд по −1.8% — так плохо бэктест не бывает
    r = judge(live, REF, np.random.default_rng(0))
    assert r["статус"] == "стоп" and r["p"] < 0.01
