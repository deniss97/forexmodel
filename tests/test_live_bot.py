"""Бот контура (scripts/live_bot.py) и новые правила счёта (live_account.py) — чистые функции, без сети и файлов контура."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_account as la  # noqa: E402
import live_bot as lb  # noqa: E402

CFG = {"capital": 15000, "slot": 5000, "max_positions": 3, "lot_over_slot": True, "max_loss_pct": 10,
       "one_position_per_stock": True, "priority": ["S42", "S46", "S48"], "near_stop_pct": 1.0}
YAML = """capital: 15000            # рублей
slot: 5000                # рублей на одну сделку
strategies:               # на счёт
  S42: [sber, gazp, lkoh]
  S46: all                # пробой
  S48: all
weekends: false
"""


def test_trial_cap_and_ranges():
    with pytest.raises(ValueError, match="пробный этап"):
        lb.parse_setting("capital", "20000", CFG)
    with pytest.raises(ValueError, match="больше капитала"):
        lb.parse_setting("slot", "12000", {**CFG, "capital": 10000})
    with pytest.raises(ValueError, match="нет настройки"):
        lb.parse_setting("leverage", "4", CFG)
    assert lb.parse_setting("slot", "4 000", CFG) == ("4000", 4000)
    assert lb.parse_setting("max_slippage_pct", "0,5", CFG) == ("0.5", 0.5)
    assert lb.parse_setting("weekends", "on", CFG) == ("true", True)


def test_set_setting_keeps_comment_and_appends_missing():
    out = lb.set_setting(YAML, "slot", "4000")
    assert "slot: 4000                # рублей на одну сделку" in out
    assert yaml.safe_load(out)["slot"] == 4000
    assert yaml.safe_load(lb.set_setting(YAML, "near_stop_pct", "1.5"))["near_stop_pct"] == 1.5


def test_strategy_toggle_turns_off_and_back_on():
    off = lb.set_strategy(YAML, "S46", False)
    assert "  S46: off                # пробой" in off
    lots = {"SBER": 1, "GAZP": 10, "LKOH": 1, "TATN": 1}
    assert la.allowed_stocks(yaml.safe_load(off), lots)["S46"] == []
    on = lb.set_strategy(off, "S46", True)
    assert la.allowed_stocks(yaml.safe_load(on), lots)["S46"] == ["sber", "gazp", "lkoh", "tatn"]
    assert la.allowed_stocks(yaml.safe_load(on), lots)["S42"] == ["sber", "gazp", "lkoh"]


def pos(**kw):
    return {"стратегия": "S42", "бумага": "SBER", "вход_в": pd.Timestamp("2026-10-23 17:01"), "стоп": 290.0,
            "стоп стратегии": 285.0, "активация": 310.0, "отступ": 12.0, "исполнение": "вручную", **kw}


def test_position_alerts():
    kinds = lambda px, **kw: [k for k, _ in lb.position_alerts(pos(**kw), px, CFG)]  # noqa: E731
    assert kinds(300.0) == []
    assert kinds(292.0) == ["близко к стопу"]                       # 0.7% до стопа 290
    assert kinds(289.0) == ["у стопа"]
    assert kinds(311.0) == ["трейлинг"]
    assert kinds(296.0, **{"стоп стратегии": 294.0}) == ["близко к стопу"]   # трейлинг подтянул стоп стратегии
    assert "/close SBER" in lb.position_alerts(pos(), 289.0, CFG)[0][1]
    assert "брокера" in lb.position_alerts(pos(исполнение="API"), 289.0, CFG)[0][1]


def test_pid_is_short_and_stable():
    assert la.pid(pos()) == "S42:SBER:2610231701"
    assert len(("cm:" + la.pid({**pos(), "стратегия": "S48", "бумага": "TRNFP"})).encode()) <= 64


def test_weekend_signals_go_to_alternatives_when_off():
    sat = pd.Timestamp("2026-10-31 12:01")
    s = {"стратегия": "S42", "бумага": "SBER", "вход_в": sat, "цена": 300.0, "стоп": 290.0, "срок": sat + pd.Timedelta(hours=240)}
    take, alt = la.allocate([s], [], {**CFG, "weekends": False}, {"SBER": 1})
    assert not take and alt[0]["причина"].startswith("выходные")
    take, alt = la.allocate([s], [], CFG, {"SBER": 1})                # по умолчанию выходные берём
    assert take and not alt


def test_volatility_flags_wide_last_hour():
    hours = pd.date_range("2026-09-21 10:00", periods=24 * 30, freq="h")
    rows = []
    for h in hours:
        wide = h == hours[-1]
        for m in range(3):
            rows.append({"begin": h + pd.Timedelta(minutes=m), "high": 101.0 + (2 if wide else 0), "low": 99.0, "close": 100.0})
    d = pd.DataFrame(rows)
    last, base, at = la.volatility(d, hours[-1] + pd.Timedelta(hours=1, minutes=5))
    assert at == hours[-1] and base == pytest.approx(2.0) and last == pytest.approx(4.0)


def test_append_csv_aligns_columns(tmp_path):
    p = tmp_path / "t.csv"
    la.append_csv(p, [{"a": 1, "b": 2}])
    la.append_csv(p, [{"b": 3, "a": 4}])                              # другой порядок
    la.append_csv(p, [{"a": 5, "c": 6}])                              # новое поле — файл переписан
    d = pd.read_csv(p)
    assert d["a"].tolist() == [1, 4, 5] and d["b"].tolist()[:2] == [2, 3] and np.isnan(d["b"].iloc[2]) and d["c"].iloc[2] == 6


NOW = pd.Timestamp("2026-07-15 12:05")


def div(amount: float, last_buy: str) -> dict:
    return {"dividendNet": {"units": int(amount), "nano": round((amount - int(amount)) * 1e9)}, "lastBuyDate": last_buy + "T00:00:00Z"}


def test_dividend_alert_warns_when_gap_beats_stop():
    p = {"стратегия": "S42", "бумага": "SBER", "цена": 300.0, "стоп": 294.0, "стоп стратегии": 290.0}
    m = la.dividend_alert(p, div(34.84, "2026-07-17"), NOW, days=3)
    assert "сработает" in m and "34.84" in m and "/close SBER" in m
    assert la.dividend_alert(p, div(34.84, "2026-07-20"), NOW, days=3) is None          # дальше трёх дней
    assert "можно держать" in la.dividend_alert(p, div(3.0, "2026-07-16"), NOW, days=3)  # 1% при стопе в 2%


def test_account_pnl_counts_closed_and_open():
    trades = pd.DataFrame([{"цена": 100.0, "цена выхода": 110.0, "штук": 10},
                           {"цена": 199.0, "цена_входа": 200.0, "цена выхода": 190.0, "штук": 5},
                           {"цена": 50.0, "цена выхода": None, "штук": 3}])          # без цены выхода — не считается
    open_ = [{"бумага": "GAZP", "статус": "открыта", "цена": 50.0, "штук": 10}, {"бумага": "LKOH", "статус": "к исполнению", "цена": 1.0, "штук": 1}]
    assert la.account_pnl(trades, open_, {"GAZP": 55.0, "LKOH": 9.0}) == pytest.approx(100 - 50 + 50)


def test_drawdown_alert_levels_once_until_new_peak():
    st, cfg = {}, {"capital": 15000, "dd_warn_pct": 8, "dd_stop_pct": 12}
    seen = [la.drawdown_alert(x, st, cfg) for x in (0, 600, -700, -800, -1300, -1500, 700, -700)]
    kinds = [None if m is None else ("стоп" if m.startswith("🔴") else "внимание") for m in seen]
    assert kinds == [None, None, "внимание", None, "стоп", None, None, "внимание"]


def test_append_threshold_is_separate(tmp_path):
    import moex_data
    moex_data.check_disk(tmp_path, 0)                                   # дописка: порог задан явно
    with pytest.raises(OSError, match="запись данных остановлена"):
        moex_data.check_disk(tmp_path, 1e9)
    assert moex_data.MIN_FREE_APPEND_GB < moex_data.MIN_FREE_GB
