"""Проба P42 (созвон 04.11): S42 с порогом входа ниже — тем же путём сигнал → счёт → бот, но 1 лот и свои файлы."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_account as la  # noqa: E402
import live_bot as lb  # noqa: E402
import trial_args as ta  # noqa: E402

from forexmodel.simulation.exits import trailing_exit  # noqa: E402

CFG = {"capital": 15000, "slot": 5000, "max_positions": 3, "lot_over_slot": True, "max_loss_pct": 10,
       "one_position_per_stock": True, "priority": ["S42", "S46", "S48", "P42"], "fixed_lots": {"LKOH": 1}, "trial_lots": 1}
LOTS = {"SBER": 10, "GAZP": 10, "LKOH": 1}
AT = pd.Timestamp("2026-11-06 11:01")


def sig(code="P42", paper="SBER", price=300.0):
    return {"стратегия": code, "бумага": paper, "вход_в": AT, "цена": price, "стоп": price * 0.985,
            "срок": AT + pd.Timedelta(hours=240), "активация": price * 1.01, "отступ": 4.0}


def test_trial_takes_exactly_trial_lots_not_slot():
    take, alt = la.allocate([sig()], [], CFG, LOTS)
    assert not alt and take[0]["лоты"] == 1 and take[0]["штук"] == 10 and take[0]["сумма"] == 3000.0
    take, _ = la.allocate([sig()], [], {**CFG, "trial_lots": 2}, LOTS)
    assert take[0]["лоты"] == 2


def test_trial_respects_one_position_per_stock_and_money():
    held = [{**sig("S42"), "лоты": 1, "сумма": 3000.0}]
    take, alt = la.allocate([sig()], held, CFG, LOTS)
    assert not take and alt[0]["причина"] == "бумага уже в позиции"
    take, alt = la.allocate([sig(price=20000.0)], [], CFG, LOTS)
    assert not take and alt[0]["причина"].startswith("не хватает денег на пробу")


def test_trial_signals_and_exits_use_own_state():
    state = {"решения": {"sber": {"решение": "открыть", "вход_в": str(AT), "цена": 300.0, "стоп": 295.6, "срок_часов": 240,
                                  "активация": 303.0, "отступ": 4.4}}}
    s, pos = la.s42_signals(state, ["sber"], la.TRIAL)
    assert s[0]["стратегия"] == "P42" and s[0]["бумага"] == "SBER"
    p = {**s[0], "штук": 10, "лоты": 1, "сумма": 3000.0, "стоп стратегии": 295.6}
    still, closed = la.exits([p], {}, {}, {"SBER": {"открыта": True, "вход": AT, "стоп": 297.0}})
    assert still and still[0]["стоп стратегии"] == 297.0 and not closed
    still, closed = la.exits([p], {}, {}, {})
    assert not still and closed[0]["причина"] == "позиция стратегии не найдена"


def test_signal_before_entry_minute_survives_second_pass():
    """06.11: «ОТКРЫТЬ» в HH:00 с входом стратегии в HH:01 — у решения есть «вход_в», но нет «вход». Второй проход
    счёта в том же часе (после S46/S48) не должен снимать сигнал как «сделка стратегии уже закрылась»."""
    state = {"решения": {"sber": {"решение": "открыть", "вход_в": str(AT), "цена": 293.87, "стоп": 287.05, "срок_часов": 2,
                                  "активация": 297.28, "отступ": 6.82}}}
    for code in ("S42", la.TRIAL):
        s, pos = la.s42_signals(state, ["sber"], code)
        p = {**s[0], "штук": 1, "лоты": 1, "сумма": 293.87, "стоп стратегии": 287.05, "статус": "к исполнению"}
        still, closed = la.exits([p], pos if code == "S42" else {}, {}, pos if code != "S42" else {})
        assert still and not closed, code


def test_trial_message_is_marked():
    take, _ = la.allocate([sig()], [], CFG, LOTS)
    text = la.msg_open(take[0], {**CFG, "mode": "button", "trial_impulse_atr": 1.5})
    assert text.startswith("🧪 ПРОБА — ОТКРЫТЬ P42 · SBER") and "1.5 ATR" in text and "/close SBER" in text
    take, _ = la.allocate([sig("S42")], [], CFG, LOTS)
    assert la.msg_open(take[0]).startswith("🟢 ОТКРЫТЬ S42")


def test_trial_args_and_bot_toggle():
    assert ta.trial_args({"strategies": {"S42": ["sber"], "P42": False}}) == ""
    assert ta.trial_args({"strategies": {"P42": ["sber", "gazp"]}, "trial_impulse_atr": 1.5}) == "--only sber gazp --impulse-atr 1.5"
    assert ta.trial_args({"strategies": {"P42": ["sber"]}, "trial_gate": False}).endswith("--no-gate")
    text = "strategies:\n  S42: [sber, gazp, lkoh]\n  P42: off                # проба\n"
    on = yaml.safe_load(lb.set_strategy(text, "P42", True))
    assert ta.trial_args(on).startswith("--only sber gazp lkoh --impulse-atr 1.5")
    assert lb.parse_setting("trial_impulse_atr", "1,5", {}) == ("1.5", 1.5)
    assert lb.parse_setting("trial_gate", "off", {}) == ("false", False)


def test_trailing_exit_loss_cap():
    """H71: потолок убытка — стоп не дальше max_loss_pct % от входа; дальше 6 ATR — ничего не меняет."""
    low = np.array([99.0, 96.5, 90.0])
    high = np.array([100.5, 100.0, 99.0])
    close = np.array([100.0, 97.0, 91.0])
    kw = dict(side="buy", entry_price=100.0, atr=2.0, sl_atr=6.0, trail_atr=6.0, activate_atr=3.0)
    assert trailing_exit(high, low, close, **kw)[2] == "timeout"          # стоп 6 ATR = 88, минимум 90 — не задет
    j, px, why = trailing_exit(high, low, close, max_loss_pct=3.0, **kw)
    assert (j, px, why) == (1, 97.0, "stop_loss")
    j, px, why = trailing_exit(high, low, close, max_loss_pct=20.0, **kw)
    assert why == "timeout"
