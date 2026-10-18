"""Решающая часть исполнителя живого контура (scripts/live_trade.py: plan) — без обращения к брокеру."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_trade import plan  # noqa: E402

NOW = pd.Timestamp("2026-10-20 12:02")
OPEN = {"решение": "открыть", "сторона": 1, "вход_в": pd.Timestamp("2026-10-20 12:01"), "цена": 280.0, "atr": 1.0,
        "стоп": 274.0, "активация": 283.0, "отступ": 6.0, "срок_часов": 240.0, "стоп_atr": 6, "активация_atr": 3}
REC = {"lots": 3, "entry": 280.0, "stop": 274.0, "activation": 283.0, "indent": 6.0, "deadline": "2026-10-30 12:02"}
LOTS = {"sber": 10}


def run(dec, held=None, state=None, stops=None, capital=10_000, now=NOW, price=280.0):
    return plan({"sber": dec}, held or {}, state or {}, stops or {}, {"sber": price}, LOTS, capital, now)


def test_open_signal_without_position_buys_on_capital():
    (a,) = run(OPEN)
    assert a.kind == "buy" and a.lots == 3            # 10 000 / (280 × 10) = 3 лота
    assert a.extra["стоп_atr"] == 6 and a.extra["активация_atr"] == 3 and a.extra["отступ"] == 6.0


def test_short_signal_is_not_traded():
    (a,) = run({**OPEN, "сторона": -1})
    assert a.kind == "alert"


def test_not_enough_capital_alerts():
    (a,) = run(OPEN, capital=1_000)
    assert a.kind == "alert"


def test_position_closed_by_broker_cancels_remaining_stop():
    (a,) = run({"решение": "закрыто"}, held={"sber": 0}, state={"sber": REC}, stops={"sber": 1})
    assert a.kind == "cancel_stops"


def test_deadline_sells():
    (a,) = run({"решение": "держать"}, held={"sber": 3}, state={"sber": REC}, stops={"sber": 2},
               now=pd.Timestamp("2026-10-30 12:05"))
    assert a.kind == "sell" and a.lots == 3


def test_missing_stops_are_restored():
    (a,) = run({"решение": "держать"}, held={"sber": 3}, state={"sber": REC}, stops={"sber": 0})
    assert a.kind == "place_stops" and a.stop == 274.0 and a.activation == 283.0


def test_unknown_broker_position_is_not_touched():
    (a,) = run(OPEN, held={"sber": 5})
    assert a.kind == "alert"


def test_missed_entry_is_not_chased():
    (a,) = run({"решение": "держать", "сторона": 1})
    assert a.kind == "alert"


def test_nothing_to_do():
    assert run({"решение": "ничего"}) == []


class FakeBroker:
    """Подставной брокер: покупка исполняется по 281 (а не по цене сигнала 280), стоп-заявки запоминаются."""

    def __init__(self):
        self.calls = []

    def market(self, acc, uid, lots, buy):
        self.calls.append(("market", lots, buy))
        return {"executedOrderPrice": {"units": "281", "nano": 0}, "executedCommission": {"units": "0", "nano": 420000000}}

    def stop_loss(self, acc, share, lots, stop):
        self.calls.append(("stop_loss", lots, round(stop, 6)))
        return "sl-1"

    def trailing(self, acc, share, lots, activation, indent):
        self.calls.append(("trailing", lots, round(activation, 6), round(indent, 6)))
        return "tr-1"

    def stop_orders(self, acc):
        return []


def test_execute_places_stops_from_actual_entry(tmp_path, monkeypatch):
    import live_trade

    monkeypatch.setattr(live_trade, "ORDERS", tmp_path / "orders.csv")
    monkeypatch.setattr(live_trade, "OUT", tmp_path)
    share = type("S", (), {"uid": "u-sber", "lot": 10})()
    state, broker = {}, FakeBroker()
    live_trade.execute(broker, "acc", {"sber": share}, run(OPEN), state, send=True, now=NOW)
    assert broker.calls == [("market", 3, True), ("stop_loss", 3, 275.0), ("trailing", 3, 284.0, 6.0)]
    assert state["sber"]["entry"] == 281.0 and state["sber"]["deadline"] == str(NOW + pd.Timedelta(hours=240))
    assert (tmp_path / "orders.csv").exists()


def test_dry_run_sends_nothing():
    import live_trade

    broker = FakeBroker()
    lines = live_trade.execute(broker, "acc", {"sber": object()}, run(OPEN), {}, send=False, now=NOW)
    assert broker.calls == [] and lines and lines[0].startswith("было бы")
