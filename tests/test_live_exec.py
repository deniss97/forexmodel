"""Исполнение по кнопке «Исполнить» (scripts/live_exec.py) — на поддельном брокере, без сети."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_exec as lx  # noqa: E402

FILL, NEW = "EXECUTION_REPORT_STATUS_FILL", "EXECUTION_REPORT_STATUS_NEW"
CFG = {"max_slippage_pct": 0.3, "order_attempts": 3, "order_wait_sec": 4}
P = {"лоты": 3, "цена": 300.0, "стоп": 285.0, "активация": 309.0, "отступ": 18.0}


def q(x: float) -> dict:
    return {"units": int(x), "nano": round((x - int(x)) * 1e9)}


class Share:
    ticker, uid, lot = "SBER", "u1", 1


class FakeAPI:
    def __init__(self, prices=(300.5,), fills=(3,), money=100_000.0, held=None, stops=None, stop_fails=False):
        self.prices, self.fills, self.money_ = list(prices), list(fills), money
        self.held, self.stops, self.stop_fails = held or {}, list(stops or []), stop_fails
        self.orders, self.cancelled, self.placed, self.sold, self.state = [], [], [], [], {}

    def last_price(self, uid):
        return self.prices.pop(0) if len(self.prices) > 1 else self.prices[0]

    def money(self, acc):
        return self.money_

    def limit(self, acc, share, lots, price, buy):
        n, oid = self.fills.pop(0), f"o{len(self.orders)}"
        self.orders.append((lots, price))
        self.state[oid] = {"orderId": oid, "executionReportStatus": FILL if n == lots else NEW, "lotsExecuted": n,
                           "executedOrderPrice": q(price - 0.2)}
        return dict(self.state[oid])

    def order_state(self, acc, oid):
        return dict(self.state[oid])

    def cancel_order(self, acc, oid):
        self.cancelled.append(oid)
        self.state[oid]["executionReportStatus"] = "EXECUTION_REPORT_STATUS_CANCELLED"

    def stop_loss(self, acc, share, lots, stop):
        if self.stop_fails:
            raise RuntimeError("HTTP 400")
        self.placed.append(("стоп", lots, stop))
        return "sl1"

    def trailing(self, acc, share, lots, act, indent):
        self.placed.append(("трейлинг", lots, act, indent))
        return "tr1"

    def stop_orders(self, acc):
        return self.stops

    def cancel_stop(self, acc, sid):
        self.stops = [o for o in self.stops if o["stopOrderId"] != sid]
        self.cancelled.append(sid)

    def positions(self, acc):
        return self.held

    def market(self, acc, uid, lots, buy):
        self.sold.append(lots)
        return {"orderId": "m1", "executedOrderPrice": q(310.0), "executedCommission": q(0.5)}


@pytest.fixture(autouse=True)
def orders_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lx, "ORDERS", tmp_path / "orders.csv")


def nosleep(_):
    pass


def test_filled_at_once_places_stop_and_trailing_at_limit_price():
    api = FakeAPI()
    r = lx.open_position(api, "acc", Share(), P, CFG, sleep=nosleep)
    assert r.ok and r.lots == 3 and r.stop_ids == ["sl1", "tr1"]
    assert api.orders == [(3, pytest.approx(300.9))]                 # лимит = сигнал + 0.3%
    assert api.placed == [("стоп", 3, 285.0), ("трейлинг", 3, 309.0, 18.0)]


def test_not_filled_retries_and_cancels_each_order():
    api = FakeAPI(fills=(0, 0, 0))
    r = lx.open_position(api, "acc", Share(), P, CFG, sleep=nosleep)
    assert not r.ok and len(api.orders) == 3 and len(api.cancelled) == 3 and not api.placed


def test_partial_fill_buys_the_rest_on_next_attempt():
    api = FakeAPI(fills=(1, 2))
    r = lx.open_position(api, "acc", Share(), P, CFG, sleep=nosleep)
    assert r.ok and r.lots == 3 and [o[0] for o in api.orders] == [3, 2]
    assert api.placed[0] == ("стоп", 3, 285.0)


def test_price_above_slippage_limit_posts_nothing():
    api = FakeAPI(prices=(302.0,))
    r = lx.open_position(api, "acc", Share(), P, CFG, sleep=nosleep)
    assert not r.ok and not api.orders and "допуска" in r.why


def test_not_enough_money_posts_nothing():
    api = FakeAPI(money=500.0)
    r = lx.open_position(api, "acc", Share(), P, CFG, sleep=nosleep)
    assert not r.ok and not api.orders and "нужно" in r.why


def test_stop_order_failure_is_reported_but_position_kept():
    api = FakeAPI(stop_fails=True)
    r = lx.open_position(api, "acc", Share(), P, CFG, sleep=nosleep)
    assert r.ok and r.lots == 3 and "СТОП-ЗАЯВКИ НЕ ВЫСТАВЛЕНЫ" in r.why


def test_no_trailing_without_activation():
    api = FakeAPI()
    r = lx.open_position(api, "acc", Share(), {**P, "активация": None, "отступ": None}, CFG, sleep=nosleep)
    assert r.ok and r.stop_ids == ["sl1"]


def test_close_cancels_only_this_stock_stops_and_sells_held():
    api = FakeAPI(held={"u1": 3}, stops=[{"instrumentUid": "u1", "stopOrderId": "s1"}, {"instrumentUid": "u2", "stopOrderId": "s2"}])
    r = lx.close_position(api, "acc", Share(), 5, sleep=nosleep)
    assert r.ok and api.sold == [3] and r.price == 310.0
    assert [o["stopOrderId"] for o in api.stops] == ["s2"]


def test_close_when_nothing_held_is_gone():
    r = lx.close_position(FakeAPI(), "acc", Share(), 3, sleep=nosleep)
    assert not r.ok and r.gone
