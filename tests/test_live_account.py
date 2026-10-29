"""Решения счёта по сигналам (scripts/live_account.py: allocate, exits) — без брокера и файлов."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_account import allocate, exits  # noqa: E402

CFG = {"capital": 15000, "slot": 5000, "max_positions": 3, "lot_over_slot": True, "max_loss_pct": 10,
       "one_position_per_stock": True, "priority": ["S42", "S46", "S48"]}
LOTS = {"SBER": 1, "GAZP": 10, "LKOH": 1, "TATN": 1}
T = pd.Timestamp("2026-10-26 12:01")


def sig(strategy="S46", stock="SBER", price=290.0, stop=280.0, at=T):
    return {"стратегия": strategy, "бумага": stock, "вход_в": at, "цена": price, "стоп": stop, "срок": at + pd.Timedelta(hours=240)}


def pos(stock, amount, strategy="S46"):
    return {"стратегия": strategy, "бумага": stock, "вход_в": T - pd.Timedelta(days=1), "цена": 100.0, "сумма": amount,
            "стоп стратегии": 95.0, "стоп": 95.0, "штук": 1}


def test_slot_rounds_down_to_whole_lots():
    (t,), alt = allocate([sig("S46", "GAZP", price=114.75, stop=110)], [], CFG, LOTS)
    assert t["лоты"] == 4 and t["штук"] == 40 and abs(t["сумма"] - 4590) < 1e-6   # 5000 // 1147.5 = 4 лота по 10 шт.
    assert alt == []


def test_lot_over_slot_taken_when_cash_allows():
    (t,), _ = allocate([sig("S42", "LKOH", price=6144.5, stop=5800)], [], CFG, LOTS)
    assert t["лоты"] == 1 and t["сумма"] == 6144.5                # слот 5 тыс. меньше лота — берём 1 лот, денег 15 тыс.


def test_lot_over_slot_skipped_without_cash():
    take, (a,) = allocate([sig("S42", "LKOH", price=6144.5, stop=5800)], [pos("TATN", 5000), pos("SBER", 5000)], CFG, LOTS)
    assert take == [] and a["причина"].startswith("не хватает денег на лот")   # свободно 5 тыс. < 6.1 тыс.


def test_max_positions_and_one_per_stock():
    held = [pos("TATN", 4000), pos("GAZP", 4000)]
    signals = [sig("S46", "TATN"), sig("S46", "SBER"), sig("S48", "LKOH", price=6000, stop=5700)]   # S46 раньше S48
    take, alt = allocate(signals, held, CFG, LOTS)
    assert [t["бумага"] for t in take] == ["SBER"]
    reasons = {a["бумага"]: a["причина"] for a in alt}
    assert reasons["TATN"] == "бумага уже в позиции" and reasons["LKOH"] == "нет свободного слота"


def test_priority_and_time_order():
    later = sig("S42", "TATN", price=600, stop=580, at=T + pd.Timedelta(hours=1))
    take, _ = allocate([later, sig("S48", "SBER"), sig("S42", "GAZP", price=110, stop=100)], [pos("LKOH", 6000), pos("NVTK", 4000)], CFG, LOTS)
    assert [(t["стратегия"], t["бумага"]) for t in take] == [("S42", "GAZP")]   # раньше по времени, при равенстве — S42


def test_stop_capped_at_max_loss():
    (t,), _ = allocate([sig("S46", "SBER", price=300, stop=250)], [], CFG, LOTS)
    assert t["стоп"] == 270 and t["стоп стратегии"] == 250         # стоп стратегии −16.7% → потолок −10%
    (t,), _ = allocate([sig("S46", "SBER", price=300, stop=290)], [], CFG, LOTS)
    assert t["стоп"] == 290                                         # ближе потолка — стоп стратегии


def test_kill_switch_and_disabled_strategy():
    take, (a,) = allocate([sig()], [], CFG, LOTS, stop_on=True)
    assert take == [] and a["причина"] == "стоп-кран"
    take, (a,) = allocate([sig("S48")], [], CFG, LOTS, disabled={"S48"})
    assert take == [] and a["причина"].startswith("стратегия остановлена")


def test_exits_follow_strategy_trades():
    p42 = {**pos("SBER", 3000, "S42"), "вход_в": T}
    p46 = {**pos("TATN", 3000, "S46"), "вход_в": T}
    p48 = {**pos("NVTK", 3000, "S48"), "вход_в": T}
    s42 = {"SBER": {"открыта": False, "вход": T, "выход": T + pd.Timedelta(hours=5), "причина": "trailing_stop", "цена_выхода": 310.0}}
    it = {("S46", "TATN", T): {"открыта": True, "стоп": 96.5}}
    still, closed = exits([p42, p46, p48], s42, it)
    assert [p["бумага"] for p in still] == ["TATN"] and still[0]["стоп стратегии"] == 96.5
    by = {p["бумага"]: p for p in closed}
    assert by["SBER"]["причина"] == "trailing_stop" and by["SBER"]["цена выхода"] == 310.0
    assert by["NVTK"]["причина"] == "позиция стратегии не найдена"


# 29.10 (Денис): LKOH — всегда ровно 1 лот, деньги на него зарезервированы, пока его нет в позиции
FIX = {**CFG, "fixed_lots": {"LKOH": 1}, "strategies": {"S42": ["sber", "gazp", "lkoh"]}}
LOTS3 = {"SBER": 1, "GAZP": 10, "LKOH": 1}
PRICES = {"SBER": 298.0, "GAZP": 117.0, "LKOH": 6300.0}


def three():
    m = pd.Timedelta(minutes=1)
    return [sig("S42", "SBER", 298.0, 290.0, T), sig("S42", "GAZP", 117.0, 114.0, T + m), sig("S42", "LKOH", 6300.0, 6100.0, T + 2 * m)]


def test_without_fixed_lots_lkoh_is_squeezed_out():
    take, alt = allocate(three(), [], {**FIX, "fixed_lots": {}}, LOTS3, prices=PRICES)
    assert [p["бумага"] for p in take] == ["SBER", "GAZP"] and alt[0]["бумага"] == "LKOH"


def test_fixed_lkoh_always_one_lot_and_all_three_fit():
    take, alt = allocate(three(), [], FIX, LOTS3, prices=PRICES)
    assert not alt and [p["бумага"] for p in take] == ["SBER", "GAZP", "LKOH"]
    assert take[2]["лоты"] == 1 and take[2]["сумма"] == 6300.0
    assert take[1]["лоты"] == 3                                     # GAZP ужат резервом под LKOH: 3 лота, не 4
    assert sum(p["сумма"] for p in take) <= FIX["capital"]


def test_reserve_released_when_lkoh_held():
    held = [{**pos("LKOH", 6300.0, "S42"), "бумага": "LKOH"}]
    take, _ = allocate([sig("S42", "SBER", 298.0, 290.0)], held, FIX, LOTS3, prices=PRICES)
    assert take[0]["штук"] == 16                                    # полный слот 5 тыс.: резерв больше не нужен
