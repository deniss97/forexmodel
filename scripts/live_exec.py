"""Исполнение сделок счёта через API Т-Инвестиций — по кнопке «Исполнить» в боте (scripts/live_bot.py), ТЗ инференса п. 2.

open_position — купить позицию, которую контур взял на счёт (reports/live/account_state.json):
  1. цена сейчас не выше цены сигнала + допуск проскальзывания (max_slippage_pct) — иначе ждём и пробуем снова;
  2. денег на счёте хватает на остаток позиции по предельной цене;
  3. лимитная заявка на покупку не дороже этой цены; не исполнилась за order_wait_sec — снимаем и повторяем,
     всего order_attempts попыток; частичное исполнение — остаток снимается, берём сколько купили;
  4. у брокера: стоп-лосс по стопу позиции и, если в сигнале есть включение и отступ (S42), трейлинг-стоп.
close_position — снять стоп-заявки бумаги и продать по рынку (несколько попыток).
Обе функции работают с любым объектом с методами TInvest — тесты подставляют поддельный брокер (tests/test_live_exec.py).
Каждая заявка — строка в reports/live/orders.csv.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ROOT / "reports" / "live" / "orders.csv"
DONE = {"EXECUTION_REPORT_STATUS_FILL", "EXECUTION_REPORT_STATUS_REJECTED", "EXECUTION_REPORT_STATUS_CANCELLED"}


@dataclass
class Result:
    ok: bool
    lots: int = 0                  # исполнено лотов
    price: float = 0.0             # средняя цена одной бумаги
    commission: float = 0.0
    stop_ids: list = field(default_factory=list)
    log: list = field(default_factory=list)   # что происходило, по шагам — уходит в ответ бота
    why: str = ""                  # почему не исполнено или исполнено не полностью
    gone: bool = False             # закрывать нечего: бумаги на счёте нет


def num(q) -> float:
    """Quotation / MoneyValue брокера → число."""
    if not q:
        return 0.0
    return int(q.get("units", 0) or 0) + int(q.get("nano", 0) or 0) / 1e9


def log_order(row: dict) -> None:
    """Строка журнала заявок — по столбцам файла (у покупки и продажи разные поля; новые поля — файл переписывается)."""
    ORDERS.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if not ORDERS.exists():
        new.to_csv(ORDERS, index=False)
        return
    cols = pd.read_csv(ORDERS, nrows=0).columns.tolist()
    if set(new.columns) <= set(cols):
        new.reindex(columns=cols).to_csv(ORDERS, mode="a", header=False, index=False)
    else:
        pd.concat([pd.read_csv(ORDERS), new], ignore_index=True).to_csv(ORDERS, index=False)


def _wait_fill(api, acc: str, r: dict, wait: float, sleep) -> dict:
    """Ждём исполнения заявки до wait секунд; не исполнилась — снимаем. Возвращает последнее состояние заявки."""
    oid, waited = r.get("orderId"), 0.0
    while r.get("executionReportStatus") not in DONE and waited < wait:
        sleep(2)
        waited += 2
        r = api.order_state(acc, oid)
    if r.get("executionReportStatus") not in DONE:
        try:
            api.cancel_order(acc, oid)
        except Exception:                  # уже исполнилась или снята — состояние ниже скажет
            pass
        r = api.order_state(acc, oid)
    return r


def open_position(api, acc: str, share, p: dict, cfg: dict, sleep=time.sleep, now=None) -> Result:
    """Купить p["лоты"] лотов бумаги share по сигналу p (цена, стоп, у S42 — активация и отступ трейлинга)."""
    want = int(p["лоты"])
    cap = float(p["цена"]) * (1 + float(cfg.get("max_slippage_pct", 0.3)) / 100)
    attempts, wait = int(cfg.get("order_attempts", 3)), float(cfg.get("order_wait_sec", 20))
    got, cost, comm, log = 0, 0.0, 0.0, []
    for k in range(1, attempts + 1):
        last = float(api.last_price(share.uid))
        if last > cap:
            log.append(f"попытка {k}: цена {last:g} выше допуска {cap:.2f} (сигнал {p['цена']:g} + {cfg.get('max_slippage_pct', 0.3)}%)")
            if k < attempts:
                sleep(wait)
            continue
        need = (want - got) * share.lot * cap
        money = float(api.money(acc))
        if money < need:
            log.append(f"на счёте {money:,.0f} ₽, а на {want - got} лот. по {cap:.2f} нужно {need:,.0f} ₽".replace(",", " "))
            break
        r = _wait_fill(api, acc, api.limit(acc, share, want - got, cap, buy=True), wait, sleep)
        n = int(r.get("lotsExecuted", 0) or 0)
        if n:
            px = num(r.get("executedOrderPrice")) or num(r.get("averagePositionPrice")) or last
            got, cost, comm = got + n, cost + n * px, comm + num(r.get("executedCommission"))
            log_order({"время": now or pd.Timestamp.now(), "бумага": share.ticker, "действие": "купить (лимит)", "лотов": n, "цена": px,
                       "лимит": round(cap, 4), "комиссия": num(r.get("executedCommission")), "заявка": r.get("orderId")})
        log.append(f"попытка {k}: лимит {cap:.2f}, исполнено {n} из {want - got + n} лот.")
        if got >= want:
            break
    if not got:
        return Result(False, log=log, why=log[-1] if log else "заявка не исполнилась")
    res = Result(True, got, cost / got, comm, log=log, why="" if got == want else f"исполнено {got} из {want} лот.")
    try:                                   # стопы у брокера — на исполненное количество
        res.stop_ids.append(api.stop_loss(acc, share, got, float(p["стоп"])))
        if p.get("активация") and p.get("отступ"):
            res.stop_ids.append(api.trailing(acc, share, got, float(p["активация"]), float(p["отступ"])))
        log.append(f"у брокера: стоп-лосс {float(p['стоп']):.2f}" + (f", трейлинг с {float(p['активация']):.2f}, отступ {float(p['отступ']):.2f}"
                                                                   if len(res.stop_ids) > 1 else ""))
    except Exception as e:                 # позиция куплена, а стопа нет — главное, чтобы об этом узнали
        res.why = (res.why + "; " if res.why else "") + f"СТОП-ЗАЯВКИ НЕ ВЫСТАВЛЕНЫ: {str(e)[:150]} — поставьте стоп вручную"
        log.append(res.why)
    return res


def close_position(api, acc: str, share, pieces: int, attempts: int = 3, sleep=time.sleep, now=None, why: str = "закрыть") -> Result:
    """Снять стоп-заявки бумаги и продать по рынку не больше pieces штук (сколько есть на счёте)."""
    log = []
    for o in api.stop_orders(acc):
        if o.get("instrumentUid") == share.uid:
            api.cancel_stop(acc, o["stopOrderId"])
            log.append("снята стоп-заявка")
    held = api.positions(acc).get(share.uid, 0)
    lots = min(held, int(pieces)) // share.lot
    if lots <= 0:
        return Result(False, log=log, why=f"на счёте нет {share.ticker} — позиция уже закрыта (стоп или вручную)", gone=True)
    for k in range(1, attempts + 1):
        try:
            r = api.market(acc, share.uid, lots, buy=False)
        except Exception as e:
            log.append(f"попытка {k}: {str(e)[:120]}")
            sleep(3 * k)
            continue
        px = num(r.get("executedOrderPrice"))
        log_order({"время": now or pd.Timestamp.now(), "бумага": share.ticker, "действие": f"продать (рынок, {why})", "лотов": lots,
                   "цена": px, "комиссия": num(r.get("executedCommission")), "заявка": r.get("orderId")})
        log.append(f"продано {lots} лот. по {px:g}")
        return Result(True, lots, px, num(r.get("executedCommission")), log=log)
    return Result(False, log=log, why="продажа не прошла: " + (log[-1] if log else ""))
