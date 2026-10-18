"""Исполнитель живого контура в Т-Инвестициях (задача T30): решения scripts/live_signals.py → заявки брокера.

Первый этап запуска — GAZP, SBER, LKOH на бирже только в лонг (docs/results/launch_plan.md). Каждый запуск
(по cron, после свежих минут) сверяет три вещи — решение контура, позицию у брокера и записанное состояние — и:
  * «открыть», позиции нет → купить по рынку на капитал бумаги, сразу выставить у брокера стоп-лосс (вход − 6 ATR)
    и трейлинг-стоп (активация вход + 3 ATR, отступ 6 ATR) — от фактической цены входа, ATR — из сигнала;
  * позиция есть, вышел срок 240 ч → снять стоп-заявки, продать по рынку;
  * позиция записана, а у брокера её нет → закрыта стопом или трейлингом брокера: снять оставшуюся стоп-заявку;
  * позиция есть, а стоп-заявок нет → выставить заново;
  * любое другое расхождение (у брокера позиция, которой нет в состоянии, и т. п.) — не трогать, тревога для разбора.
Решающая часть — функция plan() без обращения к брокеру (её проверяют тесты tests/test_live_trade.py).

Режимы: по умолчанию песочница и --dry (только показать, что было бы сделано); --send — отправлять заявки;
--live — боевой счёт, только вместе с --account и --i-understand. Состояние — reports/live/broker_state.json,
журнал заявок — reports/live/orders.csv.

    python scripts/live_trade.py                         # песочница, только показать
    python scripts/live_trade.py --send                  # песочница, с заявками
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live"
STATE = OUT / "broker_state.json"
ORDERS = OUT / "orders.csv"
STOCKS = ["sber", "gazp", "lkoh"]


@dataclass
class Action:
    kind: str                     # buy | sell | cancel_stops | place_stops | alert
    inst: str
    lots: int = 0
    stop: float = 0.0
    activation: float = 0.0
    indent: float = 0.0
    why: str = ""
    extra: dict = field(default_factory=dict)


def plan(decisions: dict[str, dict], held: dict[str, int], state: dict, stops: dict[str, int], prices: dict[str, float],
         lots_of: dict[str, int], capital: float, now: pd.Timestamp) -> list[Action]:
    """Что сделать. decisions — инструмент → машинные поля решения (`_м` из live_signals); held — бумаг у брокера;
    state — записанные позиции; stops — активных стоп-заявок у брокера; prices — последняя цена; lots_of — лотность."""
    acts = []
    for inst, d in decisions.items():
        have, rec, n_stops = held.get(inst, 0), state.get(inst), stops.get(inst, 0)
        if rec and have == 0:                                   # закрыта стопом / трейлингом брокера
            acts.append(Action("cancel_stops" if n_stops else "alert", inst, why="позиция закрыта у брокера (стоп или трейлинг)",
                               extra={"closed": True}))
            continue
        if have and not rec:
            acts.append(Action("alert", inst, why=f"у брокера {have} бумаг, которых нет в состоянии исполнителя — разобрать вручную"))
            continue
        if rec:
            if now >= pd.Timestamp(rec["deadline"]):
                acts.append(Action("sell", inst, lots=rec["lots"], why=f"вышел срок {rec['deadline']}"))
            elif n_stops == 0:
                acts.append(Action("place_stops", inst, lots=rec["lots"], stop=rec["stop"], activation=rec["activation"],
                                   indent=rec["indent"], why="стоп-заявок нет — выставить заново"))
            continue
        if d.get("решение") == "открыть":
            if d.get("сторона", 1) < 0:
                acts.append(Action("alert", inst, why="сигнал в шорт — на бирже торгуем только лонг"))
                continue
            px = prices.get(inst) or d["цена"]
            lots = int(capital // (px * lots_of[inst]))
            if lots < 1:
                acts.append(Action("alert", inst, why=f"капитала {capital:.0f} не хватает на 1 лот по {px}"))
                continue
            acts.append(Action("buy", inst, lots=lots, why=f"сигнал: вход в {d['вход_в']}", extra={
                "atr": d["atr"], "стоп_atr": d["стоп_atr"], "активация_atr": d["активация_atr"], "отступ": d["отступ"],
                "срок_часов": d["срок_часов"]}))
        elif d.get("решение") == "держать":
            acts.append(Action("alert", inst, why="контур держит позицию, которой у исполнителя нет (вход пропущен) — не догоняем"))
    return acts


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(s: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def log_order(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(ORDERS, mode="a", header=not ORDERS.exists(), index=False)


def execute(api, acc: str, shares: dict, acts: list[Action], state: dict, send: bool, now: pd.Timestamp) -> list[str]:
    from forexmodel.broker.tinvest import q2d

    out = []
    for a in acts:
        sh = shares.get(a.inst)
        line = f"[{a.inst}] {a.kind}: {a.why}"
        if a.kind == "alert" or not send:
            out.append(("ТРЕВОГА " if a.kind == "alert" else "было бы: ") + line + (f", лотов {a.lots}" if a.lots else ""))
            if a.kind == "alert" and a.extra.get("closed"):
                state.pop(a.inst, None)
            continue
        if a.kind == "buy":
            r = api.market(acc, sh.uid, a.lots, buy=True)
            entry = float(q2d(r.get("executedOrderPrice"))) or float(api.last_price(sh.uid))
            atr = a.extra["atr"]
            stop, act_px, indent = entry - a.extra["стоп_atr"] * atr, entry + a.extra["активация_atr"] * atr, a.extra["отступ"]
            sl = api.stop_loss(acc, sh, a.lots, stop)
            tr = api.trailing(acc, sh, a.lots, act_px, indent)
            state[a.inst] = {"lots": a.lots, "entry": entry, "entry_time": str(now), "atr": atr, "stop": stop, "activation": act_px,
                             "indent": indent, "deadline": str(now + pd.Timedelta(hours=a.extra["срок_часов"])), "stop_ids": [sl, tr]}
            log_order({"время": now, "инструмент": a.inst, "действие": "купить", "лотов": a.lots, "цена": entry,
                       "комиссия": float(q2d(r.get("executedCommission"))), "стоп": round(stop, 4), "активация": round(act_px, 4), "отступ": round(indent, 4)})
            out.append(f"{line} → куплено {a.lots} лотов по {entry}, стоп {stop:.2f}, трейлинг с {act_px:.2f}")
        elif a.kind == "sell":
            for o in api.stop_orders(acc):
                if o.get("instrumentUid") == sh.uid:
                    api.cancel_stop(acc, o["stopOrderId"])
            r = api.market(acc, sh.uid, a.lots, buy=False)
            px = float(q2d(r.get("executedOrderPrice")))
            log_order({"время": now, "инструмент": a.inst, "действие": "продать (срок)", "лотов": a.lots, "цена": px,
                       "комиссия": float(q2d(r.get("executedCommission")))})
            state.pop(a.inst, None)
            out.append(f"{line} → продано по {px}")
        elif a.kind == "cancel_stops":
            for o in api.stop_orders(acc):
                if o.get("instrumentUid") == sh.uid:
                    api.cancel_stop(acc, o["stopOrderId"])
            log_order({"время": now, "инструмент": a.inst, "действие": "закрыта брокером; снята оставшаяся стоп-заявка"})
            state.pop(a.inst, None)
            out.append(f"{line} → снята оставшаяся стоп-заявка")
        elif a.kind == "place_stops":
            sl = api.stop_loss(acc, sh, a.lots, a.stop)
            tr = api.trailing(acc, sh, a.lots, a.activation, a.indent)
            state[a.inst]["stop_ids"] = [sl, tr]
            out.append(f"{line} → выставлены заново")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--send", action="store_true", help="отправлять заявки (без флага — только показать)")
    ap.add_argument("--live", action="store_true", help="боевой счёт (по умолчанию песочница)")
    ap.add_argument("--account", default=None, help="номер счёта (обязателен для --live)")
    ap.add_argument("--i-understand", action="store_true", help="подтверждение для --live --send")
    ap.add_argument("--capital", type=float, default=100_000, help="капитал на одну бумагу, рублей")
    ap.add_argument("--now", default=None, help="момент данных (для проверки на истории)")
    a = ap.parse_args(argv)
    if a.live and a.send and not (a.account and a.i_understand):
        print("боевой счёт с заявками: нужны --account и --i-understand")
        return 2
    from forexmodel.broker.tinvest import TInvest
    from live_signals import decisions as get_decisions

    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    api = TInvest(sandbox=not a.live)
    acc = a.account or (api.accounts()[0]["id"] if api.accounts() else api.open_sandbox_account())
    tick = {i: live["instruments"][i]["ticker"] for i in STOCKS}
    shares = {i: api.share(t) for i, t in tick.items()}
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now().floor("min")
    dec = {r["инструмент"]: r["_м"] for r in get_decisions(STOCKS, pd.Timestamp(a.now) if a.now else None)}
    pos = api.positions(acc)
    held = {i: pos.get(s.uid, 0) // s.lot for i, s in shares.items()}
    so = api.stop_orders(acc)
    stops = {i: sum(1 for o in so if o.get("instrumentUid") == s.uid) for i, s in shares.items()}
    prices = {i: float(api.last_price(s.uid)) for i, s in shares.items()}
    state = load_state()
    acts = plan(dec, held, state, stops, prices, {i: s.lot for i, s in shares.items()}, a.capital, now)
    lines = execute(api, acc, shares, acts, state, a.send, now)
    if a.send:
        save_state(state)
    print(f"{'БОЕВОЙ СЧЁТ' if a.live else 'песочница'}, {'заявки отправляются' if a.send else 'только показать'}; {now}")
    for i in STOCKS:
        print(f"  {i}: решение «{dec.get(i, {}).get('решение', '—')}», у брокера {held.get(i, 0)} лотов, стоп-заявок {stops.get(i, 0)}")
    print("\n".join(lines) if lines else "  действий нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
