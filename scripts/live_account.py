"""Счёт по сигналам контура (ТЗ инференса 26.10 и условия Дениса): какие сигналы берём на счёт, что держим, когда
закрываем, и уведомления — сделки на первом этапе делаем руками.

Каждый час после сигналов (S42 — scripts/live_signals.py → reports/live/s42_state.json; S46, S48 —
scripts/live_iterator.py → reports/live/iterator_state.json):
  1. выходы: позиция счёта закрывается, когда закрылась сделка стратегии (стоп, трейлинг, сигнал выхода, срок) —
     уведомление «ЗАКРЫТЬ» (стоп и трейлинг стоят у брокера — «проверить, сработал ли»);
  2. новые сигналы — по настройкам configs/account.yaml: слот (5 тыс. ₽ на сделку, вниз до целых лотов; лот дороже
     слота — один лот, если свободных денег хватает), не больше max_positions позиций, одна позиция на бумагу,
     порядок — по времени входа, при равенстве — priority. Стоп — стоп стратегии, но не дальше −max_loss_pct% от входа.
     Взятый сигнал — уведомление «ОТКРЫТЬ»; не взятый — строка в журнале альтернативных сделок с причиной
     (reports/live/alt_journal.csv), а когда сделка стратегии закроется — её результат там же: сколько упустили;
  3. стоп-кран: configs/account.yaml stop: true или файл reports/live/STOP — новые сделки не открываются; стратегия,
     остановленная сверкой с симуляцией (reports/live/strategy_stop.json, scripts/live_deviation.py), — тоже;
  4. данные бумаги не обновлялись больше 2 часов в торговое время — уведомление (раз в день на бумагу);
  5. позиции у брокера (только чтение, если есть токен): взятый сигнал, которого через 30 минут нет на счёте, —
     уведомление «сделка не открыта»; бумаги на счёте, которых нет у контура, — тоже.
Состояние — reports/live/account_state.json, история сделок счёта — reports/live/account_trades.csv.

    python scripts/live_account.py              # каждый час — scripts/queues/live_hourly.sh
    python scripts/live_account.py --no-broker  # без обращения к брокеру
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import yaml

import notify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live"
CFG = ROOT / "configs" / "account.yaml"
STATE = OUT / "account_state.json"
TRADES = OUT / "account_trades.csv"
ALT = OUT / "alt_journal.csv"
STOP_FILE = OUT / "STOP"
STRATEGY_STOP = OUT / "strategy_stop.json"
REASON = {"stop_loss": "стоп-лосс", "trailing_stop": "трейлинг-стоп", "timeout": "по сроку", "signal": "сигнал выхода стратегии",
          "take_profit": "тейк"}
STOPS_AT_BROKER = {"stop_loss", "trailing_stop"}   # эти выходы исполняет стоп-заявка у брокера — проверить, сработала ли


def ts(x) -> pd.Timestamp | None:
    return None if x in (None, "", "NaT", "None") or (isinstance(x, float) and pd.isna(x)) else pd.Timestamp(x)


# ---------------------------------------------------------------- сигналы и позиции стратегий

def s42_signals(state: dict, stocks: list[str]) -> tuple[list[dict], dict]:
    """Решения S42 → свежие сигналы «открыть» и состояние позиций стратегии по бумагам."""
    sig, pos = [], {}
    for inst, m in (state.get("решения") or {}).items():
        if inst not in stocks or not m:
            continue
        r = m.get("решение")
        if r == "открыть":
            at = ts(m.get("вход_в"))
            srok = ts(m.get("срок")) or at + pd.Timedelta(hours=float(m.get("срок_часов", 240)))
            sig.append({"стратегия": "S42", "бумага": inst.upper(), "вход_в": at, "цена": float(m["цена"]), "стоп": float(m["стоп"]),
                        "срок": srok, "активация": m.get("активация"), "отступ": m.get("отступ")})
        if r in ("держать", "открыть") and m.get("вход") is not None:
            pos[inst.upper()] = {"открыта": True, "вход": ts(m["вход"]), "стоп": m.get("стоп")}
        elif r == "закрыто":
            pos[inst.upper()] = {"открыта": False, "вход": ts(m["вход"]), "выход": ts(m["выход"]), "причина": m.get("причина"),
                                 "цена_выхода": m.get("цена_выхода")}
    return sig, pos


def iterator_signals(state: dict, allowed: dict) -> tuple[list[dict], dict]:
    """Сигналы S46, S48 и их позиции: ключ (стратегия, бумага, вход)."""
    sig = [{**s, "вход_в": ts(s["вход_в"]), "срок": ts(s["срок"])} for s in state.get("сигналы", [])
           if s["бумага"].lower() in allowed.get(s["стратегия"], [])]
    pos = {}
    for p in state.get("позиции", []):
        pos[(p["стратегия"], p["бумага"], ts(p["open_dt"]))] = {"открыта": True, "стоп": p["стоп"]}
    for p in state.get("закрыто", []):
        pos[(p["стратегия"], p["бумага"], ts(p["open_dt"]))] = {"открыта": False, "выход": ts(p["close_dt"]),
                                                                "причина": p["exit_reason"], "цена_выхода": p["exit_price"]}
    return sig, pos


# ---------------------------------------------------------------- решения счёта (без брокера — тесты)

def allocate(signals: list[dict], positions: list[dict], cfg: dict, lots: dict, stop_on: bool = False,
             disabled: set | frozenset = frozenset()) -> tuple[list[dict], list[dict]]:
    """Какие сигналы берём на счёт: (взятые с лотами, суммой и стопом; не взятые — с причиной)."""
    cash = cfg["capital"] - sum(p["сумма"] for p in positions)
    held = {p["бумага"] for p in positions}
    free = cfg["max_positions"] - len(positions)
    rank = {s: k for k, s in enumerate(cfg.get("priority", []))}
    take, alt = [], []
    for s in sorted(signals, key=lambda s: (s["вход_в"], rank.get(s["стратегия"], 99))):
        lot = int(lots.get(s["бумага"], 1))
        cost = lot * s["цена"]
        n, reason = 0, None
        if stop_on:
            reason = "стоп-кран"
        elif s["стратегия"] in disabled:
            reason = "стратегия остановлена сверкой с симуляцией"
        elif cfg.get("one_position_per_stock", True) and s["бумага"] in held:
            reason = "бумага уже в позиции"
        elif free <= 0:
            reason = "нет свободного слота"
        else:
            n = int(min(cfg["slot"], cash) // cost)
            if n == 0 and cfg.get("lot_over_slot", True) and cash >= cost:
                n = 1
            if n == 0:
                reason = f"не хватает денег на лот: лот {f(cost, 0)} ₽, свободно {f(cash, 0)} ₽"
        if reason:
            alt.append({**s, "причина": reason})
            continue
        floor = s["цена"] * (1 - cfg["max_loss_pct"] / 100)
        take.append({**s, "лоты": n, "лот": lot, "штук": n * lot, "сумма": n * cost, "стоп стратегии": s["стоп"],
                     "стоп": max(s["стоп"], floor)})
        cash -= n * cost
        free -= 1
        held.add(s["бумага"])
    return take, alt


def exits(positions: list[dict], s42: dict, it: dict) -> tuple[list[dict], list[dict]]:
    """Позиции счёта → (ещё открытые с текущим стопом стратегии, закрытые с причиной и ценой выхода)."""
    still, closed = [], []
    for p in positions:
        if p["стратегия"] == "S42":
            q = s42.get(p["бумага"])
            same = q is not None and q.get("вход") == p["вход_в"]
        else:
            q = it.get((p["стратегия"], p["бумага"], p["вход_в"]))
            same = q is not None
        if same and q["открыта"]:
            stop = q.get("стоп")
            still.append({**p, "стоп стратегии": float(stop) if stop is not None else p["стоп стратегии"]})
        elif same:
            closed.append({**p, "выход": q.get("выход"), "причина": q.get("причина"), "цена выхода": q.get("цена_выхода")})
        else:                                 # сделки стратегии больше нет (закрыта давно или данные пересчитаны)
            closed.append({**p, "выход": None, "причина": "позиция стратегии не найдена", "цена выхода": None})
    return still, closed


# ---------------------------------------------------------------- сообщения

def f(x, d=2) -> str:
    return "—" if x is None else f"{x:,.{d}f}".replace(",", " ")


def msg_open(p: dict) -> str:
    trail = (f"\nТрейлинг-стоп: включение {f(p['активация'])}, отступ {f(p['отступ'])} — выставить у брокера."
             if p.get("активация") else "")
    cap = "" if p["стоп"] == p["стоп стратегии"] else f" (стоп стратегии {f(p['стоп стратегии'])} дальше −10% — ставим потолок)"
    return (f"🟢 ОТКРЫТЬ {p['стратегия']} · {p['бумага']}\nКупить {p['штук']} шт. ({p['лоты']} лот.) по рынку ≈ {f(p['цена'])} ₽ "
            f"на {f(p['сумма'], 0)} ₽.\nСтоп-лосс: {f(p['стоп'])} ({(p['стоп'] / p['цена'] - 1) * 100:+.1f}%){cap}.{trail}\n"
            f"Вход стратегии {p['вход_в']:%d.%m %H:%M}; закрыть не позже {p['срок']:%d.%m %H:%M}.")


def msg_close(p: dict) -> str:
    why = REASON.get(p.get("причина"), p.get("причина") or "")
    px = p.get("цена выхода")
    res = f" → {f(px)} ({(px / p['цена'] - 1) * 100:+.2f}% до издержек)" if px else ""
    act = ("проверить, что стоп-заявка у брокера сработала; если нет — продать по рынку"
           if p.get("причина") in STOPS_AT_BROKER else "продать по рынку")
    return (f"🔴 ЗАКРЫТЬ {p['стратегия']} · {p['бумага']} — {why}\n{p['штук']} шт.: {act}.\nВход {f(p['цена'])}{res}.")


# ---------------------------------------------------------------- брокер (только чтение)

def account_id(api, want: str | None) -> str:
    """Счёт из настроек (номер или его конец), иначе первый счёт токена."""
    accs = [a["id"] for a in api.accounts()]
    return next((a for a in accs if want and str(a).endswith(str(want))), accs[0])


def broker_view(lots: dict, want: str | None = None) -> dict | None:
    """Бумаги на счёте (тикер → штук) и рубли; {"ошибка": …} — нет токена или сбой (контур работает и без этого)."""
    try:
        from forexmodel.broker.tinvest import TInvest

        api = TInvest(sandbox=False)
        acc = account_id(api, want)
        uids = json.loads((OUT / "uids.json").read_text()) if (OUT / "uids.json").exists() else {}
        for t in lots:
            if t not in uids:
                uids[t] = api.share(t).uid
        (OUT / "uids.json").write_text(json.dumps(uids, ensure_ascii=False, indent=1))
        held = api.positions(acc)
        back = {u: t for t, u in uids.items()}
        return {"бумаги": {back[u]: q for u, q in held.items() if u in back and q}, "рубли": float(api.money(acc))}
    except Exception as e:                    # нет токена, сеть, лимит — без брокера
        return {"ошибка": str(e)[:200]}


# ---------------------------------------------------------------- запуск

def load_state() -> dict:
    if not STATE.exists():
        return {"позиции": [], "алерты": {}, "стоп_кран": False}
    s = json.loads(STATE.read_text(encoding="utf-8"))
    for p in s["позиции"]:
        for k in ("вход_в", "срок", "назначено"):
            p[k] = ts(p.get(k))
    return s


def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-broker", action="store_true")
    a = ap.parse_args(argv)
    now = pd.Timestamp.now("Europe/Moscow").tz_localize(None).floor("min")
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    lots = {k: v["lot"] for k, v in json.loads((ROOT / cfg["lots_file"]).read_text()).items()}
    allowed = {code: [s.lower() for s in lots] if v == "all" else v for code, v in cfg["strategies"].items()}
    s42_state = json.loads((OUT / "s42_state.json").read_text(encoding="utf-8")) if (OUT / "s42_state.json").exists() else {}
    it_state = json.loads((OUT / "iterator_state.json").read_text(encoding="utf-8")) if (OUT / "iterator_state.json").exists() else {}
    sig42, pos42 = s42_signals(s42_state, allowed.get("S42", []))
    sig_it, pos_it = iterator_signals(it_state, allowed)
    st = load_state()
    msgs = []

    # 1. выходы
    still, closed = exits(st["позиции"], pos42, pos_it)
    for p in closed:
        msgs.append(msg_close(p))
    append_csv(TRADES, [{**p, "закрыто": now} for p in closed])

    # 2. стоп-кран и новые сигналы
    stop_on = bool(cfg.get("stop")) or STOP_FILE.exists()
    if stop_on != st.get("стоп_кран", False):
        msgs.append("⛔ Стоп-кран включён: новые сделки не открываются. Открытые позиции: " + (", ".join(f"{p['стратегия']} {p['бумага']}" for p in still) or "нет")
                    if stop_on else "✅ Стоп-кран выключен: контур снова берёт сигналы.")
    disabled = set(json.loads(STRATEGY_STOP.read_text())) if STRATEGY_STOP.exists() else set()
    seen = {(p["стратегия"], p["бумага"], str(p["вход_в"])) for p in st["позиции"]}
    if ALT.exists():
        alt_old = pd.read_csv(ALT)
        seen |= set(zip(alt_old["стратегия"], alt_old["бумага"], alt_old["вход_в"].astype(str)))
    fresh = [s for s in sig42 + sig_it if (s["стратегия"], s["бумага"], str(s["вход_в"])) not in seen]
    take, alt = allocate(fresh, still, cfg, lots, stop_on, disabled)
    for p in take:
        p.update({"статус": "к исполнению", "назначено": now})
        msgs.append(msg_open(p))
    append_csv(ALT, [{**{k: s.get(k) for k in ("стратегия", "бумага", "вход_в", "цена", "стоп", "срок", "причина")}, "запуск": now}
                     for s in alt])

    # 3. позиции у брокера: взятые сигналы исполнены ли
    positions = still + take
    if not a.no_broker and cfg.get("mode") == "manual" and positions:
        b = broker_view(lots, cfg.get("account"))
        if b and "бумаги" in b:
            for p in positions:
                have = b["бумаги"].get(p["бумага"], 0)
                if p.get("статус") == "к исполнению":
                    if have >= p["штук"]:
                        p["статус"] = "открыта"
                    elif now - p["назначено"] > pd.Timedelta(minutes=30) and not p.get("напомнили"):
                        msgs.append(f"⚠️ Сделка не открыта: {p['стратегия']} {p['бумага']} — на счёте {have} шт. из {p['штук']} через 30 минут "
                                    f"после сигнала. Открыть по рынку или отменить: файл reports/live/account_state.json.")
                        p["напомнили"] = True
            extra = set(b["бумаги"]) - {p["бумага"] for p in positions}
            key = "лишние " + ",".join(sorted(extra))
            if extra and st["алерты"].get(key) != str(now.date()):
                msgs.append("⚠️ На счёте бумаги, которых нет у контура: " + ", ".join(f"{t} {b['бумаги'][t]} шт." for t in sorted(extra)))
                st["алерты"][key] = str(now.date())

    # 4. свежесть данных в торговое время (будни 07–23 ч по Москве)
    data = {**{k.upper(): ts(v.get("данные_до")) for k, v in (s42_state.get("решения") or {}).items() if v},
            **{k: ts(v["до"]) for k, v in (it_state.get("данные") or {}).items()}}
    if now.weekday() < 5 and 8 <= now.hour <= 23:
        stale = sorted(t for t, d in data.items() if d is not None and now - d > pd.Timedelta(hours=2)
                       and st["алерты"].get("данные " + t) != str(now.date()))
        if stale:
            msgs.append("⚠️ Данные не обновлялись больше 2 часов: " + ", ".join(f"{t} (до {data[t]:%d.%m %H:%M})" for t in stale))
            for t in stale:
                st["алерты"]["данные " + t] = str(now.date())

    # 5. сводка раз в день
    if now.hour == 23 and st.get("сводка") != str(now.date()):
        used = sum(p["сумма"] for p in positions)
        lines = [f"📋 Счёт на {now:%d.%m %H:%M}: в позициях {f(used, 0)} ₽ из {f(cfg['capital'], 0)} ₽, позиций {len(positions)} из {cfg['max_positions']}."]
        lines += [f"• {p['стратегия']} {p['бумага']}: {p['штук']} шт. с {p['вход_в']:%d.%m}, стоп {f(p['стоп'])}, до {p['срок']:%d.%m}" for p in positions]
        if ALT.exists():
            n = (pd.to_datetime(pd.read_csv(ALT)["запуск"]) >= now.normalize()).sum()
            lines.append(f"Альтернативных сигналов за день: {n} (reports/live/alt_journal.csv).")
        msgs.append("\n".join(lines))
        st["сводка"] = str(now.date())

    st.update({"позиции": positions, "стоп_кран": stop_on, "запуск": now})
    STATE.write_text(json.dumps(st, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    for m in msgs:
        notify.send(m)
        print(m, "\n", flush=True)
    print(f"[счёт] позиций {len(positions)}, взято {len(take)}, в альтернативные {len(alt)}, закрыто {len(closed)}, стоп-кран {'вкл' if stop_on else 'выкл'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
