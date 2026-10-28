"""Счёт по сигналам контура (ТЗ инференса Тимура, решения 26.10 и 28.10): какие сигналы берём на счёт, что держим,
когда закрываем, и уведомления. Сделки — по кнопкам в группе (mode: button): «Исполнить» — бот покупает через API
(scripts/live_bot.py, live_exec.py), «Купил сам» — сделка руками в приложении; mode: manual — только уведомления.

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
     уведомление «сделка не открыта»; бумаги на счёте, которых нет у контура, — тоже;
  6. выходные (weekends: false, созвон 26.10): сигналы торгов выходного дня на счёт не берём — в журнал альтернативных;
  7. тревоги ТЗ: сигнал не взят из-за лимита (нет слота или денег) и повышенная волатильность — ход последнего часа
     в volatility_x раз больше обычного (медиана часов за 20 дней), раз в день на бумагу;
  8. пробный этап (28.10): количество бумаг у брокера против позиции счёта (бумаг нет — позиция закрывается в счёте);
     отсечка дивидендов по позиции за dividend_days дней (стоп у брокера сработает на разрыве, бэктест держал бы);
     просадка результата счёта от пика — «внимание» dd_warn_pct и «стоп» dd_stop_pct % депозита; мало места на диске.
Состояние — reports/live/account_state.json (меняет и бот — под общей блокировкой state_lock), история сделок счёта —
reports/live/account_trades.csv.

    python scripts/live_account.py              # каждый час — scripts/queues/live_hourly.sh
    python scripts/live_account.py --no-broker  # без обращения к брокеру
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import _bootstrap  # noqa: F401
import notify
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live"
CFG = ROOT / "configs" / "account.yaml"
STATE = OUT / "account_state.json"
TRADES = OUT / "account_trades.csv"
ALT = OUT / "alt_journal.csv"
STOP_FILE = OUT / "STOP"
STRATEGY_STOP = OUT / "strategy_stop.json"
LOCK = OUT / ".account.lock"
DISK_WARN_GB = 3.0                  # меньше — тревога раз в день (почасовая дописка минуток встанет при 1 ГБ, moex_data)
LEVELS = {"внимание": 1, "стоп": 2}  # пороги просадки счёта, по возрастанию
LIVE_FILES = {"SBER": "live/SBER_1m_live.csv", "GAZP": "live/GAZP_1m_live.csv", "LKOH": "live/LKOH_1m_live.csv"}  # остальные — moex/<T>_1m.csv
REASON = {"stop_loss": "стоп-лосс", "trailing_stop": "трейлинг-стоп", "timeout": "по сроку", "signal": "сигнал выхода стратегии",
          "take_profit": "тейк"}
STOPS_AT_BROKER = {"stop_loss", "trailing_stop"}   # эти выходы исполняет стоп-заявка у брокера — проверить, сработала ли


def ts(x) -> pd.Timestamp | None:
    return None if x in (None, "", "NaT", "None") or (isinstance(x, float) and pd.isna(x)) else pd.Timestamp(x)


@contextmanager
def state_lock(timeout: float = 180.0):
    """Общая блокировка состояния счёта: почасовой запуск и бот не пишут его одновременно."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as fh:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > timeout:
                    raise TimeoutError("состояние счёта занято другим процессом") from None
                time.sleep(0.5)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def pid(p: dict) -> str:
    """Короткий ключ позиции для кнопок: S42:SBER:2610231701 (стратегия, бумага, вход стратегии)."""
    return f"{p['стратегия']}:{p['бумага']}:{pd.Timestamp(p['вход_в']):%y%m%d%H%M}"


def allowed_stocks(cfg: dict, lots: dict) -> dict[str, list[str]]:
    """Бумаги стратегий на счёте: all — все из lots_file, список — эти, off / false / пусто — стратегия на счёт не берётся."""
    out = {}
    for code, v in (cfg.get("strategies") or {}).items():
        out[code] = [s.lower() for s in lots] if v == "all" else ([s.lower() for s in v] if isinstance(v, list) else [])
    return out


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
        elif not cfg.get("weekends", True) and pd.Timestamp(s["вход_в"]).weekday() >= 5:
            reason = "выходные: торги выходного дня на счёт не берём"
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


def msg_open(p: dict, cfg: dict | None = None) -> str:
    trail = (f"\nТрейлинг-стоп: включение {f(p['активация'])}, отступ {f(p['отступ'])} — выставить у брокера."
             if p.get("активация") else "")
    cap = "" if p["стоп"] == p["стоп стратегии"] else f" (стоп стратегии {f(p['стоп стратегии'])} дальше −10% — ставим потолок)"
    text = (f"🟢 ОТКРЫТЬ {p['стратегия']} · {p['бумага']}\nКупить {p['штук']} шт. ({p['лоты']} лот.) по рынку ≈ {f(p['цена'])} ₽ "
            f"на {f(p['сумма'], 0)} ₽.\nСтоп-лосс: {f(p['стоп'])} ({(p['стоп'] / p['цена'] - 1) * 100:+.1f}%){cap}.{trail}\n"
            f"Вход стратегии {p['вход_в']:%d.%m %H:%M}; закрыть не позже {p['срок']:%d.%m %H:%M}.")
    if cfg and cfg.get("mode") == "button":
        lim = p["цена"] * (1 + float(cfg.get("max_slippage_pct", 0.3)) / 100)
        text += (f"\n\n«Исполнить» — бот купит через API не дороже {f(lim)} ₽ (допуск {cfg.get('max_slippage_pct', 0.3)}%) и сам "
                 f"поставит стоп{' и трейлинг' if p.get('активация') else ''}. «Купил сам» — если покупаете в приложении.")
    return text


def open_buttons(p: dict) -> list[list[tuple[str, str]]]:
    k = pid(p)
    return [[("✅ Исполнить", "x:" + k), ("👌 Купил сам", "m:" + k)], [("⏭ Пропустить", "s:" + k)]]


def msg_close(p: dict) -> str:
    why = REASON.get(p.get("причина"), p.get("причина") or "")
    px = p.get("цена выхода")
    res = f" → {f(px)} ({(px / p['цена'] - 1) * 100:+.2f}% до издержек)" if px else ""
    act = ("проверить, что стоп-заявка у брокера сработала; если нет — продать по рынку"
           if p.get("причина") in STOPS_AT_BROKER else "продать по рынку")
    return (f"🔴 ЗАКРЫТЬ {p['стратегия']} · {p['бумага']} — {why}\n{p['штук']} шт.: {act}.\nВход {f(p['цена'])}{res}.")


def close_buttons(p: dict) -> list[list[tuple[str, str]]]:
    k = pid(p)
    return [[("🔴 Закрыть через API", "c:" + k), ("👌 Закрыл сам", "cm:" + k)]]


def num(q) -> float:
    """MoneyValue / Quotation брокера → число."""
    return int(q.get("units", 0) or 0) + int(q.get("nano", 0) or 0) / 1e9 if q else 0.0


def dividend_alert(p: dict, d: dict, now: pd.Timestamp, days: int = 3) -> str | None:
    """Отсечка по позиции счёта: последний день покупки в ближайшие days дней → тревога. В бэктесте цены с поправкой на
    дивиденды и разрыва нет, а стоп у брокера стоит на реальной цене и на разрыве в день отсечки сработает."""
    last_buy = pd.Timestamp(d["lastBuyDate"]).tz_convert(None) if pd.Timestamp(d["lastBuyDate"]).tzinfo else pd.Timestamp(d["lastBuyDate"])
    if not (now.normalize() <= last_buy <= now.normalize() + pd.Timedelta(days=days)):
        return None
    amount, price = num(d.get("dividendNet")), float(p.get("цена_входа") or p["цена"])
    stop = max(float(p["стоп"]), float(p.get("стоп стратегии") or 0))
    gap, room = amount / price * 100, (1 - stop / price) * 100
    head = (f"📅 Отсечка {p['бумага']} ({p['стратегия']}): последний день покупки {last_buy:%d.%m}, дивиденд {amount:g} ₽ ≈ "
            f"{gap:.1f}% цены. На следующий день цена откроется примерно на дивиденд ниже.")
    if gap < room:
        return head + f" Стоп {stop:g} ({room:.1f}% от входа) дальше разрыва — можно держать, дивиденд придёт."
    return (head + f" Стоп {stop:g} всего в {room:.1f}% от входа — на разрыве он сработает, хотя бэктест держал бы позицию. "
            f"Решите до {last_buy:%d.%m}: опустить стоп на {amount:g} ₽ (в приложении: снять и поставить заново) — "
            f"как в бэктесте, дивиденд придёт; или закрыть до отсечки — /close {p['бумага']}.")


def account_pnl(trades: pd.DataFrame | None, positions: list[dict], prices: dict) -> float:
    """Результат счёта в рублях: закрытые сделки счёта (вход → выход) и открытые позиции по последней цене. Пополнения и
    выводы денег не влияют — в отличие от стоимости портфеля у брокера."""
    total = 0.0
    if trades is not None and len(trades) and "цена выхода" in trades:
        t = trades[trades["цена выхода"].notna()]
        entry = t["цена_входа"].fillna(t["цена"]) if "цена_входа" in t else t["цена"]
        total += float(((t["цена выхода"].astype(float) - entry.astype(float)) * t["штук"].astype(float)).sum())
    for p in positions:
        if p.get("статус") == "открыта" and p["бумага"] in prices:
            total += (prices[p["бумага"]] - float(p.get("цена_входа") or p["цена"])) * float(p["штук"])
    return total


def drawdown_alert(pnl: float, st: dict, cfg: dict) -> str | None:
    """Просадка от пика результата в % депозита: «внимание» и «стоп» (разбор 28.10, раздел 7) — по одному разу, пока
    результат не обновит пик. Решение остановиться — за людьми: бот только сообщает."""
    peak = max(float(st.get("пик_результата", 0.0)), pnl)
    st["пик_результата"] = peak
    if pnl >= peak:
        st["просадка_уровень"] = None
        return None
    dd = (peak - pnl) / cfg["capital"] * 100
    hit = next((lv for lv, key in (("стоп", "dd_stop_pct"), ("внимание", "dd_warn_pct")) if dd >= float(cfg.get(key, 12 if lv == "стоп" else 8))), None)
    if not hit or LEVELS[hit] <= LEVELS.get(st.get("просадка_уровень") or "", 0):
        return None
    st["просадка_уровень"] = hit
    head = f"Просадка счёта −{dd:.1f}% депозита: результат {pnl:+,.0f} ₽ при пике {peak:+,.0f} ₽.".replace(",", " ")
    if hit == "внимание":
        return f"🟠 {head} Порог «внимание» (−{cfg.get('dd_warn_pct', 8)}%): проверить исполнение и данные — /status, /journal. Торгуем дальше."
    return (f"🔴 {head} Порог «стоп» (−{cfg.get('dd_stop_pct', 12)}%). По договорённости: /stop и разбор за 1–2 дня — исполнение и данные, "
            f"сверка с бэктестом в 23:00, рынок. Если всё исправно — /resume (разбор 28.10, раздел 7).")


def tail_minutes(path: Path, nbytes: int = 3_500_000) -> pd.DataFrame:
    """Хвост файла минуток (≈ 40 торговых дней) — без чтения всего файла."""
    with open(path, "rb") as fh:
        head = fh.readline().decode().strip().split(",")
        fh.seek(0, 2)
        fh.seek(max(fh.tell() - nbytes, 0))
        lines = fh.read().decode(errors="ignore").splitlines()[1:]
    rows = [x.split(",") for x in lines if x.count(",") == len(head) - 1 and not x.startswith(head[0] + ",")]
    d = pd.DataFrame(rows, columns=head)
    for c in ("high", "low", "close"):
        d[c] = d[c].astype(float)
    d["begin"] = pd.to_datetime(d["begin"])
    return d


def volatility(d: pd.DataFrame, now: pd.Timestamp, days: int = 20) -> tuple[float, float, pd.Timestamp] | None:
    """Ход последнего завершённого часа (максимум − минимум к цене, %) и медиана таких часов за days дней до него."""
    h = d.assign(hour=d["begin"].dt.floor("h")).groupby("hour").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    h = h[h.index < now.floor("h")]                          # текущий час ещё не закрыт
    if len(h) < 50:
        return None
    rng = (h["high"] - h["low"]) / h["close"] * 100
    last = rng.index[-1]
    base = rng[(rng.index < last) & (rng.index >= last - pd.Timedelta(days=days))]
    return float(rng.iloc[-1]), float(base.median()), last


# ---------------------------------------------------------------- брокер (только чтение)

def account_id(api, want: str | None) -> str:
    """Счёт из настроек (номер или его конец), иначе первый счёт токена."""
    accs = [a["id"] for a in api.accounts()]
    return next((a for a in accs if want and str(a).endswith(str(want))), accs[0])


def broker_view(lots: dict, want: str | None = None, sandbox: bool = False) -> dict | None:
    """Бумаги на счёте (тикер → штук) и рубли; {"ошибка": …} — нет токена или сбой (контур работает и без этого)."""
    try:
        from forexmodel.broker.tinvest import TInvest

        api = TInvest(sandbox=sandbox)
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
        return {"позиции": [], "алерты": {}, "стоп_кран": False, "к_закрытию": {}}
    s = json.loads(STATE.read_text(encoding="utf-8"))
    for p in s["позиции"]:
        for k in ("вход_в", "срок", "назначено"):
            p[k] = ts(p.get(k))
    s.setdefault("алерты", {})
    s.setdefault("к_закрытию", {})
    return s


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, default=str, indent=1), encoding="utf-8")


def append_csv(path: Path, rows: list[dict]) -> None:
    """Дописать строки по столбцам файла; у строк новые поля (позиции из бота богаче) — файл переписывается целиком."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if not path.exists():
        new.to_csv(path, index=False)
        return
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    if set(new.columns) <= set(cols):
        new.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)
    else:
        pd.concat([pd.read_csv(path), new], ignore_index=True).to_csv(path, index=False)


def volatility_alerts(stocks: set[str], cfg: dict, st: dict, now: pd.Timestamp) -> list[str]:
    """Ход последнего часа в volatility_x раз больше обычного — раз в день на бумагу (ТЗ: повышенная волатильность)."""
    out, x = [], float(cfg.get("volatility_x", 2.5))
    for t in sorted(stocks):
        path = ROOT / "data" / "raw" / LIVE_FILES.get(t, f"moex/{t}_1m.csv")
        key = "волатильность " + t
        if not path.exists() or st["алерты"].get(key) == str(now.date()):
            continue
        try:
            v = volatility(tail_minutes(path), now)
        except Exception:                      # файл дописывается или битая строка — в следующий час
            continue
        if v and v[1] > 0 and v[0] >= x * v[1] and now - v[2] <= pd.Timedelta(hours=2):
            out.append(f"🌪 Повышенная волатильность {t}: за час {v[2]:%H:%M} ход {v[0]:.2f}% при обычных {v[1]:.2f}% (×{v[0] / v[1]:.1f}).")
            st["алерты"][key] = str(now.date())
    return out


def run(cfg: dict, now: pd.Timestamp, use_broker: bool) -> tuple[list[tuple[str, list | None]], str]:
    """Один часовой шаг счёта под блокировкой: (сообщения с кнопками, строка итога)."""
    lots = {k: v["lot"] for k, v in json.loads((ROOT / cfg["lots_file"]).read_text()).items()}
    allowed = allowed_stocks(cfg, lots)
    button = cfg.get("mode") == "button"
    s42_state = json.loads((OUT / "s42_state.json").read_text(encoding="utf-8")) if (OUT / "s42_state.json").exists() else {}
    it_state = json.loads((OUT / "iterator_state.json").read_text(encoding="utf-8")) if (OUT / "iterator_state.json").exists() else {}
    sig42, pos42 = s42_signals(s42_state, allowed.get("S42", []))
    sig_it, pos_it = iterator_signals(it_state, allowed)
    prices = {**{k.upper(): float(v["цена"]) for k, v in (s42_state.get("решения") or {}).items() if v and v.get("цена") is not None},
              **{k: float(v["цена"]) for k, v in (it_state.get("данные") or {}).items() if v.get("цена") is not None}}
    st = load_state()
    msgs: list[tuple[str, list | None]] = []

    # 1. выходы: не исполненный сигнал просто снимается, открытая позиция — «ЗАКРЫТЬ» с кнопками
    still, closed = exits(st["позиции"], pos42, pos_it)
    for p in closed:
        if p.get("статус") == "к исполнению":
            msgs.append((f"⏹ Сигнал {p['стратегия']} {p['бумага']} снят: сделка стратегии уже закрылась, а позиция не открывалась — "
                         f"покупать не нужно.", None))
            continue
        msgs.append((msg_close(p), close_buttons(p) if button else None))
        st["к_закрытию"][pid(p)] = {"стратегия": p["стратегия"], "бумага": p["бумага"], "штук": p["штук"], "цена": p["цена"],
                                    "исполнение": p.get("исполнение"), "причина": p.get("причина"), "когда": str(now)}
    st["к_закрытию"] = {k: v for k, v in st["к_закрытию"].items() if now - pd.Timestamp(v["когда"]) < pd.Timedelta(days=3)}
    append_csv(TRADES, [{**p, "закрыто": now} for p in closed if p.get("статус") != "к исполнению"])

    # 2. стоп-кран и новые сигналы
    stop_on = bool(cfg.get("stop")) or STOP_FILE.exists()
    if stop_on != st.get("стоп_кран", False):
        msgs.append(("⛔ Стоп-кран включён: новые сделки не открываются. Открытые позиции: " + (", ".join(f"{p['стратегия']} {p['бумага']}" for p in still) or "нет")
                     if stop_on else "✅ Стоп-кран выключен: контур снова берёт сигналы.", None))
    disabled = set(json.loads(STRATEGY_STOP.read_text())) if STRATEGY_STOP.exists() else set()
    seen = {(p["стратегия"], p["бумага"], str(p["вход_в"])) for p in st["позиции"]}
    for path in (ALT, TRADES):               # не взятые и уже закрытые (в том числе через бот) повторно не берём
        if path.exists():
            old = pd.read_csv(path)
            seen |= set(zip(old["стратегия"], old["бумага"], old["вход_в"].astype(str)))
    fresh = [s for s in sig42 + sig_it if (s["стратегия"], s["бумага"], str(s["вход_в"])) not in seen]
    take, alt = allocate(fresh, still, cfg, lots, stop_on, disabled)
    for p in take:
        p.update({"статус": "к исполнению", "назначено": now})
        msgs.append((msg_open(p, cfg), open_buttons(p) if button else None))
    append_csv(ALT, [{**{k: s.get(k) for k in ("стратегия", "бумага", "вход_в", "цена", "стоп", "срок", "причина")}, "запуск": now}
                     for s in alt])
    if alt:                                    # ТЗ: о неоткрытии и причине; лимит сделок — отдельной тревогой
        limit = [s for s in alt if s["причина"].startswith(("нет свободного слота", "не хватает денег"))]
        other = [s for s in alt if s not in limit]
        if limit:
            msgs.append(("⚠️ Лимит сделок исчерпан — сигнал не взят:\n" + "\n".join(
                f"• {s['стратегия']} {s['бумага']} по {f(s['цена'])}: {s['причина']}" for s in limit), None))
        if other:
            msgs.append(("ℹ️ Сигнал не взят на счёт (в журнал альтернативных):\n" + "\n".join(
                f"• {s['стратегия']} {s['бумага']} по {f(s['цена'])}: {s['причина']}" for s in other), None))

    # 3. позиции у брокера: взятые сигналы исполнены ли (куплено руками — тоже видно)
    positions = still + take
    if use_broker and cfg.get("mode") in ("manual", "button") and positions:
        b = broker_view(lots, cfg.get("account"), sandbox=cfg.get("broker") == "sandbox")
        if b and "бумаги" in b:
            for p in positions:
                have = b["бумаги"].get(p["бумага"], 0)
                if p.get("статус") == "к исполнению":
                    if have >= p["штук"]:
                        p.update({"статус": "открыта", "исполнение": p.get("исполнение") or "вручную"})
                    elif now - p["назначено"] > pd.Timedelta(minutes=30) and not p.get("напомнили"):
                        msgs.append((f"⚠️ Сделка не открыта: {p['стратегия']} {p['бумага']} — на счёте {have} шт. из {p['штук']} через 30 минут "
                                     f"после сигнала. " + ("Нажмите «Исполнить» или «Пропустить»." if button else
                                                           "Открыть по рынку или отменить: файл reports/live/account_state.json."),
                                     open_buttons(p) if button else None))
                        p["напомнили"] = True
                elif p.get("статус") == "открыта" and have != p["штук"]:
                    k = f"количество {pid(p)} {have}"
                    if have == 0:              # стоп у брокера или продали в приложении: позиции нет — закрываем в счёте
                        p["статус"] = "нет у брокера"
                        msgs.append((f"🔔 {p['стратегия']} {p['бумага']}: у брокера бумаг больше нет — сработал стоп или продано вручную. "
                                     f"Позиция закрыта в счёте контура.", None))
                    elif st["алерты"].get(k) != str(now.date()):
                        msgs.append((f"⚠️ {p['стратегия']} {p['бумага']}: у брокера {have} шт., у контура {p['штук']} — "
                                     + ("часть продана вручную? Остаток закрыть — /close, или докупить в приложении." if have < p["штук"]
                                        else "докуплено вручную? Лишнее контур не ведёт: стопы стоят только на его часть."), None))
                        st["алерты"][k] = str(now.date())
            gone = [p for p in positions if p.get("статус") == "нет у брокера"]
            append_csv(TRADES, [{**p, "выход": now, "причина": "нет у брокера (стоп или продано вручную)", "цена выхода": prices.get(p["бумага"]),
                                 "закрыто": now} for p in gone])          # цена выхода — последняя цена данных, приближённо
            positions = [p for p in positions if p.get("статус") != "нет у брокера"]
            extra = set(b["бумаги"]) - {p["бумага"] for p in positions}
            key = "лишние " + ",".join(sorted(extra))
            if extra and st["алерты"].get(key) != str(now.date()):
                msgs.append(("⚠️ На счёте бумаги, которых нет у контура: " + ", ".join(f"{t} {b['бумаги'][t]} шт." for t in sorted(extra)), None))
                st["алерты"][key] = str(now.date())

    # 3а. отсечки дивидендов по позициям счёта — из API Т-Инвестиций, раз в день на позицию
    todo = [p for p in positions if st["алерты"].get("дивиденды " + pid(p)) != str(now.date())]
    if use_broker and todo and now.hour >= 8:
        try:
            from forexmodel.broker.tinvest import TInvest

            api, days = TInvest(sandbox=cfg.get("broker") == "sandbox"), int(cfg.get("dividend_days", 3))
            for p in todo:
                for d in api.dividends(api.share(p["бумага"]).uid, now - pd.Timedelta(days=1), now + pd.Timedelta(days=days + 7)):
                    m = dividend_alert(p, d, now, days)
                    if m:
                        msgs.append((m, None))
                st["алерты"]["дивиденды " + pid(p)] = str(now.date())
        except Exception as e:                 # нет токена или сеть — проверим в следующий час
            print(f"[счёт] дивиденды не проверены: {str(e)[:150]}")

    # 3б. просадка счёта по его сделкам: «внимание» и «стоп» (решение остановиться — за людьми)
    st["результат"] = account_pnl(pd.read_csv(TRADES) if TRADES.exists() else None, positions, prices)
    m = drawdown_alert(st["результат"], st, cfg)
    if m:
        msgs.append((m, None))

    # 3в. место на диске: при 1 ГБ контур перестанет дописывать минутки (28.10 его съели соседние проекты)
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free < DISK_WARN_GB and st["алерты"].get("диск") != str(now.date()):
        msgs.append((f"💾 На диске сервера свободно {free:.1f} ГБ. При меньше 1 ГБ контур перестанет дописывать минутки и считать сигналы. "
                     f"Место занимают и соседние проекты на сервере.", None))
        st["алерты"]["диск"] = str(now.date())

    # 4. свежесть данных и волатильность в торговое время (будни 07–23 ч по Москве)
    data = {**{k.upper(): ts(v.get("данные_до")) for k, v in (s42_state.get("решения") or {}).items() if v},
            **{k: ts(v["до"]) for k, v in (it_state.get("данные") or {}).items()}}
    if now.weekday() < 5 and 8 <= now.hour <= 23:
        stale = sorted(t for t, d in data.items() if d is not None and now - d > pd.Timedelta(hours=2)
                       and st["алерты"].get("данные " + t) != str(now.date()))
        if stale:
            msgs.append(("⚠️ Данные не обновлялись больше 2 часов: " + ", ".join(f"{t} (до {data[t]:%d.%m %H:%M})" for t in stale), None))
            for t in stale:
                st["алерты"]["данные " + t] = str(now.date())
        watch = {s.upper() for v in allowed.values() for s in v} | {p["бумага"] for p in positions}
        msgs += [(m, None) for m in volatility_alerts(watch, cfg, st, now)]

    # 5. сводка раз в день
    if now.hour == 23 and st.get("сводка") != str(now.date()):
        used = sum(p["сумма"] for p in positions)
        lines = [f"📋 Счёт на {now:%d.%m %H:%M}: в позициях {f(used, 0)} ₽ из {f(cfg['capital'], 0)} ₽, позиций {len(positions)} из {cfg['max_positions']}."]
        lines += [f"• {p['стратегия']} {p['бумага']}: {p['штук']} шт. с {p['вход_в']:%d.%m}, стоп {f(p['стоп'])}, до {p['срок']:%d.%m} — {p.get('статус')}"
                  for p in positions]
        if ALT.exists():
            n = (pd.to_datetime(pd.read_csv(ALT)["запуск"]) >= now.normalize()).sum()
            lines.append(f"Альтернативных сигналов за день: {n} (reports/live/alt_journal.csv).")
        msgs.append(("\n".join(lines), None))
        st["сводка"] = str(now.date())

    st.update({"позиции": positions, "стоп_кран": stop_on, "запуск": now})
    save_state(st)
    return msgs, (f"[счёт] позиций {len(positions)}, взято {len(take)}, в альтернативные {len(alt)}, закрыто {len(closed)}, "
                  f"стоп-кран {'вкл' if stop_on else 'выкл'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-broker", action="store_true")
    a = ap.parse_args(argv)
    now = pd.Timestamp.now("Europe/Moscow").tz_localize(None).floor("min")
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    with state_lock():
        msgs, summary = run(cfg, now, not a.no_broker)
    for m, buttons in msgs:                    # сеть — уже без блокировки
        notify.send(m, buttons=buttons)
        print(m, "\n", flush=True)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
