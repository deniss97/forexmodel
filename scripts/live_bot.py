"""Бот контура в Telegram (ТЗ инференса Тимура, решения 26.10 и 28.10) — служба forexmodel-bot (systemd пользователя).

Слушает только группу из configs/telegram.yaml (chat_id): писать боту в личку бессмысленно. Команды:
  /status — счёт: стоп-кран, деньги и слоты, позиции с ценой и расстоянием до стопа, сигналы, ждущие решения,
            бумажные позиции S42, S46, S48 и свежесть данных;
  /balance — деньги и бумаги у брокера по всем счетам токена;
  /stop — стоп-кран сразу; /resume — снять стоп-кран (с подтверждением);
  /close SBER — закрыть позицию через API; /closeall — закрыть всё и включить стоп-кран (оба — с подтверждением);
  /risk — настройки риска; /set slot 4000 — изменить настройку; /strategy S46 on|off — стратегия на счёте
            (с подтверждением; капитал на пробном этапе — не больше 15 тыс. ₽, созвон 26.10);
  /journal — последние сделки счёта, заявки и альтернативные сигналы; /help;
  /check — проверка перед торговлей: контур, данные, токен и права, деньги, тариф, стоп-кран, диск, архив, расписание;
  /ask вопрос — вопрос о контуре, сделках, рисках: отвечает Claude Code на этом сервере (claude -p в папке проекта,
            только чтение файлов — ничего не меняет и не торгует), по одному вопросу за раз;
  /chart — картинка «состояние контура» (scripts/live_charts.py; сама приходит каждый день в 23 ч).
Кнопки: под «ОТКРЫТЬ» — «Исполнить» (покупка через API: scripts/live_exec.py — лимит с допуском проскальзывания, повторы,
стоп-лосс и трейлинг у брокера), «Купил сам», «Пропустить»; под «ЗАКРЫТЬ» — «Закрыть через API», «Закрыл сам».
Раз в monitor_min минут — цены открытых позиций: близко к стопу, цена дошла до стопа, включился трейлинг; позиция,
купленная через API, пропала у брокера — сработал стоп или трейлинг. Каждая тревога — один раз на позицию.
Состояние счёта меняется под общей блокировкой с почасовым запуском (live_account.state_lock); своё — reports/live/bot_state.json.

    python scripts/live_bot.py          # как служба: systemctl --user start forexmodel-bot
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

import _bootstrap  # noqa: F401
import live_account as la
import live_exec as lx
import notify
import pandas as pd
import yaml

ROOT, OUT = la.ROOT, la.OUT
TG_CFG = ROOT / "configs" / "telegram.yaml"
BOT_STATE = OUT / "bot_state.json"
TRIAL_CAP = 15000                         # созвон 26.10: на пробном этапе не больше 15 тыс. ₽
SETTINGS = {                              # ключ: (тип, от, до, что это)
    "capital": (int, 1000, TRIAL_CAP, "рублей на счёте стратегий (пробный этап — не больше 15 000)"),
    "slot": (int, 500, TRIAL_CAP, "рублей на одну сделку, вниз до целых лотов"),
    "max_positions": (int, 1, 10, "позиций одновременно"),
    "max_loss_pct": (float, 1, 20, "стоп не дальше этого % от входа"),
    "max_slippage_pct": (float, 0, 2, "допуск проскальзывания при покупке через API, %"),
    "order_attempts": (int, 1, 5, "попыток выставить заявку"),
    "near_stop_pct": (float, 0.2, 5, "тревога «близко к стопу»: % до стопа"),
    "volatility_x": (float, 1.5, 10, "тревога о волатильности: ход часа во столько раз больше обычного"),
    "weekends": (bool, None, None, "брать на счёт сигналы торгов выходного дня"),
    "dd_warn_pct": (float, 2, 50, "просадка счёта от пика, % депозита: тревога «внимание»"),
    "dd_stop_pct": (float, 3, 60, "просадка счёта от пика, % депозита: тревога «стоп» (/stop — решение за вами)"),
    "dividend_days": (int, 1, 10, "за сколько дней предупреждать об отсечке дивидендов по позиции"),
}
STRATEGY_ON = {"S42": "[sber, gazp, lkoh]", "S46": "all", "S48": "all"}
COMMANDS = [("status", "Счёт, позиции, стоп-кран"), ("balance", "Деньги и бумаги у брокера"), ("stop", "Включить стоп-кран"),
            ("resume", "Снять стоп-кран"), ("close", "Закрыть позицию: /close SBER"), ("closeall", "Закрыть всё и включить стоп-кран"),
            ("risk", "Настройки риска"), ("set", "Изменить настройку: /set slot 4000"),
            ("strategy", "Стратегия на счёте: /strategy S46 off"), ("journal", "Последние сделки и сигналы"),
            ("check", "Проверка перед торговлей"), ("ask", "Вопрос о контуре: /ask почему нет сделок?"),
            ("chart", "Картинка: состояние контура"), ("help", "Что умеет бот")]
HELP = """🤖 Бот контура forexmodel

Сигналы приходят сами: «🟢 ОТКРЫТЬ» с кнопками «Исполнить» (бот купит через API и поставит стоп), «Купил сам», «Пропустить»; «🔴 ЗАКРЫТЬ» — «Закрыть через API» или «Закрыл сам».

/status — счёт и позиции
/balance — деньги и бумаги у брокера
/stop — стоп-кран: новые сделки не открываются
/resume — снять стоп-кран
/close SBER — закрыть позицию через API
/closeall — закрыть всё и включить стоп-кран
/risk — настройки риска; /set slot 4000 — изменить
/strategy S46 on — стратегия на счёте (off — выключить)
/journal — последние сделки и сигналы
/check — проверка перед торговлей
/chart — картинка: позиции, результат, стратегии (сама — каждый день в 23:00)
/ask вопрос — спросить Claude о контуре, сделках, рисках (например: /ask почему сегодня не было сделок?)

Тревоги: близко к стопу, цена у стопа, включился трейлинг, отсечка дивидендов, бумаг у брокера не столько, сколько у контура, просадка счёта (−8% / −12%), лимит сделок, волатильность, данные не обновляются, мало места на диске.
Опасные команды — с кнопкой подтверждения; капитал пробного этапа — не больше 15 000 ₽."""
_shares: dict = {}
CLAUDE = os.path.expanduser("~/.local/bin/claude")
ASKING = threading.Lock()
ASK_RULES = """Ты отвечаешь в Telegram-группе «forexmodel» (Денис и Тимур) на вопрос о живом торговом контуре проекта forexmodel:
стратегия S42 (импульс) на SBER, GAZP, LKOH через Т-Инвестиции, S46 и S48 — на бумаге, бот @MeLan_ITB_1_bot.
Правила: только читать файлы (Read, Grep, Glob) — ничего не меняй, не торгуй, команды не запускай. Отвечай по-русски,
простыми словами и коротко (до 1500 знаков), без таблиц и разметки Markdown — Telegram покажет текст как есть.
Цифры бери из файлов и не выдумывай; если данных нет — так и скажи и подскажи, где их взять или какую команду бота нажать.
Где смотреть: reports/live/account_state.json (счёт), s42_state.json и iterator_state.json (стратегии),
alt_journal.csv (не взятые сигналы), account_trades.csv (сделки счёта), orders.csv (заявки через API),
notifications.log (что бот писал в группу), logs/<дата>.log (почасовые запуски), deviation.md (сверка с бэктестом),
configs/account.yaml (настройки), docs/results/inference_tz.md (как устроен контур), docs/results/bot_guide/bot_guide.html
(инструкция к боту), docs/results/launch_questions.md (разборы: портфель, стопы, плечо, просадки),
docs/meetings/ (протоколы созвонов), docs/hypotheses.md (банк гипотез)."""


def now() -> pd.Timestamp:
    return pd.Timestamp.now("Europe/Moscow").tz_localize(None).floor("s")


def f(x, d: int = 2) -> str:
    return la.f(x, d)


# ---------------------------------------------------------------- Telegram

class Telegram:
    def __init__(self, token: str, chat: int):
        self.base, self.chat = f"https://api.telegram.org/bot{token}/", int(chat)

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        req = urllib.request.Request(self.base + method, json.dumps(params or {}).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def say(self, text: str, buttons=None) -> None:
        msg = {"chat_id": self.chat, "text": text[:4000], "disable_web_page_preview": True}
        if buttons:
            msg["reply_markup"] = notify.keyboard(buttons)
        for k in range(3):
            try:
                self.call("sendMessage", msg)
                notify._log(text, True)
                return
            except Exception:
                time.sleep(3 * (k + 1))
        notify._log(text, False)

    def drop_buttons(self, message: dict) -> None:
        try:
            self.call("editMessageReplyMarkup", {"chat_id": self.chat, "message_id": message["message_id"],
                                                 "reply_markup": {"inline_keyboard": []}})
        except Exception:                      # сообщение старое или кнопок уже нет
            pass


def who(user: dict) -> str:
    return (user.get("first_name") or "") + (f" (@{user['username']})" if user.get("username") else "")


# ---------------------------------------------------------------- настройки (configs/account.yaml с комментариями)

def load_cfg() -> dict:
    return yaml.safe_load(la.CFG.read_text(encoding="utf-8"))


def write_cfg(text: str) -> None:
    tmp = la.CFG.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, la.CFG)


def parse_setting(key: str, raw: str, cfg: dict) -> tuple[str, object]:
    """Значение настройки из команды → (как записать в YAML, значение). Ошибка — ValueError с понятным текстом."""
    if key not in SETTINGS:
        raise ValueError(f"нет настройки «{key}»; есть: {', '.join(SETTINGS)}")
    typ, lo, hi, _ = SETTINGS[key]
    raw = raw.strip().lower().replace(" ", "").replace(",", ".")
    if typ is bool:
        if raw not in ("on", "off", "true", "false", "да", "нет", "1", "0"):
            raise ValueError("нужно on или off")
        v = raw in ("on", "true", "да", "1")
        return ("true" if v else "false"), v
    try:
        v = typ(float(raw)) if typ is int else float(raw)
    except ValueError:
        raise ValueError(f"«{raw}» — не число") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key} — от {lo:g} до {hi:g}" + (" (пробный этап, созвон 26.10)" if hi == TRIAL_CAP else ""))
    if key == "slot" and v > cfg["capital"]:
        raise ValueError(f"слот больше капитала ({cfg['capital']})")
    if key == "capital" and v < cfg["slot"]:
        raise ValueError(f"капитал меньше слота ({cfg['slot']})")
    return (str(v) if typ is int else f"{v:g}"), v


def set_setting(text: str, key: str, value: str) -> str:
    """Поменять значение ключа верхнего уровня, сохранив комментарий; нет ключа — дописать строку."""
    pat = re.compile(rf"^({re.escape(key)}:[ \t]*)([^#\n]*?)([ \t]*#.*)?$", re.M)
    if pat.search(text):
        return pat.sub(lambda m: m.group(1) + value + (m.group(3) or ""), text, count=1)
    return text.rstrip("\n") + f"\n{key}: {value}\n"


def set_strategy(text: str, code: str, on: bool) -> str:
    """Стратегия в блоке strategies: on — прежний набор бумаг (STRATEGY_ON), off — на счёт не берётся."""
    value = STRATEGY_ON[code] if on else "off"
    pat = re.compile(rf"^([ \t]+{re.escape(code)}:[ \t]*)([^#\n]*?)([ \t]*#.*)?$", re.M)
    if pat.search(text):
        return pat.sub(lambda m: m.group(1) + value + (m.group(3) or ""), text, count=1)
    return re.sub(r"^(strategies:.*)$", lambda m: m.group(1) + f"\n  {code}: {value}", text, count=1, flags=re.M)


def risk_text(cfg: dict) -> str:
    lines = ["⚙️ Настройки риска (configs/account.yaml)"]
    for k, (_, _, _, what) in SETTINGS.items():
        v = cfg.get(k)
        lines.append(f"• {k} = {('on' if v else 'off') if isinstance(v, bool) else v} — {what}")
    if cfg.get("fixed_lots"):
        lines.append("• всегда ровно: " + ", ".join(f"{t} — {n} лот" for t, n in cfg["fixed_lots"].items())
                     + " (лот дороже слота; деньги на него зарезервированы, пока бумаги нет в позиции; меняется только в configs/account.yaml)")
    on = {c: v for c, v in (cfg.get("strategies") or {}).items()}
    lines.append("• стратегии: " + ", ".join(f"{c} {'выкл' if not v else ('все бумаги' if v == 'all' else ', '.join(map(str, v)).upper())}"
                                            for c, v in on.items()))
    lines.append(f"• режим: {cfg.get('mode')}, брокер: {'песочница' if cfg.get('broker') == 'sandbox' else 'боевой счёт'}")
    lines.append("\nИзменить: /set slot 4000 · /set weekends on · /strategy S46 on")
    return "\n".join(lines)


# ---------------------------------------------------------------- брокер

def broker(cfg: dict):
    from forexmodel.broker.tinvest import TInvest

    api = TInvest(sandbox=cfg.get("broker") == "sandbox")
    return api, la.account_id(api, cfg.get("account"))


def share(api, ticker: str):
    key = (api.sandbox, ticker.upper())
    if key not in _shares:
        _shares[key] = api.share(ticker.upper())
    return _shares[key]


def position_alerts(p: dict, px: float, cfg: dict) -> list[tuple[str, str]]:
    """Тревоги по цене открытой позиции: (вид, текст). Стоп — больший из стопа позиции и текущего стопа стратегии."""
    stop = max(float(p["стоп"]), float(p.get("стоп стратегии") or 0))
    api = p.get("исполнение") == "API"
    out = []
    if px <= stop:
        out.append(("у стопа", f"🔻 {p['бумага']}: цена {px:g} дошла до стопа {stop:g}. " +
                    ("Стоп у брокера должен был сработать — бот проверит позицию." if api else
                     f"Если стоп в приложении не стоит — продайте: /close {p['бумага']}.")))
    elif (px / stop - 1) * 100 <= float(cfg.get("near_stop_pct", 1.0)):
        out.append(("близко к стопу", f"⚠️ {p['бумага']} близко к стопу: цена {px:g}, стоп {stop:g} ({(px / stop - 1) * 100:.1f}% до стопа)."))
    act = p.get("активация")
    if act and px >= float(act):
        out.append(("трейлинг", f"📈 {p['бумага']}: цена {px:g} дошла до включения трейлинга {float(act):g}. " +
                    ("Трейлинг у брокера включился." if api else
                     f"Если трейлинг не стоит в приложении — подтяните стоп: на {f(float(p.get('отступ') or 0))} ₽ ниже максимума.")))
    return out


# ---------------------------------------------------------------- состояние

def load_bot() -> dict:
    s = json.loads(BOT_STATE.read_text(encoding="utf-8")) if BOT_STATE.exists() else {}
    return {"offset": 0, "pending": {}, "alerts": {}, "started": None, **s}


def save_bot(s: dict) -> None:
    BOT_STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def find(st: dict, key: str) -> dict | None:
    return next((p for p in st["позиции"] if la.pid(p) == key), None)


def stop_on(cfg: dict) -> bool:
    return bool(cfg.get("stop")) or la.STOP_FILE.exists()


# ---------------------------------------------------------------- ответы на команды

def status_text(cfg: dict) -> str:
    with la.state_lock():
        st = la.load_state()
    t = now()
    lines = [f"📊 Счёт · {t:%d.%m %H:%M}",
             f"Стоп-кран: {'ВКЛЮЧЁН ⛔ — новые сделки не открываются' if stop_on(cfg) else 'выключен ✅'}"
             + (f"\n  {la.STOP_FILE.read_text(encoding='utf-8').strip().splitlines()[0][:200]}" if la.STOP_FILE.exists() else ""),
             f"Режим: {'кнопки «Исполнить» / «Купил сам»' if cfg.get('mode') == 'button' else cfg.get('mode')}, "
             f"{'песочница' if cfg.get('broker') == 'sandbox' else 'боевой счёт'}; выходные: {'берём' if cfg.get('weekends', True) else 'без новых сигналов'}"]
    lots = {k: v["lot"] for k, v in json.loads((ROOT / cfg["lots_file"]).read_text()).items()}
    allowed = la.allowed_stocks(cfg, lots)
    lines.append("На счёт: " + "; ".join(f"{c} — {', '.join(s.upper() for s in v) if len(v) < 6 else f'{len(v)} бумаг'}" for c, v in allowed.items() if v)
                 + ("; бумага: " + ", ".join(c for c, v in allowed.items() if not v) if any(not v for v in allowed.values()) else ""))
    pos = st["позиции"]
    used = sum(p["сумма"] for p in pos)
    lines.append(f"Деньги: в позициях {f(used, 0)} ₽ из {f(cfg['capital'], 0)} ₽, позиций {len(pos)} из {cfg['max_positions']}")
    res, peak = float(st.get("результат", 0.0)), float(st.get("пик_результата", 0.0))
    lines.append(f"Результат сделок счёта: {res:+,.0f} ₽, пик {peak:+,.0f} ₽, просадка {(peak - res) / cfg['capital'] * 100:.1f}% депозита".replace(",", " ")
                 + f" (тревоги при −{cfg.get('dd_warn_pct', 8)}% и −{cfg.get('dd_stop_pct', 12)}%)")
    prices = {}
    if pos:
        try:
            api, _ = broker(cfg)
            prices = {p["бумага"]: float(api.last_price(share(api, p["бумага"]).uid)) for p in pos}
        except Exception as e:
            lines.append(f"(цены у брокера не получены: {str(e)[:80]})")
    lines.append("\nПозиции счёта:" if pos else "\nПозиций на счёте нет.")
    for p in pos:
        entry = float(p.get("цена_входа") or p["цена"])
        px = prices.get(p["бумага"])
        stop = max(float(p["стоп"]), float(p.get("стоп стратегии") or 0))
        now_s = f" → {px:g} ({(px / entry - 1) * 100:+.2f}%, {(px - entry) * p['штук']:+.0f} ₽), до стопа {(px / stop - 1) * 100:.1f}%" if px else ""
        lines.append(f"• {p['стратегия']} {p['бумага']} — {p.get('статус')}{' (' + p['исполнение'] + ')' if p.get('исполнение') else ''}: "
                     f"{p['штук']} шт. по {entry:g}{now_s}; стоп {stop:g}; закрыть до {pd.Timestamp(p['срок']):%d.%m %H:%M}")
    if st.get("к_закрытию"):
        lines.append("Ждут закрытия: " + ", ".join(f"{v['стратегия']} {v['бумага']} ({v['штук']} шт.)" for v in st["к_закрытию"].values()))
    # бумажные позиции стратегий
    paper = []
    s42 = json.loads((OUT / "s42_state.json").read_text(encoding="utf-8")) if (OUT / "s42_state.json").exists() else {}
    for inst, m in (s42.get("решения") or {}).items():
        if m and m.get("вход") and m.get("решение") in ("держать", "открыть"):
            paper.append(f"S42 {inst.upper()} с {pd.Timestamp(m['вход']):%d.%m} {m['цена_входа']:g} → {m['цена']:g} "
                         f"({(m['цена'] / m['цена_входа'] - 1) * 100:+.1f}%), стоп {m['стоп']:g}{', трейлинг' if m.get('трейлинг_вкл') else ''}")
    it = json.loads((OUT / "iterator_state.json").read_text(encoding="utf-8")) if (OUT / "iterator_state.json").exists() else {}
    last = {k: v.get("цена") for k, v in (it.get("данные") or {}).items()}
    for p in it.get("позиции", []):
        px = last.get(p["бумага"])
        paper.append(f"{p['стратегия']} {p['бумага']} с {pd.Timestamp(p['open_dt']):%d.%m} {p['open_price']:g}"
                     + (f" → {px:g} ({(px / p['open_price'] - 1) * 100:+.1f}%)" if px else ""))
    lines.append("\nБумага (стратегии без денег): " + ("\n• " + "\n• ".join(paper) if paper else "позиций нет"))
    data = [la.ts(m.get("данные_до")) for m in (s42.get("решения") or {}).values() if m]
    if data:
        lines.append(f"\nДанные: до {max(d for d in data if d is not None):%d.%m %H:%M}")
    return "\n".join(lines)


def balance_text(cfg: dict) -> str:
    api, _ = broker(cfg)
    uids = json.loads((OUT / "uids.json").read_text()) if (OUT / "uids.json").exists() else {}
    back = {u: t for t, u in uids.items()}
    lines = [f"💰 Т-Инвестиции{' — ПЕСОЧНИЦА' if api.sandbox else ''}"]
    for a in api.accounts():
        pf = api.portfolio(a["id"])
        lines.append(f"\nСчёт …{str(a['id'])[-4:]} {a.get('name') or ''}: всего {f(lx.num(pf.get('totalAmountPortfolio')), 0)} ₽, "
                     f"деньги {f(lx.num(pf.get('totalAmountCurrencies')), 0)} ₽")
        for x in pf.get("positions", []):
            if x.get("instrumentType") == "currency":
                continue
            q, avg, cur = lx.num(x.get("quantity")), lx.num(x.get("averagePositionPrice")), lx.num(x.get("currentPrice"))
            lines.append(f"• {back.get(x.get('instrumentUid'), x.get('ticker') or x.get('figi'))}: {q:g} шт., средняя {avg:g}, сейчас {cur:g}, "
                         f"{lx.num(x.get('expectedYield')):+.0f} ₽")
        so = [o for o in api.stop_orders(a["id"])] if a.get("id") else []
        if so:
            lines.append(f"Стоп-заявок активно: {len(so)}")
    return "\n".join(lines)


def journal_text() -> str:
    lines = ["🧾 Журнал"]
    for title, path, cols in (("Сделки счёта", la.TRADES, ["стратегия", "бумага", "вход_в", "штук", "причина", "закрыто"]),
                              ("Заявки через API", lx.ORDERS, ["время", "бумага", "действие", "лотов", "цена"]),
                              ("Не взятые сигналы", la.ALT, ["стратегия", "бумага", "вход_в", "цена", "причина"])):
        if not path.exists():
            lines.append(f"\n{title}: пусто")
            continue
        d = pd.read_csv(path).tail(5)
        lines.append(f"\n{title} (последние {len(d)}):")
        for _, r in d.iterrows():
            lines.append("• " + " · ".join(str(r[c])[:40] for c in cols if c in r and pd.notna(r[c])))
    return "\n".join(lines)


# ---------------------------------------------------------------- проверка перед торговлей, вопросы, картинка

def check_facts(cfg: dict) -> dict:
    """Всё, что нужно для /check: контур, данные, брокер (только чтение), диск, архив, расписание."""
    t = now()
    with la.state_lock():
        st = la.load_state()
    s42 = json.loads((OUT / "s42_state.json").read_text(encoding="utf-8")) if (OUT / "s42_state.json").exists() else {}
    it = json.loads((OUT / "iterator_state.json").read_text(encoding="utf-8")) if (OUT / "iterator_state.json").exists() else {}
    log = OUT / "logs" / f"{pd.Timestamp.now():%Y-%m-%d}.log"
    last_block = log.read_text(encoding="utf-8", errors="ignore").split("\n20")[-1] if log.exists() else ""
    d42 = [la.ts(m.get("данные_до")) for m in (s42.get("решения") or {}).values() if m and m.get("данные_до")]
    dit = [la.ts(v.get("до")) for v in (it.get("данные") or {}).values() if v.get("до")]
    man = OUT / "yadisk_manifest.json"
    ups = [y.get("uploaded") for v in (json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}).values() for y in v.values()]
    try:
        cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        cron = ""
    lots = {k: v["lot"] for k, v in json.loads((ROOT / cfg["lots_file"]).read_text()).items()}
    f = {"now": t, "run": la.ts(st.get("запуск")), "run_error": "Traceback" in last_block, "stop_on": stop_on(cfg),
         "data42": max(d42) if d42 else None, "data_it": max(dit) if dit else None, "trading": t.weekday() < 5 and 7 <= t.hour < 24,
         "cfg": cfg, "allowed": la.allowed_stocks(cfg, lots), "positions": st["позиции"],
         "strategy_stop": json.loads(la.STRATEGY_STOP.read_text()) if la.STRATEGY_STOP.exists() else [],
         "disk_gb": shutil.disk_usage(ROOT).free / 1024 ** 3, "yadisk": os.path.ismount(os.path.expanduser("~/yadisk")),
         "archive": pd.Timestamp(max(ups)) if ups else None, "cron_hourly": "live_hourly.sh" in cron, "cron_archive": "yadisk_nightly.sh" in cron,
         "traded": la.TRADES.exists() or any(p.get("исполнение") for p in st["позиции"])}
    try:
        api, acc = broker(cfg)
        info = api.call("UsersService", "GetInfo")
        accs = {a["id"]: a for a in api.accounts()}
        held = api.positions(acc)
        tickers = sorted({x.upper() for v in f["allowed"].values() for x in v})
        uids = {share(api, t2).uid: t2 for t2 in tickers}
        mine = {p["бумага"] for p in st["позиции"]}
        foreign = {uids.get(u, u[:8]): q for u, q in held.items() if q and uids.get(u) not in mine}   # бумаги вне позиций контура
        f.update({"broker": True, "tariff": info.get("tariff"), "account": acc, "access": accs.get(acc, {}).get("accessLevel", ""),
                  "money": float(api.money(acc)), "foreign": foreign, "stop_orders": len(api.stop_orders(acc)),
                  "no_api": [t2 for t2 in tickers if not share(api, t2).api_trade]})
    except Exception as e:
        f.update({"broker": False, "broker_error": str(e)[:150]})
    return f


def check_items(f: dict) -> list[tuple[str, str]]:
    """Факты → строки проверки: ok — в порядке, warn — заметить, bad — поправить до торговли, hand — проверить руками."""
    cfg, t, out = f["cfg"], f["now"], []
    if f["run"] is None or (t - f["run"]) > pd.Timedelta(minutes=75):
        out.append(("bad", f"Контур: последний запуск {f['run']:%d.%m %H:%M} — больше часа назад" if f["run"] else "Контур: ещё не запускался"))
    else:
        out.append(("warn" if f["run_error"] else "ok", f"Контур: последний запуск {f['run']:%H:%M}" + (", в логе ошибка — см. reports/live/logs" if f["run_error"] else ", без ошибок")))
    for name, d in (("S42 (SBER, GAZP, LKOH)", f["data42"]), ("S46/S48 (19 бумаг)", f["data_it"])):
        stale = d is None or (f["trading"] and t - d > pd.Timedelta(hours=2))
        out.append(("bad" if stale else "ok", f"Данные {name}: до {d:%d.%m %H:%M}" if d is not None else f"Данные {name}: нет"))
    if not f.get("broker"):
        out.append(("bad", f"Брокер: не ответил — {f.get('broker_error', '')}"))
    else:
        full = "FULL" in str(f["access"])
        out.append(("ok" if full else "warn", f"Токен и счёт …{str(f['account'])[-4:]}: " + ("полный доступ — «Исполнить» сможет покупать" if full
                                                                                    else f"доступ {f['access'] or 'неизвестен'} — «Исполнить» не сработает, «Купил сам» — да")))
        out.append(("ok", f"Тариф: {f['tariff']}" + ("; комиссию сверим по первой сделке" if not f["traded"] else "")))
        cap = float(cfg["capital"])
        out.append(("ok" if f["money"] >= cap * 0.98 else "warn", f"Деньги на счёте: {la.f(f['money'], 0)} ₽ при капитале в настройках {la.f(cap, 0)} ₽"))
        out.append(("ok" if not f["foreign"] else "warn", "Посторонних бумаг на счёте нет" if not f["foreign"]
                    else "На счёте бумаги вне контура: " + ", ".join(f"{k} {v} шт." for k, v in f["foreign"].items())))
        out.append(("ok" if not f["no_api"] else "bad", "Торговля через API по бумагам стратегий доступна" if not f["no_api"]
                    else "Через API нельзя торговать: " + ", ".join(f["no_api"])))
    on = [c for c, v in f["allowed"].items() if v]
    out.append(("ok", f"Настройки: на счёт {', '.join(on) or 'ничего'}, капитал {la.f(float(cfg['capital']), 0)} ₽, слот {la.f(float(cfg['slot']), 0)} ₽, "
                f"режим {cfg.get('mode')}, {'песочница' if cfg.get('broker') == 'sandbox' else 'боевой счёт'}, выходные {'берём' if cfg.get('weekends', True) else 'нет'}"
                + "".join(f", {t} — всегда {n} лот" for t, n in (cfg.get("fixed_lots") or {}).items())))
    if f["strategy_stop"]:
        out.append(("warn", "Сверка с бэктестом остановила: " + ", ".join(f["strategy_stop"])))
    out.append(("warn" if f["stop_on"] else "ok", "Стоп-кран включён — сделки на счёт не берутся (снять: /resume)" if f["stop_on"] else "Стоп-кран выключен — торгуем"))
    out.append(("bad" if f["disk_gb"] < 1 else "warn" if f["disk_gb"] < 3 else "ok", f"Диск сервера: свободно {f['disk_gb']:.0f} ГБ"))
    fresh = f["archive"] is not None and t - f["archive"] < pd.Timedelta(hours=30)
    out.append(("ok" if f["yadisk"] and fresh else "warn", ("Яндекс Диск подключён" if f["yadisk"] else "Яндекс Диск не подключён")
                + (f", архив минуток обновлён {f['archive']:%d.%m %H:%M}" if f["archive"] is not None else ", архива минуток ещё нет")))
    out.append(("ok" if f["cron_hourly"] and f["cron_archive"] else "bad", "Расписание: почасовой контур и ночной архив стоят" if f["cron_hourly"] and f["cron_archive"]
                else "Расписание: нет " + ", ".join(n for n, ok in (("почасового контура", f["cron_hourly"]), ("ночного архива", f["cron_archive"])) if not ok)))
    out.append(("hand", "Проверить руками в приложении: заявкам по API не нужно SMS-подтверждение; за «Премиум» нет абонплаты"))
    return out


def check_text(cfg: dict) -> str:
    items = check_items(check_facts(cfg))
    icon = {"ok": "✅", "warn": "⚠️", "bad": "❌", "hand": "☑️"}
    bad, warn = [x for k, x in items if k == "bad"], [x for k, x in items if k == "warn"]
    if bad:
        verdict = "❌ До торговли поправить: " + "; ".join(x.split(":")[0] for x in bad)
    elif stop_on(cfg):
        verdict = "Всё готово, кроме решения: торговля начнётся после /resume (вместе)" + (" — и посмотрите замечания" if len(warn) > 1 else "")
    else:
        verdict = "Работает" + (", есть замечания" if warn else ", замечаний нет")
    return f"🩺 Проверка перед торговлей · {now():%d.%m %H:%M}\n\n" + "\n".join(f"{icon[k]} {x}" for k, x in items) + f"\n\n{verdict}"


def split_text(text: str, size: int = 3800) -> list[str]:
    parts, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > size and cur:
            parts.append(cur)
            cur = ""
        cur += para + "\n"
    return parts + ([cur] if cur.strip() else [])


def ask_claude(question: str, name: str) -> str:
    """Вопрос → Claude Code на этом сервере (claude -p в папке проекта), только чтение файлов."""
    prompt = f"Вопрос от {name}: {question}\n\nСостояние контура сейчас (вывод /status):\n{status_text(load_cfg())}"
    r = subprocess.run([CLAUDE, "-p", prompt, "--append-system-prompt", ASK_RULES, "--no-session-persistence", "--strict-mcp-config",
                        "--allowedTools", "Read,Grep,Glob", "--disallowedTools", "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task",
                        "--output-format", "text"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.stdout.strip() or "Ответа нет: " + ((r.stderr.strip().splitlines() or ["пусто"])[-1])[:300]


def do_ask(tg: Telegram, question: str, name: str) -> None:
    if not question.strip():
        tg.say("Напишите вопрос после команды, например: /ask почему сегодня не было сделок?")
        return
    if not ASKING.acquire(blocking=False):
        tg.say("Ещё отвечаю на предыдущий вопрос — подождите минуту.")
        return
    tg.say(f"🤔 Думаю над вопросом — {name}. Обычно 1–2 минуты: читаю файлы контура.")

    def work():
        try:
            answer = ask_claude(question, name)
        except subprocess.TimeoutExpired:
            answer = "Не успел ответить за 5 минут — попробуйте вопрос покороче."
        except Exception as e:
            answer = f"Не получилось ответить: {str(e)[:200]}"
        finally:
            ASKING.release()
        for i, chunk in enumerate(split_text(answer)):
            tg.say(("💬 " if i == 0 else "") + chunk)

    threading.Thread(target=work, daemon=True).start()


def do_chart(tg: Telegram) -> None:
    """Картинка отдельным процессом: matplotlib не раздувает память бота; отправляет сам скрипт."""
    def work():
        r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-W", "ignore", str(ROOT / "scripts" / "live_charts.py"), "--send"],
                           cwd=ROOT, capture_output=True, text=True, timeout=180)
        if "доставлено" not in r.stdout or "не доставлено" in r.stdout:
            tg.say("Картинка не получилась: " + ((r.stderr or r.stdout).strip().splitlines() or ["без подробностей"])[-1][:300])

    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------- действия

def do_resume(tg: Telegram, name: str) -> None:
    la.STOP_FILE.unlink(missing_ok=True)
    cfg = load_cfg()
    if cfg.get("stop"):
        write_cfg(set_setting(la.CFG.read_text(encoding="utf-8"), "stop", "false"))
    tg.say(f"✅ Стоп-кран снят — {name}. Новые сигналы берутся на счёт со следующего часового запуска (в :05).")


def do_stop(tg: Telegram, name: str, why: str = "") -> None:
    la.STOP_FILE.write_text(f"Стоп-кран {now():%d.%m %H:%M} — {name}{': ' + why if why else ''}\nСнять: /resume в группе или rm reports/live/STOP\n",
                            encoding="utf-8")
    tg.say(f"⛔ Стоп-кран включён — {name}{': ' + why if why else ''}. Новые сделки не открываются; открытые позиции не трогаем.")


def do_close(tg: Telegram, key: str, name: str, why: str = "по команде") -> None:
    """Закрыть через API: позиция счёта или ждущая закрытия (после «ЗАКРЫТЬ»)."""
    cfg = load_cfg()
    with la.state_lock():
        st = la.load_state()
        p = find(st, key) or st["к_закрытию"].get(key)
    if not p:
        tg.say(f"Позиции {key} уже нет в состоянии счёта.")
        return
    api, acc = broker(cfg)
    r = lx.close_position(api, acc, share(api, p["бумага"]), int(p["штук"]), attempts=int(cfg.get("order_attempts", 3)), now=now(), why=why)
    with la.state_lock():
        st = la.load_state()
        q = find(st, key)
        if r.ok or r.gone:                     # продали или продавать уже нечего
            if q:
                st["позиции"] = [x for x in st["позиции"] if la.pid(x) != key]
                la.append_csv(la.TRADES, [{**q, "выход": now(), "причина": f"закрыта через бот ({why})", "цена выхода": r.price or None,
                                           "закрыто": now()}])
            st["к_закрытию"].pop(key, None)
            la.save_state(st)
    head = f"{p['стратегия']} {p['бумага']}"
    if r.ok:
        entry = float(p.get("цена_входа") or p["цена"])
        tg.say(f"🔴 Закрыто через API: {head} — {r.lots} лот. по {r.price:g} ({(r.price / entry - 1) * 100:+.2f}% от входа {entry:g}). — {name}\n"
               + "\n".join(r.log))
    else:
        tg.say(f"{'ℹ️' if r.gone else '❌'} {head}: {r.why}\n" + "\n".join(r.log))


def do_open(tg: Telegram, key: str, name: str) -> None:
    """«Исполнить»: позиция «к исполнению» → покупка через API → стопы у брокера."""
    cfg = load_cfg()
    with la.state_lock():
        st = la.load_state()
        p = find(st, key)
        if not p or p.get("статус") != "к исполнению":
            tg.say(f"Сигнал {key} уже не ждёт исполнения (статус: {p.get('статус') if p else 'нет в счёте'}).")
            return
        if stop_on(cfg):
            tg.say("⛔ Включён стоп-кран — сначала /resume.")
            return
        if now() > pd.Timestamp(p["срок"]):
            tg.say(f"Сигнал {key} устарел: срок сделки стратегии {pd.Timestamp(p['срок']):%d.%m %H:%M} прошёл.")
            return
        p["статус"] = "исполняется"
        la.save_state(st)
    tg.say(f"⏳ Покупаю {p['стратегия']} {p['бумага']}: {p['лоты']} лот. не дороже {p['цена'] * (1 + float(cfg.get('max_slippage_pct', 0.3)) / 100):.2f} — {name}")
    try:
        api, acc = broker(cfg)
        sh = share(api, p["бумага"])
        r = lx.open_position(api, acc, sh, p, cfg, now=now())
    except Exception as e:
        r, sh = lx.Result(False, why=f"ошибка брокера: {str(e)[:200]}"), None
    with la.state_lock():
        st = la.load_state()
        q = find(st, key)
        if q is not None:
            if r.ok:
                lot = sh.lot
                q.update({"статус": "открыта", "исполнение": "API", "цена_входа": r.price, "лоты": r.lots, "штук": r.lots * lot,
                          "сумма": r.lots * lot * r.price, "стоп_заявки": r.stop_ids, "исполнено_в": now(), "кто": name})
            else:
                q["статус"] = "к исполнению"
                q["ошибка"] = r.why
            la.save_state(st)
    if r.ok:
        tg.say(f"✅ Куплено через API: {p['стратегия']} {p['бумага']} — {r.lots * sh.lot} шт. ({r.lots} лот.) по {r.price:g} на "
               f"{f(r.lots * sh.lot * r.price, 0)} ₽ (сигнал {p['цена']:g}).{' ' + r.why if r.why else ''}\n" + "\n".join(r.log))
    else:
        tg.say(f"❌ Не исполнено: {p['стратегия']} {p['бумага']} — {r.why}\n" + "\n".join(r.log)
               + "\nСигнал остался «к исполнению»: можно нажать «Исполнить» ещё раз, «Купил сам» или «Пропустить».",
               buttons=la.open_buttons(p))


def do_manual(tg: Telegram, key: str, name: str, opened: bool) -> None:
    """«Купил сам» (opened) или «Закрыл сам»."""
    with la.state_lock():
        st = la.load_state()
        if opened:
            p = find(st, key)
            if not p or p.get("статус") not in ("к исполнению",):
                tg.say(f"Сигнал {key} уже не ждёт исполнения.")
                return
            p.update({"статус": "открыта", "исполнение": "вручную", "исполнено_в": now(), "кто": name})
        else:
            p = st["к_закрытию"].pop(key, None) or find(st, key)
            st["позиции"] = [x for x in st["позиции"] if la.pid(x) != key]
        la.save_state(st)
    if not p:
        tg.say(f"Позиции {key} уже нет.")
    elif opened:
        trail = f" и трейлинг (включение {f(float(p['активация']))}, отступ {f(float(p['отступ']))})" if p.get("активация") else ""
        tg.say(f"👌 {p['стратегия']} {p['бумага']}: отмечено «куплено вручную» — {name}. Поставьте в приложении стоп-лосс {f(float(p['стоп']))}{trail}.")
    else:
        tg.say(f"👌 {p['стратегия']} {p['бумага']}: отмечено «закрыто вручную» — {name}.")


def do_skip(tg: Telegram, key: str, name: str) -> None:
    with la.state_lock():
        st = la.load_state()
        p = find(st, key)
        if not p or p.get("статус") != "к исполнению":
            tg.say(f"Сигнал {key} уже не ждёт исполнения.")
            return
        st["позиции"] = [x for x in st["позиции"] if la.pid(x) != key]
        la.append_csv(la.ALT, [{**{k: p.get(k) for k in ("стратегия", "бумага", "вход_в", "цена", "стоп", "срок")},
                                "причина": f"пропущено вручную ({name})", "запуск": now()}])
        la.save_state(st)
    tg.say(f"⏭ {p['стратегия']} {p['бумага']} пропущен — {name}. Сигнал записан в журнал альтернативных: видно, сколько упустили.")


def run_confirmed(tg: Telegram, a: dict, name: str) -> None:
    kind, args = a["action"], a["args"]
    if kind == "resume":
        do_resume(tg, name)
    elif kind == "close":
        do_close(tg, args["key"], name)
    elif kind == "closeall":
        do_stop(tg, name, "закрыть всё")
        with la.state_lock():
            keys = [la.pid(p) for p in la.load_state()["позиции"] if p.get("статус") in ("открыта", "исполняется")]
        for k in keys:
            do_close(tg, k, name, "закрыть всё")
        if not keys:
            tg.say("Открытых позиций нет.")
    elif kind == "set":
        write_cfg(set_setting(la.CFG.read_text(encoding="utf-8"), args["key"], args["value"]))
        tg.say(f"⚙️ {args['key']} = {args['value']} — {name}. Действует со следующего часового запуска.")
    elif kind == "strategy":
        write_cfg(set_strategy(la.CFG.read_text(encoding="utf-8"), args["code"], args["on"]))
        tg.say(f"⚙️ {args['code']} на счёте: {'включена' if args['on'] else 'выключена (остаётся на бумаге)'} — {name}.")


def ask(tg: Telegram, bst: dict, action: str, args: dict, text: str, name: str) -> None:
    k = uuid.uuid4().hex[:8]
    bst["pending"][k] = {"action": action, "args": args, "at": str(now()), "who": name}
    tg.say(text + "\nПодтвердить?", buttons=[[("✅ Да", "ok:" + k), ("✖️ Нет", "no:" + k)]])


def on_command(tg: Telegram, bst: dict, text: str, name: str) -> None:
    parts = text.split()
    cmd, args = parts[0].split("@")[0].lstrip("/").lower(), parts[1:]
    cfg = load_cfg()
    if cmd in ("start", "help"):
        tg.say(HELP)
    elif cmd == "status":
        tg.say(status_text(cfg))
    elif cmd == "balance":
        tg.say(balance_text(cfg))
    elif cmd == "stop":
        do_stop(tg, name, " ".join(args))
    elif cmd == "resume":
        if not stop_on(cfg):
            tg.say("Стоп-кран и так выключен.")
        else:
            ask(tg, bst, "resume", {}, f"Снять стоп-кран? Новые сигналы снова пойдут на счёт (капитал {f(cfg['capital'], 0)} ₽, "
                                       f"слот {f(cfg['slot'], 0)} ₽, стратегии: {', '.join(c for c, v in cfg['strategies'].items() if v)}).", name)
    elif cmd == "close":
        if not args:
            tg.say("Какую бумагу? Например: /close SBER")
            return
        with la.state_lock():
            ps = [p for p in la.load_state()["позиции"] if p["бумага"] == args[0].upper() and p.get("статус") in ("открыта", "исполняется")]
        if not ps:
            tg.say(f"Открытой позиции {args[0].upper()} на счёте нет (/status).")
            return
        p = ps[0]
        ask(tg, bst, "close", {"key": la.pid(p)}, f"Закрыть через API {p['стратегия']} {p['бумага']}: продать {p['штук']} шт. по рынку и снять стоп-заявки?", name)
    elif cmd == "closeall":
        ask(tg, bst, "closeall", {}, "Закрыть ВСЕ позиции счёта через API по рынку и включить стоп-кран?", name)
    elif cmd == "risk":
        tg.say(risk_text(cfg))
    elif cmd == "set":
        if len(args) < 2:
            tg.say("Например: /set slot 4000 · /set weekends on\n\n" + risk_text(cfg))
            return
        try:
            yv, v = parse_setting(args[0].lower(), " ".join(args[1:]), cfg)
        except ValueError as e:
            tg.say(f"Не получится: {e}")
            return
        ask(tg, bst, "set", {"key": args[0].lower(), "value": yv}, f"Изменить {args[0].lower()}: {cfg.get(args[0].lower())} → {yv}?", name)
    elif cmd == "strategy":
        code = args[0].upper() if args else ""
        if code not in STRATEGY_ON or len(args) < 2 or args[1].lower() not in ("on", "off"):
            tg.say("Например: /strategy S46 on или /strategy S46 off (есть S42, S46, S48)")
            return
        on = args[1].lower() == "on"
        ask(tg, bst, "strategy", {"code": code, "on": on},
            f"{'Включить' if on else 'Выключить'} {code} на счёте?" + (" (созвон 26.10: пока на деньги только S42)" if on and code != "S42" else ""), name)
    elif cmd == "check":
        tg.say(check_text(cfg))
    elif cmd == "ask":
        do_ask(tg, text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else "", name)
    elif cmd == "chart":
        do_chart(tg)
    elif cmd == "journal":
        tg.say(journal_text())
    else:
        tg.say("Не знаю такой команды. /help — что умею.")


def on_callback(tg: Telegram, bst: dict, q: dict) -> None:
    data, name = q.get("data", ""), who(q.get("from", {}))
    kind, _, key = data.partition(":")
    try:
        tg.call("answerCallbackQuery", {"callback_query_id": q["id"], "text": "принято"})
    except Exception:
        pass
    if kind in ("ok", "no"):
        a = bst["pending"].pop(key, None)
        tg.drop_buttons(q["message"])
        if not a:
            tg.say("Это подтверждение уже не действует.")
        elif kind == "no":
            tg.say(f"Отменено — {name}.")
        elif now() - pd.Timestamp(a["at"]) > pd.Timedelta(minutes=10):
            tg.say("Подтверждение просрочено (больше 10 минут) — повторите команду.")
        else:
            run_confirmed(tg, a, name)
        return
    tg.drop_buttons(q["message"])
    if kind == "x":
        do_open(tg, key, name)
    elif kind == "m":
        do_manual(tg, key, name, opened=True)
    elif kind == "s":
        do_skip(tg, key, name)
    elif kind == "c":
        do_close(tg, key, name, "по сигналу ЗАКРЫТЬ")
    elif kind == "cm":
        do_manual(tg, key, name, opened=False)


def monitor(tg: Telegram, bst: dict) -> None:
    """Цены открытых позиций: тревоги ТЗ (близко к стопу, у стопа, трейлинг) и закрытие стопом у брокера."""
    t = now()
    if t.hour < 7:                             # ночью торгов нет
        return
    cfg = load_cfg()
    with la.state_lock():
        ps = [p for p in la.load_state()["позиции"] if p.get("статус") == "открыта"]
    if not ps:
        return
    api, acc = broker(cfg)
    held = api.positions(acc) if any(p.get("исполнение") == "API" for p in ps) else {}
    for p in ps:
        sh = share(api, p["бумага"])
        alerts = position_alerts(p, float(api.last_price(sh.uid)), cfg)
        if p.get("исполнение") == "API" and held.get(sh.uid, 0) == 0:
            alerts.append(("у брокера закрыта", f"🔔 {p['стратегия']} {p['бумага']}: позиции у брокера больше нет — сработал стоп или трейлинг. "
                                               f"Контур отметит выход в ближайший час."))
        for kind, text in alerts:
            k = f"{kind} {la.pid(p)}"
            if k not in bst["alerts"]:
                tg.say(text)
                bst["alerts"][k] = str(t)
    bst["alerts"] = {k: v for k, v in bst["alerts"].items() if t - pd.Timestamp(v) < pd.Timedelta(days=14)}


def main() -> int:
    tcfg = yaml.safe_load(TG_CFG.read_text(encoding="utf-8"))
    tg = Telegram(tcfg["token"], tcfg["chat_id"])
    bst = load_bot()
    tg.call("setMyCommands", {"commands": [{"command": c, "description": d} for c, d in COMMANDS]})
    if not bst["started"] or now() - pd.Timestamp(bst["started"]) > pd.Timedelta(hours=6):
        tg.say("🤖 Бот контура запущен. /help — что умею, /status — счёт.")
    bst["started"] = str(now())
    save_bot(bst)
    last_monitor = 0.0
    while True:
        try:
            ups = tg.call("getUpdates", {"offset": bst["offset"], "timeout": 50, "allowed_updates": ["message", "callback_query"]},
                          timeout=65).get("result", [])
            for u in ups:
                bst["offset"] = u["update_id"] + 1
                save_bot(bst)                  # обработанное не повторится после перезапуска
                try:
                    if "callback_query" in u and u["callback_query"].get("message", {}).get("chat", {}).get("id") == tg.chat:
                        on_callback(tg, bst, u["callback_query"])
                    elif u.get("message", {}).get("chat", {}).get("id") == tg.chat and u["message"].get("text", "").startswith("/"):
                        on_command(tg, bst, u["message"]["text"], who(u["message"].get("from", {})))
                except Exception as e:
                    traceback.print_exc()
                    tg.say(f"⚠️ Ошибка бота: {str(e)[:300]}")
                save_bot(bst)
        except urllib.error.HTTPError as e:     # 409 — бота слушает ещё кто-то; 5xx — Telegram
            print(f"{now()} Telegram HTTP {e.code}", flush=True)
            time.sleep(30)
        except Exception as e:                  # сеть
            print(f"{now()} {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(10)
        if time.time() - last_monitor > 60 * float(load_cfg().get("monitor_min", 5)):
            last_monitor = time.time()
            try:
                monitor(tg, bst)
                save_bot(bst)
            except Exception as e:
                print(f"{now()} monitor: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
