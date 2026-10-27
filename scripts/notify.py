"""Уведомления контура (ТЗ инференса 26.10: сигналы, сделки, ошибки данных — в Telegram): всегда пишутся в журнал
reports/live/notifications.log, в Telegram — если задан бот.

Бот создаётся у @BotFather; токен и chat_id — в configs/telegram.yaml (в .gitignore, права только у владельца):
    token: "123456789:AA..."
    chat_id: 123456789          # свой id (написать боту и открыть https://api.telegram.org/bot<токен>/getUpdates) или id группы
Без файла сообщения только пишутся в журнал. Проверка: python scripts/notify.py "проверка связи".
Кнопки под сообщением (buttons — ряды пар «надпись, данные») нажимают в группе; нажатия разбирает бот scripts/live_bot.py.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "telegram.yaml"
LOG = ROOT / "reports" / "live" / "notifications.log"


def _log(text: str, sent: bool) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"--- {stamp} {'отправлено' if sent else 'не отправлено (нет бота или ошибка)'}\n{text}\n")


def keyboard(buttons: list[list[tuple[str, str]]]) -> dict:
    """[[("✅ Исполнить", "x:…"), …], …] → reply_markup Telegram (данные кнопки — до 64 байт)."""
    return {"inline_keyboard": [[{"text": t, "callback_data": d[:64]} for t, d in row] for row in buttons]}


def send(text: str, retries: int = 3, buttons: list[list[tuple[str, str]]] | None = None) -> bool:
    """Сообщение в Telegram; True — доставлено. Несколько попыток (ТЗ: при сбое — повтор), всё — в журнал."""
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else None
    ok = False
    if cfg and cfg.get("token") and cfg.get("chat_id"):
        msg = {"chat_id": cfg["chat_id"], "text": text[:4000], "disable_web_page_preview": True}
        if buttons:
            msg["reply_markup"] = keyboard(buttons)
        body = json.dumps(msg).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{cfg['token']}/sendMessage", body,
                                     {"Content-Type": "application/json"})
        for k in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    ok = bool(json.load(r).get("ok"))
                    break
            except Exception:                    # сеть, ограничение частоты — повтор с паузой
                time.sleep(3 * (k + 1))
    _log(text, ok)
    return ok


if __name__ == "__main__":
    print("доставлено" if send(" ".join(sys.argv[1:]) or "проверка связи") else f"не доставлено — см. {LOG.relative_to(ROOT)}")
