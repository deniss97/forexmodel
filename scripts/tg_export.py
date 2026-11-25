"""Разовая выгрузка истории Telegram-канала (в том числе закрытого платного) для проверки сигналов автора (H78).

Запуск на любом ноутбуке с Python 3.9+ (Windows / macOS / Linux):
    pip install telethon
    python tg_export.py                 # спросит ключи, телефон, код и канал
    python tg_export.py --no-media      # только текст, быстрее
Ключи api_id / api_hash — с my.telegram.org (API development tools), можно передать --api-id/--api-hash или
переменными окружения TG_API_ID / TG_API_HASH. Ключи принадлежат приложению, входить можно любым аккаунтом;
после первого входа они запоминаются в tg_keys.json рядом со скриптом. На Windows проще запускать START.bat.

Что делает:
  1. Входит в аккаунт (телефон → код из Telegram → облачный пароль, если есть). Сессия хранится ТОЛЬКО в памяти,
     файла сессии на диске нет.
  2. Показывает список каналов и групп аккаунта, ищет по части названия; выбрать номер.
  3. Выгружает все сообщения от старых к новым в tg_<канал>_<дата>/:
        messages.jsonl — все поля по сообщению (UTC и МСК, правки, ответы, пересылки, просмотры, реакции, медиа),
        messages.csv   — то же коротко, для Excel,
        meta.json      — канал, сколько выгружено, когда, версия скрипта,
        media/         — картинки (и файлы до --media-max-mb), имя = id сообщения.
     Пишет по ходу: если выгрузка оборвалась, повторный запуск с той же папкой (--out) продолжит с места обрыва.
  4. Упаковывает: <папка>_text.zip (текст, маленький — сразу в чат) и <папка>_media.zip (картинки).
  5. Спрашивает, выгрузить ли ещё канал; в конце выходит из аккаунта (сессия закрывается и на серверах
     Telegram) — на ноутбуке доступа не остаётся.

Запрет пересылки в канале на выгрузку не влияет: скрипт читает сообщения как приложение Telegram.
Содержимое платного канала — только для своей проверки, не публиковать.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "2026-11-21"
MSK = timezone(timedelta(hours=3))
KEYS = Path(__file__).resolve().parent / "tg_keys.json"  # api_id / api_hash приложения

try:
    from telethon.errors import FloodWaitError, SessionPasswordNeededError
    from telethon.sessions import StringSession
    from telethon.sync import TelegramClient
except ImportError:
    sys.exit("Нет библиотеки telethon. Установите:  pip install telethon")

for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ask(prompt: str, secret: bool = False) -> str:
    while True:
        v = (getpass.getpass(prompt) if secret else input(prompt)).strip()
        if v:
            return v


def iso(d: datetime | None, tz=timezone.utc) -> str | None:
    return d.astimezone(tz).isoformat(timespec="seconds") if d else None


def media_kind(msg) -> str | None:
    if not msg.media:
        return None
    if msg.photo:
        return "photo"
    if msg.video or msg.gif:
        return "video"
    if msg.voice or msg.audio:
        return "audio"
    if msg.sticker:
        return "sticker"
    if msg.poll:
        return "poll"
    if msg.web_preview:
        return "link"
    if msg.document:
        return "file"
    return type(msg.media).__name__


def file_size(msg) -> int:
    f = msg.file
    return int(f.size or 0) if f else 0


def reactions(msg) -> dict | None:
    r = getattr(msg, "reactions", None)
    if not r or not getattr(r, "results", None):
        return None
    out = {}
    for x in r.results:
        key = getattr(x.reaction, "emoticon", None) or type(x.reaction).__name__
        out[key] = x.count
    return out


def forward(msg) -> dict | None:
    f = msg.fwd_from
    if not f:
        return None
    return {"from_name": f.from_name, "from_id": getattr(f.from_id, "channel_id", None) or getattr(f.from_id, "user_id", None),
            "date_utc": iso(f.date), "post_author": f.post_author}


def row(msg, media_file: str | None) -> dict:
    poll = None
    if msg.poll:
        p = msg.poll.poll
        poll = {"question": getattr(p.question, "text", p.question),
                "answers": [getattr(a.text, "text", a.text) for a in p.answers]}
    return {
        "id": msg.id,
        "date_utc": iso(msg.date),
        "date_msk": iso(msg.date, MSK),
        "edit_date_utc": iso(msg.edit_date),
        "text": msg.message or "",
        "text_md": msg.text or "",
        "views": msg.views,
        "forwards": msg.forwards,
        "replies": msg.replies.replies if msg.replies else None,
        "reactions": reactions(msg),
        "reply_to": msg.reply_to.reply_to_msg_id if msg.reply_to else None,
        "grouped_id": msg.grouped_id,
        "post_author": msg.post_author,
        "pinned": bool(msg.pinned),
        "fwd": forward(msg),
        "media": media_kind(msg),
        "media_size": file_size(msg) or None,
        "media_file": media_file,
        "poll": poll,
        "service": msg.action is not None and type(msg.action).__name__,
    }


def login(args) -> TelegramClient:
    saved = {}
    if KEYS.exists():
        try:
            saved = json.loads(KEYS.read_text(encoding="utf-8"))
            print(f"Ключи api_id / api_hash взяты из {KEYS.name}")
        except Exception:
            saved = {}
    api_id = args.api_id or os.environ.get("TG_API_ID") or saved.get("api_id") or ask("api_id (число с my.telegram.org): ")
    api_hash = args.api_hash or os.environ.get("TG_API_HASH") or saved.get("api_hash") or ask("api_hash: ")
    client = TelegramClient(StringSession(), int(api_id), api_hash, device_model="tg_export", app_version=VERSION)
    client.flood_sleep_threshold = 300  # короткие «подождите N с» от Telegram — просто ждать
    client.connect()
    phone = args.phone or ask("Телефон аккаунта (+7...): ")
    client.send_code_request(phone)
    code = ask("Код из Telegram (придёт в приложение Telegram, не SMS): ")
    try:
        client.sign_in(phone, code)
    except SessionPasswordNeededError:
        client.sign_in(password=ask("Облачный пароль (двухэтапная аутентификация): ", secret=True))
    me = client.get_me()
    if not saved:  # ключи приложения (не аккаунта) — запомнить, чтобы в следующий раз не спрашивать
        try:
            KEYS.write_text(json.dumps({"api_id": str(api_id), "api_hash": api_hash}), encoding="utf-8")
        except Exception:
            pass
    print(f"Вход: {me.first_name or ''} {me.last_name or ''} (@{me.username or '-'})")
    return client


def choose(client: TelegramClient, query: str | None):
    dialogs = [d for d in client.get_dialogs() if d.is_channel or d.is_group]
    while True:
        q = (query or input("Часть названия канала (Enter — показать все): ")).strip().lower()
        query = None
        found = [d for d in dialogs if q in (d.name or "").lower()]
        if not found:
            print("Не нашлось, попробуйте ещё раз.")
            continue
        for i, d in enumerate(found[:50], 1):
            kind = "канал" if d.is_channel and not d.is_group else "группа"
            print(f"{i:3}. {d.name}  [{kind}]")
        if len(found) > 50:
            print(f"... и ещё {len(found) - 50}, уточните название")
        v = input("Номер (Enter — искать заново): ").strip()
        if v.isdigit() and 1 <= int(v) <= min(len(found), 50):
            return found[int(v) - 1].entity


def slug(name: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    return (s or "channel")[:40]


def export(client: TelegramClient, entity, out: Path, media: bool, media_max_mb: float, limit: int | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "media").mkdir(exist_ok=True)
    jl = out / "messages.jsonl"
    last_id = 0
    if jl.exists():  # продолжить с места обрыва
        with jl.open(encoding="utf-8") as f:
            for line in f:
                try:
                    last_id = max(last_id, json.loads(line)["id"])
                except Exception:
                    pass
        if last_id:
            print(f"Продолжаю после сообщения {last_id}")
    total = client.get_messages(entity, limit=0).total
    print(f"Сообщений в канале: {total}")
    n = n_media = n_err = 0
    with jl.open("a", encoding="utf-8") as f:
        it = client.iter_messages(entity, reverse=True, min_id=last_id, limit=limit, wait_time=1)
        while True:
            try:
                msg = next(it)
            except StopIteration:
                break
            except FloodWaitError as e:  # длинная пауза от Telegram — ждём и продолжаем
                print(f"Telegram просит подождать {e.seconds} с ...")
                time.sleep(e.seconds + 1)
                it = client.iter_messages(entity, reverse=True, min_id=last_id, wait_time=1)
                continue
            mf = None
            if media and msg.media and media_kind(msg) in ("photo", "file", "video", "audio") and \
                    (msg.photo or file_size(msg) <= media_max_mb * 1e6):
                try:
                    p = client.download_media(msg, str(out / "media" / str(msg.id)))
                    if p:
                        mf = "media/" + Path(p).name
                        n_media += 1
                except Exception as e:  # защищённое медиа и т. п. — не останавливаемся
                    n_err += 1
                    print(f"  медиа {msg.id} не скачалось: {type(e).__name__}")
            f.write(json.dumps(row(msg, mf), ensure_ascii=False) + "\n")
            f.flush()
            last_id = msg.id
            n += 1
            if n % 200 == 0:
                print(f"  {n} новых ... {iso(msg.date, MSK)[:10]}")
    print(f"Выгружено новых: {n}, медиа: {n_media}, ошибок медиа: {n_err}")
    return {"total_in_channel": total, "new": n, "media": n_media, "media_errors": n_err}


def write_csv(out: Path) -> int:
    rows = [json.loads(x) for x in (out / "messages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r: r["id"])
    cols = ["id", "date_msk", "edit_date_utc", "views", "reply_to", "media", "media_file", "text"]
    with (out / "messages.csv").open("w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel видит кириллицу
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) if r.get(c) is not None else "" for c in cols])
    return len(rows)


def pack(out: Path) -> list[Path]:
    made = []
    text_zip = out.with_name(out.name + "_text.zip")
    with zipfile.ZipFile(text_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("messages.jsonl", "messages.csv", "meta.json"):
            if (out / name).exists():
                z.write(out / name, f"{out.name}/{name}")
    made.append(text_zip)
    files = sorted((out / "media").glob("*"))
    if files:
        media_zip = out.with_name(out.name + "_media.zip")
        with zipfile.ZipFile(media_zip, "w", zipfile.ZIP_STORED) as z:  # картинки уже сжаты
            for p in files:
                z.write(p, f"{out.name}/media/{p.name}")
        made.append(media_zip)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Выгрузка истории Telegram-канала (H78)")
    ap.add_argument("--api-id")
    ap.add_argument("--api-hash")
    ap.add_argument("--phone")
    ap.add_argument("--channel", help="часть названия канала (иначе спросит)")
    ap.add_argument("--out", help="папка выгрузки (повторный запуск с ней же — продолжить)")
    ap.add_argument("--no-media", action="store_true", help="только текст")
    ap.add_argument("--media-max-mb", type=float, default=20, help="файлы крупнее не качать (картинки — всегда)")
    ap.add_argument("--limit", type=int, help="не больше N сообщений (для пробы)")
    args = ap.parse_args()

    client = login(args)
    query = args.channel
    try:
        while True:
            entity = choose(client, query)
            query = None
            title = getattr(entity, "title", None) or str(entity.id)
            out = Path(args.out) if args.out else Path(f"tg_{slug(title)}_{datetime.now(MSK):%Y%m%d}")
            print(f"Канал: {title} → {out.resolve()}")
            stats = export(client, entity, out, not args.no_media, args.media_max_mb, args.limit)
            n = write_csv(out)
            meta = {"channel": title, "channel_id": entity.id, "username": getattr(entity, "username", None),
                    "noforwards": getattr(entity, "noforwards", None), "messages_in_file": n,
                    "exported_at_msk": iso(datetime.now(timezone.utc), MSK), "script_version": VERSION, **stats}
            (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            for z in pack(out):
                print(f"Готово: {z.resolve()}  ({z.stat().st_size / 1e6:.1f} МБ)")
            print("Эти .zip — прислать в исследовательский чат.")
            if os.name == "nt":  # Windows: сразу открыть папку с архивами
                try:
                    os.startfile(str(out.resolve().parent))
                except Exception:
                    pass
            if args.out or input("Выгрузить ещё канал? (y/N): ").strip().lower() not in ("y", "д", "да", "yes"):
                break
    except KeyboardInterrupt:
        print("\nОстановлено. Повторный запуск с --out <та же папка> продолжит выгрузку.")
    finally:
        try:
            client.log_out()
            print("Вышел из аккаунта: на этом компьютере доступа к Telegram не осталось.")
        except Exception:
            print("Выйти автоматически не удалось: Telegram → Настройки → Устройства → завершить сеанс «tg_export».")
        client.disconnect()

if __name__ == "__main__":
    main()
