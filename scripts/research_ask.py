"""/ask в исследовательском чате (Денис, 20.11: «разрешаю присылать архивы и файлы по нашему запросу через /ask»).

Вопрос → Claude Code на сервере (claude -p в папке проекта, только чтение файлов). Если просят файл, таблицу, картинку
или архив, Claude в конце ответа пишет строки «ФАЙЛ: путь/от/корня/проекта» (файл или папка). Бот проверяет пути и
отправляет: один файл до 45 МБ — как есть; несколько файлов или папка — одним .zip; больше 45 МБ (предел Telegram
для бота — 50 МБ) — на Яндекс Диск в forexmodel/для_чата/ и в чат ссылка.

Что никогда не отправляется: всё вне папки проекта, токены и ключи (configs/telegram.yaml, configs/tinvest_token,
configs/certs/, configs/account.yaml, файлы с token/secret/key/password/.env в имени), .git.

Из командной строки (проверка без Telegram):
    python scripts/research_ask.py "пришли таблицу случаев H94"
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live" / "ask_out"
CLAUDE = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
TG_LIMIT = 45 * 1024 ** 2
MAX_TOTAL = 3 * 1024 ** 3            # больше 3 ГБ за один запрос не собираем — диск сервера
YD_DIR = "yadisk:forexmodel/для_чата"
YD_LOCAL = Path.home() / "yadisk" / "server-archive" / "forexmodel"   # архив проекта на Диске (ссылки из data/, reports/)
DENY = re.compile(r"(^|/)(\.git|\.venv|certs)(/|$)|token|secret|passw|\.env$|\.env/|(^|/)\.ssh|\.key$|\.pem$|"
                  r"^configs/(telegram\.yaml|account\.yaml|tinvest_token)", re.I)
FILE_LINE = re.compile(r"^\s*ФАЙЛ:\s*(.+?)\s*$", re.M)

RULES = """Ты отвечаешь в Telegram, в исследовательском чате проекта forexmodel (Денис и Тимур): гипотезы, проверки,
данные, отчёты. Только читай файлы (Read, Grep, Glob) — ничего не меняй и не запускай. Отвечай по-русски, простыми
словами, коротко (до 1500 знаков), без таблиц и Markdown — Telegram покажет текст как есть. Цифры — только из файлов.

Где что лежит: docs/results/ (отчёты по гипотезам: <код>.md и папка с картинками и таблицами), docs/experiments/
(журнал экспериментов: catalog.yaml, README.md), docs/hypotheses.md и docs/hypothesis_bank/journal.json (банк),
docs/meetings/ (протоколы созвонов), data/raw/ (минутки: *_1m*.csv, moex/, moex/BR_1m_cont.csv — нефть Brent),
data/derived/ (промежуточные таблицы, например h94_events.parquet), data/calendar/ (EIA), scripts/ (код проверок),
reports/ (служебные отчёты).

ФАЙЛЫ. Если просят прислать файл, таблицу, картинку, код, данные или архив — найди нужное (Glob, Grep) и в самом конце
ответа напиши по строке на каждый путь от корня проекта:
ФАЙЛ: docs/results/h94/counts.csv
Можно указать папку — бот упакует её в архив; несколько строк — тоже один архив. Бот сам отправит, а если больше 45 МБ —
положит на Яндекс Диск и пришлёт ссылку. Не указывай токены, ключи, пароли, configs/telegram.yaml, configs/account.yaml,
configs/tinvest_token, configs/certs — их не отправляем никогда. Если файла нет — так и скажи, ничего не придумывай.
Если файлы не просили — строк ФАЙЛ: не пиши."""


def ask_claude(question: str, name: str) -> str:
    prompt = f"Вопрос из исследовательского чата от {name}: {question}"
    r = subprocess.run([CLAUDE, "-p", prompt, "--append-system-prompt", RULES, "--no-session-persistence", "--strict-mcp-config",
                        "--allowedTools", "Read,Grep,Glob", "--disallowedTools", "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task",
                        "--output-format", "text"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    return r.stdout.strip() or "Ответа нет: " + ((r.stderr.strip().splitlines() or ["пусто"])[-1])[:300]


def safe(rel: str) -> tuple[Path | None, str]:
    """Путь от корня проекта → (путь, '') или (None, почему нельзя)."""
    rel = rel.strip().strip("`\"'«»")
    p = Path(os.path.normpath(ROOT / rel))          # без раскрытия ссылок: часть данных — ссылки на архив Диска
    try:
        r = p.relative_to(ROOT).as_posix()
    except ValueError:
        return None, f"{rel}: вне проекта"
    real = p.resolve()
    if not (real.is_relative_to(ROOT) or real.is_relative_to(YD_LOCAL)):
        return None, f"{rel}: ведёт вне проекта"
    if DENY.search(r):
        return None, f"{rel}: служебный файл с ключами — не отправляется"
    if not p.exists():
        return None, f"{rel}: нет такого файла"
    return p, ""


def walk(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    out = []
    for x in sorted(p.rglob("*")):
        if not x.is_file():
            continue
        rel = x.relative_to(ROOT).as_posix() if x.is_relative_to(ROOT) else None   # симлинк на Диск — по своему пути
        if rel is not None and not DENY.search(rel):
            out.append(x)
    return out


def package(paths: list[Path]) -> tuple[Path, bool]:
    """→ (что отправить, временный ли это архив)."""
    if len(paths) == 1 and paths[0].is_file():
        return paths[0], False
    OUT.mkdir(parents=True, exist_ok=True)
    name = (paths[0].name if len(paths) == 1 else "файлы") + time.strftime("_%Y%m%d_%H%M%S") + ".zip"
    z = OUT / name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in paths:
            base = p.parent
            for f in walk(p):
                zf.write(f, f.relative_to(base).as_posix())
    return z, True


def yadisk_link(path: Path) -> str | None:
    """Большой файл → Яндекс Диск (forexmodel/для_чата/) и публичная ссылка."""
    try:
        subprocess.run(["rclone", "copy", str(path), YD_DIR], check=True, capture_output=True, timeout=3600)
        r = subprocess.run(["rclone", "link", f"{YD_DIR}/{path.name}"], check=True, capture_output=True, text=True, timeout=300)
        return r.stdout.strip() or None
    except Exception:
        return None


def send_document(token: str, chat: int, path: Path, caption: str = "") -> bool:
    boundary = "----forexmodel" + str(int(time.time() * 1000))
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n{chat}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n{caption[:1024]}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{path.name}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n').encode()
    body = head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendDocument", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return bool(json.load(r).get("ok"))
        except Exception:
            time.sleep(5 * (k + 1))
    return False


def deliver(rels: list[str], send_file, say) -> None:
    """Пути из ответа → файлы в чат (send_file(path, caption) -> bool, say(text))."""
    ok, bad = [], []
    for rel in rels:
        p, why = safe(rel)
        (ok.append(p) if p else bad.append(why))
    if bad:
        say("Не отправляю: " + "; ".join(bad))
    if not ok:
        return
    total = sum(f.stat().st_size for p in ok for f in walk(p))
    if total > MAX_TOTAL:
        say(f"Слишком много: {total / 1024 ** 3:.1f} ГБ за раз. Попросите часть (бумагу, год) — или Денис выгрузит с сервера.")
        return
    if len(ok) > 1 or ok[0].is_dir() or total > TG_LIMIT:
        say(f"📦 Собираю {'архив' if (len(ok) > 1 or ok[0].is_dir()) else 'файл'} ({total / 1024 ** 2:.1f} МБ)…")
    out, tmp = package(ok)
    size = out.stat().st_size
    try:
        if size <= TG_LIMIT:
            if not send_file(out, f"{out.name} · {size / 1024 ** 2:.1f} МБ"):
                say(f"Не получилось отправить {out.name} — попробуйте ещё раз.")
        else:
            say(f"{out.name} — {size / 1024 ** 2:.0f} МБ, больше предела Telegram. Кладу на Яндекс Диск…")
            link = yadisk_link(out)
            say(f"🔗 {out.name}: {link}" if link else "Не получилось положить на Яндекс Диск — напишите Денису.")
    finally:
        if tmp:
            out.unlink(missing_ok=True)


def split_answer(answer: str) -> tuple[str, list[str]]:
    rels = FILE_LINE.findall(answer)
    return FILE_LINE.sub("", answer).strip(), rels


if __name__ == "__main__":
    a = ask_claude(" ".join(sys.argv[1:]), "проверка")
    text, rels = split_answer(a)
    print(text)
    print("файлы:", rels, [safe(r)[1] or "ok" for r in rels])
