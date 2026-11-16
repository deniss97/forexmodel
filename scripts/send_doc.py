"""Файл (PDF отчёта и т. п.) в группу Telegram с подписью; в журнал reports/live/notifications.log — «[файл …]».
    python scripts/send_doc.py <файл> --caption-file <текст.md> [--reply-to N]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import yaml

from notify import CFG, _log


def document(path: Path, caption: str = "", reply_to: int | None = None, retries: int = 3) -> int | None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8")) if CFG.exists() else None
    mid = None
    if cfg and cfg.get("token") and cfg.get("chat_id"):
        boundary = "----forexmodel" + str(int(time.time() * 1000))
        fields = [("chat_id", cfg["chat_id"]), ("caption", caption[:1024])]
        if reply_to:
            fields.append(("reply_to_message_id", int(reply_to)))
        parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode() for k, v in fields]
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{path.name}"\r\n'
                      f'Content-Type: application/octet-stream\r\n\r\n').encode() + path.read_bytes() + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{cfg['token']}/sendDocument", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        for k in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    res = json.load(r)
                    if res.get("ok"):
                        mid = res["result"]["message_id"]
                    break
            except Exception:
                time.sleep(3 * (k + 1))
    _log(f"[файл {path.name}] {caption}", mid is not None)
    return mid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--caption-file")
    ap.add_argument("--reply-to", type=int)
    a = ap.parse_args()
    cap = Path(a.caption_file).read_text(encoding="utf-8").strip() if a.caption_file else ""
    m = document(Path(a.file), cap, a.reply_to)
    print(f"отправлено, сообщение {m}" if m else "не отправлено — см. reports/live/notifications.log")
