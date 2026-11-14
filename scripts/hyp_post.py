"""Сообщения в группу проекта о работе над гипотезами (просьба Дениса 12.11).

* start  — «начинаю гипотезу Hxx»: подробное описание простыми словами (то, что подтверждено в плане банка);
* result — итог ответом (reply) на сообщение start этой гипотезы, с картинками, если есть;
* note   — обычное сообщение (например, о новом в банке гипотез).

Номера сообщений start хранятся в reports/live/hyp_messages.json, поэтому результат приходит ответом на нужное сообщение.
Текст берётся из файла (--file) или из аргумента; длиннее ~4000 знаков режется на части.

    python scripts/hyp_post.py note --file msg.md
    python scripts/hyp_post.py start H40 --file h40_start.md
    python scripts/hyp_post.py result H40 --file h40_result.md --photo reports/h40/groups.png reports/h40/trades.png
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import notify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "reports" / "live" / "hyp_messages.json"
LIMIT = 3900


def chunks(text: str) -> list[str]:
    """Режет по абзацам, чтобы части были не длиннее лимита Telegram."""
    out, cur = [], ""
    for par in text.strip().split("\n\n"):
        while len(par) > LIMIT:
            out.append(par[:LIMIT]); par = par[LIMIT:]
        if len(cur) + len(par) + 2 > LIMIT:
            out.append(cur); cur = par
        else:
            cur = f"{cur}\n\n{par}" if cur else par
    if cur:
        out.append(cur)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["start", "result", "note"])
    ap.add_argument("code", nargs="?")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--photo", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.kind == "note" and a.code and not a.text:      # note "текст" — текст попадает в позицию кода
        a.text, a.code = a.code, None
    text = Path(a.file).read_text(encoding="utf-8") if a.file else (a.text or "")
    if not text.strip():
        ap.error("пустой текст")
    store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    code = (a.code or "").upper()
    if a.kind != "note" and not code:
        ap.error("нужен код гипотезы")

    reply = None
    if a.kind == "result":
        reply = (store.get(code) or {}).get("start")
        if not reply:
            print(f"нет сообщения start для {code} — результат уйдёт отдельным сообщением")
    first = None
    for part in chunks(text):
        mid = notify.send_id(part, reply_to=reply if first is None else first)
        if mid is None:
            print("не доставлено — см. reports/live/notifications.log"); return 1
        first = first or mid
    for ph in a.photo:
        notify.photo(Path(ph), reply_to=first or reply)
    if a.kind != "note":
        rec = store.setdefault(code, {})
        rec[a.kind] = first
        rec[a.kind + "_at"] = datetime.now().isoformat(timespec="seconds")
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"отправлено, сообщение {first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
