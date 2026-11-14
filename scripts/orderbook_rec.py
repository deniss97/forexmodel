"""Снимки стакана 21 акции (глубина 20) через API Т-Инвестиций — бесплатно копим свои данные о стакане (обсуждение 14.11:
прежде чем покупать журнал заявок Мосбиржи, накопить и проверить идеи на своих снимках).

Каждую минуту (cron) — 4 снимка, раз в 15 с: по всем бумагам параллельно GetOrderBook depth 20. Строка JSON на снимок:
{"s": бумага, "t": время стакана UTC, "lp": последняя цена, "b": [[цена, лоты] × 20], "a": [[цена, лоты] × 20]}.
На сервере — только текущий час (Денис 14.11: «данные сразу на Яндекс Диск»): data/raw/orderbook/<дата>_<ЧЧ>.jsonl
(все бумаги в одном файле, час по МСК); законченный час → gzip → yadisk:forexmodel/стакан/<год>/<дата>/<ЧЧ>.jsonl.gz
(rclone) и удаляется с сервера — 17 файлов в день, ~1 МБ каждый. Вне торговых часов (06:50–23:55 МСК) — только выгрузка. Учёт — reports/live/orderbook_manifest.json; сбой — в группу не чаще раза в 6 часов.

cron (часы сервера +05):  * * * * *  /home/dev/forexmodel/scripts/queues/orderbook_rec.sh
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forexmodel.broker.tinvest import TInvest, q2d  # noqa: E402

OUT = ROOT / "data" / "raw" / "orderbook"
UIDS = ROOT / "reports" / "live" / "orderbook_uids.json"
MANIFEST = ROOT / "reports" / "live" / "orderbook_manifest.json"
RCLONE = Path.home() / "bin" / "rclone"
REMOTE = "yadisk:forexmodel/стакан"
MSK = timezone(timedelta(hours=3))
SECIDS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
          "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS", "MOEX", "MTSS"]
SNAPS, EVERY = 4, 15


def uids(api: TInvest) -> dict[str, str]:
    m = json.loads(UIDS.read_text()) if UIDS.exists() else {}
    for s in SECIDS:
        if s not in m:
            m[s] = api.share(s).uid
    UIDS.parent.mkdir(parents=True, exist_ok=True)
    UIDS.write_text(json.dumps(m, indent=1))
    return m


def px(lv: list[dict]) -> list[list]:
    return [[float(q2d(x["price"])), int(x["quantity"])] for x in lv]


def snap(api: TInvest, uid: str) -> dict | None:
    try:
        r = api.call("MarketDataService", "GetOrderBook", {"instrumentId": uid, "depth": 20}, retries=2)
    except Exception:  # noqa: BLE001 — один пропущенный снимок не страшен
        return None
    return {"t": r.get("orderbookTs"), "lp": float(q2d(r.get("lastPrice"))) if r.get("lastPrice") else None,
            "b": px(r.get("bids", [])), "a": px(r.get("asks", []))}


def upload_finished(current: str, man: dict) -> list[str]:
    """Все часовые файлы, кроме текущего часа, → Диск; после успешной выгрузки — удалить с сервера."""
    errors = []
    for f in sorted(OUT.glob("*.jsonl")):
        if f.stem >= current:
            continue
        date, hh = f.stem.split("_")
        gz = f.with_suffix(".jsonl.gz")
        with open(f, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        r = subprocess.run([str(RCLONE), "copyto", str(gz), f"{REMOTE}/{date[:4]}/{date}/{hh}.jsonl.gz",
                            "--stats", "0", "--retries", "3"], capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            man.setdefault(date, {})[hh] = {"lines": sum(1 for _ in open(f)), "mb": round(gz.stat().st_size / 2**20, 2)}
            f.unlink()
        else:
            errors.append(f"{f.stem}: не выгрузилось на Диск")
        gz.unlink(missing_ok=True)
    return errors


def main() -> int:
    now = datetime.now(MSK)
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    current = now.strftime("%Y-%m-%d_%H")
    errors = upload_finished(current, man)
    if not (now.replace(hour=6, minute=50) <= now <= now.replace(hour=23, minute=55)):
        MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        return 0
    api = TInvest(sandbox=False)
    m = uids(api)
    start = time.time()
    with ThreadPoolExecutor(8) as ex:
        for k in range(SNAPS):
            res = dict(zip(SECIDS, ex.map(lambda s: snap(api, m[s]), SECIDS)))
            hour_file = OUT / f"{datetime.now(MSK).strftime('%Y-%m-%d_%H')}.jsonl"
            with open(hour_file, "a") as fh:
                for s, r in res.items():
                    if r and (r["b"] or r["a"]):
                        fh.write(json.dumps({"s": s, **r}, separators=(",", ":")) + "\n")
            if k < SNAPS - 1:
                time.sleep(max(0.0, start + EVERY * (k + 1) - time.time()))
    if errors:
        last = man.get("_alert")
        if not last or time.time() - float(last) > 6 * 3600:
            import notify  # noqa: PLC0415
            notify.send("⚠️ Запись стакана (Т-Инвестиции → Яндекс Диск):\n" + "\n".join(errors[:10]))
            man["_alert"] = time.time()
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
