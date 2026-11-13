"""Обезличенные сделки Т-Инвестиций по 21 акции → Яндекс Диск (forexmodel/лента Т-Инвестиций/<бумага>/<год>/<дата>.csv.gz).

Источник — тот же, что в скрипте Тимура (T_trades_data.ipynb, прислан 12.11): открытый архив
https://invest-public-api.tinkoff.ru/history-trades/<дата>?instrumentId=<бумага>_TQBR (без токена, лимит 30 запросов в
минуту; колонки TRADE_TS, TICKER_CC, DIRECTION, PRICE, QUANTITY, TRADE_SOURCE, INSTRUMENT_UID — как у ленты ALGOPACK на
Диске, которая кончается 06.04.2026). Файл «все инструменты за день» (без instrumentId) с нашего сервера не скачать:
сервер отдаёт ~450 КБ/с и закрывает поток через 60–80 с, а день весит 50–90 МБ (проверено 12.11). По бумаге день —
0.03–1 МБ. Сервер подписан сертификатом Минцифры (Russian Trusted Root CA) — его нет в системном хранилище, поэтому он
лежит в configs/certs/ и передаётся только этим запросам.

Файл проверяется на целостность gzip (оборван — повтор), день по всем бумагам уходит на Диск одним rclone copy, с
сервера удаляется. Пары «бумага/дата», которые есть на Диске, пропускаются; пустые (нет торгов) записываются в учёт
reports/live/yadisk_tinvest_manifest.json и больше не запрашиваются.

    python scripts/tinvest_tape.py                         # с 07.04.2026 по вчера (по Москве), только недостающее
    python scripts/tinvest_tape.py LKOH --start 2026-04-07 --end 2026-04-10
    python scripts/tinvest_tape.py --alert                 # сбой — уведомление в группу (для cron)

cron (часы сервера +05) — раз в сутки, мимо почасового контура:
    15 6 * * *  /home/dev/forexmodel/scripts/queues/tinvest_tape_nightly.sh
"""
from __future__ import annotations

import argparse
import gzip
import json
import ssl
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CA = ROOT / "configs" / "certs" / "russian_trusted_root_ca.pem"
MANIFEST = ROOT / "reports" / "live" / "yadisk_tinvest_manifest.json"
TMP = ROOT / "data" / "raw" / "tinvest_tape_tmp"
RCLONE = Path.home() / "bin" / "rclone"
REMOTE = "yadisk:forexmodel/лента Т-Инвестиций"
URL = "https://invest-public-api.tinkoff.ru/history-trades/{d}?instrumentId={secid}_TQBR"
FIRST = "2026-04-07"   # лента ALGOPACK на Диске («обезличенные сделки», все инструменты) кончается 06.04.2026
PAUSE = 1.5            # лимит сервера — 30 запросов в минуту
# как в scripts/moex_trades.py: 19 бумаг контура + MOEX, MTSS
SECIDS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
          "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS", "MOEX", "MTSS"]
HEADER = b"TRADE_TS,TICKER_CC,DIRECTION,PRICE,QUANTITY,TRADE_SOURCE,INSTRUMENT_UID"
MSK = timezone(timedelta(hours=3))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def on_disk() -> set[str]:
    """Уже лежащие на Диске пары «бумага/дата»."""
    r = subprocess.run([str(RCLONE), "lsf", "-R", "--files-only", REMOTE], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and "directory not found" not in r.stderr:
        raise RuntimeError(r.stderr[-300:])
    return {p.split("/")[0] + "/" + Path(p).name.removesuffix(".csv.gz") for p in r.stdout.split() if p.endswith(".csv.gz")}


def gzip_ok(path: Path) -> tuple[bool, int]:
    """Файл распаковывается до конца и начинается с ожидаемого заголовка; число строк данных."""
    try:
        with gzip.open(path, "rb") as f:
            if f.readline().strip() != HEADER:
                return False, 0
            n = sum(1 for _ in f)
        return True, n
    except (OSError, EOFError):
        return False, 0


def download(secid: str, d: str, ctx: ssl.SSLContext, tries: int = 5) -> tuple[str, Path | None, int]:
    """('ok', файл, строк) | ('empty', None, 0) | ('fail', None, 0)."""
    out = TMP / secid / d[:4] / f"{d}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(d=d, secid=secid), headers={"User-Agent": "forexmodel-tape/1.0"})
            with urllib.request.urlopen(req, timeout=90, context=ctx) as r, open(out, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            ok, n = gzip_ok(out)
            if ok:
                return ("ok", out, n) if n else ("empty", None, 0)
            print(f"  {secid} {d}: файл оборван, повтор {k + 1}", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                out.unlink(missing_ok=True)
                return "empty", None, 0
            print(f"  {secid} {d}: HTTP {e.code}, повтор {k + 1}", flush=True)
            if e.code == 429:
                time.sleep(60)
        except Exception as e:  # noqa: BLE001
            print(f"  {secid} {d}: {type(e).__name__}: {e}, повтор {k + 1}", flush=True)
        time.sleep(5 * (k + 1))
    out.unlink(missing_ok=True)
    return "fail", None, 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("secids", nargs="*", default=SECIDS)
    ap.add_argument("--start", default=FIRST)
    ap.add_argument("--end")
    ap.add_argument("--alert", action="store_true")
    a = ap.parse_args(argv)

    m = load_manifest()
    have = on_disk()
    empty = set(m.get("_empty", []))
    end = date.fromisoformat(a.end) if a.end else datetime.now(MSK).date() - timedelta(days=1)
    ctx = ssl.create_default_context(cafile=str(CA))

    errors = []
    d = date.fromisoformat(a.start)
    while d <= end:
        ds = d.isoformat()
        d += timedelta(days=1)
        todo = [s for s in a.secids if f"{s}/{ds}" not in have and f"{s}/{ds}" not in empty]
        if not todo:
            continue
        t0, got, rows = time.time(), [], 0
        for s in todo:
            status, path, n = download(s, ds, ctx)
            time.sleep(PAUSE)
            if status == "empty":
                empty.add(f"{s}/{ds}")
            elif status == "fail":
                errors.append(f"{s} {ds}: не скачалось")
            else:
                got.append(s)
                rows += n
                m.setdefault(s, {})[ds] = n
        if got:
            r = subprocess.run([str(RCLONE), "copy", str(TMP), REMOTE, "--stats", "0", "--retries", "3"],
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                errors.append(f"{ds}: не выгрузилось на Диск")
                for s in got:
                    m[s].pop(ds, None)
        shutil.rmtree(TMP, ignore_errors=True)
        m["_empty"] = sorted(empty)
        save_manifest(m)
        print(f"{ds}: {len(got)} бумаг, {rows:,} сделок, {time.time() - t0:.0f} с", flush=True)

    if errors:
        print("Сбои:\n" + "\n".join(errors[:50]), flush=True)
        if a.alert:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            import notify
            notify.send("⚠️ Лента Т-Инвестиций (раз в сутки на Яндекс Диск):\n" + "\n".join(errors[:10])
                        + "\nПовтор — следующей ночью.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
