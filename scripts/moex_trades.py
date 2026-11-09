"""Сделки Мосбиржи текущего торгового дня с направлением (ISS, бесплатно, задержка 15 мин) → по дням на Яндекс Диск.

Историю сделок биржа бесплатно не отдаёт — только сделки текущего дня, а лента Т-Инвестиций (GetLastTrades) — только за
последний час (проверено 23.10 и 09.11). Лента ALGOPACK на Диске («обезличенные сделки») кончается 06.04.2026. Поэтому
свежую ленту копим сами (созвон 09.11, решение Дениса): каждые 15 минут дозабираем новые сделки по номеру
(tradeno + next_trade, без повторной загрузки дня), а законченные дни уходят на Диск.

  * горячий день — data/raw/moex_trades/<бумага>/<дата>.csv (дописывается; дубли по TRADENO не попадают);
  * законченный день (дата раньше сегодняшней по Москве) → свёртка agg/<дата>.json (часовые свечи и объём покупок /
    продаж по рублю цены — её читает страница H84 «Объём по уровням LKOH» и разборы ленты), затем
    yadisk:forexmodel/лента ISS/<бумага>/<год>/<дата>.csv.gz через rclone copyto (API, сверка хэша); после загрузки
    сырой файл на сервере удаляется, учёт — reports/live/yadisk_trades_manifest.json. Сбой — уведомление в группу.

Дата — календарная (TRADEDATE): сделки сессии выходного дня биржа относит к торговому дню понедельника, а здесь они
лежат в своих днях. Колонки: TRADENO, TRADEDATE, TRADETIME, PRICE, QUANTITY, BUYSELL (инициатор: B / S),
TRADINGSESSION, TRADE_SESSION_DATE.

cron (часы сервера +05): каждые 15 минут, мимо почасового контура в :00
    5,20,35,50 * * * *  /home/dev/forexmodel/scripts/queues/moex_trades_nightly.sh

    python scripts/moex_trades.py                 # все бумаги контура: дозабрать сделки, выгрузить законченные дни
    python scripts/moex_trades.py LKOH SBER       # только эти
    python scripts/moex_trades.py --no-upload     # без выгрузки на Диск
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "moex_trades"
MANIFEST = ROOT / "reports" / "live" / "yadisk_trades_manifest.json"
RCLONE = Path.home() / "bin" / "rclone"
REMOTE = "yadisk:forexmodel/лента ISS"
ISS = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities/{secid}/trades.json"
COLS = ["TRADENO", "TRADEDATE", "TRADETIME", "PRICE", "QUANTITY", "BUYSELL", "TRADINGSESSION", "TRADE_SESSION_DATE"]
PAGE = 5000
# 19 бумаг контура (S42 — SBER, GAZP, LKOH; S46 / S48 — 16 бумаг индекса) + MOEX, MTSS из первых выгрузок
SECIDS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
          "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS", "MOEX", "MTSS"]


def get(url: str, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "forexmodel-trades/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))
    return {}


def fetch(secid: str, board: str = "TQBR", after: int | None = None) -> pd.DataFrame:
    """Сделки текущего торгового дня по бумаге — все или только с номером больше `after`, страницами по 5000."""
    parts = []
    start = 0
    while True:
        q = f"?iss.meta=off&iss.only=trades&limit={PAGE}&trades.columns={','.join(COLS)}"
        q += f"&tradeno={after}&next_trade=1" if after else f"&start={start}"
        j = get(ISS.format(board=board, secid=secid) + q)["trades"]
        df = pd.DataFrame(j["data"], columns=j["columns"])
        if df.empty:
            break
        parts.append(df[COLS])
        if len(df) < PAGE:
            break
        if after:
            after = int(df["TRADENO"].max())
        else:
            start += len(df)
        time.sleep(0.2)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def aggregate(day: pd.DataFrame) -> dict:
    """Сделки одного календарного дня → часовые свечи и объём покупок / продаж по рублю цены."""
    d = day.copy()
    d["ts"] = pd.to_datetime(d["TRADEDATE"] + " " + d["TRADETIME"])
    d = d.sort_values(["ts", "TRADENO"])
    d["hour"] = d["ts"].dt.floor("1h")
    g = d.groupby("hour")
    candles = [[h.strftime("%Y-%m-%dT%H:%M"), float(r.o), float(r.h), float(r.l), float(r.c), int(r.v)]
               for h, r in pd.DataFrame({"o": g["PRICE"].first(), "h": g["PRICE"].max(), "l": g["PRICE"].min(),
                                         "c": g["PRICE"].last(), "v": g["QUANTITY"].sum()}).iterrows()]
    d["lvl"] = np.floor(d["PRICE"].astype(float)).astype(int)
    d["buy"] = np.where(d["BUYSELL"] == "B", d["QUANTITY"], 0)
    d["sell"] = np.where(d["BUYSELL"] == "S", d["QUANTITY"], 0)
    lv = d.groupby("lvl")[["buy", "sell"]].sum()
    return {"candles": candles, "levels": [[int(k), int(r.buy), int(r.sell)] for k, r in lv.iterrows()],
            "trades": int(len(d)), "undirected": int((~d["BUYSELL"].isin(["B", "S"])).sum())}


def last_tradeno(base: Path, manifest: dict) -> int | None:
    """Номер последней сохранённой сделки (номера растут сквозь дни): по горячему файлу, а после выгрузки дня — по
    манифесту, иначе сразу после полуночи ISS отдал бы вчерашний день заново."""
    known = [manifest.get(base.name, {}).get("_last_tradeno") or 0]
    files = sorted(base.glob("*.csv"))
    if files:
        tail = pd.read_csv(files[-1], usecols=["TRADENO"])["TRADENO"]
        known.append(int(tail.max()) if len(tail) else 0)
    return max(known) or None


def migrate_gz(base: Path) -> None:
    """Сырые дни старого формата (<дата>.csv.gz, собирались целиком) → горячий .csv, чтобы выгрузка шла одним путём."""
    for gz in base.glob("*.csv.gz"):
        plain = gz.with_suffix("")
        if not plain.exists():
            with gzip.open(gz, "rt") as src, open(plain, "w") as dst:
                shutil.copyfileobj(src, dst)
        gz.unlink()


def collect(secid: str, manifest: dict, board: str = "TQBR") -> int:
    base = OUT / secid
    base.mkdir(parents=True, exist_ok=True)
    migrate_gz(base)
    after = last_tradeno(base, manifest)
    new = fetch(secid, board, after)
    if after is not None and not new.empty:
        new = new[new["TRADENO"] > after]
    if new.empty:
        return 0
    for date, part in new.groupby("TRADEDATE"):
        raw = base / f"{date}.csv"
        part.sort_values("TRADENO").to_csv(raw, mode="a", header=not raw.exists(), index=False)
    return len(new)


def finished_days(today: str) -> list[tuple[str, Path]]:
    """Горячие файлы дней раньше сегодняшнего (по Москве): сессия кончилась, хвост уже дозабран."""
    return [(p.parent.name, p) for p in sorted(OUT.glob("*/*.csv")) if p.stem < today]


def upload_day(secid: str, raw: Path, manifest: dict) -> None:
    """Законченный день: свёртка → .csv.gz на Диск → сырой файл с сервера удалить."""
    date = raw.stem
    day = pd.read_csv(raw, dtype={"TRADINGSESSION": str}).drop_duplicates("TRADENO").sort_values("TRADENO")
    (raw.parent / "agg").mkdir(exist_ok=True)
    (raw.parent / "agg" / f"{date}.json").write_text(json.dumps(aggregate(day), ensure_ascii=False,
                                                                separators=(",", ":")), encoding="utf-8")
    gz = raw.with_name(f"{date}.csv.gz")
    day.to_csv(gz, index=False, compression="gzip")
    try:
        subprocess.run([str(RCLONE), "copyto", str(gz), f"{REMOTE}/{secid}/{date[:4]}/{date}.csv.gz", "--stats", "0",
                        "--retries", "3"], check=True, capture_output=True, text=True, timeout=1800)
    finally:
        gz.unlink(missing_ok=True)
    m = manifest.setdefault(secid, {})
    m["_last_tradeno"] = max(int(m.get("_last_tradeno") or 0), int(day["TRADENO"].max()))
    m[date] = {"trades": int(len(day)), "undirected": int((~day["BUYSELL"].isin(["B", "S"])).sum()),
                                            "uploaded": pd.Timestamp.now().isoformat(timespec="seconds")}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    raw.unlink()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("secids", nargs="*", default=SECIDS)
    ap.add_argument("--board", default="TQBR")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args(argv)
    errors = []
    got = {}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    for s in a.secids:
        try:
            got[s.upper()] = collect(s.upper(), manifest, a.board)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s}: сбор — {str(e)[:150]}")
    print("[сбор] " + (", ".join(f"{k} +{v}" for k, v in got.items() if v) or "новых сделок нет"), flush=True)
    if not a.no_upload:
        today = pd.Timestamp.now(tz="Europe/Moscow").strftime("%Y-%m-%d")
        for secid, raw in finished_days(today):
            try:
                upload_day(secid, raw, manifest)
                print(f"[Диск] {secid} {raw.stem}: выгружено", flush=True)
            except Exception as e:  # noqa: BLE001  — сырой день остаётся на сервере, повтор через 15 минут
                msg = e.stderr.strip().splitlines()[-1] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
                errors.append(f"{secid} {raw.stem}: выгрузка — {msg[:150]}")
    for e in errors:
        print(e, file=sys.stderr, flush=True)
    alert(errors, manifest)
    return 1 if errors else 0


def alert(errors: list[str], manifest: dict) -> None:
    """Сбой — в группу, но не чаще раза в 6 часов (скрипт идёт каждые 15 минут)."""
    if not errors:
        return
    last = manifest.get("_alert")
    if last and pd.Timestamp.now() - pd.Timestamp(last) < pd.Timedelta("6h"):
        return
    import notify
    notify.send("⚠️ Лента сделок Мосбиржи (сбор каждые 15 мин, выгрузка на Яндекс Диск):\n" + "\n".join(errors[:10])
                + "\nНевыгруженные дни лежат на сервере, повтор — автоматически.")
    manifest["_alert"] = pd.Timestamp.now().isoformat(timespec="seconds")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
