"""Сделки Мосбиржи текущего торгового дня с направлением (ISS, бесплатно, задержка 15 мин) → по дням на сервере.

Для страницы «Объём по уровням LKOH» (H84, для Тагира): лента сделок ALGOPACK у нас кончается 06.04.2026, а историю сделок
биржа бесплатно не отдаёт — только сделки текущего дня. Поэтому сделки собираются каждую ночь и сразу сворачиваются в то,
что нужно странице: часовые свечи и объём покупок / продаж по уровням цены с шагом 1 ₽ (сторона — BUYSELL, инициатор
сделки). Сырые сделки — data/raw/moex_trades/<бумага>/<дата>.csv.gz (дубли по TRADENO убираются), свёртки —
data/raw/moex_trades/<бумага>/agg/<дата>.json. Дата — календарная (TRADEDATE): сделки сессии выходного дня биржа относит
к торговому дню понедельника, а здесь они лежат в своих днях.

cron (часы сервера +05): 00:20 и 06:20 по Москве — после вечерней сессии и до утренней (повтор не задваивает):
    20 2,8 * * *  /home/dev/forexmodel/scripts/queues/moex_trades_nightly.sh

    python scripts/moex_trades.py LKOH            # собрать сделки текущего дня и пересчитать свёртки
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "moex_trades"
ISS = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities/{secid}/trades.json"
COLS = ["TRADENO", "TRADEDATE", "TRADETIME", "PRICE", "QUANTITY", "BUYSELL", "TRADINGSESSION", "TRADE_SESSION_DATE"]
PAGE = 5000


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


def fetch(secid: str, board: str = "TQBR") -> pd.DataFrame:
    """Все сделки текущего торгового дня по бумаге — постранично по 5000."""
    parts, start = [], 0
    while True:
        j = get(ISS.format(board=board, secid=secid) + f"?iss.meta=off&iss.only=trades&start={start}&limit={PAGE}")["trades"]
        df = pd.DataFrame(j["data"], columns=j["columns"])
        if df.empty:
            break
        parts.append(df[COLS])
        if len(df) < PAGE:
            break
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


def collect(secid: str, board: str = "TQBR") -> list[str]:
    new = fetch(secid, board)
    if new.empty:
        print(f"[{secid}] сделок нет (выходной или биржа не ответила)", flush=True)
        return []
    base = OUT / secid
    (base / "agg").mkdir(parents=True, exist_ok=True)
    days = []
    for date, part in new.groupby("TRADEDATE"):
        raw = base / f"{date}.csv.gz"
        old = pd.read_csv(raw, dtype={"TRADINGSESSION": str}) if raw.exists() else pd.DataFrame(columns=COLS)
        allt = pd.concat([old, part], ignore_index=True).drop_duplicates("TRADENO").sort_values("TRADENO")
        allt.to_csv(raw, index=False, compression="gzip")
        agg = aggregate(allt)
        (base / "agg" / f"{date}.json").write_text(json.dumps(agg, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"[{secid}] {date}: сделок {len(allt)} (+{len(allt) - len(old)}), часов {len(agg['candles'])}, "
              f"уровней {len(agg['levels'])}, без направления {agg['undirected']}", flush=True)
        days.append(str(date))
    return days


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("secids", nargs="*", default=["LKOH"])
    ap.add_argument("--board", default="TQBR")
    a = ap.parse_args(argv)
    ok = True
    for s in a.secids:
        try:
            collect(s.upper(), a.board)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[{s}] ошибка: {e}", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
