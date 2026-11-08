"""H84 (Тимур, для Тагира): страница «Объём по уровням LKOH» — часовые свечи недели и объём покупок / продаж по уровням цены.

Данные по неделям (с понедельника 00:00 МСК, вместе с сессиями выходного дня):
  * история — секундные агрегаты обезличенных сделок data/raw/orderflow_lkoh_1s.parquet (TQBR, 01.2024 — 06.04.2026);
    уровень секунды — её средневзвешенная цена вниз до рубля; считается один раз → reports/h84/history.json (--history);
  * свежие дни — сделки Мосбиржи, которые каждую ночь собирает scripts/moex_trades.py (data/raw/moex_trades/LKOH/agg).
Покупка / продажа — по стороне инициатора сделки. Шаги 5 и 10 ₽ страница собирает сама из шага 1 ₽.

Выход — две версии одной страницы из docs/results/h84_volume/page.template.html:
  * docs/results/h84_volume/index.html — артефакт (Plotly с cdnjs, данные внутри);
  * docs/results/h84_volume/lkoh_volume_levels.html — один файл для Тагира: Plotly и данные внутри, открывается двойным
    щелчком без интернета и без входа на claude.ai.

    python scripts/h84_volume_weeks.py --history     # пересчитать историю из ленты (долго, один раз)
    python scripts/h84_volume_weeks.py               # собрать страницу: история + собранные дни
"""
from __future__ import annotations

import argparse
import gc
import json
import urllib.request

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from experiments import ROOT

SRC = ROOT / "data" / "raw" / "orderflow_lkoh_1s.parquet"
AGG = ROOT / "data" / "raw" / "moex_trades" / "LKOH" / "agg"
CACHE = ROOT / "reports" / "h84"
HIST = CACHE / "history.json"
DIR = ROOT / "docs" / "results" / "h84_volume"
TEMPLATE = DIR / "page.template.html"
PLOTLY_URL = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.34.0/plotly.min.js"
PLOTLY = CACHE / "plotly-2.34.0.min.js"


def monday(ts: pd.Series | pd.DatetimeIndex):
    return ts.normalize() - pd.to_timedelta(ts.weekday, unit="D")


def build_history() -> dict:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(SRC)
    hours, levels = [], []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=["time", "price", "vwap", "volume", "buy_volume", "sell_volume"]).to_pandas()
        t = pd.to_datetime(df["time"])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(None)
        df["time"] = t + pd.Timedelta(hours=3)                                   # UTC → МСК
        df = df[df["volume"] > 0].sort_values("time")
        g = df.groupby(df["time"].dt.floor("1h"))
        hours.append(pd.DataFrame({"t0": g["time"].min(), "o": g["price"].first(), "h": g["price"].max(),
                                   "l": g["price"].min(), "c": g["price"].last(), "v": g["volume"].sum()}))
        px = df["vwap"].where(df["vwap"] > 0, df["price"])
        df["lvl"] = np.floor(px).astype("int32")
        df["week"] = monday(df["time"].dt)
        levels.append(df.groupby(["week", "lvl"])[["buy_volume", "sell_volume"]].sum())
        del df, g, t, px
        gc.collect()
    H = pd.concat(hours).rename_axis("hour").reset_index().sort_values(["hour", "t0"])
    H = H.groupby("hour").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
    L = pd.concat(levels).groupby(level=[0, 1]).sum().reset_index()
    H["week"] = monday(H.index)
    weeks = {}
    for wk, hc in H.groupby("week"):
        lv = L[L["week"] == wk].sort_values("lvl")
        weeks[wk.strftime("%Y-%m-%d")] = {
            "candles": [[ts.strftime("%Y-%m-%dT%H:%M"), float(r.o), float(r.h), float(r.l), float(r.c), int(r.v)] for ts, r in hc.iterrows()],
            "levels": [[int(r.lvl), int(round(r.buy_volume)), int(round(r.sell_volume))] for r in lv.itertuples()],
        }
    out = {"from": H.index.min().strftime("%Y-%m-%d"), "to": H.index.max().strftime("%Y-%m-%d"), "weeks": weeks}
    CACHE.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"история: недель {len(weeks)}, {out['from']} … {out['to']} → {HIST.relative_to(ROOT)}", flush=True)
    return out


def recent() -> tuple[dict, list[str]]:
    """Свёртки собранных дней → недели (свечи подряд, объём по уровням суммой)."""
    weeks, days = {}, []
    for p in sorted(AGG.glob("*.json")) if AGG.exists() else []:
        d = pd.Timestamp(p.stem)
        a = json.loads(p.read_text(encoding="utf-8"))
        if not a.get("candles"):
            continue
        days.append(p.stem)
        wk = (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d")
        w = weeks.setdefault(wk, {"candles": [], "lv": {}})
        w["candles"] += a["candles"]
        for lvl, b, s in a["levels"]:
            x = w["lv"].setdefault(lvl, [0, 0])
            x[0] += b; x[1] += s
    for w in weeks.values():
        w["candles"].sort(key=lambda c: c[0])
        w["levels"] = [[k, v[0], v[1]] for k, v in sorted(w.pop("lv").items())]
    return weeks, days


def merge(hist: dict, rec: dict) -> list[dict]:
    allw = {}
    for src in (hist, rec):
        for wk, w in src.items():
            if wk not in allw:
                allw[wk] = {"candles": list(w["candles"]), "levels": [list(x) for x in w["levels"]]}
                continue
            m = allw[wk]                                                   # неделя на стыке источников
            seen = {c[0] for c in m["candles"]}
            m["candles"] = sorted(m["candles"] + [c for c in w["candles"] if c[0] not in seen], key=lambda c: c[0])
            lv = {k: [b, s] for k, b, s in m["levels"]}
            for k, b, s in w["levels"]:
                x = lv.setdefault(k, [0, 0]); x[0] += b; x[1] += s
            m["levels"] = [[k, v[0], v[1]] for k, v in sorted(lv.items())]
    return [{"start": wk, **allw[wk]} for wk in sorted(allw) if len(allw[wk]["candles"]) >= 3 and allw[wk]["levels"]]


def plotly_js() -> str:
    if not PLOTLY.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(PLOTLY_URL, headers={"User-Agent": "forexmodel-h84/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            PLOTLY.write_bytes(r.read())
    return PLOTLY.read_text(encoding="utf-8")


def pages(data: dict) -> None:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    head, body = tpl.split("<!--BODY-->", 1)
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = body.replace("/*DATA*/", js, 1)
    (DIR / "index.html").write_text(head + body.replace("<!--PLOTLY-->", f'<script src="{PLOTLY_URL}"></script>', 1), encoding="utf-8")
    inline = "<script>\n" + plotly_js().replace("</script", "<\\/script") + "\n</script>"
    local = ("<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
             "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
             + head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "") + "</head>\n<body>\n"
             + body.replace("<!--PLOTLY-->", inline, 1) + "\n</body>\n</html>\n")
    (DIR / "lkoh_volume_levels.html").write_text(local, encoding="utf-8")
    for f in ("index.html", "lkoh_volume_levels.html"):
        print(f"→ {(DIR / f).relative_to(ROOT)} ({(DIR / f).stat().st_size / 1e6:.1f} МБ)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", action="store_true", help="пересчитать историю из секундной ленты")
    a = ap.parse_args(argv)
    hist = build_history() if a.history or not HIST.exists() else json.loads(HIST.read_text(encoding="utf-8"))
    rec, days = recent()
    weeks = merge(hist["weeks"], rec)
    meta = {"ticker": "LKOH", "board": "TQBR", "unit": "акций (лот LKOH — 1 акция)", "tz": "МСК",
            "from": hist["from"], "to": max([hist["to"]] + days),
            "history": f"{hist['from']} — {hist['to']}", "fresh": f"{days[0]} — {days[-1]}" if days else "",
            "weeks": len(weeks),
            "source": "обезличенные сделки Мосбиржи: секундные агрегаты ALGOPACK до " + hist["to"]
                      + (f", дальше — сделки биржи, собранные с {days[0]}" if days else ""),
            "side": "по стороне инициатора сделки", "level": "цена сделки вниз до рубля",
            "built": pd.Timestamp.now(tz="Europe/Moscow").strftime("%d.%m.%Y %H:%M")}
    pages({"meta": meta, "weeks": weeks})
    print(f"недель {len(weeks)}; история {meta['history']}; свежие дни: {meta['fresh'] or 'нет'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
