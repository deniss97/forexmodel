"""Часовые свечи и признаки перебора для живого контура — дописываются по новым минутам каждый час (ТЗ инференса 26.10).

Ночной пересчёт (live_iterator до 26.10) каждый раз разбирал файл минуток целиком (2 млн строк) и строил ~100 фильтров
перебора — ~40 с на бумагу. Для сигналов S46 и S48 раз в час так нельзя: на 19 бумаг ушло бы 13 минут. Здесь:
  * часовые свечи с поправкой на дивиденды лежат в reports/live/bars/<бумага>.pkl; каждый запуск читает только хвост
    файла минуток (с начала последней свечи — она могла быть неполной) и пересчитывает последние часы. Минуты после
    последней отсечки поправки не требуют; если в data/dividends/moex_dividends.csv появилась новая отсечка — полная
    пересборка;
  * из фильтров перебора считаются только те, что нужны S46 и S48 (EMA200 4ч, EMA200 дни, Supertrend 4ч), остальные
    помечены «нет фильтра» (99) — движку перебора они не нужны.
Свечи те же, что у полной сборки (resample по ровным часам), поэтому сделки совпадают с журналом — проверено на SBER и
TATN (tests / docs/results/inference_tz.md).
"""
from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from experiments import NEW_STOCKS, ROOT

OUT = ROOT / "reports" / "live" / "bars"
FEAT = OUT / "feat"
DIVS = ROOT / "data" / "dividends" / "moex_dividends.csv"
LAUNCH = ["sber", "gazp", "lkoh"]
NEED_GATES = {"ema_price:200 @4h", "ema_price:200 @24h", "supertrend:10:3 @4h"}    # фильтры и выход S46, S48


def minutes_csv(inst: str) -> Path:
    """SBER, GAZP, LKOH — живые файлы контура; новые бумаги H66 — data/raw/moex (докачка в тот же файл)."""
    if inst in LAUNCH:
        return ROOT / "data" / "raw" / "live" / f"{inst.upper()}_1m_live.csv"
    if inst in NEW_STOCKS:
        return ROOT / "data" / "raw" / "moex" / f"{inst.upper()}_1m.csv"
    raise KeyError(inst)


def tail_minutes(csv: Path, since: pd.Timestamp, time_col: str = "begin") -> pd.DataFrame:
    """Минуты с time >= since: поиск места в файле делением пополам по байтам (время в файле по возрастанию)."""
    key = since.strftime("%Y-%m-%d %H:%M:%S").encode()
    size = csv.stat().st_size
    with open(csv, "rb") as fh:
        header = fh.readline().decode().strip().split(",")
        ti = header.index(time_col)
        first = fh.tell()

        def time_after(pos: int) -> bytes | None:
            fh.seek(pos)
            if pos > first:
                fh.readline()                          # дочитать оборванную строку
            line = fh.readline()
            return line.split(b",")[ti] if line.strip() else None

        lo, hi = first, size
        while hi - lo > 1 << 16:
            mid = (lo + hi) // 2
            t = time_after(mid)
            if t is None or t >= key:
                hi = mid
            else:
                lo = mid
        fh.seek(lo)
        if lo > first:
            fh.readline()
        data = fh.read()
    df = pd.read_csv(io.BytesIO(data), names=header, header=None)
    df = df[df[time_col].astype(str) >= key.decode()]
    out = pd.DataFrame({"time": pd.to_datetime(df[time_col])})
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = df[c].astype(float).to_numpy()
    return out.reset_index(drop=True)


def dividends(inst: str) -> pd.DataFrame:
    from forexmodel.data.dividends import load_dividends

    return load_dividends(DIVS, inst.upper())


def hourly(inst: str) -> pd.DataFrame:
    """Часовые свечи с поправкой на дивиденды, дописанные до последней минуты файла."""
    from forexmodel.data.dividends import adjust_for_dividends
    from forexmodel.data.loader import load_minute_compact, resample_ohlcv

    OUT.mkdir(parents=True, exist_ok=True)
    p, csv, divs = OUT / f"{inst}.pkl", minutes_csv(inst), dividends(inst)
    key = [(str(a), float(b)) for a, b in zip(divs["ex_time"], divs["amount"])]
    if p.exists():
        saved = pd.read_pickle(p)
        if saved["divs"] == key:
            h = saved["bars"]
            last = h["time"].iloc[-1]
            new = tail_minutes(csv, last)               # с начала последней свечи: она могла дополниться
            if len(new):
                h = pd.concat([h[h["time"] < last], resample_ohlcv(new, "1h")], ignore_index=True)
                pd.to_pickle({"bars": h, "divs": key}, p)
            return h
    m = load_minute_compact(csv, "begin", None, ["volume"])         # полная сборка: впервые или новая отсечка
    m = adjust_for_dividends(m, divs)
    h = resample_ohlcv(m, "1h").reset_index(drop=True)
    pd.to_pickle({"bars": h, "divs": key}, p)
    return h


def gates(h: pd.DataFrame) -> np.ndarray:
    """Фильтры перебора в порядке trend_grid.gate_items(): нужные S46 и S48 — посчитаны, остальные 99 («нет фильтра»)."""
    import trend_grid as tg
    from trend_filter_lab import base_config

    tg._W["base"] = dataclasses.replace(base_config(), **tg.GATE_C)
    items = tg.gate_items()
    out = np.full((len(items), len(h)), 99, np.int8)
    b = h[["time", "open", "high", "low", "close"]]
    for k, (_, name, params) in enumerate(items):
        if name in NEED_GATES:
            g = tg.gate_values(b, params)
            out[k] = np.where(np.isfinite(g), g, 99).astype(np.int8)
    return out


def features(inst: str) -> Path:
    """Признаки для движка перебора по свежим свечам → reports/live/bars/feat/<бумага>.pkl; возвращает папку."""
    from night_search import build_features

    FEAT.mkdir(parents=True, exist_ok=True)
    h = hourly(inst)
    pd.to_pickle({"bars": h[["time", "open", "high", "low", "close"]], "f": build_features(h, inst), "gates": gates(h)},
                 FEAT / f"{inst}.pkl")
    return FEAT
