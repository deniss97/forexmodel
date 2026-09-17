"""Загрузка минутных котировок и ресемплинг на рабочий ТФ."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

OHLC = ("open", "high", "low", "close")


def load_minute_csv(
    path: Path | str,
    time_col: str = "begin",
    volume_candidates: Sequence[str] = ("value", "volume"),
) -> pd.DataFrame:
    """Читает минутный CSV и приводит его к виду time/open/high/low/close[/volume].

    Объём НЕ выбрасывается (в ноутбуке он терялся дважды: при чтении CSV и при
    ресемплинге). Относительный объём на баре пробоя — один из немногих
    признаков, реально разделяющих истинные и ложные пробои.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл котировок не найден: {path}. Укажите правильный путь в configs/*.yaml -> data.csv_path"
        )

    df = pd.read_csv(path)

    if time_col not in df.columns:
        fallbacks = [c for c in ("time", "begin", "datetime", "date") if c in df.columns]
        if not fallbacks:
            raise KeyError(f"В {path.name} нет колонки времени {time_col!r}; есть: {list(df.columns)}")
        log.warning("Колонки %r нет, беру %r", time_col, fallbacks[0])
        time_col = fallbacks[0]

    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["time"])

    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise KeyError(f"В {path.name} нет колонок {missing}; есть: {list(df.columns)}")

    vol_col = next((c for c in volume_candidates if c in df.columns), None)
    cols = ["time", *OHLC]
    if vol_col is not None:
        if vol_col != "volume" and "volume" in df.columns:
            # в CSV бывают обе колонки (value и volume) — иначе rename создаст дубликат
            df = df.drop(columns=["volume"])
        df = df.rename(columns={vol_col: "volume"})
        cols.append("volume")
        log.info("Объём берётся из колонки %r", vol_col)
    else:
        log.warning("Колонка объёма не найдена (искал %s) — объёмные признаки будут пропущены", list(volume_candidates))

    df = df[cols].sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    log.info("Загружено %d минутных баров: %s .. %s", len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Ресемплинг OHLC(V). В отличие от исходного `resample_tf`, объём сохраняется."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(timeframe).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.reset_index()


def slice_time(
    df: pd.DataFrame,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """Полуинтервал [start, end): единая точка правды для всех нарезок."""
    out = df
    if start is not None:
        out = out[out[time_col] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out[time_col] < pd.Timestamp(end)]
    return out.reset_index(drop=True)


def minute_window_for(
    df_tf: pd.DataFrame, df_min: pd.DataFrame, horizon_minutes: int, time_col: str = "time"
) -> pd.DataFrame:
    """Минутные бары, покрывающие разметку/симуляцию для куска df_tf.

    Нужны минуты вплоть до конца горизонта ПОСЛЕДНЕГО бара, иначе последние
    метки/сделки молча теряются.
    """
    if df_tf.empty:
        return df_min.iloc[:0].copy()

    start = pd.Timestamp(df_tf[time_col].iloc[0])
    end = pd.Timestamp(df_tf[time_col].iloc[-1]) + pd.Timedelta(minutes=2 * horizon_minutes)
    return df_min[(df_min[time_col] >= start) & (df_min[time_col] <= end)].reset_index(drop=True)
