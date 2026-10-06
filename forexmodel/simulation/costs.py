"""Издержки брокера: спред и перенос позиции через ночь (своп, «форвардные пункты»).

Профиль брокера — YAML в configs/costs/ (напр. alfaforex.yaml): по инструменту спред и свопы лонга и
шорта либо в % цены, либо в пунктах (тогда нужны размер пункта `point` и цена `ref_price`, по которой
пункты переводятся в %: условия даются на сегодня, а пункты — в деньгах, поэтому для всей истории
берётся их доля от текущей цены).

Перенос: в момент `rollover_hour` (время котировок) после каждого торгового дня пн–пт; с дня
`triple_weekday` (0 — пн) — трижды, за выходные; суббота и воскресенье — без переноса. Если
`triple_weekday` не задан, каждая календарная ночь — по разу.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def rollover_count(open_dt, close_dt, hour: int = 0, triple_weekday: Optional[int] = 2) -> int:
    """Сколько переносов (с учётом тройного) приходится на позицию [open_dt, close_dt].

    Перенос после торгового дня D — в D + hour, если hour ≥ 12 (вечер того же дня), иначе в
    D + 1 день + hour (после полуночи).
    """
    open_dt, close_dt = pd.Timestamp(open_dt), pd.Timestamp(close_dt)
    if close_dt <= open_dt:
        return 0
    shift = pd.Timedelta(hours=hour) + (pd.Timedelta(0) if hour >= 12 else pd.Timedelta(days=1))
    n = 0
    for d in pd.date_range(open_dt.normalize() - pd.Timedelta(days=1), close_dt.normalize(), freq="D"):
        moment = d + shift
        if open_dt < moment <= close_dt:
            if triple_weekday is None:
                n += 1
            elif d.dayofweek < 5:
                n += 3 if d.dayofweek == triple_weekday else 1
    return n


def profile_entry(path: str | Path, instrument: str) -> dict:
    """Условия брокера по инструменту, переведённые в % цены."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if instrument not in data["instruments"]:
        raise KeyError(f"в профиле {path} нет инструмента {instrument!r}")
    e = dict(data["instruments"][instrument])
    if "point" in e:
        k = e["point"] / e["ref_price"] * 100.0                   # 1 пункт в % цены
        e.setdefault("spread_pct", e.get("spread_points", 0.0) * k)
        e.setdefault("swap_long_pct", e.get("swap_long_points", 0.0) * k)
        e.setdefault("swap_short_pct", e.get("swap_short_points", 0.0) * k)
    for key in ("spread_pct", "swap_long_pct", "swap_short_pct"):
        e[key] = float(e.get(key, 0.0))
    e.setdefault("rollover_hour", 0)
    e.setdefault("triple_weekday", 2)
    return e


def apply_profile(sim_cfg, entry: dict) -> None:
    """Спред брокера — вместо комиссии за круг; свопы и моменты переноса — в настройки симуляции."""
    sim_cfg.commission_pct = entry["spread_pct"]
    sim_cfg.swap_long_pct = entry["swap_long_pct"]
    sim_cfg.swap_short_pct = entry["swap_short_pct"]
    sim_cfg.swap_rollover_hour = int(entry["rollover_hour"])
    sim_cfg.swap_triple_weekday = entry["triple_weekday"]


def recost(trades: pd.DataFrame, entry: dict) -> pd.DataFrame:
    """Пересчёт результата готовых сделок под другие издержки (путь сделки от издержек не зависит).

    Нужны колонки open_dt, close_dt, side, gross_unit_pct, size; результат — колонки rollovers,
    swap_pct, commission_pct, profit_pct (как у симулятора).
    """
    t = trades.copy()
    if not len(t):
        return t
    n = np.array([rollover_count(a, b, entry["rollover_hour"], entry["triple_weekday"])
                  for a, b in zip(pd.to_datetime(t["open_dt"]), pd.to_datetime(t["close_dt"]))])
    size = t["size"].to_numpy(dtype=float) if "size" in t else np.ones(len(t))
    swap = np.where(t["side"].to_numpy() == "buy", entry["swap_long_pct"], entry["swap_short_pct"]) * n
    t["rollovers"] = n
    t["swap_pct"] = size * swap
    t["commission_pct"] = size * entry["spread_pct"]
    t["profit_pct"] = size * (t["gross_unit_pct"].to_numpy(dtype=float) - entry["spread_pct"] + swap)
    return t
