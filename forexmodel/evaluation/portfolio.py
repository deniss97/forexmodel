"""Портфель стратегий на общем капитале (задача T22, гипотеза H53).

Стратегия здесь — стандартный журнал сделок (forexmodel.evaluation.experiments.standardize). Капитал делится
поровну между стратегиями, внутри стратегии — поровну между инструментами и прогонами; позиции разных
стратегий на одном инструменте независимы (у каждой своя доля капитала). Результат сделки в портфеле —
profit_pct × вес.

Сумма процентов портфеля из двух стратегий — среднее их результатов, поэтому сравнивать портфель с одной
стратегией по итогу нельзя: главные метрики — не зависящие от плеча t месячных результатов и отношение
итога к просадке.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .experiments import PERIODS

HOUR = pd.Timedelta(hours=1)


def combine(parts: Dict[str, pd.DataFrame], instruments: Iterable[str]) -> pd.DataFrame:
    """Сделки стратегий одним списком с колонками strategy и w (доля капитала сделки)."""
    inst = list(instruments)
    out = []
    for name, t in parts.items():
        x = t[t["instrument"].isin(inst)].copy()
        runs = x.groupby("instrument")["run"].nunique()
        x["strategy"] = name
        x["w"] = x["instrument"].map(lambda i: 1.0 / len(parts) / len(inst) / runs.get(i, 1))
        out.append(x)
    return pd.concat(out, ignore_index=True)


def _period(x: pd.DataFrame, period: str) -> pd.DataFrame:
    a, b = PERIODS[period]
    if a is not None:
        x = x[x["open_dt"] >= a]
    if b is not None:
        x = x[x["open_dt"] < b]
    return x


def monthly(x: pd.DataFrame) -> pd.Series:
    """Результат портфеля по месяцам закрытия сделок, % капитала (пустые месяцы — ноль)."""
    wp = (x["w"] * x["profit_pct"]).groupby(x["close_dt"].dt.to_period("M")).sum()
    if wp.empty:
        return wp
    full = pd.period_range(wp.index.min(), wp.index.max(), freq="M")
    return wp.reindex(full, fill_value=0.0)


def metrics(x: pd.DataFrame, period: str = "проверка") -> dict:
    """Метрики портфеля: итог, PF, без 2 лучших, просадка, итог/просадка, t месяцев, сделок в год."""
    x = _period(x, period).sort_values("close_dt")
    if x.empty:
        return {"сделок": 0}
    wp = (x["w"] * x["profit_pct"]).to_numpy(float)
    eq = np.cumsum(wp)
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min())
    m = monthly(x)
    years = (x["close_dt"].max() - x["open_dt"].min()).days / 365.25
    by_year = x.assign(wp=wp).groupby(x["open_dt"].dt.year)["wp"].sum()
    loss = -wp[wp < 0].sum()
    total = float(wp.sum())
    return {
        "сделок": len(x), "в_год": len(x) / max(years, 1e-9), "итог": total,
        "PF": float(wp[wp > 0].sum() / loss) if loss > 0 else np.inf,
        "без_2_лучших": float(total - np.sort(wp)[-2:].sum()) if len(wp) > 2 else total,
        "просадка": dd, "итог_к_просадке": total / -dd if dd < 0 else np.inf,
        "t_мес": float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 1 and m.std(ddof=1) > 0 else np.nan,
        "лет_в_плюсе": f"{int((by_year > 0).sum())}/{len(by_year)}",
    }


def positions(t: pd.DataFrame, grid: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    """Позиция стратегии по часам на каждом инструменте: +1 лонг, −1 шорт, 0 нет позиции."""
    out = {}
    for inst, g in t.groupby("instrument"):
        p = np.zeros(len(grid), dtype=np.int8)
        a = grid.searchsorted(g["open_dt"].dt.floor("h"))
        b = grid.searchsorted(g["close_dt"].dt.floor("h"))
        for i, j, s in zip(a, b, g["side"].to_numpy()):
            p[i:max(j, i + 1)] = 1 if s > 0 else -1
        out[inst] = p
    return out


def overlap(cand: pd.DataFrame, base: pd.DataFrame, period: str = "проверка") -> dict:
    """Насколько кандидат торгует тогда же, когда база: доля часов кандидата в позиции, когда база в позиции на том же
    инструменте; из них — доля в ту же сторону; корреляция месячных результатов."""
    c, b = _period(cand, period), _period(base, period)
    if c.empty or b.empty:
        return {}
    start = min(c["open_dt"].min(), b["open_dt"].min()).floor("h")
    end = max(c["close_dt"].max(), b["close_dt"].max()).ceil("h")
    grid = pd.date_range(start, end, freq="h")
    pc, pb = positions(c, grid), positions(b, grid)
    held = both = same = 0
    for inst, p in pc.items():
        q = pb.get(inst, np.zeros_like(p))
        on = p != 0
        held += int(on.sum())
        both += int((on & (q != 0)).sum())
        same += int((on & (q == p)).sum())
    mc, mb = monthly(c), monthly(b)
    j = pd.concat([mc, mb], axis=1).fillna(0.0)
    return {"совпадение": both / held if held else np.nan, "в_ту_же_сторону": same / both if both else np.nan,
            "корр_мес": float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 2 else np.nan}


def concurrency(x: pd.DataFrame, period: str = "проверка") -> dict:
    """Сколько позиций портфеля открыто одновременно: максимум и в среднем по часам, когда хоть одна открыта."""
    x = _period(x, period)
    if x.empty:
        return {}
    grid = pd.date_range(x["open_dt"].min().floor("h"), x["close_dt"].max().ceil("h"), freq="h")
    n = np.zeros(len(grid), dtype=np.int16)
    a = grid.searchsorted(x["open_dt"].dt.floor("h"))
    b = grid.searchsorted(x["close_dt"].dt.floor("h"))
    for i, j in zip(a, b):
        n[i:max(j, i + 1)] += 1
    on = n[n > 0]
    return {"макс_позиций": int(n.max()), "ср_позиций": float(on.mean()) if len(on) else 0.0}
