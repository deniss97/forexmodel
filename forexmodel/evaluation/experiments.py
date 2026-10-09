"""Единый протокол оценки экспериментов: одинаковые метрики на одном и том же периоде проверки.

Каждый эксперимент — набор сделок по инструментам (у одного инструмента может быть несколько прогонов,
например seed модели: тогда каждый прогон весит 1/n). Метрики считаются после издержек самого
эксперимента для периодов «выбор» (сделки, открытые до 2021 года) и «проверка» (с 1 января 2021),
по каждому инструменту и по портфелю (инструменты поровну).

Метрики: сделок, итог %, доля выигрышей, ожидание на сделку (средний результат), средний выигрыш и
проигрыш, профит-фактор, итог без 2 лучших сделок, максимальная просадка (п.п., без реинвестирования),
t средней сделки, годы в плюсе.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2021-01-01")
PERIODS = {"выбор": (None, TEST_START), "проверка": (TEST_START, None), "всё": (None, None)}
CORE = ("silver", "lkoh", "gazp", "sber")
STANDARD_COLS = ["instrument", "run", "side", "open_dt", "close_dt", "open_price", "exit_price",
                 "gross_pct", "profit_pct", "exit_reason"]


def standardize(t: pd.DataFrame, instrument: str, run: str) -> pd.DataFrame:
    """Сделки симулятора -> стандартные колонки."""
    out = pd.DataFrame({
        "instrument": instrument, "run": run,
        "side": np.where(t["side"].astype(str).isin(["buy", "1", "2"]), 1, -1),
        "open_dt": pd.to_datetime(t["open_dt"]), "close_dt": pd.to_datetime(t["close_dt"]),
        "open_price": t["open_price"].astype(float), "exit_price": t["exit_price"].astype(float),
        "gross_pct": t.get("gross_unit_pct", t["gross_pct"]).astype(float), "profit_pct": t["profit_pct"].astype(float),
        "exit_reason": t["exit_reason"].astype(str),
    })
    return out


def _weights(t: pd.DataFrame, instruments: Iterable[str]) -> pd.Series:
    """Вес сделки: 1/(число инструментов) × 1/(число прогонов инструмента)."""
    inst = list(instruments)
    runs = t.groupby("instrument")["run"].nunique()
    return t["instrument"].map(lambda i: 1.0 / len(inst) / runs.get(i, 1))


def metrics(t: pd.DataFrame, instruments: Optional[Iterable[str]] = None, period: str = "проверка") -> dict:
    """Метрики набора сделок (инструменты поровну, прогоны инструмента поровну)."""
    instruments = list(instruments) if instruments is not None else sorted(t["instrument"].unique())
    a, b = PERIODS[period]
    x = t[t["instrument"].isin(instruments)]
    if a is not None:
        x = x[x["open_dt"] >= a]
    if b is not None:
        x = x[x["open_dt"] < b]
    if x.empty:
        return {"сделок": 0}
    x = x.sort_values("close_dt")
    w = _weights(x, instruments)
    p = x["profit_pct"].to_numpy(float)
    wp = w.to_numpy() * p
    n_runs = x.groupby("instrument")["run"].nunique().mean()
    win, loss = p > 0, p <= 0
    gain, lossum = wp[p > 0].sum(), -wp[p < 0].sum()
    eq = np.cumsum(wp)
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    years = x.assign(wp=wp).groupby(x["open_dt"].dt.year)["wp"].sum()
    top2 = np.sort(wp)[-2:].sum() if len(wp) > 2 else 0.0

    def tstat(v: np.ndarray) -> float:
        s = v.std(ddof=1) if len(v) > 1 else np.nan
        return float(v.mean() / (s / np.sqrt(len(v)))) if s and s > 0 else np.nan

    # несколько прогонов (seed) одного инструмента похожи друг на друга: t по каждому прогону и среднее,
    # а не по слитым сделкам (иначе t завышен в √n раз)
    if x.groupby("instrument")["run"].nunique().max() > 1:
        t_val = float(np.nanmean([tstat(g["profit_pct"].to_numpy(float)) for _, g in x.groupby("run")]))
    else:
        t_val = tstat(p)
    return {
        "сделок": int(round(len(x) / n_runs)),
        "итог": float(wp.sum()),
        "выигрышей": float(np.average(win, weights=w)) * 100,
        "ожидание": float(np.average(p, weights=w)),
        "ср_выигрыш": float(np.average(p[win], weights=w[win])) if win.any() else np.nan,
        "ср_проигрыш": float(np.average(p[loss], weights=w[loss])) if loss.any() else np.nan,
        "PF": float(gain / lossum) if lossum > 0 else np.inf,
        "без_2_лучших": float(wp.sum() - top2),
        "просадка": dd,
        "t": t_val,
        "лет_в_плюсе": f"{int((years > 0).sum())}/{len(years)}",
        "с": x["open_dt"].min().strftime("%Y-%m"), "по": x["open_dt"].max().strftime("%Y-%m"),
    }


def scopes(instruments: Iterable[str]) -> Dict[str, list]:
    """Портфели и инструменты, по которым считаются метрики эксперимента."""
    inst = list(instruments)
    out = {"все": inst}
    if all(i in inst for i in CORE) and len(inst) > len(CORE):
        out["портфель 4"] = list(CORE)
    for i in inst:
        out[i] = [i]
    return out
