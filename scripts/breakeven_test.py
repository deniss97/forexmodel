"""Перенос стопа в безубыток после хода +X ATR — для импульсной стратегии.

Разбор сделок (docs/results/strategy_viz): 28–49% проигрышей сначала уходили в нашу сторону на
3 ATR и больше, потом закрывались в минус. Проверка: базовый выход (240 ч, стоп и трейлинг 6
ATR, активация 3) плюс правило — как только лучший ход достиг `be` ATR, стоп не ниже цены
входа + `offset` ATR. Движок и сигналы — как в impulse_universe (гейт C, 6 ч ≥ 3 ATR),
выбор до 2021, проверка после.

    python scripts/breakeven_test.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from exit_grid import metrics
from forexmodel.data.loader import load_minute_compact
from impulse_universe import hourly, signal
from pattern_lab import ALL_INSTRUMENTS, apply_exit, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]


def trail_be(h, l, c, entry, atr, sgn, sl, trail, act, be, off):
    """Трейлинг как в entry_rules_lab._trailing + пол стопа на entry + off·ATR после хода ≥ be·ATR."""
    if sgn > 0:
        best = np.maximum.accumulate(np.concatenate([[entry], h[:-1]]))
        gain = best - entry
        stop = np.where(gain >= act * atr, np.maximum(entry - sl * atr, best - trail * atr), entry - sl * atr)
        if be is not None:
            stop = np.where(gain >= be * atr, np.maximum(stop, entry + off * atr), stop)
        hit = np.flatnonzero(l <= stop)
    else:
        best = np.minimum.accumulate(np.concatenate([[entry], l[:-1]]))
        gain = entry - best
        stop = np.where(gain >= act * atr, np.minimum(entry + sl * atr, best + trail * atr), entry + sl * atr)
        if be is not None:
            stop = np.where(gain >= be * atr, np.minimum(stop, entry - off * atr), stop)
        hit = np.flatnonzero(h >= stop)
    if hit.size:
        j = hit[0]
        return sgn * (stop[j] - entry) / entry * 100, j
    j = len(c) - 1
    return sgn * (c[j] - entry) / entry * 100, j


def run(minute, sig, cfg, commission, be, off):
    sim = cfg.simulation
    t = minute["time"].to_numpy()
    hi, lo, cl, op = (minute[x].to_numpy(dtype=float) for x in ("high", "low", "close", "open"))
    rows, busy = [], np.datetime64("1970-01-01")
    for tc, s, atr in sig.itertuples(index=False):
        et = np.datetime64(tc) + np.timedelta64(1, "m")
        if et <= busy or not np.isfinite(atr):
            continue
        a = np.searchsorted(t, et, "left")
        b = np.searchsorted(t, et + np.timedelta64(cfg.horizon_minutes, "m"), "right") - 1
        if a >= len(t) or b <= a:
            continue
        g, j = trail_be(hi[a:b + 1], lo[a:b + 1], cl[a:b + 1], op[a], atr, s, sim.sl_atr, sim.trail_atr,
                        sim.activate_atr, be, off)
        busy = t[a + j]
        rows.append((tc, g))
    r = pd.DataFrame(rows, columns=["time", "gross"])
    r["net"] = r["gross"] - commission
    r["год"] = pd.to_datetime(r["time"]).dt.year
    return r


def main() -> int:
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    variants = [("база", None, 0.0)] + [(f"безубыток после +{be} ATR", be, 0.0) for be in (2.0, 3.0, 4.0, 6.0)] + \
               [("стоп в +1 ATR после +4 ATR", 4.0, 1.0)]
    trades = []
    for inst in ALL_INSTRUMENTS:
        cfg = apply_exit(instrument_cfg(inst), "240:6:6:3")
        minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
        sig = signal(hourly(minute, cfg), "atr", 3.0, "gate")
        for name, be, off in variants:
            trades.append(run(minute, sig, cfg, cfg.simulation.commission_pct, be, off).assign(инструмент=inst, вариант=name))
        print(f"[{inst}] готово", flush=True)
        del minute
    t = pd.concat(trades, ignore_index=True).sort_values("time")
    rows = []
    for v, g in t.groupby("вариант", sort=False):
        for per, x in (("выбор", g[g["год"] < 2021]), ("проверка", g[g["год"] >= 2021]), ("всё", g)):
            n = x["инструмент"].nunique()
            rows.append({"вариант": v, "период": per, **metrics(x["net"] / n, x["год"])})
            for inst, xi in x.groupby("инструмент"):
                rows.append({"вариант": v, "период": per, "инструмент": inst, **metrics(xi["net"], xi["год"])})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "reports" / "patterns" / "breakeven.csv", index=False)
    port = r[r["инструмент"].isna()]
    print(port.drop(columns=["инструмент"]).round(3).to_string(index=False))
    inst = r[r["инструмент"].notna() & (r["период"] == "всё")]
    print(inst.pivot_table(index="инструмент", columns="вариант", values="итог").round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
