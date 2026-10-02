"""Импульсная стратегия: выход по развороту тренд-фильтра вместо (или вместе с) выходом по времени.

Вход — как в docs/results/patterns.md (ход за 6 ч ≥ 3 ATR в сторону гейта C, через минуту после
закрытия часа). Выход — трейлинг 6 ATR (активация +3 ATR) и предел по времени, плюс, по варианту,
разворот фильтра выхода: «opposite» — фильтр показал противоположное направление, «not_aligned» —
ушёл в нейтраль или против. Значение фильтра на часе T известно на закрытии T + 1 ч, на минуты
переносится с этого момента; выход — по close минуты, в которую стал известен разворот.

Фильтры выхода (docs/results/trend_filter_ema.md): гейт C, ноутбук v3, «плато» и «№1» из подбора
тренд-фильтра (лучшие по следованию тренду, медленные).

    python scripts/trend_exit_lab.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from entry_rules_lab import _trailing
from exit_grid import metrics
from forexmodel.data.loader import load_minute_compact
from forexmodel.features.trend import add_trend_filter
from impulse_universe import hourly, signal
from pattern_lab import ALL_INSTRUMENTS, apply_exit, instrument_cfg
from trend_filter_lab import NB_V3, base_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "trend_lab"
CORE = ["silver", "lkoh", "gazp", "sber"]


def exit_filters() -> dict:
    tuned = json.loads((OUT / "tuned_sets.json").read_text(encoding="utf-8"))
    base = base_config()
    return {"гейт C": base, "ноутбук v3": dataclasses.replace(base, **NB_V3),
            "подбор: плато": dataclasses.replace(base, **tuned["plateau"]),
            "подбор: №1": dataclasses.replace(base, **tuned["top1"])}


def to_minutes(minute_t: np.ndarray, hour_t: pd.Series, values: np.ndarray) -> np.ndarray:
    """Значение часа T — на минуты с T + 1 ч (после закрытия часа)."""
    lk = pd.DataFrame({"time": hour_t + pd.Timedelta("1h"), "v": values}).dropna().sort_values("time")
    m = pd.merge_asof(pd.DataFrame({"time": minute_t}), lk, on="time", direction="backward")
    return m["v"].to_numpy(dtype=float)


def simulate(minute, sig, cfg, commission, trend_min=None, mode="opposite", slip_pct=0.0):
    sim = cfg.simulation
    t = minute["time"].to_numpy()
    hi, lo, cl, op = (minute[c].to_numpy(dtype=float) for c in ("high", "low", "close", "open"))
    rows, busy = [], np.datetime64("1970-01-01")
    for tc, s, atr in sig.itertuples(index=False):
        et = np.datetime64(tc) + np.timedelta64(1, "m")
        if et <= busy or not np.isfinite(atr):
            continue
        a = np.searchsorted(t, et, "left")
        b = np.searchsorted(t, et + np.timedelta64(cfg.horizon_minutes, "m"), "right") - 1
        if a >= len(t) or b <= a:
            continue
        fill = op[a] * (1 + s * slip_pct / 100)
        g, j, _ = _trailing(hi[a:b + 1], lo[a:b + 1], cl[a:b + 1], fill, atr, s, sim.sl_atr, sim.trail_atr,
                            sim.activate_atr)
        reason = "трейлинг/стоп/время"
        if trend_min is not None:
            w = trend_min[a + 1:a + j + 1]
            bad = (w == -s) if mode == "opposite" else ((w * s) <= 0)
            bad &= np.isfinite(w)
            k = np.flatnonzero(bad)
            if k.size:
                j = int(k[0]) + 1
                g = s * (cl[a + j] - fill) / fill * 100
                reason = "разворот"
        busy = t[a + j]
        rows.append((tc, s, g, t[a], t[a + j], reason))
    r = pd.DataFrame(rows, columns=["time", "side", "gross", "open_time", "exit_time", "reason"])
    r["net"] = r["gross"] - commission
    return r


def main() -> int:
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    filters = exit_filters()
    variants = [("240 ч, без разворота", 240, None, None)]
    for hours in (240, 720):
        for fname in filters:
            for mode in ("opposite", "not_aligned"):
                variants.append((f"{hours} ч, разворот «{fname}» ({'против' if mode == 'opposite' else 'нейтраль/против'})",
                                 hours, fname, mode))
    variants.append(("720 ч, без разворота", 720, None, None))
    trades = []
    for inst in ALL_INSTRUMENTS:
        cfg0 = instrument_cfg(inst)
        minute = load_minute_compact(ROOT / cfg0.data.csv_path, cfg0.data.time_col, ROOT / "reports" / "_cache")
        h = hourly(minute, cfg0)
        sig = signal(h, "atr", 3.0, "gate")
        bars = h[["time", "open", "high", "low", "close"]]
        tmin = {name: to_minutes(minute["time"].to_numpy(), h["time"], add_trend_filter(bars, tc)["trend_4h"].to_numpy())
                for name, tc in filters.items()}
        for vname, hours, fname, mode in variants:
            cfg = apply_exit(instrument_cfg(inst), f"{hours}:6:6:3")
            for slip in (0.0, 0.1):
                tr = simulate(minute, sig, cfg, cfg.simulation.commission_pct, tmin.get(fname), mode or "opposite", slip)
                tr["год"] = pd.to_datetime(tr["time"]).dt.year
                trades.append(tr.assign(инструмент=inst, вариант=vname, проскальзывание=slip))
        print(f"[{inst}] готово", flush=True)
        del minute
    t = pd.concat(trades, ignore_index=True).sort_values("exit_time")
    t.to_csv(OUT / "trend_exit_trades.csv", index=False)
    rows = []
    for (v, slip), g in t.groupby(["вариант", "проскальзывание"], sort=False):
        for sname, insts in (("все 8", list(ALL_INSTRUMENTS)), ("ядро 4", CORE)):
            x = g[g["инструмент"].isin(insts)]
            for per, m in (("2015–20", x["год"] < 2021), ("2021–26", x["год"] >= 2021), ("всё", x["год"] > 0)):
                xx = x[m]
                rows.append({"вариант": v, "проскальзывание": slip, "набор": sname, "период": per,
                             **metrics(xx["net"] / len(insts), xx["год"]),
                             "ч_в_сделке": (pd.to_datetime(xx["exit_time"]) - pd.to_datetime(xx["open_time"])).dt.total_seconds().mean() / 3600,
                             "доля_выходов_по_развороту": (xx["reason"] == "разворот").mean()})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "trend_exit.csv", index=False)
    for sname in ("ядро 4", "все 8"):
        for slip in (0.0, 0.1):
            x = r[(r["набор"] == sname) & (r["проскальзывание"] == slip)]
            print(f"\n{sname}, проскальзывание {slip}%: итог / просадка / t по периодам")
            p = x.pivot_table(index="вариант", columns="период", values=["итог", "просадка", "t"], sort=False)
            p[("ч", "")] = x[x["период"] == "всё"].set_index("вариант")["ч_в_сделке"]
            p[("по развороту", "")] = x[x["период"] == "всё"].set_index("вариант")["доля_выходов_по_развороту"]
            print(p.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
