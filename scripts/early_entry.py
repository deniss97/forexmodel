"""Ранний вход по импульсу: проверять условие каждые 15/30 минут, а не на закрытии часа.

Идея встречи 2026-09-28: «если шоковое изменение цены, не ждать целый час». Импульс тот же,
что в docs/results/patterns.md: ход за 6 часов ≥ x ATR часа в сторону тренд-гейта C. Но
ход считается на закрытии каждого бара шага `step` (15m / 30m / 1h) по окну 6 часов, ATR
и гейт берутся с последнего ЗАКРЫТОГО часа. Вход — через минуту после закрытия бара шага,
выход и комиссия одинаковы для всех шагов, позиции не перекрываются. Так разница между
шагами — только в моменте входа.

    python scripts/early_entry.py --exit 48:4:4:2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from entry_rules_lab import _trailing, trend_variant
from forexmodel.data.loader import load_minute_compact, resample_ohlcv
from forexmodel.features import indicators as ind
from pattern_lab import INSTRUMENTS, apply_exit, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]


def signals(minute: pd.DataFrame, cfg, step: str, window_h: int, x: float) -> pd.DataFrame:
    h = resample_ohlcv(minute, "1h")
    h["atr"] = ind.atr(h, cfg.features.atr_period)
    h["gate"] = trend_variant(h, "C", True, cfg, hold_h=2)
    # значения часа известны после его закрытия: time + 1h
    hc = pd.DataFrame({"t_known": h["time"] + pd.Timedelta("1h"), "atr": h["atr"], "gate": h["gate"]})
    b = resample_ohlcv(minute, step)
    b["t_close"] = b["time"] + pd.Timedelta(step)
    b = pd.merge_asof(b.sort_values("t_close"), hc, left_on="t_close", right_on="t_known", direction="backward")
    # ход за window_h часов по закрытиям баров шага: close сейчас − close бара, закрывшегося window_h назад
    past = pd.merge_asof(b[["t_close"]].assign(t_back=b["t_close"] - pd.Timedelta(hours=window_h)),
                         b[["t_close", "close"]].rename(columns={"t_close": "t_prev", "close": "c_prev"}),
                         left_on="t_back", right_on="t_prev", direction="backward")
    b["move"] = (b["close"] - past["c_prev"].to_numpy()) / b["atr"]
    side = np.where(b["move"] >= x, 1, np.where(b["move"] <= -x, -1, 0))
    b["side"] = np.where(side == b["gate"], side, 0)
    return b[b["side"] != 0][["t_close", "side", "atr"]].reset_index(drop=True)


def simulate(minute: pd.DataFrame, sig: pd.DataFrame, cfg, commission: float, delay_min: int = 1,
             slip_pct: float = 0.0, slip_atr: float = 0.0) -> pd.DataFrame:
    """Сделки без перекрытия. Вход — open минуты через `delay_min` после закрытия бара сигнала,
    хуже на `slip_pct` % цены и `slip_atr` ATR (против сделки); стоп и трейлинг — от цены исполнения."""
    sim = cfg.simulation
    t = minute["time"].to_numpy()
    hi, lo, cl, op = (minute[c].to_numpy(dtype=float) for c in ("high", "low", "close", "open"))
    rows, busy_until = [], np.datetime64("1970-01-01")
    for tc, s, atr in sig.itertuples(index=False):
        entry_t = np.datetime64(tc) + np.timedelta64(delay_min, "m")
        if entry_t <= busy_until or not np.isfinite(atr):
            continue
        a = np.searchsorted(t, entry_t, "left")
        b = np.searchsorted(t, entry_t + np.timedelta64(cfg.horizon_minutes, "m"), "right") - 1
        if a >= len(t) or b <= a:
            continue
        fill = op[a] * (1 + s * slip_pct / 100) + s * slip_atr * atr
        g, j, first_stop = _trailing(hi[a:b + 1], lo[a:b + 1], cl[a:b + 1], fill, atr, s, sim.sl_atr, sim.trail_atr,
                                     sim.activate_atr)
        # причина выхода: стоп (трейлинг ещё не включился), трейлинг, время
        reason = "stop" if first_stop else ("time" if j == b - a and (s * (cl[b] - fill) / fill * 100) == g else "trail")
        busy_until = t[a + j]
        rows.append((tc, s, fill, g, t[a], t[a + j], reason))
    r = pd.DataFrame(rows, columns=["time", "side", "entry", "gross", "open_time", "exit_time", "reason"])
    r["net"] = r["gross"] - commission
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exit", default="48:4:4:2")
    ap.add_argument("--steps", nargs="+", default=["60min", "30min", "15min"])
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--atr-mult", type=float, default=3.0)
    ap.add_argument("--split-year", type=int, default=2021)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    rows = []
    for inst in INSTRUMENTS:
        cfg = apply_exit(instrument_cfg(inst), args.exit)
        minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
        for step in args.steps:
            tr = simulate(minute, signals(minute, cfg, step, args.window, args.atr_mult), cfg,
                          cfg.simulation.commission_pct)
            tr["год"] = pd.to_datetime(tr["time"]).dt.year
            for per, g in tr.groupby(np.where(tr["год"] < args.split_year, "выбор", "проверка")):
                for side, x in (("все", g), ("лонг", g[g.side > 0]), ("шорт", g[g.side < 0])):
                    n, sd = len(x), x["net"].std()
                    rows.append({"инструмент": inst, "шаг": step, "период": per, "сторона": side, "сделок": n,
                                 "нетто": x["net"].mean(), "t": x["net"].mean() / (sd / np.sqrt(n)) if n > 1 else np.nan,
                                 "winrate": (x["net"] > 0).mean() * 100, "итог_%": x["net"].sum()})
            print(f"[{inst}] {step}: {len(tr)} сделок", flush=True)
        del minute
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "reports" / "patterns" / f"early_entry_{args.exit.replace(':', '-')}.csv", index=False)
    for v in ("нетто", "t", "сделок", "итог_%"):
        print(f"\n{v}")
        print(r[r["сторона"] == "все"].pivot_table(index="шаг", columns=["инструмент", "период"], values=v).round(3)
              .to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
