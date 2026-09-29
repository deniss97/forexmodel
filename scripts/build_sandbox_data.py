"""Данные «песочницы» карты сделок: все варианты паттерна и выхода по всем инструментам.

Для каждого инструмента, варианта порога (`VARIANTS`) и выхода (`EXITS`) — сделки быстрого
движка (`early_entry.simulate`, без комиссии: она, проскальзывание и своп задаются на странице).
У каждой сделки:
  * результат до комиссии, %; время входа и выхода; число ночей в позиции (смен календарной
    даты по UTC — грубая мера свопа: тройной своп в среду покрывает выходные);
  * старший тренд на момент сигнала по дневкам, только по ЗАКРЫТЫМ дням (до даты сигнала):
    цена к SMA50, наклон SMA50 за 10 дней, цена к SMA200, ход за 20 дней — по каждому
    +1 / −1 / 0 (нет данных);
  * флаг «внутри сделки был ночной гэп ≥ 4%» (у акций это в основном дивидендные отсечки).

    python scripts/build_sandbox_data.py      # -> docs/results/strategy_viz/data/sandbox.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from early_entry import simulate
from forexmodel.data.loader import load_minute_compact
from impulse_universe import hourly, signal
from pattern_lab import ALL_INSTRUMENTS, apply_exit, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "strategy_viz" / "data" / "sandbox.json"
VARIANTS = {"3atr": ("atr", 3.0, "6 ч ≥ 3 ATR"), "2.5atr": ("atr", 2.5, "6 ч ≥ 2.5 ATR"),
            "4atr": ("atr", 4.0, "6 ч ≥ 4 ATR"), "1.5pct": ("pct", 1.5, "6 ч ≥ 1.5%")}
EXITS = {"240_6": ("240:6:6:3", "240 ч, трейлинг 6 ATR"), "240_3": ("240:3:3:1.5", "240 ч, трейлинг 3 ATR"),
         "120_6": ("120:6:6:3", "120 ч, трейлинг 6 ATR"), "48_4": ("48:4:4:2", "48 ч, трейлинг 4 ATR")}
MARKET = {"silver": "fx", "gold": "fx", "eth": "crypto"}          # остальные — акции MOEX


def daily_trend(h: pd.DataFrame) -> pd.DataFrame:
    """Состояние старшего тренда на каждую дату — по закрытиям ПРЕДЫДУЩИХ дней."""
    d = h.set_index("time")["close"].resample("1D").last().dropna()
    sma50, sma200 = d.rolling(50).mean(), d.rolling(200).mean()
    st = pd.DataFrame({
        "sma50": np.sign(d - sma50), "slope50": np.sign(sma50 - sma50.shift(10)),
        "sma200": np.sign(d - sma200), "ret20": np.sign(d - d.shift(20)),
    }).fillna(0).astype(int)
    st.index = st.index + pd.Timedelta("1D")        # доступно со следующего дня
    return st


def main() -> int:
    logging.disable(logging.WARNING)
    out = {"variants": {k: v[2] for k, v in VARIANTS.items()}, "exits": {k: v[1] for k, v in EXITS.items()},
           "trend_cols": ["sma50", "slope50", "sma200", "ret20"],
           "cols": ["вход", "выход", "сторона", "до_комиссии_%", "ночей", "sma50", "slope50", "sma200", "ret20", "гэп"],
           "instruments": {}}
    for inst in ALL_INSTRUMENTS:
        base = instrument_cfg(inst)
        minute = load_minute_compact(ROOT / base.data.csv_path, base.data.time_col, ROOT / "reports" / "_cache")
        h = hourly(minute, base)
        st = daily_trend(h)
        gap_ok = h["time"].diff() > pd.Timedelta("8h")
        gap = (h["open"] / h["close"].shift() - 1).abs() >= 0.04
        gap_t = h.loc[gap_ok & gap, "time"].to_numpy()
        runs = {}
        for vk, (kind, thr, _) in VARIANTS.items():
            sig = signal(h, kind, thr, "gate")
            for ek, (spec, _) in EXITS.items():
                cfg = apply_exit(instrument_cfg(inst), spec)
                t = simulate(minute, sig, cfg, 0.0)
                day = pd.to_datetime(t["time"]).dt.normalize()
                tr = st.reindex(day, method="ffill").to_numpy()
                op, ex = pd.to_datetime(t["open_time"]), pd.to_datetime(t["exit_time"])
                nights = (ex.dt.normalize() - op.dt.normalize()).dt.days.to_numpy()
                g = [bool(((gap_t > o) & (gap_t <= e)).any()) for o, e in zip(op.to_numpy(), ex.to_numpy())]
                runs[f"{vk}|{ek}"] = [
                    [int(o.timestamp()), int(e.timestamp()), int(s), round(float(gr), 3), int(n), *map(int, trow), int(gg)]
                    for o, e, s, gr, n, trow, gg in zip(op, ex, t["side"], t["gross"], nights, tr, g)]
        out["instruments"][inst] = {"market": MARKET.get(inst, "stock"), "runs": runs}
        print(f"[{inst}] {sum(len(v) for v in runs.values())} сделок во всех вариантах", flush=True)
        del minute
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT} {OUT.stat().st_size / 1e6:.1f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
