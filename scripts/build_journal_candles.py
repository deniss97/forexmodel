"""Свечи для страницы журнала экспериментов (docs/experiments/site/data/candles/<инструмент>.json).

По каждому инструменту — часовые свечи и слои графика: зоны тренд-гейта C (как в стратегии: обновление
каждый час, удержание 2 часа), зоны трёхслойного фильтра (режим по дневной EMA200, как в H58–H60: их показывают
эксперименты с `zones: tl` в каталоге), ATR и EMA5 4-часовых закрытий. Слои считаются на тех же ценах, что и
стратегия (у акций — с обратной корректировкой на дивиденды), а затем переводятся в масштаб сырых свечей:
на графике котировки биржи, цены сделок страница получает в том же масштабе (scripts/experiments.py).

    python scripts/build_journal_candles.py            # все инструменты
    python scripts/build_journal_candles.py silver     # только эти
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.data.dividends import adjust_for_dividends, adjustment_factors, load_dividends
from forexmodel.data.loader import load_minute_compact
from forexmodel.features.trend import add_trend_filter
from experiments import NEW_STOCKS
from impulse_universe import hourly
from pattern_lab import ALL_INSTRUMENTS, instrument_cfg
from trend_filter_lab import base_config
from trend_grid import GATE_C

# трёхслойный фильтр — как scripts/trend_follow.py (TL) и trend_model_gate.py
THREE_LAYER = {"mode": "score", "score_regime_tf": "24h"}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "experiments" / "site" / "data" / "candles"
DIVS = ROOT / "data" / "dividends" / "moex_dividends.csv"
STOCKS = {"lkoh", "gazp", "sber", "moex", "mtss", *NEW_STOCKS}


def build(inst: str) -> dict:
    cfg = instrument_cfg(inst)
    minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
    minute = minute.sort_values("time").reset_index(drop=True)
    factor = np.ones(len(minute))
    if inst in STOCKS:
        divs = load_dividends(DIVS, inst)
        factor = adjustment_factors(minute, divs)
        minute = adjust_for_dividends(minute, divs)
    h = hourly(minute, cfg)
    # множитель корректировки на час: по первой минуте часа (внутри часа отсечек нет)
    f = factor[np.clip(np.searchsorted(minute["time"].to_numpy(), h["time"].to_numpy(), "left"), 0, len(factor) - 1)]
    del minute
    four = h.set_index("time")["close"].resample("4h").last().dropna()
    e = four.ewm(span=5, adjust=False).mean()
    e.index = e.index + pd.Timedelta("4h")
    ema4 = pd.merge_asof(h[["t_close"]], e.rename("ema").reset_index().rename(columns={"time": "t_close"}),
                         on="t_close", direction="backward")["ema"].to_numpy()
    raw = {k: h[k].to_numpy() / f for k in ("open", "high", "low", "close")}
    med = float(np.nanmedian(raw["close"]))
    dec = int(min(4, max(0, 4 - math.floor(math.log10(med)))))
    t = ((h["time"] - pd.Timestamp(0)) // pd.Timedelta("1s")).to_numpy().astype(np.int64)
    r = lambda a, d=dec: [round(float(x), d) if np.isfinite(x) else None for x in a]  # noqa: E731
    gate = np.nan_to_num(h["gate"].to_numpy(dtype=float)).astype(int)
    ch = np.flatnonzero(np.r_[True, gate[1:] != gate[:-1]])
    tc = dataclasses.replace(dataclasses.replace(base_config(), **GATE_C), **THREE_LAYER)   # как trend_grid.gate_values
    tl = np.nan_to_num(add_trend_filter(h[["time", "open", "high", "low", "close"]], tc)["trend_4h"].to_numpy(dtype=float)).astype(int)
    ch_tl = np.flatnonzero(np.r_[True, tl[1:] != tl[:-1]])
    return {
        "id": inst, "decimals": dec,
        # время свечи: t0 (unix, с) + накопленная сумма шагов dt (часы)
        "t0": int(t[0]), "dt": (np.diff(t, prepend=t[0]) // 3600).astype(int).tolist(),
        "o": r(raw["open"]), "h": r(raw["high"]), "l": r(raw["low"]), "c": r(raw["close"]),
        "gate": [[int(i), int(gate[i])] for i in ch],           # [индекс свечи, -1/0/1] с которого действует
        "gate_tl": [[int(i), int(tl[i])] for i in ch_tl],       # то же для трёхслойного фильтра
        "atr": r(h["atr"].to_numpy() / f, dec + 1),
        "ema": r(ema4 / f),
        "atr_period": int(cfg.features.atr_period),
    }


def main(argv=None) -> int:
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    for inst in (argv or list(ALL_INSTRUMENTS)):
        d = build(inst)
        p = OUT / f"{inst}.json"
        p.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"[{inst}] {len(d['c'])} свечей, {len(d['gate'])} смен гейта → {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} МБ)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
