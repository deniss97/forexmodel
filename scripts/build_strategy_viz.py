"""Данные для обозревателя импульсной стратегии (docs/results/strategy_viz/index.html).

Для каждого инструмента: часовые свечи, зоны тренд-гейта C, EMA5 4-часовых закрытий (серая
зона из обсуждения 09-29) и сделки штатного симулятора из walk-forward выхода 240 ч / 6 ATR
(reports/walk_forward/<inst>_exit_240h6__impulse_6_3_trades.csv). По каждой сделке —
величина импульса на входе и лучший / худший ход внутри сделки (MFE / MAE, в ATR) по
часовым high/low.

    python scripts/build_strategy_viz.py            # все инструменты, у которых есть сделки
    cd docs/results/strategy_viz && python -m http.server 8000   # открыть http://localhost:8000
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

from forexmodel.data.loader import load_minute_compact
from forexmodel.features.trend import add_trend_filter, reset_ema
from impulse_universe import hourly
from pattern_lab import ALL_INSTRUMENTS, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "strategy_viz"
WF = ROOT / "reports" / "walk_forward"
TITLES = {"silver": "Серебро · XAGUSD", "gold": "Золото · XAUUSD", "lkoh": "Лукойл · LKOH",
          "gazp": "Газпром · GAZP", "sber": "Сбербанк · SBER", "moex": "Мосбиржа · MOEX", "mtss": "МТС · MTSS",
          "eth": "Эфир · ETHUSDT"}
REASONS = {"stop_loss": "стоп", "trailing_stop": "трейлинг", "timeout": "время", "take_profit": "тейк",
           "trend_flip": "разворот"}


def gate_variants() -> dict:
    """Варианты тренд-фильтра для зон на графике (поверх гейта C из configs/silver.yaml)."""
    from trend_filter_lab import NB_V3, base_config

    v = {
        "C + сброс по EMA5 4ч": dict(ema_reset_period=5, ema_reset_tf="htf"),
        "C + сброс по EMA5 часа": dict(ema_reset_period=5),
        "ноутбук v3 (гистерезис)": NB_V3,
        "ноутбук v3 + сброс по EMA5 4ч": {**NB_V3, "ema_reset_period": 5, "ema_reset_tf": "htf"},
    }
    best = ROOT / "reports" / "trend_lab" / "tune_best.json"
    if best.exists():
        v["лучший из подбора"] = json.loads(best.read_text(encoding="utf-8"))["params"]
    return {name: dataclasses.replace(base_config(), **kw) for name, kw in v.items()}


def rle(a: np.ndarray) -> list:
    a = np.nan_to_num(a)
    ch = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    return [[int(i), int(a[i])] for i in ch]


def build(inst: str) -> dict | None:
    tf = WF / f"{inst}_exit_240h6__impulse_6_3_trades.csv"
    if not tf.exists():
        print(f"[{inst}] нет сделок {tf.name}, пропуск")
        return None
    cfg = instrument_cfg(inst)
    minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
    h = hourly(minute, cfg)
    del minute
    # EMA5 4-часовых закрытий на часовой сетке (как в impulse_universe: последний закрытый 4-часовой бар)
    four = h.set_index("time")["close"].resample("4h").last().dropna()
    e = four.ewm(span=5, adjust=False).mean()
    e.index = e.index + pd.Timedelta("4h")
    ema4 = pd.merge_asof(h[["t_close"]], e.rename("ema").reset_index().rename(columns={"time": "t_close"}),
                         on="t_close", direction="backward")["ema"].to_numpy()

    med = float(h["close"].median())
    dec = int(min(4, max(0, 4 - math.floor(math.log10(med)))))
    t = ((h["time"] - pd.Timestamp(0)) // pd.Timedelta("1s")).to_numpy().astype(np.int64)
    r = lambda a: [round(float(x), dec) if np.isfinite(x) else None for x in a]  # noqa: E731
    gate = h["gate"].to_numpy()
    ch = np.flatnonzero(np.r_[True, gate[1:] != gate[:-1]])
    # варианты фильтра для зон и линии EMA, по которым работает сброс
    bars = h[["time", "open", "high", "low", "close"]]
    variants = gate_variants()
    gates = {name: rle(add_trend_filter(bars, tc)["trend_4h"].to_numpy(dtype=float)) for name, tc in variants.items()}
    any_tc = next(iter(variants.values()))
    ema1 = reset_ema(bars, dataclasses.replace(any_tc, ema_reset_period=5, ema_reset_tf="base"))
    ema4_live = reset_ema(bars, dataclasses.replace(any_tc, ema_reset_period=5, ema_reset_tf="htf"))

    tr = pd.read_csv(tf, parse_dates=["signal_dt", "open_dt", "close_dt"])
    hi, lo, atr_h, mv = h["high"].to_numpy(), h["low"].to_numpy(), h["atr"].to_numpy(), h["move_atr"].to_numpy()
    times = h["time"].to_numpy()
    rows = []
    for x in tr.itertuples(index=False):
        a = int(np.searchsorted(times, np.datetime64(x.open_dt.floor("h")), "left"))
        b = int(np.searchsorted(times, np.datetime64(x.close_dt.floor("h")), "right")) - 1
        si = int(np.searchsorted(times, np.datetime64(x.signal_dt), "left"))
        d = 1 if x.side == "buy" else -1
        atr = float(x.atr_at_entry)
        if b >= a and atr > 0:
            fav = (hi[a:b + 1].max() - x.open_price) if d > 0 else (x.open_price - lo[a:b + 1].min())
            adv = (x.open_price - lo[a:b + 1].min()) if d > 0 else (hi[a:b + 1].max() - x.open_price)
            mfe, mae = float(fav) / atr, float(adv) / atr
        else:
            mfe = mae = float("nan")
        rows.append([int(x.open_dt.timestamp()), int(x.close_dt.timestamp()), d,
                     round(float(x.open_price), dec), round(float(x.exit_price), dec),
                     round(float(x.profit_pct), 3), REASONS.get(x.exit_reason, x.exit_reason),
                     round(float(x.minutes_in_trade) / 60, 1),
                     round(mfe, 2) if np.isfinite(mfe) else None, round(mae, 2) if np.isfinite(mae) else None,
                     round(float(abs(mv[si])) if si < len(mv) and np.isfinite(mv[si]) else float("nan"), 2),
                     round(atr / float(x.open_price) * 100, 3)])
    return {
        "id": inst, "title": TITLES.get(inst, inst), "decimals": dec,
        "commission": cfg.simulation.commission_pct,
        # время свечи: t0 (unix, с) + накопленная сумма шагов dt (часы)
        "t0": int(t[0]), "dt": (np.diff(t, prepend=t[0]) // 3600).astype(int).tolist(),
        "o": r(h["open"]), "h": r(h["high"]), "l": r(h["low"]), "c": r(h["close"]),
        "ema": r(ema4),
        "ema1h": r(ema1),
        "ema4h_live": r(ema4_live),
        "gates": gates,
        "gate": [[int(i), int(gate[i])] for i in ch],
        "trade_cols": ["вход", "выход", "сторона", "цена_входа", "цена_выхода", "результат_%", "причина_выхода",
                       "часов", "mfe_atr", "mae_atr", "импульс_atr", "atr_%"],
        "trades": rows,
    }


def main(argv=None) -> int:
    logging.disable(logging.WARNING)
    insts = argv or list(ALL_INSTRUMENTS)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    manifest = []
    for inst in insts:
        d = build(inst)
        if d is None:
            continue
        p = OUT / "data" / f"{inst}.json"
        p.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        pnl = pd.Series([x[5] for x in d["trades"]])
        years = pd.Series([pd.Timestamp(x[0], unit="s").year for x in d["trades"]])
        eq = pnl.cumsum()
        loss = -pnl[pnl < 0].sum()
        yr = pnl.groupby(years).sum()
        manifest.append({"id": inst, "title": d["title"], "trades": len(pnl),
                         "winrate": round(float((pnl > 0).mean() * 100), 1), "total": round(float(pnl.sum()), 1),
                         "ex_top2": round(float(pnl.drop(pnl.nlargest(2).index).sum()), 1),
                         "pf": round(float(pnl[pnl > 0].sum() / loss), 2) if loss > 0 else None,
                         "max_dd": round(float((eq - eq.cummax().clip(lower=0)).min()), 1),
                         "years": f"{int((yr > 0).sum())}/{len(yr)}",
                         "first_year": int(years.min()), "last_year": int(years.max())})
        print(f"[{inst}] {len(d['c'])} часов, {len(d['trades'])} сделок, {p.stat().st_size / 1e6:.1f} МБ", flush=True)
    order = list(ALL_INSTRUMENTS)
    manifest.sort(key=lambda m: order.index(m["id"]))
    (OUT / "data" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
