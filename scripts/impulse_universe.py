"""Импульсная стратегия на многих инструментах: порог в ATR или в %, серая зона тренд-фильтра.

Вопросы встречи 2026-09-29:
  * работает ли стратегия на других инструментах (GAZP, SBER, MOEX, MTSS, ETHUSDT) —
    если сигнал настоящий, он сохранится на десятках инструментов;
  * порог в процентах цены вместо ATR («прошёл полпроцента — пройдёт ли дальше»);
  * меньший порог — больше сделок;
  * тренд-фильтр с «серой зоной»: закрытие по другую сторону EMA5 против тренда → нейтраль.

Импульс: ход close за 6 часов ≥ порога (в ATR часа или в % цены), в сторону хода, сторону
подтверждает гейт C с последнего закрытого часа. Выход — `--exit` (по умолчанию 240 ч /
стоп и трейлинг 6 ATR / активация 3), движок `early_entry.simulate`, комиссия из конфига.
Период выбора — до `--split-year`, проверки — после.

    python scripts/impulse_universe.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from early_entry import simulate
from entry_rules_lab import trend_variant
from exit_grid import metrics
from forexmodel.data.loader import load_minute_compact, resample_ohlcv
from forexmodel.features import indicators as ind
from pattern_lab import ALL_INSTRUMENTS, apply_exit, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]


def hourly(minute: pd.DataFrame, cfg) -> pd.DataFrame:
    """Часовые бары: ATR, гейт C и две серые зоны (EMA5 часа и EMA5 4-часовых закрытий)."""
    h = resample_ohlcv(minute, "1h")
    h["atr"] = ind.atr(h, cfg.features.atr_period)
    h["gate"] = trend_variant(h, "C", True, cfg, hold_h=2)
    h["t_close"] = h["time"] + pd.Timedelta("1h")
    ema1 = h["close"].ewm(span=5, adjust=False).mean()
    h4 = resample_ohlcv(minute, "4h")
    h4 = pd.DataFrame({"t_close": h4["time"] + pd.Timedelta("4h"), "ema4": h4["close"].ewm(span=5, adjust=False).mean()})
    ema4 = pd.merge_asof(h[["t_close"]], h4, on="t_close", direction="backward")["ema4"].to_numpy()
    g = h["gate"].to_numpy()
    c = h["close"].to_numpy()
    for name, ema in (("gate_ema1", ema1.to_numpy()), ("gate_ema4", ema4)):
        gg = g.copy()
        gg[(g > 0) & (c < ema)] = 0
        gg[(g < 0) & (c > ema)] = 0
        h[name] = gg
    h["move_atr"] = (h["close"] - h["close"].shift(6)) / h["atr"]
    h["move_pct"] = (h["close"] / h["close"].shift(6) - 1) * 100
    return h


def signal(h: pd.DataFrame, kind: str, thr: float, gate: str) -> pd.DataFrame:
    m = h["move_atr"] if kind == "atr" else h["move_pct"]
    side = np.where(m >= thr, 1, np.where(m <= -thr, -1, 0))
    side = np.where(side == h[gate].to_numpy(), side, 0)
    s = h[side != 0]
    return pd.DataFrame({"t_close": s["t_close"], "side": side[side != 0], "atr": s["atr"]}).reset_index(drop=True)


VARIANTS = ([("atr", x, "gate") for x in (2.0, 2.5, 3.0, 4.0)]
            + [("pct", x, "gate") for x in (0.5, 1.0, 1.5, 2.0, 3.0)]
            + [("atr", 3.0, "gate_ema1"), ("atr", 3.0, "gate_ema4")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inst", nargs="+", default=list(ALL_INSTRUMENTS))
    ap.add_argument("--exit", default="240:6:6:3")
    ap.add_argument("--slip", nargs="+", type=float, default=[0.0, 0.1])
    ap.add_argument("--split-year", type=int, default=2021)
    ap.add_argument("--name", default="impulse_universe")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)

    trades = []
    for inst in args.inst:
        cfg = apply_exit(instrument_cfg(inst), args.exit)
        minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
        h = hourly(minute, cfg)
        for kind, thr, gate in VARIANTS:
            sig = signal(h, kind, thr, gate)
            for slip in args.slip:
                t = simulate(minute, sig, cfg, cfg.simulation.commission_pct, slip_pct=slip)
                t["год"] = pd.to_datetime(t["time"]).dt.year
                g = {"gate": "C", "gate_ema1": "C + EMA5 часа", "gate_ema4": "C + EMA5 4ч"}[gate]
                trades.append(t.assign(инструмент=inst, порог=f"{thr:g} {'ATR' if kind == 'atr' else '%'}", гейт=g,
                                       проскальзывание=slip))
        print(f"[{inst}] готово: {len(h)} часов", flush=True)
        del minute
    t = pd.concat(trades, ignore_index=True)
    out = ROOT / "reports" / "patterns"
    t.to_csv(out / f"{args.name}_trades.csv", index=False)

    rows = []
    keys = ["порог", "гейт", "проскальзывание"]
    for k, g in t.sort_values("time").groupby(keys):
        for per, x in (("выбор", g[g["год"] < args.split_year]), ("проверка", g[g["год"] >= args.split_year]), ("всё", g)):
            for inst, xi in x.groupby("инструмент"):
                rows.append({**dict(zip(keys, k)), "период": per, "инструмент": inst, **metrics(xi["net"], xi["год"])})
            n = x["инструмент"].nunique()
            if n:
                rows.append({**dict(zip(keys, k)), "период": per, "инструмент": f"портфель ({n})",
                             **metrics(x["net"] / n, x["год"])})
    r = pd.DataFrame(rows)
    r.to_csv(out / f"{args.name}.csv", index=False)
    base = r[(r["порог"] == "3 ATR") & (r["гейт"] == "C")]
    for per in ("выбор", "проверка", "всё"):
        print(f"\nБАЗА (6 ч / 3 ATR, гейт C), период «{per}»")
        print(base[base["период"] == per].drop(columns=["порог", "гейт", "период"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
