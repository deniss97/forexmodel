"""Импульсная стратегия: подбор выхода (удержание × трейлинг) и проверка проскальзыванием.

Стратегия — docs/results/patterns.md §5: ход за 6 ч ≥ 3 ATR часа в сторону тренд-гейта C,
вход после закрытия часа. Движок — `early_entry.simulate` (сделки только по сигналам,
трейлинг на минутках, без перекрытия, комиссия из конфига).

--mode grid: удержание `--hours` × трейлинг `--trails` (в ATR часа; стоп = трейлинг,
  активация = трейлинг / 2). Выход выбирается по ПОРТФЕЛЮ трёх инструментов (1/3 капитала
  на каждый) на периоде выбора (< `--split-year`) по отношению итога к макс. просадке;
  проверка — после. Печатается вся сетка, чтобы видеть, гладко ли.
--mode slippage: для выхода `--exit` — задержка входа (минут после закрытия часа, цены
  реальные) и проскальзывание против сделки (% цены или доля ATR); стоп и трейлинг
  считаются от цены исполнения.

    python scripts/exit_grid.py --mode grid
    python scripts/exit_grid.py --mode slippage --exit 120:6:6:3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from early_entry import signals, simulate
from forexmodel.data.loader import load_minute_compact
from pattern_lab import INSTRUMENTS, apply_exit, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]


def metrics(p: pd.Series, years: pd.Series) -> dict:
    """p — % сделок в порядке закрытия; итог без реинвестирования, как в отчётах пайплайна."""
    if not len(p):
        return {"сделок": 0}
    p = p.reset_index(drop=True)
    eq = p.cumsum()
    loss = -p[p < 0].sum()
    y = p.groupby(years.reset_index(drop=True)).sum()
    dd = float((eq - eq.cummax().clip(lower=0)).min())
    return {"сделок": len(p), "итог": p.sum(), "на_сделку": p.mean(),
            "t": p.mean() / (p.std() / np.sqrt(len(p))) if len(p) > 1 else np.nan,
            "PF": p[p > 0].sum() / loss if loss > 0 else np.inf, "winrate": (p > 0).mean() * 100,
            "просадка": dd, "итог/просадка": p.sum() / -dd if dd < 0 else np.inf,
            "лет_в_плюсе": f"{int((y > 0).sum())}/{len(y)}"}


def summarize(trades: pd.DataFrame, split_year: int, keys: list[str]) -> pd.DataFrame:
    rows = []
    trades = trades.sort_values("exit_order")
    for k, g in trades.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        for per, x in (("выбор", g[g["год"] < split_year]), ("проверка", g[g["год"] >= split_year]), ("всё", g)):
            for inst, xi in [("портфель", x)] + list(x.groupby("инструмент")):
                w = 1 / 3 if inst == "портфель" else 1.0          # портфель: 1/3 капитала на инструмент
                rows.append({**dict(zip(keys, k)), "период": per, "инструмент": inst,
                             **metrics(xi["net"] * w, xi["год"])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["grid", "slippage"], default="grid")
    ap.add_argument("--hours", nargs="+", type=int, default=[48, 72, 120, 168, 240])
    ap.add_argument("--trails", nargs="+", type=float, default=[3, 4, 6, 8])
    ap.add_argument("--exit", default="120:6:6:3", help="для slippage: часы:стоп:трейлинг:активация")
    ap.add_argument("--delays", nargs="+", type=int, default=[1, 5, 15, 30, 60])
    ap.add_argument("--slip-pct", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    ap.add_argument("--slip-atr", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    ap.add_argument("--split-year", type=int, default=2021)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)

    all_tr = []
    for inst in INSTRUMENTS:
        base = instrument_cfg(inst)
        minute = load_minute_compact(ROOT / base.data.csv_path, base.data.time_col, ROOT / "reports" / "_cache")
        sig = signals(minute, base, "60min", 6, 3.0)
        comm = base.simulation.commission_pct
        if args.mode == "grid":
            runs = [({"часы": hrs, "трейлинг": tr}, f"{hrs}:{tr}:{tr}:{tr / 2}", {})
                    for hrs in args.hours for tr in args.trails]
        else:
            runs = [({"вариант": "база"}, args.exit, {})]
            runs += [({"вариант": f"задержка {d} мин"}, args.exit, {"delay_min": d}) for d in args.delays if d != 1]
            runs += [({"вариант": f"проскальзывание {x}%"}, args.exit, {"slip_pct": x}) for x in args.slip_pct]
            runs += [({"вариант": f"проскальзывание {x} ATR"}, args.exit, {"slip_atr": x}) for x in args.slip_atr]
        for key, spec, kw in runs:
            cfg = apply_exit(instrument_cfg(inst), spec)
            tr = simulate(minute, sig, cfg, comm, **kw)
            tr["год"] = pd.to_datetime(tr["time"]).dt.year
            tr["exit_order"] = pd.to_datetime(tr["time"])     # порядок входа ≈ порядок закрытия (без перекрытия)
            all_tr.append(tr.assign(инструмент=inst, **key))
        print(f"[{inst}] {len(runs)} вариантов готово", flush=True)
        del minute
    t = pd.concat(all_tr, ignore_index=True)
    keys = ["часы", "трейлинг"] if args.mode == "grid" else ["вариант"]
    res = summarize(t, args.split_year, keys)
    name = args.name or f"exit_{args.mode}"
    out = ROOT / "reports" / "patterns"
    res.to_csv(out / f"{name}.csv", index=False)
    t.to_csv(out / f"{name}_trades.csv", index=False)

    port = res[res["инструмент"] == "портфель"]
    for per in ("выбор", "проверка", "всё"):
        x = port[port["период"] == per]
        print(f"\nПОРТФЕЛЬ, период «{per}»")
        if args.mode == "grid":
            for v in ("итог", "просадка", "итог/просадка", "t"):
                print(f"  {v}:")
                print(x.pivot_table(index="часы", columns="трейлинг", values=v).round(2).to_string())
        else:
            print(x.drop(columns=["период", "инструмент"]).round(3).to_string(index=False))
    if args.mode == "grid":
        sel = port[port["период"] == "выбор"].sort_values("итог/просадка", ascending=False).head(5)
        print("\nЛучшие по итог/просадка на периоде выбора:")
        print(sel[["часы", "трейлинг", "итог", "просадка", "итог/просадка", "t"]].round(2).to_string(index=False))
        best = sel.iloc[0]
        chk = res[(res["часы"] == best["часы"]) & (res["трейлинг"] == best["трейлинг"])]
        print(f"\nВыбран выход {int(best['часы'])} ч / трейлинг {best['трейлинг']} ATR — все периоды и инструменты:")
        print(chk.drop(columns=["часы", "трейлинг"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
