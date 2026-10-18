"""Прозрачность итератора (задача T36, созвон 18.10): как перебор собирает стратегию и как она торгует.

Четыре стратегии из финалистов честного перебора H54 (scripts/second_search.py) на наборе для запуска:
  1. сделки — тем же часовым движком, что считал перебор (strategy_search.trades, сверен с минутным
     симулятором), пишутся журналом сделок reports/walk_forward/<рынок>_wt<n>__eng_trades.csv — чтобы их можно было
     посмотреть на графике в журнале экспериментов (S45–S48);
  2. H61 — наложение на тренд-фильтр: брать сделку, только если трёхслойный фильтр смотрит в её сторону;
  3. H62 — согласие с импульсом S42: сделка одной стратегии берётся, если другая в этот момент держит позицию
     в ту же сторону на том же рынке (или открыла её за последние 24 ч).
Рынки считаются по одному (память машины ограничена). Отчёт — docs/results/iterator_walkthrough.md.

    python scripts/iterator_walkthrough.py trades      # сделки движка → reports/walk_forward
    python scripts/iterator_walkthrough.py report      # H61, H62 и описание (после пересборки журнала)
"""
from __future__ import annotations

import gc
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import second_search as s2      # подменяет входы: акции только в лонг
import strategy_search as ss

ROOT = ss.ROOT
WF = ROOT / "reports" / "walk_forward"
DOC = ROOT / "docs" / "results" / "iterator_walkthrough.md"
SET = s2.SET
STRATS = {
    1: {"entry": "NR7 → пробой", "f1": "только лонг", "f2": "ADX 1ч > 25", "exit": "Дончиан 20 дней · стоп 6", "atr": 10, "size": "по риску"},
    2: {"entry": "пробой 55 дней", "f1": "EMA200 4ч", "f2": "ATR14/ATR100 > 1", "exit": "Supertrend 4ч · стоп 3", "atr": 20, "size": "по риску"},
    3: {"entry": "keltner:20:10:2 @24h", "f1": "EMA200 дни", "f2": "нет", "exit": "обратный сигнал · стоп 6", "atr": 14, "size": "по риску"},
    4: {"entry": "пробой 20 дней + 0.25 ATR", "f1": "нет", "f2": "EMA200 дни", "exit": "Дончиан 20 дней · стоп 3", "atr": 10, "size": "1"},
}
EXP_ID = {1: "S45", 2: "S46", 3: "S47", 4: "S48"}
KIND = {0: "stop_loss", 1: "take_profit", 2: "timeout", 3: "signal"}


def load_inst(inst: str):
    from forexmodel.simulation.costs import profile_entry

    ins = ss.Inst(inst)
    if inst in s2.STOCKS:
        e = profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst)
        ins.spread, ins.swap = e["spread_pct"], {1: e["swap_long_pct"], -1: e["swap_short_pct"]}
    return ins


def trades_frame(ins, prm: dict) -> pd.DataFrame:
    i, s = ss.entries_for(ins, prm)
    if not len(i):
        return pd.DataFrame()
    r = ss.simulate(ins, prm, i, s)
    k = ss.chain(r["i"], r["x"])
    r = {key: v[k] for key, v in r.items()}
    gross = r["gross"]
    return pd.DataFrame({
        "signal_dt": pd.to_datetime(ins.t[r["i"]]), "open_dt": pd.to_datetime(r["t_entry"]), "close_dt": pd.to_datetime(r["t_exit"]),
        "side": np.where(r["s"] > 0, "buy", "sell"), "open_price": r["entry_px"], "exit_price": r["exit_px"],
        "size": r["size"], "gross_unit_pct": gross, "gross_pct": gross * r["size"], "profit_pct": r["net"],
        "exit_reason": [KIND[int(x)] for x in r["kind"]], "atr_at_entry": r["atr"], "rollovers": r["nroll"]})


def make_trades() -> int:
    for inst in SET:                                  # по одному рынку — память
        ins = load_inst(inst)
        for n, prm in STRATS.items():
            t = trades_frame(ins, prm)
            t = t[(t["open_dt"] >= "2016-01-01")]
            t.to_csv(WF / f"{inst}_wt{n}__eng_trades.csv", index=False)
            print(f"[{inst}] стратегия {n}: {len(t)} сделок", flush=True)
        del ins
        gc.collect()
    return 0


def main(argv) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "trades":
        return make_trades()
    if cmd == "report":
        from iterator_report import report
        return report()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
