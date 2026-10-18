"""Журнал сделок для проверок тренд-фильтра 18–19.10 — чтобы они стояли в журнале экспериментов на общем протоколе.

  * H58 (M19, L09) — сделки модели CatBoost (M13 — серебро, 3 seed; L07 — LKOH), которые трёхслойный фильтр
    пропускает: лонг при фильтре вверх, шорт при фильтре вниз на последнем закрытом часе до входа
    (как scripts/trend_model_gate.py);
  * H59 (S49) — сделки по самому трёхслойному фильтру на наборе для запуска (как scripts/trend_follow.py);
  * H60 (S50) — акции в лонг по фильтру «держать до противоположного сигнала» (как scripts/trend_hold.py).

Пишет reports/walk_forward/<прогон>_trades.csv; издержки пересчитывает сборка журнала по профилю эксперимента.

    python scripts/trend_journal.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import trend_grid as tg
from experiments import WF, load_catalog
from forexmodel.evaluation.experiments import standardize
from trend_follow import SET, STOCKS, TL, follow
from trend_model_gate import at_entry


def main() -> int:
    tg._init()
    data = tg._W["data"]
    gates = {i: np.nan_to_num(tg.gate_values(data[i]["bars"], TL)) for i in SET}
    _, exps = load_catalog()
    for mid in ("M13", "L07"):
        for inst, runs in exps[mid]["sources"].items():
            for run in runs:
                raw = pd.read_csv(WF / f"{run}_trades.csv", parse_dates=["open_dt", "close_dt"])
                s = at_entry(standardize(raw, inst, run), data[inst]["bars"], gates[inst])
                side = np.where(raw["side"].astype(str).isin(["buy", "1", "2"]), 1, -1)
                keep = raw[s == side]
                keep.to_csv(WF / f"{run}__tl_trades.csv", index=False)
                print(f"[{mid}] {run}: {len(keep)} из {len(raw)}", flush=True)
    for inst in SET:
        bars = data[inst]["bars"]
        t = follow(bars, gates[inst], inst in STOCKS)
        t.to_csv(WF / f"{inst}_tf3__follow_trades.csv", index=False)
        msg = f"[S49] {inst}: {len(t)}"
        if inst in STOCKS:
            h = follow(bars, gates[inst], True, hold=True)
            h.to_csv(WF / f"{inst}_tf3hold__follow_trades.csv", index=False)
            msg += f"; [S50] {len(h)}"
        print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
