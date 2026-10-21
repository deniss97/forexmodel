"""Перепроверка S49 (H59, 22.10): следуют ли сделки трёхслойному фильтру и что закрашено на графике журнала.

Денис по графику журнала: сделки открываются поздно, не закрываются при переключении фильтра, открываются в нейтрали.
Здесь по каждому рынку набора для запуска: с каким фильтром совпадают зоны графика (docs/experiments/site/data/candles),
стоит ли трёхслойный фильтр в сторону сделки на часе решения, выходит ли сделка сразу после его выключения, и насколько
вход позже гейта C, когда тот уже был в ту же сторону.

    python scripts/trend_follow_check.py
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import trend_grid as tg
from experiments import ROOT, WF
from trend_follow import SET, STOCKS, TL

CANDLES = ROOT / "docs" / "experiments" / "site" / "data" / "candles"


def zones(path, key: str) -> pd.Series:
    """Зоны фильтра со страницы журнала по часам: [индекс свечи, -1/0/1] с которого действует."""
    d = json.loads(path.read_text(encoding="utf-8"))
    t = pd.to_datetime(d["t0"] + np.cumsum(np.array(d["dt"])) * 3600, unit="s")
    g = np.zeros(len(t), int)
    ch = d.get(key) or []
    for k, (i, v) in enumerate(ch):
        g[i:ch[k + 1][0] if k + 1 < len(ch) else len(t)] = v
    return pd.Series(g, index=t)


def main() -> int:
    tg._init()
    rows = []
    for inst in SET:
        bars = tg._W["data"][inst]["bars"].reset_index(drop=True)
        tm = pd.to_datetime(bars["time"])
        tl = np.nan_to_num(tg.gate_values(bars, TL)).astype(int)
        gc = np.nan_to_num(tg.gate_values(bars, {})).astype(int)
        chart = zones(CANDLES / f"{inst}.json", "gate").reindex(tm.values).to_numpy()
        chart_tl = zones(CANDLES / f"{inst}.json", "gate_tl").reindex(tm.values).to_numpy()   # слой с 22.10
        tr = pd.read_csv(WF / f"{inst}_tf3__follow_trades.csv", parse_dates=["open_dt", "close_dt"])
        pos = pd.Series(np.arange(len(tm)), index=tm.values)
        # время сделки — открытие часа + 1 минута (scripts/trend_follow.py)
        e = pos.reindex((tr["open_dt"] - pd.Timedelta(minutes=1)).values).to_numpy().astype(int)
        x = pos.reindex((tr["close_dt"] - pd.Timedelta(minutes=1)).values).to_numpy().astype(int)
        s = np.where(tr["side"].eq("buy"), 1, -1)
        lag = []
        for a, sd in zip(e, s):
            if gc[a - 1] == sd:
                j = a - 1
                while j > 0 and gc[j] == sd:
                    j -= 1
                lag.append(a - 1 - j)
        rows.append({"рынок": inst, "сделок": len(tr),
                     "зоны gate = гейт C": np.mean(chart == gc), "зоны gate = трёхслойный": np.mean(chart == tl),
                     "зоны gate_tl = трёхслойный": np.mean(chart_tl == tl),
                     "вход: трёхслойный в сторону сделки": np.mean(tl[e - 1] == s),
                     "вход: гейт C в сторону сделки": np.mean(gc[e - 1] == s),
                     "выход сразу после выключения трёхслойного": np.mean(tl[x - 1] != s),
                     "вход позже гейта C, ч (медиана)": float(np.median(lag)) if lag else np.nan,
                     "сделок короче 3 ч": np.mean(x - e <= 3), "только лонг": inst in STOCKS})
    out = pd.DataFrame(rows).set_index("рынок")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(out.T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
