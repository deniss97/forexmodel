"""Контекст импульса: подтверждение связанным инструментом, час суток, день недели, сила хода.

Берёт исходы кандидатов из scripts/impulse_gate_tuner.py events (сделка 240 ч / 6 ATR для каждого
часа с ходом ≥ 2.5 ATR), оставляет сигналы базовой стратегии (6 ч ≥ 3 ATR по гейту C) и смотрит,
как средний результат сделки (до комиссии, все кандидаты, включая перекрывающиеся — так больше
наблюдений) зависит от:
  * подтверждения: у связанного инструмента (серебро ↔ золото; у акций — другие акции MOEX) в тот
    же час ход за 6 ч ≥ 1.5 ATR в ту же сторону / против / нет хода;
  * часа суток и дня недели сигнала;
  * силы хода (3–3.5 / 3.5–4 / 4–5 / > 5 ATR).
Периоды 2015–2020 и 2021–2026 — отдельно: интересны только различия с одним знаком в обоих.

    python scripts/impulse_context.py
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.features.trend import add_trend_filter
from impulse_gate_tuner import EVENTS, WINDOWS
from trend_filter_lab import base_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audit"
PEERS = {"silver": ["gold"], "gold": ["silver"], "lkoh": ["gazp", "sber", "moex", "mtss"],
         "gazp": ["lkoh", "sber", "moex", "mtss"], "sber": ["lkoh", "gazp", "moex", "mtss"],
         "moex": ["lkoh", "gazp", "sber", "mtss"], "mtss": ["lkoh", "gazp", "sber", "moex"]}


def move6(bars: pd.DataFrame) -> pd.Series:
    pc = bars["close"].shift(1)
    tr = np.maximum(bars["high"] - bars["low"], np.maximum((bars["high"] - pc).abs(), (bars["low"] - pc).abs()))
    atr = tr.rolling(14).mean()
    return pd.Series(((bars["close"] - bars["close"].shift(6)) / atr).to_numpy(), index=bars["time"])


def main() -> int:
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    data = pd.read_pickle(EVENTS)
    tc = base_config()
    moves = {inst: move6(ev["bars"]) for inst, ev in data.items()}
    frames = []
    for inst, ev in data.items():
        gate = add_trend_filter(ev["bars"], tc)["trend_4h"].to_numpy(dtype=float)[ev["rows"]]
        m6 = ev["mv"][:, WINDOWS.index(6)]
        side = np.sign(np.nan_to_num(m6))
        ok = (np.abs(np.nan_to_num(m6)) >= 3.0) & (side == gate)
        gross = np.where(side > 0, ev["g_long"], ev["g_short"])
        t_sig = pd.DatetimeIndex(ev["t_entry"] - np.timedelta64(61, "m"))      # начало часа сигнала
        d = pd.DataFrame({"inst": inst, "time": t_sig, "year": ev["year"], "side": side, "size": np.abs(m6),
                          "gross": gross})[ok]
        d = d[np.isfinite(d["gross"])]
        # подтверждение связанными инструментами в тот же час
        peer = np.zeros(len(d))
        for p in PEERS.get(inst, []):
            pm = moves[p].reindex(d["time"]).to_numpy()
            peer += np.where(np.abs(np.nan_to_num(pm)) >= 1.5, np.sign(np.nan_to_num(pm)) * d["side"].to_numpy(), 0)
        d["подтверждение"] = np.select([peer > 0, peer < 0], ["в ту же сторону", "против"], "нет хода")
        d["час"] = d["time"].dt.hour
        d["день"] = d["time"].dt.dayofweek
        d["сила"] = pd.cut(d["size"], [3, 3.5, 4, 5, 100], labels=["3–3.5", "3.5–4", "4–5", "> 5"], right=False)
        d["период"] = np.where(d["year"] < 2021, "2015–20", "2021–26")
        d["net"] = d["gross"] - ev["commission"]
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D.to_csv(OUT / "impulse_context_events.csv", index=False)
    print(f"сигналов базовой стратегии (все кандидаты, с перекрытиями): {len(D)}")

    def table(col, by_inst=False):
        idx = [col] if not by_inst else ["inst", col]
        g = D.groupby(idx + ["период"], observed=True)["net"].agg(["mean", "size"])
        t = g["mean"].unstack("период").round(3)
        n = g["size"].unstack("период")
        t.columns = [f"нетто {c}" for c in t.columns]
        n.columns = [f"n {c}" for c in n.columns]
        return pd.concat([t, n], axis=1)

    print("\nПО ПОДТВЕРЖДЕНИЮ связанным инструментом (ход ≥ 1.5 ATR за 6 ч в тот же час), % на сделку до/после комиссии 0.04")
    print(table("подтверждение").to_string())
    print("\n  то же по инструментам:")
    print(table("подтверждение", by_inst=True).to_string())
    print("\nПО СИЛЕ ХОДА, ATR")
    print(table("сила").to_string())
    print("\nПО ДНЮ НЕДЕЛИ (0 — понедельник)")
    print(table("день").to_string())
    print("\nПО ЧАСУ СИГНАЛА, форекс (серебро, золото; время источника)")
    Df = D[D["inst"].isin(["silver", "gold"])]
    print(Df.groupby(["час", "период"])["net"].agg(["mean", "size"]).unstack("период").round(3).to_string())
    print("\nПО ЧАСУ СИГНАЛА, акции MOEX (МСК)")
    Ds = D[D["inst"].isin(["lkoh", "gazp", "sber", "moex", "mtss"])]
    print(Ds.groupby(["час", "период"])["net"].agg(["mean", "size"]).unstack("период").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
