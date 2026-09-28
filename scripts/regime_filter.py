"""Признак бокового рынка для импульсной стратегии (docs/results/patterns.md).

Импульс «6 ч ≥ 3 ATR по тренду» на серебре убыточен все 4 года 2015–2018 и прибылен все 8
лет 2019–2026. Здесь ищется каузальный признак режима, который отличает такие периоды.
Все признаки считаются на закрытии бара входа только по прошлому:

  er_20d     — эффективность движения Кауфмана за 20 дней: |ход| / сумма |ходов часа|;
  vr_60d     — отношение дисперсий суточных и часовых доходностей за 60 дней (>1 — тренд);
  atr_pct    — ATR часа в % цены (в тихие годы ход мал относительно комиссии);
  vol_ratio  — ATR к медиане за 20 дней;
  adx_d      — среднее |наклона| EMA50 за 20 дней в ATR (сила трендов);
  last10     — средний результат 10 последних ЗАКРЫТЫХ сделок самой стратегии.

Пороги (терцили) — по сделкам периода выбора (до `--split-year`), проверка — после.
Для каждого признака: результат сделки по терцилям в обоих периодах и на всех
инструментах, затем правило «торговать только в лучших терцилях» на проверке.

    python scripts/regime_filter.py --exit 48:4:4:2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from entry_rules_lab import sequential
from pattern_lab import INSTRUMENTS, add_features, bars, patterns

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["er_20d", "vr_60d", "atr_pct", "vol_ratio", "adx_d", "last10"]


def regime_features(h: pd.DataFrame) -> pd.DataFrame:
    c = h["close"]
    n = 24 * 20
    h = h.copy()
    h["er_20d"] = (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
    r1 = np.log(c).diff()
    r24 = np.log(c).diff(24)
    h["vr_60d"] = r24.rolling(24 * 60).var() / (24 * r1.rolling(24 * 60).var())
    h["atr_pct"] = h["atr_14"] / c * 100
    h["adx_d"] = (h["ema50"].diff() / h["atr_14"]).abs().rolling(n).mean()
    return h


def strategy_trades(h, d, allow, commission) -> pd.DataFrame:
    t = sequential(h, d, allow, commission)
    T = h["time"].to_numpy()
    i = np.searchsorted(T, t["time"].to_numpy())
    t["i"] = i
    for f in FEATS[:-1]:
        t[f] = h[f].to_numpy()[i]
    # результат 10 последних закрытых сделок: сделки идут без перекрытия, поэтому
    # к входу сделки k все сделки < k уже закрыты
    t["last10"] = t["net"].shift(1).rolling(10, min_periods=5).mean()
    t["год"] = pd.to_datetime(t["time"]).dt.year
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exit", default="48:4:4:2")
    ap.add_argument("--pattern", default="импульс 6/3.0")
    ap.add_argument("--split-year", type=int, default=2021)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    trades = []
    for inst in INSTRUMENTS:
        h, cfg = bars(inst, exit_spec=args.exit)
        h = regime_features(add_features(h))
        d = patterns(h)[args.pattern]
        t = strategy_trades(h, d, d == h["gate"].to_numpy(), cfg.simulation.commission_pct)
        trades.append(t.assign(инструмент=inst))
    t = pd.concat(trades, ignore_index=True)
    t["период"] = np.where(t["год"] < args.split_year, "выбор", "проверка")

    rows, rules = [], []
    for f in FEATS:
        for inst, g in t.groupby("инструмент"):
            sel = g[g["период"] == "выбор"][f].dropna()
            q1, q2 = np.quantile(sel, [1 / 3, 2 / 3])          # терцили — только по периоду выбора
            ter = np.select([g[f] <= q1, g[f] <= q2], ["низ", "сред"], "верх")
            ter = np.where(g[f].isna(), "нет", ter)
            for (per, tt), x in g.assign(терциль=ter).groupby(["период", "терциль"]):
                rows.append({"признак": f, "инструмент": inst, "период": per, "терциль": tt,
                             "сделок": len(x), "нетто": x["net"].mean()})
    r = pd.DataFrame(rows)
    out = ROOT / "reports" / "patterns"
    r.to_csv(out / f"regime_{args.exit.replace(':', '-')}.csv", index=False)
    piv = r[r["терциль"] != "нет"].pivot_table(index=["признак", "терциль"], columns=["инструмент", "период"],
                                               values="нетто")
    print(f"Нетто на сделку по терцилям признака (пороги — по периоду выбора), {args.pattern} по тренду, выход {args.exit}")
    print(piv.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
