"""Лаборатория паттернов: торговать явные шаблоны, а не угадывать направление каждого бара.

Идея (встреча 2026-09-28): модель не видит даже направления, поэтому вместо «когда
прибыль» размечаем ПАТТЕРНЫ, каждый из которых сам задаёт сторону сделки, и смотрим,
какие из них прибыльны: импульс продолжается, пробой диапазона продолжается, откат в
тренде выкупается, растяжение откатывает. Сторону даёт паттерн (или тренд-фильтр),
волатильность решает, торговать ли вообще (docs/results/volatility_and_tokdit.md).

Всё каузально: паттерн определяется по закрытию часового бара i, вход — через
`open_delay_minutes` после начала бара (то есть после закрытия), выход — штатный
трейлинг симулятора на минутках (`entry_rules_lab.forward_outcomes`), позиции не
перекрываются, комиссия — из конфига.

Чтобы не подгонять под историю, период делится на ВЫБОР (по умолчанию до 2021) и
ПРОВЕРКУ (2021+, по годам). Паттерн считается рабочим, только если он в плюсе после
комиссии в обоих периодах и на нескольких инструментах.

Паттерны (d — сторона, +1 лонг / −1 шорт):
  импульс k/x      — ход за k баров ≥ x ATR → в сторону хода (продолжение)
  импульс k/x разв — то же, но против хода (проверка обратной гипотезы)
  серия 3          — 3 бара подряд в одну сторону, ход ≥ 1.5 ATR → продолжение
  пробой N         — закрытие выше максимума / ниже минимума прошлых N баров
  пробой N сжатие  — то же после узкого диапазона (размах N баров ≤ 6 ATR)
  откат в тренде   — гейт C задаёт сторону, за 3 бара ход против неё ≥ 1 ATR
  растяжение x     — закрытие дальше x ATR от EMA20 → против (возврат к средней)
  по тренду        — вход по гейту C на каждом свободном баре (база)
  по тренду, EMA N — гейт C, но закрытие по другую сторону EMA N → нейтраль
                     (гипотеза со встречи: пробой средней — не разворот, а пауза)

Фильтры: все / высокая волатильность (ATR ≥ медианы за 20 дней) / по тренду (сторона
паттерна совпадает с гейтом C) / оба.

    python scripts/pattern_lab.py                       # серебро, золото, LKOH
    python scripts/pattern_lab.py --inst silver --split-year 2021
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from entry_rules_lab import forward_outcomes, sequential, trend_variant
from forexmodel.config import load_config
from forexmodel.data.loader import load_minute_compact
from forexmodel.features import indicators as ind

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = {
    # имя: (конфиг, csv минуток или None — как в конфиге)
    "silver": ("configs/silver.yaml", None),
    "gold": ("configs/silver.yaml", "data/raw/XAUUSD_1m_full.csv"),
    "lkoh": ("configs/default.yaml", None),
}


def instrument_cfg(name: str):
    path, csv = INSTRUMENTS[name]
    cfg = load_config(ROOT / path)
    if csv:
        cfg.data.csv_path = csv
    return cfg


def apply_exit(cfg, spec: str | None):
    """spec «часы:стоп:трейлинг:активация» в ATR часового бара; None — выход из конфига."""
    if spec:
        hours, sl, trail, act = (float(x) for x in spec.split(":"))
        cfg.simulation.horizon_minutes = int(hours * 60)
        cfg.simulation.sl_atr, cfg.simulation.trail_atr, cfg.simulation.activate_atr = sl, trail, act
    return cfg


def bars(name: str, refresh: bool = False, exit_spec: str | None = None) -> tuple[pd.DataFrame, object]:
    """Часовые бары с исходами лонга/шорта (трейлинг на минутках) и гейтом C; кэш в reports/_cache."""
    cfg = apply_exit(instrument_cfg(name), exit_spec)
    tag = "" if not exit_spec else "_" + exit_spec.replace(":", "-")
    cache = ROOT / "reports" / "_cache" / f"pattern_bars_{name}{tag}.pkl"
    if cache.exists() and not refresh:
        return pd.read_pickle(cache), cfg
    minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
    h = forward_outcomes(minute, cfg)
    del minute
    h["gate"] = trend_variant(h, "C", True, cfg, hold_h=2)
    cache.parent.mkdir(parents=True, exist_ok=True)
    h.to_pickle(cache)
    return h, cfg


def add_features(h: pd.DataFrame) -> pd.DataFrame:
    """Каузальные признаки на закрытии бара i (только бары ≤ i)."""
    h = h.copy()
    atr = h["atr_14"].replace(0, np.nan)
    c = h["close"]
    for k in (1, 3, 6):
        h[f"r{k}"] = (c - c.shift(k)) / atr
    up = np.sign(c.diff())
    h["same3"] = (up.rolling(3).sum().abs() == 3) * up
    for n in (24, 72):
        h[f"hi{n}"] = h["high"].shift(1).rolling(n).max()
        h[f"lo{n}"] = h["low"].shift(1).rolling(n).min()
        h[f"range{n}"] = (h[f"hi{n}"] - h[f"lo{n}"]) / atr
    for n in (20, 50):
        h[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    h["ext"] = (c - h["ema20"]) / atr
    h["vol_ratio"] = atr / atr.rolling(24 * 20, min_periods=24 * 5).median()
    return h


def gate_ema_neutral(h: pd.DataFrame, n: int) -> np.ndarray:
    g = h["gate"].to_numpy().copy()
    below, above = (h["close"] < h[f"ema{n}"]).to_numpy(), (h["close"] > h[f"ema{n}"]).to_numpy()
    g[(g > 0) & below] = 0
    g[(g < 0) & above] = 0
    return g


def patterns(h: pd.DataFrame) -> dict[str, np.ndarray]:
    g = h["gate"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for k, x in ((1, 1.5), (1, 2.5), (3, 2.0), (3, 3.0), (6, 3.0), (6, 4.0)):
        r = h[f"r{k}"].to_numpy()
        d = np.where(np.abs(r) >= x, np.sign(r), 0.0)
        out[f"импульс {k}/{x}"] = d
        out[f"импульс {k}/{x} разв"] = -d
    s = h["same3"].to_numpy()
    out["серия 3"] = np.where((s != 0) & (np.abs(h["r3"].to_numpy()) >= 1.5), s, 0.0)
    for n in (24, 72):
        c = h["close"].to_numpy()
        d = np.where(c > h[f"hi{n}"].to_numpy(), 1.0, np.where(c < h[f"lo{n}"].to_numpy(), -1.0, 0.0))
        out[f"пробой {n}"] = d
        out[f"пробой {n} сжатие"] = np.where(h[f"range{n}"].to_numpy() <= 6, d, 0.0)
    r3 = h["r3"].to_numpy()
    out["откат в тренде"] = np.where((g != 0) & (g * r3 <= -1), g, 0.0)
    ext = h["ext"].to_numpy()
    for x in (2.5, 3.5):
        out[f"растяжение {x}"] = np.where(np.abs(ext) >= x, -np.sign(ext), 0.0)
    out["по тренду"] = g.astype(float)
    for n in (20, 50):
        out[f"по тренду, EMA{n}"] = gate_ema_neutral(h, n).astype(float)
    return {k: np.nan_to_num(v) for k, v in out.items()}


def evaluate(h: pd.DataFrame, pats: dict, commission: float, split_year: int) -> pd.DataFrame:
    g = h["gate"].to_numpy()
    hv = (h["vol_ratio"] >= 1).to_numpy()
    rows = []
    for name, d in pats.items():
        filters = {"все": np.ones(len(h), bool), "выс. волат.": hv, "по тренду": d == g, "оба": hv & (d == g)}
        for fname, allow in filters.items():
            if name.startswith("по тренду") and fname in ("по тренду", "оба"):
                continue                                   # у трендовых входов совпадает по построению
            tr = sequential(h, d, allow, commission)
            if tr.empty:
                continue
            tr["год"] = pd.to_datetime(tr["time"]).dt.year
            row = {"паттерн": name, "фильтр": fname}
            for lab, part in (("выбор", tr[tr["год"] < split_year]), ("проверка", tr[tr["год"] >= split_year])):
                n = len(part)
                sd = part["net"].std()
                row[f"{lab}_n"] = n
                row[f"{lab}_до"] = part["gross"].mean() if n else np.nan
                row[f"{lab}_нетто"] = part["net"].mean() if n else np.nan
                row[f"{lab}_t"] = part["net"].mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
            chk = tr[tr["год"] >= split_year].groupby("год")["net"].sum()
            row["лет_в_плюсе"] = f"{int((chk > 0).sum())}/{len(chk)}"
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inst", nargs="+", default=list(INSTRUMENTS))
    ap.add_argument("--split-year", type=int, default=2021, help="первый год проверки")
    ap.add_argument("--refresh", action="store_true", help="пересчитать кэш исходов")
    ap.add_argument("--exit", dest="exits", nargs="+", default=[None],
                    help="варианты выхода «часы:стоп:трейлинг:активация» (ATR часа), напр. 48:4:4:2; по умолчанию — конфиг")
    ap.add_argument("--name", default="patterns")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    out_dir = ROOT / "reports" / "patterns"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    for spec in args.exits:
        for name in args.inst:
            print(f"[{name}] выход {spec or 'из конфига'}: бары и исходы...", flush=True)
            h, cfg = bars(name, args.refresh, spec)
            h = add_features(h)
            res = evaluate(h, patterns(h), cfg.simulation.commission_pct, args.split_year)
            res.insert(0, "выход", spec or "конфиг")
            res.insert(0, "инструмент", name)
            print(res.round(3).to_string(index=False), flush=True)
            tables.append(res)

    allr = pd.concat(tables)
    allr.to_csv(out_dir / f"{args.name}_all.csv", index=False)
    # кандидаты: в плюсе после комиссии и в выборе, и в проверке
    ok = allr[(allr["выбор_нетто"] > 0) & (allr["проверка_нетто"] > 0)]
    cnt = ok.groupby(["выход", "паттерн", "фильтр"]).size().rename("инструментов_в_плюсе").reset_index()
    print("\nПаттерн × фильтр в плюсе после комиссии в обоих периодах — число инструментов:")
    print(cnt.sort_values("инструментов_в_плюсе", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
