"""Ночной перебор параметров импульсной стратегии по списку ТЗ 15.10 (гипотеза H50).

Сам собирает данные, сам перебирает комбинации и сам выбирает лучшие — только по выбору 2016–2020;
проверка 2021–2026 считается для отчёта и в выборе не участвует. Переживает перезагрузку: каждый шаг
пишется на диск, повторный запуск продолжает с места остановки.

Что перебирается (одна проба = одна комбинация):
  * период ATR 10 / 14 / 20 — в пороге импульса и в стопе;
  * импульс: окно 4 / 6 / 8 ч × порог 2.5 / 3 / 3.5 ATR;
  * выход: стоп 1.5 / 2 / 2.5 / 3 ATR × тейк 2 / 3 / 4 ATR или трейлинг того же размера; стоп и
    трейлинг 6 ATR (база); Chandelier (трейлинг 3·ATR(22) с первой минуты); выход по времени 10 / 20 / 50 ч;
  * фильтр направления: без фильтра, гейт C, «всегда лонг», 111 классических (EMA 9–200, наклон EMA,
    пары 9/21, 12/26, 20/50, 50/200 и Фибоначчи, MACD 12/26/9, 8/17/9, 5/35/5, 19/39/9 × три вида
    сигнала, Supertrend, Дончиан 20/10 и 55/20, Кельтнер, Боллинджер, ADX, KAMA — на 1 ч, 4 ч, днях) и
    14 вариантов трёхслойного фильтра;
  * дополнительные фильтры поверх (каждый может быть выключен): ADX(14) > 20 / 25 на 1 ч или 4 ч;
    волатильность ATR14/ATR100 > 1 или перцентиль ATR за 250 баров > 0.5 / 0.7; RSI(14) не
    перекуплен / по ходу, RSI(2) не перекуплен; вчера NR7 / NR4 / inside bar / NR7 + inside bar
    (дневки с правильной границей дня: металлы — 17:00 по Чикаго, как сессия COMEX); уровни —
    пробой вчерашнего дня, 20 и 55 дней (в том числе на 0.25 ATR за уровнем), верх / низ 20-дневного
    диапазона, свежий 20-дневный экстремум (≤ 24 ч), пробой фрактала (K = 3, подтверждён через 3 бара),
    пробой азиатской сессии в Лондон (металлы), запас до уровня ≥ 0.5 / 1 ATR;
  * стороны: обе / только лонг / акции только лонг;
  * размер позиции: 1 или по риску (объём = риск / (стоп · ATR), нормирован на медиану — уровни риска
    0.5% и 1% отличаются лишь множителем, PF и t от него не зависят).

Как считается быстро и точно: для каждой пары «период ATR × выход» все сигналы импульса (объединение
всех окон и порогов) один раз прогоняются ШТАТНЫМ симулятором (дивиденды, стоп по гэпу, издержки B01 и
профиль биржи), а проба лишь отбирает сигналы маской и цепочкой без перекрытия, как симулятор.
Комбинация базы B01 воспроизводит её сделки (проверено в scripts/trend_grid.py).

Как ищется: сначала случайные комбинации, затем всё чаще — мутации лучших по выбору (эволюционный
поиск). Цель — PF портфеля (--scope) на выборе 2016–2020 при числе сделок не меньше доли базы.

    python scripts/night_search.py night --hours 8          # всё сразу: подготовка, поиск, отчёт
    python scripts/night_search.py prepare --workers 4      # данные и сделки (~2–3 ч, с продолжением)
    python scripts/night_search.py search --hours 8 --workers 3 --scope металлы
    python scripts/night_search.py report                   # reports/night_search/report.md
    python scripts/night_search.py finalize --top 3         # финалисты в журнал экспериментов (S36…)

Честная оценка в отчёте — walk-forward самого поиска: на год Y берётся лучшая комбинация по годам < Y
и торгует год Y. Если она не лучше базы, найденное «лучшее» — подгонка.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "night_search"
CAND = OUT / "cand"
FEAT = OUT / "feat"
INSTRUMENTS = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
STOCKS = {"lkoh", "gazp", "sber", "moex", "mtss"}
METALS = {"silver", "gold"}
SCOPES = {"все": INSTRUMENTS, "портфель 4": ["silver", "lkoh", "gazp", "sber"], "металлы": ["silver", "gold"]}
YEARS = list(range(2016, 2027))
SEL_END = 2021                       # выбор — сделки, открытые до 2021 года
DAY_START = {"silver": 17, "gold": 17}   # начало торгового дня по времени данных (металлы — Чикаго, COMEX)
ATR_PERIODS = (10, 14, 20)
WINDOWS = (4, 6, 8)
THRESH = (2.5, 3.0, 3.5)


def exits() -> dict[str, dict]:
    """Выходы: sl/trail/act — в ATR на входе, tp — тейк в ATR (exit_mode atr), hours — не дольше."""
    e = {"трейлинг 6 (база)": dict(sl=6.0, trail=6.0, act=3.0, hours=240)}
    for sl in (1.5, 2.0, 2.5, 3.0):
        for tp in (2.0, 3.0, 4.0):
            e[f"стоп {sl:g} тейк {tp:g}"] = dict(sl=sl, tp=tp, hours=240)
        e[f"стоп и трейлинг {sl:g}"] = dict(sl=sl, trail=sl, act=sl / 2, hours=240)
    e["Chandelier 3·ATR22"] = dict(sl=3.0, trail=3.0, act=0.0, hours=240, atr=22)
    for hh in (10, 20, 50):
        e[f"по времени {hh} ч"] = dict(sl=6.0, trail=6.0, act=3.0, hours=hh)
    return e


EXITS = exits()

FILTERS = {
    "adx": ["нет", "1ч > 20", "1ч > 25", "4ч > 20", "4ч > 25"],
    "vol": ["нет", "ATR14/ATR100 > 1", "перцентиль ATR > 0.5", "перцентиль ATR > 0.7"],
    "rsi": ["нет", "RSI14 не перекуплен", "RSI14 по ходу", "RSI2 не перекуплен"],
    "squeeze": ["нет", "вчера NR7", "вчера NR4", "вчера inside bar", "вчера NR7 + inside bar"],
    "level": ["нет", "пробой вчерашнего дня", "пробой 20 дней", "пробой 20 дней + 0.25 ATR", "пробой 55 дней",
              "верх 20-дн. диапазона", "свежий 20-дн. экстремум", "пробой фрактала", "пробой Азии (металлы)",
              "запас до уровня ≥ 0.5 ATR", "запас до уровня ≥ 1 ATR"],
}
SIDES = ["обе", "только лонг", "акции только лонг"]
SIZING = ["1", "по риску"]
BASE = {"atr": 14, "exit": "трейлинг 6 (база)", "window": 6, "thr": 3.0, "gate": "гейт C (база)",
        "adx": "нет", "vol": "нет", "rsi": "нет", "squeeze": "нет", "level": "нет", "side": "обе", "size": "1"}


def gate_names() -> list[str]:
    from trend_grid import gate_items

    return [name for _, name, _ in gate_items()]


# ---------------------------------------------------------------- 1. подготовка: признаки и сделки

def day_key(t: pd.Series, inst: str) -> np.ndarray:
    """Торговый день: у металлов начинается в 17:00 по времени данных (сессия COMEX), у остальных — в 0:00."""
    s = DAY_START.get(inst, 0)
    return (t + pd.Timedelta(hours=24 - s)).dt.floor("D").to_numpy() if s else t.dt.floor("D").to_numpy()


def build_features(h: pd.DataFrame, inst: str) -> dict[str, np.ndarray]:
    """Значения фильтров на каждом часовом баре — только по прошлому (бар i знает бары <= i)."""
    from forexmodel.features import indicators as ind
    from forexmodel.features.trend_score import _htf_value

    f: dict[str, np.ndarray] = {}
    c = h["close"].astype(float)
    for p in ATR_PERIODS + (22, 100):
        f[f"atr{p}"] = ind.atr(h, p).to_numpy(dtype=float)
    for p in ATR_PERIODS:
        for w in WINDOWS:
            f[f"mv{p}_{w}"] = ((c - c.shift(w)) / ind.atr(h, p)).to_numpy(dtype=float)
    f["adx1h"] = ind.adx(h, 14)["adx"].to_numpy(dtype=float)
    f["adx4h"] = _htf_value(h, "4h", lambda g: ind.adx(g, 14)["adx"])
    atr14 = pd.Series(f["atr14"])
    f["vr"] = f["atr14"] / f["atr100"]
    f["atrpct"] = atr14.rolling(250).rank(pct=True).to_numpy()
    f["rsi14"] = ind.rsi(c, 14).to_numpy(dtype=float)
    f["rsi2"] = ind.rsi(c, 2).to_numpy(dtype=float)
    # дневки с границей торгового дня; уровень дня известен только после его окончания
    day = day_key(h["time"], inst)
    d = h.groupby(day).agg(high=("high", "max"), low=("low", "min"))
    rng = d["high"] - d["low"]
    nr7, nr4 = rng <= rng.rolling(7).min(), rng <= rng.rolling(4).min()
    ib = (d["high"] < d["high"].shift(1)) & (d["low"] > d["low"].shift(1))
    prev = lambda s: s.astype(float).shift(1).reindex(day).to_numpy()  # noqa: E731
    f["nr7"], f["nr4"], f["ib"] = prev(nr7), prev(nr4), prev(ib)
    for n in (1, 20, 55):
        f[f"hi{n}"] = d["high"].rolling(n).max().shift(1).reindex(day).to_numpy()
        f[f"lo{n}"] = d["low"].rolling(n).min().shift(1).reindex(day).to_numpy()
    per_day = max(1, int(round(pd.Series(day).value_counts().median())))
    n = 20 * per_day
    f["since_hi"] = h["high"].rolling(n).apply(lambda x: n - 1 - x.argmax(), raw=True).to_numpy()
    f["since_lo"] = h["low"].rolling(n).apply(lambda x: n - 1 - x.argmin(), raw=True).to_numpy()
    # фракталы K = 3: вершина j подтверждается на баре j + 3, до этого её не видно
    k = 3
    hi, lo = h["high"].to_numpy(dtype=float), h["low"].to_numpy(dtype=float)
    fh, fl = np.full(len(h), np.nan), np.full(len(h), np.nan)
    last_h = last_l = np.nan
    for i in range(len(h)):
        j = i - k
        if j >= k:
            if hi[j] > hi[j - k:j].max() and hi[j] > hi[j + 1:i + 1].max():
                last_h = hi[j]
            if lo[j] < lo[j - k:j].min() and lo[j] < lo[j + 1:i + 1].min():
                last_l = lo[j]
        fh[i], fl[i] = last_h, last_l
    f["frac_hi"], f["frac_lo"] = fh, fl
    # азиатская сессия металлов: 18:00–01:59 по Чикаго текущего торгового дня; пробой — с 02:00 (Лондон)
    if inst in METALS:
        hour = h["time"].dt.hour.to_numpy()
        asia = (hour >= 18) | (hour < 2)
        g = pd.DataFrame({"day": day, "hi": np.where(asia, hi, np.nan), "lo": np.where(asia, lo, np.nan)})
        ah = g.groupby("day")["hi"].cummax().to_numpy()
        al = g.groupby("day")["lo"].cummin().to_numpy()
        london = (hour >= 2) & (hour < 17)
        f["asia_hi"], f["asia_lo"] = np.where(london, ah, np.nan), np.where(london, al, np.nan)
    else:
        f["asia_hi"] = f["asia_lo"] = np.full(len(h), np.nan)
    return {k2: v.astype(np.float32) for k2, v in f.items()}


def build_gates(h: pd.DataFrame) -> np.ndarray:
    """Все фильтры направления перебора ТЗ (scripts/trend_grid.py) на часовых барах: [фильтр, бар], int8."""
    import trend_grid as tg

    tg._W["base"] = dataclasses.replace(__import__("trend_filter_lab").base_config(), **tg.GATE_C)
    out = []
    for _, _, params in tg.gate_items():
        if params is None:
            out.append(np.zeros(len(h), np.int8) + 99)            # 99 — фильтра нет, сигнал проходит
            continue
        g = tg.gate_values(h[["time", "open", "high", "low", "close"]], params)
        out.append(np.where(np.isfinite(g), g, 99).astype(np.int8))
    return np.stack(out)


def prepare_instrument(inst: str) -> str:
    """Признаки, фильтры направления и сделки всех сигналов для каждой пары «период ATR × выход»."""
    logging.disable(logging.WARNING)
    from forexmodel.data.dividends import adjust_for_dividends, load_dividends
    from forexmodel.data.loader import load_minute_compact, resample_ohlcv
    from forexmodel.simulation.costs import profile_entry, recost
    from forexmodel.simulation.simulator import simulate_trades
    from trend_grid import instrument_config

    t0 = time.time()
    cfg = instrument_config(inst)
    minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache",
                                 cfg.data.volume_candidates)
    if cfg.data.dividends_path:
        minute = adjust_for_dividends(minute, load_dividends(ROOT / cfg.data.dividends_path, cfg.data.dividend_ticker))
    h = resample_ohlcv(minute, cfg.data.base_timeframe).reset_index(drop=True)
    start_year = 2021 if inst == "eth" else 2016
    fp = FEAT / f"{inst}.pkl"
    if fp.exists():
        feat = pd.read_pickle(fp)
    else:
        feat = {"bars": h[["time", "open", "high", "low", "close"]], "f": build_features(h, inst), "gates": build_gates(h)}
        pd.to_pickle(feat, fp)
        print(f"[{inst}] признаки и фильтры направления ({time.time() - t0:.0f} с)", flush=True)
    f = feat["f"]
    years = h["time"].dt.year.to_numpy()
    ok_year = (years >= start_year) & (years <= YEARS[-1])
    prof_base = profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst)
    prof_exch = profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst)
    for p in ATR_PERIODS:
        mv = np.column_stack([f[f"mv{p}_{w}"] for w in WINDOWS])
        big = np.abs(np.nan_to_num(mv)) >= min(THRESH)
        rows, sides = [], []
        for side in (1, -1):
            m = (big & (np.sign(np.nan_to_num(mv)) == side)).any(axis=1) & ok_year
            # первые окна года без истории: в walk-forward сигнал считается внутри года
            rows.append(np.flatnonzero(m))
            sides.append(np.full(m.sum(), side))
        rows, sides = np.concatenate(rows), np.concatenate(sides)
        order = np.lexsort((sides, rows))
        rows, sides = rows[order], sides[order]
        for ename, ex in EXITS.items():
            path = CAND / f"{inst}__atr{p}__{ename}.pkl"
            if path.exists():
                continue
            t1 = time.time()
            c = cfg
            s = c.simulation
            atr_stop = ex.get("atr", p)
            s.atr_col = f"atr_{atr_stop}"
            s.horizon_minutes = int(ex["hours"] * 60)
            s.sl_atr = ex["sl"]
            if "tp" in ex:
                s.exit_mode, s.tp_atr = "atr", ex["tp"]
            else:
                s.exit_mode, s.trail_atr, s.activate_atr = "trailing", ex["trail"], ex["act"]
            sig = pd.DataFrame({"time": h["time"].to_numpy()[rows], "final_class": np.where(sides > 0, 2, 0),
                                s.atr_col: f[f"atr{atr_stop}"][rows].astype(float), "_row": rows})
            trades, _ = simulate_trades(sig, minute, c)
            trades = trades.merge(sig[["time", "final_class", "_row"]].rename(columns={"time": "signal_dt"})
                                  .assign(side=lambda x: np.where(x.final_class == 2, "buy", "sell"))
                                  .drop(columns="final_class"), on=["signal_dt", "side"])
            tb = recost(trades, prof_base)
            te = recost(trades, prof_exch)
            trades["profit_base"], trades["profit_exch"] = tb["profit_pct"].to_numpy(), te["profit_pct"].to_numpy()
            trades["swap_base"], trades["swap_exch"] = tb["swap_pct"].to_numpy(), te["swap_pct"].to_numpy()
            trades["stop_pct"] = ex["sl"] * trades["atr_at_entry"] / trades["open_price"] * 100
            pd.to_pickle(trades, path.with_suffix(".tmp"))
            os.replace(path.with_suffix(".tmp"), path)
            print(f"[{inst}] ATR{p} · {ename}: {len(trades)} сделок ({time.time() - t1:.0f} с)", flush=True)
    del minute
    return f"{inst}: готово за {(time.time() - t0) / 60:.0f} мин"


def prepare(workers: int) -> int:
    for d in (CAND, FEAT):
        d.mkdir(parents=True, exist_ok=True)
    # тяжёлые инструменты первыми, чтобы не ждать их в хвосте
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(prepare_instrument, INSTRUMENTS):
            print(msg, flush=True)
    (OUT / "prepare.done").write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- 2. оценка одной комбинации

class Data:
    """Кэш признаков и сделок кандидатов (сделки одной пары «ATR × выход» по всем инструментам)."""

    def __init__(self, max_configs: int = 6):
        self.feat = {i: pd.read_pickle(FEAT / f"{i}.pkl") for i in INSTRUMENTS}
        self.gate_index = {n: k for k, n in enumerate(gate_names())}
        self.cache: OrderedDict = OrderedDict()
        self.max = max_configs

    def config(self, atr: int, exit_name: str) -> dict:
        key = (atr, exit_name)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        out = {}
        for inst in INSTRUMENTS:
            t = pd.read_pickle(CAND / f"{inst}__atr{atr}__{exit_name}.pkl")
            t = t.sort_values(["signal_dt", "side"]).reset_index(drop=True)
            r = t["_row"].to_numpy()
            f = self.feat[inst]["f"]
            sel_med = np.nanmedian(t.loc[pd.to_datetime(t["signal_dt"]).dt.year < SEL_END, "stop_pct"]) \
                if (pd.to_datetime(t["signal_dt"]).dt.year < SEL_END).any() else np.nanmedian(t["stop_pct"])
            out[inst] = {
                "row": r, "side": np.where(t["side"].to_numpy() == "buy", 1, -1).astype(np.int8),
                "sig": pd.to_datetime(t["signal_dt"]).to_numpy().astype("int64"),
                "cls": pd.to_datetime(t["close_dt"]).to_numpy().astype("int64"),
                "year": pd.to_datetime(t["signal_dt"]).dt.year.to_numpy(),
                "p_base": t["profit_base"].to_numpy(dtype=float), "p_exch": t["profit_exch"].to_numpy(dtype=float),
                "risk_size": np.clip(sel_med / t["stop_pct"].to_numpy(dtype=float), 0.25, 4.0),
                "f": {k: v[r] for k, v in f.items()}, "gates": self.feat[inst]["gates"][:, r],
                "close": self.feat[inst]["bars"]["close"].to_numpy(dtype=np.float32)[r],
            }
        self.cache[key] = out
        if len(self.cache) > self.max:
            self.cache.popitem(last=False)
        return out


def mask(ev: dict, prm: dict, inst: str, gate_k: int) -> np.ndarray:
    f, side, c = ev["f"], ev["side"], ev["close"]
    mv = f[f"mv{prm['atr']}_{prm['window']}"]
    m = (np.abs(np.nan_to_num(mv)) >= prm["thr"]) & (np.sign(np.nan_to_num(mv)) == side)
    g = ev["gates"][gate_k]
    m &= (g == 99) | (g == side)
    if prm["side"] == "только лонг" or (prm["side"] == "акции только лонг" and inst in STOCKS):
        m &= side > 0
    a = f[f"atr{prm['atr']}"]
    up = side > 0
    with np.errstate(invalid="ignore"):
        x = prm["adx"]
        if x != "нет":
            col = "adx1h" if x.startswith("1ч") else "adx4h"
            m &= f[col] > float(x.split(">")[1])
        x = prm["vol"]
        if x == "ATR14/ATR100 > 1":
            m &= f["vr"] > 1
        elif x.startswith("перцентиль"):
            m &= f["atrpct"] > float(x.split(">")[1])
        x = prm["rsi"]
        if x == "RSI14 не перекуплен":
            m &= np.where(up, f["rsi14"] < 70, f["rsi14"] > 30)
        elif x == "RSI14 по ходу":
            m &= np.where(up, f["rsi14"] > 50, f["rsi14"] < 50)
        elif x == "RSI2 не перекуплен":
            m &= np.where(up, f["rsi2"] < 90, f["rsi2"] > 10)
        x = prm["squeeze"]
        if x != "нет":
            flag = {"вчера NR7": f["nr7"], "вчера NR4": f["nr4"], "вчера inside bar": f["ib"],
                    "вчера NR7 + inside bar": f["nr7"] * f["ib"]}[x]
            m &= flag > 0
        x = prm["level"]
        if x != "нет":
            if x.startswith("пробой") and "Азии" not in x and "фрактала" not in x:
                n = {"вчерашнего дня": 1, "20 дней": 20, "20 дней + 0.25 ATR": 20, "55 дней": 55}[x[7:]]
                buf = 0.25 * a if "0.25" in x else 0.0
                m &= np.where(up, c > f[f"hi{n}"] + buf, c < f[f"lo{n}"] - buf)
            elif x == "верх 20-дн. диапазона":
                pos = (c - f["lo20"]) / np.where(f["hi20"] > f["lo20"], f["hi20"] - f["lo20"], np.nan)
                m &= np.where(up, pos >= 0.8, pos <= 0.2)
            elif x == "свежий 20-дн. экстремум":
                m &= np.where(up, f["since_hi"] <= 24, f["since_lo"] <= 24)
            elif x == "пробой фрактала":
                m &= np.where(up, c > f["frac_hi"], c < f["frac_lo"])
            elif x == "пробой Азии (металлы)":
                if inst in METALS:
                    m &= np.where(up, c > f["asia_hi"], c < f["asia_lo"])
            elif x.startswith("запас"):
                k = float(x.split("≥")[1].split()[0])
                room = np.where(up, (f["hi20"] - c) / a, (c - f["lo20"]) / a)
                m &= (room >= k) | (room < 0)                         # за уровнем — запас не ограничен
    return m


def chain(sig: np.ndarray, cls: np.ndarray, year: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Без перекрытия, как симулятор: сигнал пропускается, если он не позже выхода прошлой сделки; окна — годы."""
    out, last, last_y = [], None, None
    for i in np.flatnonzero(elig):
        if year[i] != last_y:
            last, last_y = None, year[i]
        if last is not None and sig[i] <= last:
            continue
        out.append(i)
        last = cls[i]
    return np.asarray(out, dtype=int)


def evaluate(data: Data, prm: dict, costs: str = "base", return_trades: bool = False):
    """Суммы по годам для каждого среза: {срез: {год: [сумма, плюс, минус, сделок]}} (инструменты поровну)."""
    cfgd = data.config(prm["atr"], prm["exit"])
    gk = data.gate_index[prm["gate"]]
    per_inst = {}
    for inst in INSTRUMENTS:
        ev = cfgd[inst]
        idx = chain(ev["sig"], ev["cls"], ev["year"], mask(ev, prm, inst, gk))
        p = (ev["p_base"] if costs == "base" else ev["p_exch"])[idx]
        if prm["size"] == "по риску":
            p = p * ev["risk_size"][idx]
        per_inst[inst] = (ev["year"][idx], p, idx)
    if return_trades:
        return per_inst
    res = {}
    for scope, members in SCOPES.items():
        agg = {}
        for inst in members:
            y, p, _ = per_inst[inst]
            w = 1.0 / len(members)
            for yy in np.unique(y):
                q = p[y == yy] * w
                a = agg.setdefault(int(yy), [0.0, 0.0, 0.0, 0])
                a[0] += float(q.sum()); a[1] += float(q[q > 0].sum()); a[2] += float(-q[q < 0].sum()); a[3] += int(len(q))
        res[scope] = agg
    return res


def period(agg: dict, sel: bool) -> dict:
    ys = [y for y in agg if (int(y) < SEL_END) == sel]
    s = sum(agg[y][0] for y in ys)
    pos, neg, n = (sum(agg[y][k] for y in ys) for k in (1, 2, 3))
    return {"итог": s, "PF": pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0), "сделок": n,
            "лет_в_плюсе": sum(1 for y in ys if agg[y][0] > 0), "лет": len(ys)}


def score(res: dict, scope: str, min_trades: float) -> float:
    m = period(res[scope], True)
    return float(m["PF"]) if m["сделок"] >= min_trades else 0.0


# ---------------------------------------------------------------- 3. поиск

def space() -> dict[str, list]:
    return {"atr": list(ATR_PERIODS), "exit": list(EXITS), "window": list(WINDOWS), "thr": list(THRESH),
            "gate": gate_names(), **FILTERS, "side": SIDES, "size": SIZING}


def sample(rng: random.Random, sp: dict) -> dict:
    prm = {k: rng.choice(v) for k, v in sp.items()}
    for k in FILTERS:                           # половина проб — без каждого фильтра (простые комбинации)
        if rng.random() < 0.5:
            prm[k] = "нет"
    return prm


def mutate(rng: random.Random, sp: dict, prm: dict) -> dict:
    out = dict(prm)
    for k in rng.sample(list(sp), rng.choice((1, 1, 2, 3))):
        out[k] = rng.choice(sp[k])
    return out


DB = OUT / "trials.sqlite"


def db():
    import sqlite3

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS trials (key TEXT, scope TEXT, costs TEXT, score REAL, robust REAL,
                   params TEXT, res TEXT, how TEXT, t TEXT, PRIMARY KEY (key, scope, costs))""")
    con.execute("CREATE INDEX IF NOT EXISTS by_score ON trials (scope, costs, score)")
    con.execute("CREATE INDEX IF NOT EXISTS by_robust ON trials (scope, costs, robust)")
    return con


def compact(res: dict) -> dict:
    """Хранится: по годам [сумма, плюс, минус, сделок] для всех срезов, округлено до 0.001."""
    return {sc: {str(y): [round(v[0], 3), round(v[1], 3), round(v[2], 3), v[3]] for y, v in a.items()}
            for sc, a in res.items()}


def load_trials(scope: str | None = None, costs: str | None = None, limit: int | None = None,
                order: str = "score") -> list[dict]:
    if not DB.exists():
        return []
    con = db()
    q = "SELECT key, scope, costs, score, robust, params, res, how, t FROM trials"
    args: list = []
    if scope:
        q += " WHERE scope = ? AND costs = ?"
        args = [scope, costs]
    if limit:
        q += f" ORDER BY {order} DESC LIMIT {int(limit)}"
    out = [{"key": r[0], "scope": r[1], "costs": r[2], "score": r[3], "robust": r[4], "params": json.loads(r[5]),
            "res": json.loads(r[6]), "how": r[7], "t": r[8]} for r in con.execute(q, args)]
    con.close()
    return out


def key(prm: dict) -> str:
    return json.dumps(prm, sort_keys=True, ensure_ascii=False)


def ready_configs() -> list[tuple[int, str]]:
    """Пары «период ATR × выход», для которых сделки готовы по всем инструментам."""
    return [(p, e) for p in ATR_PERIODS for e in EXITS
            if all((CAND / f"{i}__atr{p}__{e}.pkl").exists() for i in INSTRUMENTS)]


def neighbours(rng: random.Random, sp: dict, prm: dict, k: int) -> list[dict]:
    """k соседей: ровно один параметр другой; пара «ATR × выход» та же (без подгрузки сделок)."""
    out = []
    keys = [x for x in sp if x not in ("atr", "exit")]
    for name in rng.sample(keys, min(k, len(keys))):
        q = dict(prm)
        q[name] = rng.choice([v for v in sp[name] if v != prm[name]])
        out.append(q)
    return out


def search_worker(args) -> int:
    """Пачками по 150 проб на одной паре «ATR × выход» (смена пары — подгрузка сделок, ~1 с): пара —
    случайная (разведка) или пара одной из лучших проб (развитие); внутри пачки — случайные комбинации
    и мутации лучших. Доля развития растёт с числом проб.

    Устойчивая оценка (robust): если проба попала в 100 лучших по PF выбора, считаются ещё 8 её соседей
    (один параметр другой), robust = медиана PF по пробе и соседям. Одиночный выброс среди плохих
    соседей получает низкую оценку. Развитие и финалисты — по robust."""
    wid, hours, scope, costs, min_share, seed = args
    logging.disable(logging.WARNING)
    rng = random.Random(seed * 1000 + wid + int(time.time()))
    sp = space()
    data = Data()
    base_res = evaluate(data, BASE, costs)
    min_trades = min_share * period(base_res[scope], True)["сделок"]
    con = db()
    pending: list = []

    def flush() -> None:
        if pending:
            for _ in range(20):
                try:
                    con.executemany("INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?,?)", pending)
                    con.commit()
                    break
                except Exception:  # noqa: BLE001 — база занята другим процессом
                    time.sleep(1 + rng.random())
            pending.clear()

    def exists(k: str) -> bool:
        return con.execute("SELECT 1 FROM trials WHERE key=? AND scope=? AND costs=?", (k, scope, costs)).fetchone() is not None

    def run(prm: dict) -> tuple[float, dict]:
        res = evaluate(data, prm, costs)
        return score(res, scope, min_trades), res

    def record(prm: dict, how: str, cut: float) -> None:
        k = key(prm)
        if exists(k) or any(p[0] == k for p in pending):
            return
        sc, res = run(prm)
        robust = None
        if sc > cut:                                # претендент — устойчивость по соседям
            vals = [sc] + [run(q)[0] for q in neighbours(rng, sp, prm, 8)]
            robust = float(np.median(vals))
        pending.append((k, scope, costs, sc, robust, json.dumps(prm, ensure_ascii=False),
                        json.dumps(compact(res)), how, time.strftime("%m-%d %H:%M")))
        if len(pending) >= 50:
            flush()

    if wid == 0:
        record(dict(BASE), "база", -1.0)
        flush()
    t_end, n, batch = time.time() + hours * 3600, 0, 150
    while time.time() < t_end:
        configs = ready_configs()
        if not configs:
            time.sleep(60)
            continue
        flush()
        top = load_trials(scope, costs, limit=100, order="score")
        cut = top[-1]["score"] if len(top) >= 100 else 0.0
        best = [t for t in load_trials(scope, costs, limit=30, order="robust") if t["robust"] is not None]
        exploit = bool(best) and rng.random() > max(0.3, 1.0 - n / 20000)
        cand_best = [t for t in best if (t["params"]["atr"], t["params"]["exit"]) in configs]
        if exploit and cand_best:
            pick = rng.choice(cand_best)["params"]
            cfg_pair = (pick["atr"], pick["exit"])
        else:
            cfg_pair = rng.choice(configs)
        same = [t for t in best if (t["params"]["atr"], t["params"]["exit"]) == cfg_pair]
        for _ in range(batch):
            if time.time() >= t_end:
                break
            if same and rng.random() < 0.5:
                prm, how = mutate(rng, sp, rng.choice(same)["params"]), "мутация"
            elif best and rng.random() < 0.3:
                prm, how = mutate(rng, sp, rng.choice(best)["params"]), "мутация"
            else:
                prm, how = sample(rng, sp), "разведка"
            prm["atr"], prm["exit"] = cfg_pair
            record(prm, how, cut)
            n += 1
    flush()
    con.close()
    return n


def search(hours: float, workers: int, scope: str, costs: str, min_share: float, seed: int) -> int:
    db().close()
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        counts = list(ex.map(search_worker, [(w, hours, scope, costs, min_share, seed) for w in range(workers)]))
    con = db()
    total = con.execute("SELECT COUNT(*) FROM trials WHERE scope=? AND costs=?", (scope, costs)).fetchone()[0]
    con.close()
    print(f"проб за запуск: {sum(counts)} ({(time.time() - t0) / 60:.0f} мин), всего по срезу: {total}")
    return 0


# ---------------------------------------------------------------- 4. отчёт

def describe(prm: dict) -> str:
    parts = [f"ATR{prm['atr']}", f"импульс {prm['window']} ч / {prm['thr']:g} ATR", prm["exit"], f"фильтр: {prm['gate']}"]
    parts += [f"{prm[k]}" for k in FILTERS if prm[k] != "нет"]
    if prm["side"] != "обе":
        parts.append(prm["side"])
    if prm["size"] != "1":
        parts.append("размер по риску")
    return " · ".join(parts)


def report(scope: str, costs: str) -> int:
    """Потоково по всей базе проб: лучшие по выбору (и по устойчивой оценке), место базы, связь выбор →
    проверка (выборка 50 тыс.), walk-forward самого поиска (на год Y — лучшая по PF за годы до Y)."""
    import heapq

    from scipy.stats import spearmanr

    con = db()
    n_all = con.execute("SELECT COUNT(*) FROM trials WHERE scope=? AND costs=?", (scope, costs)).fetchone()[0]
    if not n_all:
        print("проб нет — сначала search")
        return 1
    base_row = con.execute("SELECT res, score FROM trials WHERE key=? AND scope=? AND costs=?",
                           (key(BASE), scope, costs)).fetchone()
    base_res = json.loads(base_row[0])[scope] if base_row else None
    wf_years = list(range(2018, 2027))
    base_n = {y: sum(base_res[str(yy)][3] for yy in range(2016, y) if str(yy) in base_res) if base_res else 0
              for y in wf_years}
    wf_best = {y: (-1.0, None, 0.0) for y in wf_years}        # (PF прошлого, параметры, итог года Y)
    sample_sel, sample_val = [], []
    rng = random.Random(0)
    n_ok = 0
    cur = con.execute("SELECT score, params, res FROM trials WHERE scope=? AND costs=?", (scope, costs))
    for k, (sc, prm_s, res_s) in enumerate(cur):
        a = json.loads(res_s)[scope]
        arr = np.array([a.get(str(y), [0, 0, 0, 0]) for y in YEARS], dtype=float)     # [год, 4]
        if sc > 0:
            n_ok += 1
            sel, val = arr[np.array(YEARS) < SEL_END], arr[np.array(YEARS) >= SEL_END]
            pv = val[:, 1].sum() / val[:, 2].sum() if val[:, 2].sum() > 0 else np.nan
            if len(sample_sel) < 50000:
                sample_sel.append(sc); sample_val.append(pv)
            else:
                r = rng.randrange(n_ok)
                if r < 50000:
                    sample_sel[r], sample_val[r] = sc, pv
        cum = np.cumsum(arr, axis=0)
        for y in wf_years:
            past = cum[YEARS.index(y) - 1]
            if past[2] > 0 and past[3] >= 0.3 * base_n[y]:
                pf = past[1] / past[2]
                if pf > wf_best[y][0]:
                    wf_best[y] = (pf, prm_s, float(arr[YEARS.index(y), 0]))
    rho = spearmanr(sample_sel, sample_val, nan_policy="omit").statistic if len(sample_sel) > 10 else float("nan")

    def table(order: str) -> list[str]:
        rows, sig_seen = [], set()
        for t in load_trials(scope, costs, limit=200, order=order):
            sig = json.dumps(t["res"][scope], sort_keys=True)       # одинаковый результат — одна строка
            if sig not in sig_seen and len(rows) < 20:
                sig_seen.add(sig)
                rows.append(t)
        out = ["| # | выбор PF | устойч. PF | выбор итог | проверка PF | проверка итог | сделок (пров.) | лет в плюсе | комбинация |",
               "|---|---|---|---|---|---|---|---|---|"]
        test_pf, test_sum = [], []
        for i, t in enumerate(rows):
            a = t["res"][scope]
            sv, vv = period(a, True), period(a, False)
            test_pf.append(vv["PF"]); test_sum.append(vv["итог"])
            out.append(f"| {i + 1} | {sv['PF']:.2f} | {t['robust'] if t['robust'] is None else round(t['robust'], 2)} | "
                       f"{sv['итог']:+.1f}% | {vv['PF']:.2f} | {vv['итог']:+.1f}% | {vv['сделок']} | "
                       f"{vv['лет_в_плюсе']}/{vv['лет']} | {describe(t['params'])} |")
        out += ["", f"Среднее этих 20 на проверке: PF {np.nanmean(np.where(np.isinf(test_pf), np.nan, test_pf)):.2f}, "
                f"итог {np.mean(test_sum):+.1f}%.", ""]
        return out

    lines = [f"# Ночной перебор: срез «{scope}», издержки {costs}", "",
             f"Проб: {n_all}, с достаточным числом сделок на выборе: {n_ok}. Связь PF выбор → проверка "
             f"(ранговая, выборка до 50 тыс.): **{rho:.2f}** — около нуля значит, что выбор по прошлому не "
             f"переносится на будущее.", ""]
    if base_row:
        bs, bv = period(base_res, True), period(base_res, False)
        place = con.execute("SELECT COUNT(*) FROM trials WHERE scope=? AND costs=? AND score > ?",
                            (scope, costs, base_row[1])).fetchone()[0] + 1
        lines += [f"**База B01**: выбор PF {bs['PF']:.2f}, итог {bs['итог']:+.1f}% (место {place} из {n_ok}); "
                  f"проверка PF {bv['PF']:.2f}, итог {bv['итог']:+.1f}%.", ""]
    lines += ["## 20 лучших по PF на выборе 2016–2020", "",
              "Комбинации с одинаковым результатом показаны один раз (например, «акции только лонг» на металлах "
              "ничего не меняет).", "", *table("score"),
              "## 20 лучших по устойчивой оценке (медиана PF пробы и 8 соседей на выборе)", "", *table("robust")]
    lines += ["## Walk-forward самого поиска — честная оценка", "",
              "На год Y берётся комбинация с лучшим PF за годы 2016…Y−1 (сделок не меньше 30% базы) и торгует год Y.", "",
              "| год | поиск | база B01 | выбрано |", "|---|---|---|---|"]
    tot_s = tot_b = 0.0
    for y in wf_years:
        _, prm_s, v = wf_best[y]
        b = base_res.get(str(y), [0])[0] if base_res else 0.0
        if y >= SEL_END:
            tot_s, tot_b = tot_s + v, tot_b + b
        lines.append(f"| {y} | {v:+.1f}% | {b:+.1f}% | {describe(json.loads(prm_s)) if prm_s else '—'} |")
    lines += ["", f"**Сумма 2021–2026: поиск {tot_s:+.1f}% против базы {tot_b:+.1f}%.** Если поиск не лучше базы, "
              "лучшие комбинации таблиц выше — подгонка под 2016–2020.", ""]
    con.close()
    text = "\n".join(lines)
    (OUT / f"report_{scope}_{costs}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------- 5. финалисты в журнал

def finalize(scope: str, costs: str, top: int, first_id: int) -> int:
    """Лучшие по выбору комбинации → сделки (штатный симулятор + отбор) в reports/walk_forward и записи
    каталога docs/experiments/catalog.yaml (S<first_id>…); затем python scripts/experiments.py."""
    import yaml

    trials = [t for t in load_trials(scope, costs, limit=top, order="robust") if t["robust"] is not None]
    data = Data()
    cat_p, desc_p = ROOT / "docs" / "experiments" / "catalog.yaml", ROOT / "docs" / "experiments" / "descriptions.yaml"
    cat_txt, desc_txt = cat_p.read_text(encoding="utf-8"), desc_p.read_text(encoding="utf-8")
    ids = {e["id"] for e in yaml.safe_load(cat_txt)["experiments"]}
    anchor = "\n  # ---------------- модель CatBoost: серебро ----------------"
    for k, t in enumerate(trials):
        eid = f"S{first_id + k}"
        if eid in ids:
            print(f"{eid} уже в каталоге — пропуск")
            continue
        prm = t["params"]
        per = evaluate(data, prm, costs, return_trades=True)
        run = f"ns_{eid.lower()}"
        for inst, (_, _, idx) in per.items():
            src = pd.read_pickle(CAND / f"{inst}__atr{prm['atr']}__{prm['exit']}.pkl")
            src = src.sort_values(["signal_dt", "side"]).reset_index(drop=True).iloc[idx].copy()
            if prm["size"] == "по риску":
                src["size"] = data.config(prm["atr"], prm["exit"])[inst]["risk_size"][idx]
            src.drop(columns=["profit_base", "profit_exch", "swap_base", "swap_exch", "_row"]).to_csv(
                ROOT / "reports" / "walk_forward" / f"{inst}_{run}_trades.csv", index=False)
        ex = EXITS[prm["exit"]]
        exit_line = (f"{{sl: {ex['sl']}, trail: {ex.get('trail', ex['sl'])}, act: {ex.get('act', 99)}, hours: {ex['hours']}}}")
        a = t["res"][scope]
        s, v = period(a, True), period(a, False)
        src_lines = "\n".join(f"      {i}: [{i}_{run}]" for i in INSTRUMENTS)
        entry = (f"  - id: {eid}\n    family: strategy\n    title: \"Ночной перебор: №{k + 1} по устойчивой оценке ({scope})\"\n"
                 f"    change: \"{describe(prm)}\"\n    hypothesis: H50\n    exit: {exit_line}\n    baseline: B01\n"
                 f"    verdict: \"выбор PF {s['PF']:.2f}, проверка PF {v['PF']:.2f}\"\n    doc: results/night_search.md\n"
                 f"    dividends: true\n    costs: {{profile: {'base' if costs == 'base' else 'exchange'}}}\n"
                 f"    sources:\n{src_lines}\n")
        cat_txt = cat_txt.replace(anchor, "\n" + entry.rstrip("\n") + anchor, 1)
        desc_txt += (f"{eid}:\n  what: \"Комбинация параметров ТЗ, найденная ночным перебором (scripts/night_search.py): "
                     f"{describe(prm)}.\"\n  why: \"Одна из лучших по устойчивой оценке (медиана PF пробы и соседей) среза «{scope}» на выборе 2016–2020.\"\n"
                     f"  outcome: \"Выбор: PF {s['PF']:.2f}, итог {s['итог']:+.1f}%. Проверка: PF {v['PF']:.2f}, итог "
                     f"{v['итог']:+.1f}%, лет в плюсе {v['лет_в_плюсе']} из {v['лет']}.\"\n")
        print(f"{eid}: {describe(prm)}")
    cat_p.write_text(cat_txt, encoding="utf-8")
    desc_p.write_text(desc_txt, encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=4)
    for name in ("search", "night"):
        s = sub.add_parser(name)
        s.add_argument("--hours", type=float, default=8.0, help="сколько искать (подготовка — сверх этого)")
        s.add_argument("--workers", type=int, default=3)
        s.add_argument("--prepare-workers", type=int, default=4)
        s.add_argument("--scope", default="металлы", choices=list(SCOPES))
        s.add_argument("--costs", default="base", choices=["base", "exchange"],
                       help="base — B01 (CFD); exchange — акции на бирже (S24/S25)")
        s.add_argument("--min-share", type=float, default=0.3, help="минимум сделок на выборе — доля от базы")
        s.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("report")
    r.add_argument("--scope", default="металлы", choices=list(SCOPES))
    r.add_argument("--costs", default="base", choices=["base", "exchange"])
    fz = sub.add_parser("finalize")
    fz.add_argument("--scope", default="металлы", choices=list(SCOPES))
    fz.add_argument("--costs", default="base", choices=["base", "exchange"])
    fz.add_argument("--top", type=int, default=3)
    fz.add_argument("--first-id", type=int, default=36)
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        return prepare(a.workers)
    if a.cmd == "search":
        return search(a.hours, a.workers, a.scope, a.costs, a.min_share, a.seed)
    if a.cmd == "night":
        prepare(a.prepare_workers)
        search(a.hours, a.workers, a.scope, a.costs, a.min_share, a.seed)
        return report(a.scope, a.costs)
    if a.cmd == "report":
        return report(a.scope, a.costs)
    return finalize(a.scope, a.costs, a.top, a.first_id)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
