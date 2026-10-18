"""Перебор САМОСТОЯТЕЛЬНЫХ стратегий из блоков списка ТЗ 15.10 (гипотеза H50, вторая версия).

Каждая проба — новая стратегия, с импульсом не связанная: вход по сигналу индикатора, до двух фильтров,
выход, период ATR и размер позиции. Выбор — только по 2016–2020; проверка 2021–2026 — для отчёта.
Финалисты пересчитываются ШТАТНЫМ минутным симулятором с издержками B01 и попадают в журнал
экспериментов рядом с базой B01 — на одном тесте (сделки с 2021 года, те же 8 рынков).

Блоки:
  * вход (≈ 140): смена направления индикатора на 1 ч / 4 ч / днях — цена против EMA 9–200, пересечения
    EMA 9/21, 12/26, 20/50, 50/200 и Фибоначчи, наклон EMA, MACD 12/26/9, 8/17/9, 5/35/5, 19/39/9 ×
    (сигнальная / ноль / гистограмма), Supertrend (10, 3), (7, 3), (10, 2), пробой Дончиана 20 / 55,
    Кельтнера, Боллинджера, ADX с направлением по DI, KAMA; возврат к среднему RSI(14) 30/70 и RSI(2)
    10/90; пробой дня сжатия на следующий день (NR7, NR4, inside bar, NR7 + inside bar); пробой
    вчерашнего дня, 20 и 55 дней (у самого уровня и на 0.25 ATR за ним); ложный пробой 20 дней (Turtle
    Soup); пробой фрактала K = 3; пробой азиатской сессии в Лондон (металлы);
  * фильтры (до двух): цена выше / ниже EMA200 на 1 ч / 4 ч / днях или EMA50 на днях (лонг только выше),
    ADX(14) > 20 / 25 на 1 ч или 4 ч, ATR14/ATR100 > 1, перцентиль ATR за 250 баров > 0.5, только лонг;
  * выход: стоп 1.5 / 2 / 2.5 / 3 ATR × тейк 2 / 3 / 4 ATR; трейлинг 1.5 / 2 / 2.5 / 3 / 6 ATR; Chandelier
    3·ATR(22); по сигналу — разворот Supertrend (10, 3) на 1 ч / 4 ч, канал Дончиана 10 / 20 часов или
    дней, обратный сигнал входа — с защитным стопом 3 или 6 ATR; по времени 10 / 20 / 50 часов (стоп 3 ATR);
    не дольше 240 ч;
  * период ATR 10 / 14 / 20; размер — 1 или по риску (объём = риск / (стоп·ATR), нормирован на медиану).

Расчёт пробы — на часовых свечах (вход по открытию следующего часа, стоп и тейк по максимуму / минимуму
часа, при гэпе — по открытию, стоп раньше тейка, если оба в одном часе; издержки B01 — комиссия,
проскальзывание, свопы по переносам). Совпадение с минутным симулятором проверяется командой check и в
отчёте finalize (оценка движка против штатного расчёта по каждому финалисту).

    python scripts/strategy_search.py check                     # движок против минутного симулятора
    python scripts/strategy_search.py search --hours 12         # 3 процесса: металлы, портфель 4, все рынки
    python scripts/strategy_search.py report                    # reports/strategy_search/report_<срез>.md
    python scripts/strategy_search.py finalize --top 2          # штатный симулятор → журнал (S36…)
    python scripts/experiments.py                                # метрики журнала против B01

Данные — reports/night_search/feat (scripts/night_search.py prepare: свечи с дивидендами, индикаторы,
уровни с границей торгового дня; у металлов день начинается в 17:00 по Чикаго, как сессия COMEX).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "reports" / "night_search" / "feat"
OUT = ROOT / "reports" / "strategy_search"
DB = OUT / "trials.sqlite"
INSTRUMENTS = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
METALS = {"silver", "gold"}
SCOPES = {"металлы": ["silver", "gold"], "портфель 4": ["silver", "lkoh", "gazp", "sber"], "все": INSTRUMENTS}
YEARS = list(range(2016, 2027))
SEL_END = 2021
DAY_START = {"silver": 17, "gold": 17}
MAX_HOLD = 240
# B01 на выборе 2016–2020 (журнал): сделок по срезам — порог числа сделок для пробы
BASE_SEL_TRADES = {"металлы": 461, "портфель 4": 665, "все": 1196}


# ---------------------------------------------------------------- блоки

def exits() -> dict[str, dict]:
    e = {}
    for sl in (1.5, 2.0, 2.5, 3.0):
        for tp in (2.0, 3.0, 4.0):
            e[f"стоп {sl:g} · тейк {tp:g}"] = dict(sl=sl, tp=tp, H=MAX_HOLD)
    for tr in (1.5, 2.0, 2.5, 3.0, 6.0):
        e[f"трейлинг {tr:g}"] = dict(sl=tr, trail=tr, act=tr / 2, H=MAX_HOLD)
    e["Chandelier 3·ATR22"] = dict(sl=3.0, trail=3.0, act=0.0, H=MAX_HOLD, atr=22)
    for sig in ("Supertrend 1ч", "Supertrend 4ч", "Дончиан 10 ч", "Дончиан 20 ч", "Дончиан 10 дней",
                "Дончиан 20 дней", "обратный сигнал"):
        for sl in (3.0, 6.0):
            e[f"{sig} · стоп {sl:g}"] = dict(sl=sl, sig=sig, H=MAX_HOLD)
    for hh in (10, 20, 50):
        e[f"по времени {hh} ч"] = dict(sl=3.0, H=hh)
    return e


EXITS = exits()
FILTERS = ["нет", "EMA200 1ч", "EMA200 4ч", "EMA200 дни", "EMA50 дни", "ADX 1ч > 20", "ADX 1ч > 25", "ADX 4ч > 20",
           "ADX 4ч > 25", "ATR14/ATR100 > 1", "перцентиль ATR > 0.5", "только лонг"]
LEVEL_ENTRIES = ["NR7 → пробой", "NR4 → пробой", "inside bar → пробой", "NR7 + inside bar → пробой",
                 "пробой вчерашнего дня", "пробой 20 дней", "пробой 20 дней + 0.25 ATR", "пробой 55 дней",
                 "пробой 55 дней + 0.25 ATR", "ложный пробой 20 дней", "пробой фрактала", "пробой Азии (металлы)",
                 "RSI14 30/70", "RSI2 10/90"]


def classic_names() -> list[str]:
    from trend_grid import gate_items

    return [name for grp, name, _ in gate_items() if grp.startswith("H45")]


def day_key(t: pd.Series, inst: str) -> np.ndarray:
    s = DAY_START.get(inst, 0)
    return (t + pd.Timedelta(hours=24 - s)).dt.floor("D").to_numpy() if s else t.dt.floor("D").to_numpy()


def first_per_day(day: np.ndarray, m: np.ndarray) -> np.ndarray:
    s = pd.Series(m.astype(int))
    return m & (s.groupby(day).cumsum().to_numpy() == 1)


def events_from_state(s: np.ndarray) -> np.ndarray:
    prev = np.r_[0, s[:-1]]
    return np.where((s != prev) & (s == 1), 1, np.where((s != prev) & (s == -1), -1, 0)).astype(np.int8)


class Inst:
    """Всё для расчёта стратегий на одном инструменте: свечи, входы, фильтры, сигналы выхода, издержки."""

    def __init__(self, inst: str):
        from forexmodel.simulation.costs import profile_entry
        from trend_grid import gate_items

        d = pd.read_pickle(FEAT / f"{inst}.pkl")
        b, f, gates = d["bars"], d["f"], d["gates"]
        self.inst, self.n = inst, len(b)
        self.t = b["time"].to_numpy()
        self.o, self.h, self.l, self.c = (b[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        self.year = b["time"].dt.year.to_numpy()
        start = 2021 if inst == "eth" else 2016
        self.ok = (self.year >= start) & (self.year <= YEARS[-1])
        self.atr = {p: f[f"atr{p}"].astype(float) for p in (10, 14, 20, 22)}
        day = day_key(b["time"], inst)
        names = [name for _, name, _ in gate_items()]
        state = {n: gates[k].astype(np.int8) for k, n in enumerate(names)}
        c, h, l, a14 = self.c, self.h, self.l, self.atr[14]
        # входы: смена направления классических индикаторов
        self.entries = {n: events_from_state(np.where(state[n] == 99, 0, state[n])) for n in classic_names()}
        # входы по уровням, паттернам и RSI
        with np.errstate(invalid="ignore"):
            def brk(up, dn):
                return np.where(first_per_day(day, np.nan_to_num(up).astype(bool)), 1,
                                np.where(first_per_day(day, np.nan_to_num(dn).astype(bool)), -1, 0)).astype(np.int8)

            for name, flag in (("NR7 → пробой", f["nr7"]), ("NR4 → пробой", f["nr4"]), ("inside bar → пробой", f["ib"]),
                               ("NR7 + inside bar → пробой", f["nr7"] * f["ib"])):
                on = flag > 0
                self.entries[name] = brk(on & (c > f["hi1"]), on & (c < f["lo1"]))
            for n_, nm in ((1, "пробой вчерашнего дня"), (20, "пробой 20 дней"), (55, "пробой 55 дней")):
                self.entries[nm] = brk(c > f[f"hi{n_}"], c < f[f"lo{n_}"])
                if n_ > 1:
                    self.entries[nm + " + 0.25 ATR"] = brk(c > f[f"hi{n_}"] + 0.25 * a14, c < f[f"lo{n_}"] - 0.25 * a14)
            self.entries["ложный пробой 20 дней"] = brk((l < f["lo20"]) & (c > f["lo20"]), (h > f["hi20"]) & (c < f["hi20"]))
            fh_prev, fl_prev, c_prev = np.r_[np.nan, f["frac_hi"][:-1]], np.r_[np.nan, f["frac_lo"][:-1]], np.r_[np.nan, c[:-1]]
            self.entries["пробой фрактала"] = np.where((c > f["frac_hi"]) & (c_prev <= fh_prev), 1,
                                                        np.where((c < f["frac_lo"]) & (c_prev >= fl_prev), -1, 0)).astype(np.int8)
            self.entries["пробой Азии (металлы)"] = brk(c > f["asia_hi"], c < f["asia_lo"]) if inst in METALS \
                else np.zeros(self.n, np.int8)
            r14, r2 = f["rsi14"], f["rsi2"]
            r14p, r2p = np.r_[np.nan, r14[:-1]], np.r_[np.nan, r2[:-1]]
            self.entries["RSI14 30/70"] = np.where((r14 > 30) & (r14p <= 30), 1, np.where((r14 < 70) & (r14p >= 70), -1, 0)).astype(np.int8)
            self.entries["RSI2 10/90"] = np.where((r2 < 10) & (r2p >= 10), 1, np.where((r2 > 90) & (r2p <= 90), -1, 0)).astype(np.int8)
            # фильтры: разрешение лонга и шорта
            self.filters = {"нет": (np.ones(self.n, bool), np.ones(self.n, bool))}
            for nm, g in (("EMA200 1ч", "ema_price:200 @1h"), ("EMA200 4ч", "ema_price:200 @4h"),
                          ("EMA200 дни", "ema_price:200 @24h"), ("EMA50 дни", "ema_price:50 @24h")):
                self.filters[nm] = (state[g] == 1, state[g] == -1)
            for nm, col, thr in (("ADX 1ч > 20", "adx1h", 20), ("ADX 1ч > 25", "adx1h", 25), ("ADX 4ч > 20", "adx4h", 20),
                                 ("ADX 4ч > 25", "adx4h", 25)):
                m = f[col] > thr
                self.filters[nm] = (m, m)
            m = f["vr"] > 1
            self.filters["ATR14/ATR100 > 1"] = (m, m)
            m = f["atrpct"] > 0.5
            self.filters["перцентиль ATR > 0.5"] = (m, m)
            self.filters["только лонг"] = (np.ones(self.n, bool), np.zeros(self.n, bool))
            # сигналы выхода: −1 — выйти из лонга, +1 — выйти из шорта (известно на закрытии часа)
            self.xsig = {"Supertrend 1ч": np.where(state["supertrend:10:3 @1h"] == 99, 0, state["supertrend:10:3 @1h"]),
                         "Supertrend 4ч": np.where(state["supertrend:10:3 @4h"] == 99, 0, state["supertrend:10:3 @4h"])}
            for n_ in (10, 20):
                hi_n = pd.Series(h).rolling(n_).max().shift(1).to_numpy()
                lo_n = pd.Series(l).rolling(n_).min().shift(1).to_numpy()
                self.xsig[f"Дончиан {n_} ч"] = np.where(c < lo_n, -1, np.where(c > hi_n, 1, 0))
            dd = pd.DataFrame({"h": h, "l": l}).groupby(day).agg(h=("h", "max"), l=("l", "min"))
            for n_ in (10, 20):
                hi_d = dd["h"].rolling(n_).max().shift(1).reindex(day).to_numpy()
                lo_d = dd["l"].rolling(n_).min().shift(1).reindex(day).to_numpy()
                self.xsig[f"Дончиан {n_} дней"] = np.where(c < lo_d, -1, np.where(c > hi_d, 1, 0))
        self.xsig = {k: v.astype(np.int8) for k, v in self.xsig.items()}
        # издержки B01: за круг и свопы по переносам (моменты переноса и тройной — как в costs.rollover_count)
        e = profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst)
        self.spread, self.swap = e["spread_pct"], {1: e["swap_long_pct"], -1: e["swap_short_pct"]}
        hr, tw = int(e["rollover_hour"]), e["triple_weekday"]
        days = pd.date_range(b["time"].min().normalize() - pd.Timedelta(days=1), b["time"].max().normalize() + pd.Timedelta(days=12), freq="D")
        shift = pd.Timedelta(hours=hr) + (pd.Timedelta(0) if hr >= 12 else pd.Timedelta(days=1))
        w = np.where(days.dayofweek < 5, np.where(days.dayofweek == tw, 3, 1), 0) if tw is not None else np.ones(len(days))
        self.roll_t = (days + shift).to_numpy()
        self.roll_cum = np.r_[0, np.cumsum(w)]
        # медиана расстояния до стопа на выборе — нормировка размера «по риску»
        self.stop_med = {p: float(np.nanmedian((self.atr[p] / c)[self.year < SEL_END] if inst != "eth" else (self.atr[p] / c)))
                         for p in self.atr}

    def rollovers(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.roll_t, a, side="right")
        j = np.searchsorted(self.roll_t, b, side="right")
        return self.roll_cum[j] - self.roll_cum[i]


def entries_for(ins: Inst, prm: dict) -> tuple[np.ndarray, np.ndarray]:
    ev = ins.entries[prm["entry"]]
    long_ok, short_ok = np.ones(ins.n, bool), np.ones(ins.n, bool)
    for k in ("f1", "f2"):
        al, ash = ins.filters[prm[k]]
        long_ok &= al
        short_ok &= ash
    m = ins.ok & (((ev == 1) & long_ok) | ((ev == -1) & short_ok))
    m[-2:] = False
    i = np.flatnonzero(m)
    return i, ev[i].astype(int)


def first_true(m: np.ndarray) -> np.ndarray:
    return np.where(m.any(axis=1), m.argmax(axis=1), 10 ** 9)


def simulate(ins: Inst, prm: dict, i: np.ndarray, s: np.ndarray) -> dict:
    """Сделки всех входов (без учёта перекрытия), векторно по часовым свечам. Для шорта цены
    зеркалятся (−цена), и работает логика лонга."""
    ex = EXITS[prm["exit"]]
    p = ex.get("atr", prm["atr"])
    A = ins.atr[p][i]
    keep = np.isfinite(A) & (A > 0)
    i, s, A = i[keep], s[keep], A[keep]
    e = i + 1
    # горизонт — календарные часы, как в симуляторе: выход на минуте t_входа + H; k — бар, где она лежит
    # (внутри бара — выход по его открытию; в разрыве ночи / выходных — по закрытию последнего бара до разрыва)
    hz = ins.t[e] + np.timedelta64(1, "m") + np.timedelta64(int(ex["H"] * 60), "m")
    k = np.maximum(np.searchsorted(ins.t, hz, side="right") - 1, e)
    inside = hz < ins.t[k] + np.timedelta64(60, "m")
    last = k - e - inside.astype(int)               # последний бар, целиком входящий в горизонт
    keep2 = last >= 0
    i, s, A, e, hz, k, inside, last = (v[keep2] for v in (i, s, A, e, hz, k, inside, last))
    W = int(last.max()) + 2 if len(last) else 2
    idx = e[:, None] + np.arange(W)[None, :]
    valid = (idx < ins.n) & (np.arange(W)[None, :] <= last[:, None])
    idx = np.minimum(idx, ins.n - 1)
    sg = s[:, None].astype(float)
    Hh = np.where(sg > 0, ins.h[idx], -ins.l[idx])
    Ll = np.where(sg > 0, ins.l[idx], -ins.h[idx])
    O = sg * ins.o[idx]
    C = sg * ins.c[idx]
    entry = O[:, 0]
    stop0 = entry - ex["sl"] * A
    if "trail" in ex:
        prev = np.concatenate([entry[:, None], Hh[:, :-1]], axis=1)
        best = np.maximum.accumulate(prev, axis=1)
        on = (best - entry[:, None]) >= ex["act"] * A[:, None]
        stop = np.where(on, np.maximum(stop0[:, None], best - ex["trail"] * A[:, None]), stop0[:, None])
    else:
        stop = np.broadcast_to(stop0[:, None], Hh.shape)
    jS = first_true((Ll <= stop) & valid)
    jT = first_true((Hh >= (entry + ex["tp"] * A)[:, None]) & valid) if "tp" in ex else np.full(len(i), 10 ** 9)
    if "sig" in ex:
        if ex["sig"] == "обратный сигнал":
            X = ins.entries[prm["entry"]]
        else:
            X = ins.xsig[ex["sig"]]
        cond = (X[np.minimum(np.c_[i[:, None], idx], ins.n - 1)] * sg[:, :1] == -1) & np.c_[np.ones((len(i), 1), bool), valid]
        jX = first_true(cond)                      # 0 — сигнал на баре входа: выход по открытию бара входа
    else:
        jX = np.full(len(i), 10 ** 9)
    # время события: стоп / тейк — середина бара j; сигнал на баре k (k = 0 — бар сигнала) — открытие бара k
    tS, tT, tX, tL = jS + 0.5, jT + 0.5, jX.astype(float), last + 1.0
    # порядок: стоп, тейк, время, сигнал; при равенстве — раньше в списке (стоп раньше тейка в одном часе)
    kind = np.argmin(np.stack([tS, tT + 1e-3, tL, tX]), axis=0)
    rows = np.arange(len(i))
    jS_, jT_ = np.minimum(jS, W - 1), np.minimum(jT, W - 1)
    px_stop = np.minimum(O[rows, jS_], stop[rows, jS_])
    px_tp = np.maximum(O[rows, jT_], entry + ex.get("tp", 0) * A)
    jXc = np.minimum(np.maximum(jX, 0), W - 1)
    px_sig = np.where(jX == 0, entry, O[rows, jXc])
    px_last = np.where(inside, O[rows, np.minimum(last + 1, W - 1)], C[rows, last])
    exit_px = np.choose(kind, [px_stop, px_tp, px_last, px_sig])
    # бар выхода (следующий вход — только с бара позже) и время выхода для переносов
    xbar = np.choose(kind, [e + jS_, e + jT_, k, e + jXc])
    xbar = np.minimum(xbar, ins.n - 1)
    hour = np.timedelta64(60, "m")
    cl = lambda a: np.minimum(a, ins.n - 1)  # noqa: E731 — индексы за краем окна не выбираются (маска valid)
    t_exit = np.choose(kind, [ins.t[cl(e + jS_)] + np.timedelta64(59, "m"), ins.t[cl(e + jT_)] + np.timedelta64(59, "m"),
                              np.where(inside, hz, ins.t[cl(k)] + hour), ins.t[cl(e + jXc)]])
    t_entry = ins.t[e] + np.timedelta64(1, "m")
    gross = (exit_px - entry) / np.abs(entry) * 100
    nroll = ins.rollovers(t_entry, t_exit)
    swap = np.where(s > 0, ins.swap[1], ins.swap[-1]) * nroll
    size = np.ones(len(i))
    if prm["size"] == "по риску":
        size = np.clip(ins.stop_med[p] / (A / np.abs(entry)), 0.25, 4.0)
    net = size * (gross - ins.spread + swap)
    # время и цены сделки (для журнала сделок и графика; цены — без зеркала шорта), ATR на входе
    return {"i": i, "s": s, "x": xbar, "net": net, "gross": gross, "size": size, "kind": kind, "t_entry": t_entry,
            "t_exit": t_exit, "entry_px": np.abs(entry), "exit_px": np.abs(exit_px), "atr": A, "nroll": nroll}


def chain(i: np.ndarray, x: np.ndarray) -> np.ndarray:
    out, last = [], -1
    for k in range(len(i)):
        if i[k] > last:
            out.append(k)
            last = x[k]
    return np.asarray(out, dtype=int)


def trades(ins: Inst, prm: dict) -> dict:
    i, s = entries_for(ins, prm)
    if not len(i):
        return {"i": i, "s": s, "net": np.zeros(0), "year": np.zeros(0, int)}
    r = simulate(ins, prm, i, s)
    k = chain(r["i"], r["x"])
    out = {key: v[k] for key, v in r.items()}
    out["year"] = ins.year[out["i"]]
    return out


def evaluate(data: dict, prm: dict) -> dict:
    per = {inst: trades(data[inst], prm) for inst in INSTRUMENTS}
    res = {}
    for scope, members in SCOPES.items():
        agg = {}
        for inst in members:
            y, p = per[inst]["year"], per[inst]["net"] / len(members)
            for yy in np.unique(y):
                q = p[y == yy]
                a = agg.setdefault(str(int(yy)), [0.0, 0.0, 0.0, 0])
                a[0] += float(q.sum()); a[1] += float(q[q > 0].sum()); a[2] += float(-q[q < 0].sum()); a[3] += int(len(q))
        res[scope] = {k: [round(v[0], 3), round(v[1], 3), round(v[2], 3), v[3]] for k, v in agg.items()}
    return res


def period(agg: dict, sel: bool) -> dict:
    ys = [y for y in agg if (int(y) < SEL_END) == sel]
    s = sum(agg[y][0] for y in ys)
    pos, neg, n = (sum(agg[y][k] for y in ys) for k in (1, 2, 3))
    return {"итог": s, "PF": pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0), "сделок": n,
            "лет_в_плюсе": sum(1 for y in ys if agg[y][0] > 0), "лет": len(ys)}


def score(res: dict, scope: str, min_share: float) -> float:
    m = period(res[scope], True)
    return float(min(m["PF"], 10.0)) if m["сделок"] >= min_share * BASE_SEL_TRADES[scope] else 0.0


def load_data() -> dict:
    return {inst: Inst(inst) for inst in INSTRUMENTS}


# ---------------------------------------------------------------- поиск

def space() -> dict:
    names = classic_names() + LEVEL_ENTRIES
    return {"entry": names, "f1": FILTERS, "f2": FILTERS, "exit": list(EXITS), "atr": [10, 14, 20], "size": ["1", "по риску"]}


def sample(rng: random.Random, sp: dict) -> dict:
    prm = {k: rng.choice(v) for k, v in sp.items()}
    if rng.random() < 0.5:
        prm["f2"] = "нет"
    if rng.random() < 0.3:
        prm["f1"] = "нет"
    return prm


def mutate(rng: random.Random, sp: dict, prm: dict) -> dict:
    out = dict(prm)
    for k in rng.sample(list(sp), rng.choice((1, 1, 2))):
        out[k] = rng.choice(sp[k])
    return out


def key(prm: dict) -> str:
    return json.dumps(prm, sort_keys=True, ensure_ascii=False)


def db():
    import sqlite3

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS trials (key TEXT PRIMARY KEY, s_met REAL, s_p4 REAL, s_all REAL,
                   r_met REAL, r_p4 REAL, r_all REAL, params TEXT, res TEXT, how TEXT, t TEXT)""")
    for c in ("s_met", "s_p4", "s_all", "r_met", "r_p4", "r_all"):
        con.execute(f"CREATE INDEX IF NOT EXISTS i_{c} ON trials ({c})")
    return con


COL = {"металлы": "met", "портфель 4": "p4", "все": "all"}


def top(con, scope: str, by: str, limit: int) -> list[dict]:
    col = f"{by}_{COL[scope]}"
    rows = con.execute(f"SELECT params, res, s_{COL[scope]}, r_{COL[scope]} FROM trials WHERE {col} IS NOT NULL "
                       f"ORDER BY {col} DESC LIMIT {int(limit)}").fetchall()
    return [{"params": json.loads(p), "res": json.loads(r), "score": sc, "robust": rb} for p, r, sc, rb in rows]


def neighbours(rng: random.Random, sp: dict, prm: dict, k: int) -> list[dict]:
    out = []
    for name in rng.sample(list(sp), min(k, len(sp))):
        q = dict(prm)
        q[name] = rng.choice([v for v in sp[name] if v != prm[name]])
        out.append(q)
    return out


def worker(args) -> int:
    """Один процесс — одна цель (срез). Пробы общие: у каждой хранятся результаты и оценки по всем срезам.
    Претендентам в сотню лучших по своему срезу считается устойчивая оценка — медиана PF пробы и 6
    соседей (один блок другой); развитие и финалисты — по ней."""
    wid, hours, scope, min_share, seed = args
    logging.disable(logging.WARNING)
    rng = random.Random(seed * 1000 + wid + int(time.time()))
    data, sp, con = load_data(), space(), db()
    c = COL[scope]
    pending: list = []

    def flush():
        for _ in range(30):
            try:
                con.executemany("INSERT OR IGNORE INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?)", pending)
                for upd in robust_upd:
                    con.execute(f"UPDATE trials SET r_{upd[0]} = ? WHERE key = ?", (upd[1], upd[2]))
                con.commit()
                break
            except Exception:  # noqa: BLE001
                time.sleep(1 + rng.random())
        pending.clear()
        robust_upd.clear()

    robust_upd: list = []
    t_end, n = time.time() + hours * 3600, 0
    cut, best = 0.0, []
    while time.time() < t_end:
        if n % 100 == 0:
            flush()
            tops = top(con, scope, "s", 100)
            cut = tops[-1]["score"] if len(tops) >= 100 else 0.0
            best = top(con, scope, "r", 30)
        if best and rng.random() > max(0.3, 1 - n / 5000):
            prm, how = mutate(rng, sp, rng.choice(best)["params"]), "мутация"
        else:
            prm, how = sample(rng, sp), "разведка"
        k = key(prm)
        n += 1
        if con.execute("SELECT 1 FROM trials WHERE key=?", (k,)).fetchone() or any(p[0] == k for p in pending):
            continue
        res = evaluate(data, prm)
        sc = {sname: score(res, sname, min_share) for sname in SCOPES}
        rb = None
        if sc[scope] > cut:
            vals = [sc[scope]] + [score(evaluate(data, q), scope, min_share) for q in neighbours(rng, sp, prm, 6)]
            rb = float(np.median(vals))
        pending.append((k, sc["металлы"], sc["портфель 4"], sc["все"], *(rb if s == c else None for s in ("met", "p4", "all")),
                        json.dumps(prm, ensure_ascii=False), json.dumps(res), how, time.strftime("%m-%d %H:%M")))
    flush()
    con.close()
    return n


def search(hours: float, min_share: float, seed: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    db().close()
    scopes = list(SCOPES)
    with ProcessPoolExecutor(len(scopes)) as ex:
        counts = list(ex.map(worker, [(w, hours, sc, min_share, seed) for w, sc in enumerate(scopes)]))
    con = db()
    print(f"проб за запуск: {sum(counts)}, всего: {con.execute('SELECT COUNT(*) FROM trials').fetchone()[0]}")
    con.close()
    return 0


# ---------------------------------------------------------------- отчёт

def describe(prm: dict) -> str:
    f = " + ".join(x for x in (prm["f1"], prm["f2"]) if x != "нет") or "без фильтра"
    return (f"вход: {prm['entry']} · фильтр: {f} · выход: {prm['exit']} · ATR{prm['atr']}"
            + (" · размер по риску" if prm["size"] != "1" else ""))


B01 = {"металлы": {"выбор": (-11.49, 0.94), "проверка": (41.30, 1.17)},
       "портфель 4": {"выбор": (-12.29, 0.94), "проверка": (-2.32, 0.99)},
       "все": {"выбор": (-41.11, 0.79), "проверка": (-18.85, 0.95)}}
B01_YEARS = None


def report(min_share: float) -> int:
    from scipy.stats import spearmanr

    con = db()
    n_all = con.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    lines = ["# Перебор самостоятельных стратегий (H50)", "",
             f"Проб: {n_all}. Выбор — 2016–2020, проверка 2021–2026 в выборе не участвует. База B01 — импульс "
             "по гейту C с теми же издержками (из журнала). Цифры проб — часовой движок; финалисты пересчитываются "
             "штатным минутным симулятором (finalize) и попадают в журнал экспериментов.", ""]
    for scope in SCOPES:
        c = COL[scope]
        b = B01[scope]
        lines += [f"## Срез «{scope}»", "",
                  f"База B01: выбор {b['выбор'][0]:+.1f}% (PF {b['выбор'][1]:.2f}), проверка {b['проверка'][0]:+.1f}% "
                  f"(PF {b['проверка'][1]:.2f}).", ""]
        # связь выбор → проверка и walk-forward поиска — потоком по всем пробам
        sel_s, val_s, rng, n_ok = [], [], random.Random(0), 0
        wf = {y: (-1.0, None, 0.0) for y in range(2018, 2027)}
        for params, res_s in con.execute(f"SELECT params, res FROM trials WHERE s_{c} IS NOT NULL"):
            a = json.loads(res_s)[scope]
            arr = np.array([a.get(str(y), [0, 0, 0, 0]) for y in YEARS], dtype=float)
            cum = np.cumsum(arr, axis=0)
            sel, val = cum[YEARS.index(SEL_END) - 1], arr[YEARS.index(SEL_END):].sum(axis=0)
            if sel[3] >= min_share * BASE_SEL_TRADES[scope] and sel[2] > 0:
                n_ok += 1
                pair = (sel[1] / sel[2], val[1] / val[2] if val[2] > 0 else np.nan)
                if len(sel_s) < 50000:
                    sel_s.append(pair[0]); val_s.append(pair[1])
                elif rng.randrange(n_ok) < 50000:
                    r = rng.randrange(50000)
                    sel_s[r], val_s[r] = pair
            for y in wf:
                past = cum[YEARS.index(y) - 1]
                if past[2] > 0 and past[3] >= min_share * BASE_SEL_TRADES[scope] * (y - 2016) / 5:
                    pf = past[1] / past[2]
                    if pf > wf[y][0]:
                        wf[y] = (pf, params, float(arr[YEARS.index(y), 0]))
        rho = spearmanr(sel_s, val_s, nan_policy="omit").statistic if len(sel_s) > 10 else float("nan")
        lines += [f"Проб с достаточным числом сделок: {n_ok}; связь PF выбор → проверка: **{rho:.2f}**.", ""]
        for by, title in (("s", "по PF на выборе"), ("r", "по устойчивой оценке (медиана PF пробы и 6 соседей)")):
            rows, seen = [], set()
            for t in top(con, scope, by, 300):
                sig = json.dumps(t["res"][scope], sort_keys=True)
                if sig not in seen and len(rows) < 15:
                    seen.add(sig)
                    rows.append(t)
            lines += [f"### 15 лучших {title}", "",
                      "| # | выбор PF | устойч. | выбор итог | сделок (выбор) | проверка PF | проверка итог | лет в плюсе (пров.) | стратегия |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for k, t in enumerate(rows):
                sv, vv = period(t["res"][scope], True), period(t["res"][scope], False)
                rb = "—" if t["robust"] is None else f"{t['robust']:.2f}"
                lines.append(f"| {k + 1} | {sv['PF']:.2f} | {rb} | {sv['итог']:+.1f}% | {sv['сделок']} | {vv['PF']:.2f} | "
                             f"{vv['итог']:+.1f}% | {vv['лет_в_плюсе']}/{vv['лет']} | {describe(t['params'])} |")
            if rows:
                vals = [period(t["res"][scope], False)["итог"] for t in rows]
                lines += ["", f"Среднее на проверке: {np.mean(vals):+.1f}% (база {b['проверка'][0]:+.1f}%).", ""]
        lines += ["### Walk-forward самого поиска — честная оценка", "",
                  "На год Y берётся стратегия с лучшим PF за 2016…Y−1 и торгует год Y.", "",
                  "| год | поиск | стратегия |", "|---|---|---|"]
        tot = 0.0
        for y, (_, params, v) in wf.items():
            if y >= SEL_END:
                tot += v
            lines.append(f"| {y} | {v:+.1f}% | {describe(json.loads(params)) if params else '—'} |")
        lines += ["", f"**Сумма 2021–2026: поиск {tot:+.1f}% против базы B01 {b['проверка'][0]:+.1f}%.**", ""]
    con.close()
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------- штатный симулятор: проверка и финалисты

def minute_trades(inst: str, prm: dict, ins: Inst, minute: pd.DataFrame | None = None) -> pd.DataFrame:
    """Сделки стратегии штатным минутным симулятором (входы — те же события, выход — параметрами симулятора,
    выход по сигналу — close_on_trend_flip по колонке сигнала), издержки B01 по сделкам."""
    from forexmodel.simulation.costs import profile_entry, recost
    from forexmodel.simulation.simulator import simulate_trades
    from trend_grid import instrument_config

    cfg = instrument_config(inst)
    s = cfg.simulation
    ex = EXITS[prm["exit"]]
    p = ex.get("atr", prm["atr"])
    i, sd = entries_for(ins, prm)
    sig = pd.DataFrame({"time": ins.t, "final_class": np.nan, f"atr_{p}": ins.atr[p]})
    sig.loc[i, "final_class"] = np.where(sd > 0, 2, 0)
    s.atr_col, s.allow_overlapping_positions, s.use_trend_filter = f"atr_{p}", False, False
    s.horizon_minutes = int(ex["H"] * 60)
    s.sl_atr = ex["sl"]
    s.close_on_trend_flip = False
    if "tp" in ex:
        s.exit_mode, s.tp_atr = "atr", ex["tp"]
    elif "trail" in ex:
        s.exit_mode, s.trail_atr, s.activate_atr = "trailing", ex["trail"], ex["act"]
    else:
        s.exit_mode, s.trail_atr, s.activate_atr = "trailing", ex["sl"], 1e9      # только защитный стоп
    if "sig" in ex:
        X = ins.entries[prm["entry"]] if ex["sig"] == "обратный сигнал" else ins.xsig[ex["sig"]]
        sig["_exit"] = X.astype(float)
        s.close_on_trend_flip, s.trend_flip_mode, s.exit_trend_col = True, "opposite", "_exit"
    if minute is None:
        minute = load_minute(inst)
    t, _ = simulate_trades(sig[sig["time"] < pd.Timestamp("2027-01-01")], minute, cfg)
    if not len(t):
        return t
    if prm["size"] == "по риску":
        A = pd.Series(ins.atr[p], index=pd.to_datetime(ins.t)).reindex(pd.to_datetime(t["signal_dt"])).to_numpy()
        t["size"] = np.clip(ins.stop_med[p] / (A / t["open_price"].to_numpy()), 0.25, 4.0)
    return recost(t, profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst))


def load_minute(inst: str) -> pd.DataFrame:
    from forexmodel.data.dividends import adjust_for_dividends, load_dividends
    from forexmodel.data.loader import load_minute_compact
    from trend_grid import instrument_config

    cfg = instrument_config(inst)
    m = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache", cfg.data.volume_candidates)
    if cfg.data.dividends_path:
        m = adjust_for_dividends(m, load_dividends(ROOT / cfg.data.dividends_path, cfg.data.dividend_ticker))
    return m


def compare(prm: dict, data: dict, minute_cache: dict | None = None) -> pd.DataFrame:
    rows = []
    for inst in INSTRUMENTS:
        ins = data[inst]
        eng = trades(ins, prm)
        mt = minute_trades(inst, prm, ins, (minute_cache or {}).get(inst))
        for per, m_e, m_m in (("выбор", eng["year"] < SEL_END, pd.to_datetime(mt["open_dt"]).dt.year < SEL_END if len(mt) else []),
                              ("проверка", eng["year"] >= SEL_END, pd.to_datetime(mt["open_dt"]).dt.year >= SEL_END if len(mt) else [])):
            rows.append({"инструмент": inst, "период": per, "движок: сделок": int(m_e.sum()),
                         "движок: итог": float(eng["net"][m_e].sum()),
                         "минуты: сделок": int(np.sum(m_m)) if len(mt) else 0,
                         "минуты: итог": float(mt.loc[m_m, "profit_pct"].sum()) if len(mt) else 0.0})
    return pd.DataFrame(rows)


CHECKS = [
    {"entry": "donchian:55:20 @24h", "f1": "нет", "f2": "нет", "exit": "трейлинг 6", "atr": 14, "size": "1"},
    {"entry": "ema_cross:20:50 @4h", "f1": "EMA200 дни", "f2": "нет", "exit": "стоп 2 · тейк 3", "atr": 14, "size": "1"},
    {"entry": "RSI2 10/90", "f1": "нет", "f2": "нет", "exit": "по времени 10 ч", "atr": 10, "size": "по риску"},
    {"entry": "пробой 20 дней", "f1": "ADX 4ч > 20", "f2": "нет", "exit": "Supertrend 1ч · стоп 3", "atr": 20, "size": "1"},
    {"entry": "supertrend:10:3 @4h", "f1": "нет", "f2": "нет", "exit": "обратный сигнал · стоп 6", "atr": 14, "size": "1"},
    {"entry": "NR7 → пробой", "f1": "нет", "f2": "нет", "exit": "Chandelier 3·ATR22", "atr": 14, "size": "1"},
]


def check() -> int:
    """Часовой движок против штатного минутного симулятора на 6 разных стратегиях."""
    logging.disable(logging.WARNING)
    data = load_data()
    minutes = {}
    out = []
    for inst in INSTRUMENTS:
        minutes = {inst: load_minute(inst)}
        for k, prm in enumerate(CHECKS):
            ins = data[inst]
            eng = trades(ins, prm)
            mt = minute_trades(inst, prm, ins, minutes[inst])
            out.append({"стратегия": k + 1, "инструмент": inst, "движок: сделок": len(eng["net"]), "минуты: сделок": len(mt),
                        "движок: итог": round(float(eng["net"].sum()), 1),
                        "минуты: итог": round(float(mt["profit_pct"].sum()), 1) if len(mt) else 0.0})
        print(f"[{inst}] готово", flush=True)
    df = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    g = df.groupby("стратегия")[["движок: сделок", "минуты: сделок", "движок: итог", "минуты: итог"]].sum()
    g["стратегия_описание"] = [describe(p) for p in CHECKS]
    text = df.to_string(index=False) + "\n\nпо стратегиям (сумма по 8 рынкам):\n" + g.to_string()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def finalize(top_n: int, first_id: int, min_share: float) -> int:
    """Лучшие по устойчивой оценке по каждому срезу → штатный минутный симулятор → reports/walk_forward/
    <инстр>_ss_<id>_trades.csv и записи каталога журнала (S<first_id>…) с базой B01."""
    import yaml

    logging.disable(logging.WARNING)
    con = db()
    picks, seen = [], set()
    for scope in SCOPES:
        for t in top(con, scope, "r", 50):
            k = key(t["params"])
            if k not in seen and sum(1 for p in picks if p[0] == scope) < top_n:
                seen.add(k)
                picks.append((scope, t))
    con.close()
    data = load_data()
    cat_p, desc_p = ROOT / "docs" / "experiments" / "catalog.yaml", ROOT / "docs" / "experiments" / "descriptions.yaml"
    cat_txt, desc_txt = cat_p.read_text(encoding="utf-8"), desc_p.read_text(encoding="utf-8")
    ids = {e["id"] for e in yaml.safe_load(cat_txt)["experiments"]}
    anchor = "\n  # ---------------- модель CatBoost: серебро ----------------"
    fid = first_id
    lines = ["# Финалисты: часовой движок против штатного минутного симулятора", ""]
    minutes = {}
    run_trades = {}
    for inst in INSTRUMENTS:                       # минутки грузятся по одному инструменту
        minutes = {inst: load_minute(inst)}
        for k, (scope, t) in enumerate(picks):
            mt = minute_trades(inst, t["params"], data[inst], minutes[inst])
            run_trades.setdefault(k, {})[inst] = mt
        print(f"[{inst}] сделки финалистов посчитаны", flush=True)
    for k, (scope, t) in enumerate(picks):
        while f"S{fid}" in ids:
            fid += 1
        eid = f"S{fid}"
        ids.add(eid)
        fid += 1
        run = f"ss_{eid.lower()}"
        for inst, mt in run_trades[k].items():
            mt.to_csv(ROOT / "reports" / "walk_forward" / f"{inst}_{run}_trades.csv", index=False)
        prm = t["params"]
        ex = EXITS[prm["exit"]]
        sv, vv = period(t["res"][scope], True), period(t["res"][scope], False)
        exit_line = f"{{sl: {ex['sl']}, trail: {ex.get('trail', ex['sl'])}, act: {ex.get('act', 99)}, hours: {ex['H']}}}"
        src = "\n".join(f"      {i}: [{i}_{run}]" for i in INSTRUMENTS)
        entry = (f"  - id: {eid}\n    family: strategy\n    title: \"Новая стратегия №{k + 1} перебора ({scope})\"\n"
                 f"    change: \"{describe(prm)}\"\n    hypothesis: H50\n    exit: {exit_line}\n    baseline: B01\n"
                 f"    verdict: \"перебор: выбор PF {sv['PF']:.2f}, проверка PF {vv['PF']:.2f} (часовой движок)\"\n"
                 f"    doc: results/strategy_search.md\n    dividends: true\n    costs: {{profile: base}}\n    sources:\n{src}\n")
        cat_txt = cat_txt.replace(anchor, "\n" + entry.rstrip("\n") + anchor, 1)
        desc_txt += (f"{eid}:\n  what: \"Самостоятельная стратегия из блоков ТЗ 15.10, найденная перебором "
                     f"(scripts/strategy_search.py): {describe(prm)}. Сделки — штатный минутный симулятор.\"\n"
                     f"  why: \"Одна из лучших по устойчивой оценке (медиана PF пробы и соседей) среза «{scope}» на "
                     f"выборе 2016–2020.\"\n  outcome: \"Часовой движок: выбор PF {sv['PF']:.2f}, итог {sv['итог']:+.1f}%; "
                     f"проверка PF {vv['PF']:.2f}, итог {vv['итог']:+.1f}%. Цифры журнала — минутный симулятор.\"\n")
        eng_sel, eng_val = sv["итог"], vv["итог"]
        lines.append(f"* {eid} ({scope}): {describe(prm)} — движок: выбор {eng_sel:+.1f}%, проверка {eng_val:+.1f}%")
        print(f"{eid}: {describe(prm)}", flush=True)
    cat_p.write_text(cat_txt, encoding="utf-8")
    desc_p.write_text(desc_txt, encoding="utf-8")
    (OUT / "finalists.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("search")
    s.add_argument("--hours", type=float, default=12.0)
    s.add_argument("--min-share", type=float, default=0.3, help="минимум сделок на выборе — доля от B01")
    s.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("report")
    r.add_argument("--min-share", type=float, default=0.3)
    fz = sub.add_parser("finalize")
    fz.add_argument("--top", type=int, default=2)
    fz.add_argument("--first-id", type=int, default=36)
    fz.add_argument("--min-share", type=float, default=0.3)
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "check":
        return check()
    if a.cmd == "search":
        return search(a.hours, a.min_share, a.seed)
    if a.cmd == "report":
        return report(a.min_share)
    return finalize(a.top, a.first_id, a.min_share)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
