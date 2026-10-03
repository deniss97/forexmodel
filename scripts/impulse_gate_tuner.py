"""Подбор тренд-гейта и параметров импульса по результату самой импульсной стратегии.

Почему отдельно от scripts/trend_filter_lab.py tune: там цель — качество метки тренда (следование
ей как позиции), и лучший по ней фильтр оказался худшим как вход для импульса
(docs/results/trend_filter_ema.md §4). Здесь цель — итог импульсной стратегии.

Как сделано быстро и точно:
  1. `events`: для каждого часа, где ход за 4/6/8/12 ч ≥ 2.5 ATR, заранее считается сделка в сторону
     хода (вход через минуту после закрытия часа, выход 240 ч / стоп и трейлинг 6 ATR / активация 3,
     минутные цены) — исход и время выхода. Исход сделки от гейта не зависит: гейт только решает,
     брать ли её.
  2. Для варианта (гейт, окно, порог) сделки отбираются по правилу «без перекрытия» — жадной цепочкой
     по заранее посчитанным временам выхода. Это ровно то, что делает симулятор, но за миллисекунды.
  3. Гейт считается тренд-фильтром проекта (forexmodel/features/trend.py) на часовых барах.

Проверка самой процедуры подбора — walk-forward: на каждый год Y параметры выбираются только по годам
< Y (все 8 инструментов, портфель поровну) и торгуют год Y. Сравнение — с фиксированными «гейт C +
6 ч / 3 ATR» (выбраны до всякого подбора) и с выбором один раз по 2015–2020.

    python scripts/impulse_gate_tuner.py events          # исходы сделок (минутные данные, ~5 мин)
    python scripts/impulse_gate_tuner.py search --n-random 300 --workers 3
"""

from __future__ import annotations

import argparse
import dataclasses
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

from forexmodel.features.trend import add_trend_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "trend_lab"
EVENTS = OUT / "impulse_events.pkl"
INSTRUMENTS = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
WINDOWS = (4, 6, 8, 12)
THRESH = (2.5, 3.0, 3.5, 4.0)
KX = [(k, x) for k in WINDOWS for x in THRESH]
YEARS = list(range(2015, 2027))
EXIT = "240:6:6:3"


# ---------------------------------------------------------------- 1. исходы сделок

def build_events() -> int:
    from entry_rules_lab import _trailing
    from forexmodel.data.loader import load_minute_compact
    from impulse_universe import hourly
    from pattern_lab import apply_exit, instrument_cfg

    data = {}
    for inst in INSTRUMENTS:
        cfg = apply_exit(instrument_cfg(inst), EXIT)
        sim = cfg.simulation
        minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
        h = hourly(minute, cfg)
        mv = np.column_stack([((h["close"] - h["close"].shift(k)) / h["atr"]).to_numpy() for k in WINDOWS])
        cand = np.flatnonzero(np.nanmax(np.abs(np.nan_to_num(mv)), axis=1) >= min(THRESH))
        t = minute["time"].to_numpy()
        hi, lo, cl, op = (minute[c].to_numpy(dtype=float) for c in ("high", "low", "close", "open"))
        tc = h["t_close"].to_numpy()
        atr = h["atr"].to_numpy()
        res = {s: (np.full(len(cand), np.nan), np.full(len(cand), np.datetime64("NaT"), dtype="datetime64[ns]"))
               for s in (1, -1)}
        for n, r in enumerate(cand):
            sides = {int(np.sign(mv[r, j])) for j in range(len(WINDOWS)) if abs(mv[r, j]) >= min(THRESH)}
            et = tc[r] + np.timedelta64(1, "m")
            a = np.searchsorted(t, et, "left")
            b = np.searchsorted(t, et + np.timedelta64(cfg.horizon_minutes, "m"), "right") - 1
            if a >= len(t) or b <= a or not np.isfinite(atr[r]):
                continue
            for s in sides:
                g, j, _ = _trailing(hi[a:b + 1], lo[a:b + 1], cl[a:b + 1], op[a], atr[r], s, sim.sl_atr,
                                    sim.trail_atr, sim.activate_atr)
                res[s][0][n], res[s][1][n] = g, t[a + j]
        data[inst] = {
            "bars": h[["time", "open", "high", "low", "close"]].reset_index(drop=True),
            "rows": cand, "t_entry": tc[cand] + np.timedelta64(1, "m"), "mv": mv[cand],
            "g_long": res[1][0], "x_long": res[1][1], "g_short": res[-1][0], "x_short": res[-1][1],
            "year": pd.DatetimeIndex(tc[cand]).year.to_numpy(), "commission": sim.commission_pct,
        }
        print(f"[{inst}] кандидатов {len(cand)}", flush=True)
        del minute
    OUT.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, EVENTS)
    return 0


# ---------------------------------------------------------------- 2. отбор сделок

def _next_free(t_entry: np.ndarray, t_exit: np.ndarray) -> np.ndarray:
    """Первый кандидат, вход которого позже выхода этой сделки (позиции не перекрываются)."""
    x = np.where(np.isnat(t_exit), t_entry, t_exit)
    return np.searchsorted(t_entry, x, side="right")


def chain(elig: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Жадная цепочка без перекрытия: индексы взятых сделок."""
    n = len(elig)
    idx = np.where(elig, np.arange(n), n)
    ne = np.r_[np.minimum.accumulate(idx[::-1])[::-1], n]
    out, i = [], ne[0]
    while i < n:
        out.append(i)
        i = ne[nxt[i]]
    return np.asarray(out, dtype=int)


def score_inst(ev: dict, gate: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Суммы нетто по годам и числа сделок для всех (окно, порог): массивы [2, len(KX), len(YEARS)].

    [0] — сделки без перекрытия (как в торговле); [1] — плотная оценка: все подходящие импульсы,
    включая перекрывающиеся (в 20–40 раз больше наблюдений, меньше шума — для цели подбора).
    """
    sums = np.zeros((2, len(KX), len(YEARS)))
    cnts = np.zeros((2, len(KX), len(YEARS)))
    g_at = None if gate is None else np.nan_to_num(gate[ev["rows"]])
    nxt_l, nxt_s = ev["nxt_long"], ev["nxt_short"]
    yi = ev["year"] - YEARS[0]
    for q, (k, x) in enumerate(KX):
        m = ev["mv"][:, WINDOWS.index(k)]
        side = np.sign(np.nan_to_num(m))
        elig = np.abs(np.nan_to_num(m)) >= x
        if g_at is not None:
            elig &= side == g_at
        gross = np.where(side > 0, ev["g_long"], ev["g_short"])
        elig &= np.isfinite(gross)
        taken = chain(elig, np.where(side > 0, nxt_l, nxt_s))
        if taken.size:
            net = gross[taken] - ev["commission"]
            np.add.at(sums[0, q], yi[taken], net)
            np.add.at(cnts[0, q], yi[taken], 1)
        dense = np.flatnonzero(elig)
        if dense.size:
            np.add.at(sums[1, q], yi[dense], gross[dense] - ev["commission"])
            np.add.at(cnts[1, q], yi[dense], 1)
    return sums, cnts


# ---------------------------------------------------------------- 3. перебор гейтов

_W: dict = {}


def _init():
    from trend_filter_lab import base_config

    data = pd.read_pickle(EVENTS)
    for ev in data.values():
        ev["nxt_long"] = _next_free(ev["t_entry"], ev["x_long"])
        ev["nxt_short"] = _next_free(ev["t_entry"], ev["x_short"])
    _W["data"], _W["base"] = data, base_config()


def evaluate(item):
    name, params = item
    try:
        out_s, out_c = [], []
        for inst in INSTRUMENTS:
            ev = _W["data"][inst]
            if params is None:
                gate = None
            else:
                tc = dataclasses.replace(_W["base"], **params)
                gate = add_trend_filter(ev["bars"], tc)["trend_4h"].to_numpy(dtype=float)
            s, c = score_inst(ev, gate)
            out_s.append(s)
            out_c.append(c)
        return name, params, np.stack(out_s, axis=2), np.stack(out_c, axis=2)   # [вид, KX, inst, year]
    except Exception as e:  # noqa: BLE001
        return name, params, None, str(e)


def search(args) -> int:
    from trend_filter_lab import NB_V3, SPACE, _clean

    rng = random.Random(args.seed)
    tuned = json.loads((OUT / "tuned_sets.json").read_text(encoding="utf-8"))
    items = [("без гейта", None), ("гейт C", {}), ("ноутбук v3", NB_V3), ("плато (по метке)", tuned["plateau"]),
             ("C + сброс EMA5 4ч", dict(ema_reset_period=5, ema_reset_tf="htf"))]
    seen = set()
    while len(items) < args.n_random + 5:
        p = _clean({k: rng.choice(v) for k, v in SPACE.items()})
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            items.append((f"r{len(items)}", p))
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for i, r in enumerate(ex.map(evaluate, items, chunksize=2)):
            if r[2] is not None:
                res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f} с)", flush=True)
    names = [r[0] for r in res]
    params = [r[1] for r in res]
    S = np.stack([r[2] for r in res])          # [config, KX, inst, year]
    C = np.stack([r[3] for r in res])
    np.savez_compressed(OUT / "impulse_gate_search.npz", S=S, C=C)
    (OUT / "impulse_gate_search.json").write_text(json.dumps({"names": names, "params": params, "KX": KX,
                                                              "years": YEARS, "instruments": INSTRUMENTS},
                                                             ensure_ascii=False, default=str), encoding="utf-8")
    analyse(names, params, S, C)
    return 0


# ---------------------------------------------------------------- 4. анализ

def analyse(names, params, S4, C4, slip_values=(0.0, 0.1)) -> None:
    pd.set_option("display.width", 250)
    S, C = S4[:, 0], C4[:, 0]                         # торговля: сделки без перекрытия
    SD, CD = S4[:, 1], C4[:, 1]                       # плотная оценка: все подходящие импульсы
    n_inst = S.shape[2]
    base_i, base_q = names.index("гейт C"), KX.index((6, 3.0))
    yrs = np.array(YEARS)
    lines = []

    def port(s, c, slip):          # портфель поровну: [config, KX, year]
        return (s - slip * c).sum(axis=2) / n_inst

    for slip in slip_values:
        P = port(S, C, slip)                           # [cfg, kx, year]
        sel, val = yrs < 2021, yrs >= 2021
        tot_sel, tot_val = P[:, :, sel].sum(-1), P[:, :, val].sum(-1)
        from scipy.stats import spearmanr

        rho = spearmanr(tot_sel.ravel(), tot_val.ravel()).statistic
        lines.append(f"\n=== проскальзывание {slip}% ===")
        lines.append(f"ранговая корреляция выбор (2015–20) → проверка (2021–26) по {tot_sel.size} вариантам: {rho:.2f}")
        ci, qi = np.unravel_index(np.argmax(tot_sel), tot_sel.shape)
        lines.append(f"гейт C + 6 ч / 3 ATR (фиксированный): 2015–20 {tot_sel[base_i, base_q]:+.1f}%, 2021–26 {tot_val[base_i, base_q]:+.1f}%")
        lines.append(f"лучший по 2015–20: {names[ci]} {params[ci]} окно {KX[qi][0]} ч / {KX[qi][1]} ATR: "
                     f"2015–20 {tot_sel[ci, qi]:+.1f}%, 2021–26 {tot_val[ci, qi]:+.1f}%")
        for nm in ("без гейта", "ноутбук v3", "плато (по метке)", "C + сброс EMA5 4ч"):
            i = names.index(nm)
            lines.append(f"{nm} + 6 ч / 3 ATR: 2015–20 {tot_sel[i, base_q]:+.1f}%, 2021–26 {tot_val[i, base_q]:+.1f}%")
        # гейт C с другими окнами/порогами
        g = pd.DataFrame({"окно/порог": [f"{k} ч / {x} ATR" for k, x in KX], "2015–20": tot_sel[base_i],
                          "2021–26": tot_val[base_i]})
        lines.append("гейт C по окнам и порогам импульса:\n" + g.round(1).to_string(index=False))
        # плотная цель: средний результат подходящего импульса (на импульс), портфель — среднее по инструментам
        PD = ((SD - slip * CD).sum(axis=2) / n_inst)                       # [cfg, kx, year] — сумма по импульсам
        ND = CD.sum(axis=2) / n_inst
        dsel = PD[:, :, sel].sum(-1) / np.maximum(ND[:, :, sel].sum(-1), 1)
        dval = PD[:, :, val].sum(-1) / np.maximum(ND[:, :, val].sum(-1), 1)
        rho_d = spearmanr(dsel.ravel(), dval.ravel()).statistic
        rho_dt = spearmanr(dsel.ravel(), tot_val.ravel()).statistic
        lines.append(f"плотная цель (средний импульс): корреляция выбор → проверка {rho_d:.2f}; "
                     f"плотная цель 2015–20 → итог торговли 2021–26 {rho_dt:.2f}")
        # walk-forward процедуры подбора: на год Y — лучший по годам < Y
        rows = []
        for y in range(2017, 2027):
            past, now = yrs < y, yrs == y
            sc = P[:, :, past].sum(-1)
            ci, qi = np.unravel_index(np.argmax(sc), sc.shape)
            dense = PD[:, :, past].sum(-1) / np.maximum(ND[:, :, past].sum(-1), 1)
            dense = np.where(P[:, :, past].sum(-1) > 0, dense, -np.inf)      # только варианты, прибыльные в торговле
            di, dq = np.unravel_index(np.argmax(dense), dense.shape)
            order = np.argsort(sc.ravel())[::-1]
            topn = {n: np.mean([P[np.unravel_index(o, sc.shape)][now].sum() for o in order[:n]]) for n in (5, 20, 100)}
            # устойчивый выбор: лучший по медиане годовых итогов прошлого
            med = np.median(P[:, :, past], axis=-1)
            cj, qj = np.unravel_index(np.argmax(med), med.shape)
            rows.append({"год": y, "фикс. гейт C 6/3": P[base_i, base_q, now].sum(),
                         "подбор по сумме прошлого": P[ci, qi, now].sum(),
                         "выбрано": f"{names[ci]} {KX[qi][0]}ч/{KX[qi][1]}",
                         "подбор по медиане лет": P[cj, qj, now].sum(),
                         "выбрано (медиана)": f"{names[cj]} {KX[qj][0]}ч/{KX[qj][1]}",
                         "гейт C, лучший 6/3-подобный по прошлому": P[base_i, np.argmax(P[base_i, :, past].sum(-1)), now].sum(),
                         "плотная цель": P[di, dq, now].sum(), "выбрано (плотная)": f"{names[di]} {KX[dq][0]}ч/{KX[dq][1]}",
                         "среднее топ-5": topn[5], "среднее топ-20": topn[20], "среднее топ-100": topn[100],
                         "среднее всех вариантов": P[:, :, now].sum(-1).mean()})
        wf = pd.DataFrame(rows)
        tot = wf[["фикс. гейт C 6/3", "подбор по сумме прошлого", "подбор по медиане лет",
                  "гейт C, лучший 6/3-подобный по прошлому", "плотная цель", "среднее топ-5", "среднее топ-20",
                  "среднее топ-100", "среднее всех вариантов"]].sum()
        lines.append("walk-forward процедуры подбора (параметры года Y — только по годам < Y):\n"
                     + wf.round(1).to_string(index=False))
        lines.append("сумма 2017–2026: " + "; ".join(f"{k} {v:+.1f}%" for k, v in tot.items()))
    text = "\n".join(lines)
    print(text)
    (OUT / "impulse_gate_search.txt").write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("events")
    s = sub.add_parser("search")
    s.add_argument("--n-random", type=int, default=300)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=11)
    sub.add_parser("analyse")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    if args.cmd == "events":
        return build_events()
    if args.cmd == "analyse":
        meta = json.loads((OUT / "impulse_gate_search.json").read_text(encoding="utf-8"))
        z = np.load(OUT / "impulse_gate_search.npz")
        analyse(meta["names"], meta["params"], z["S"], z["C"])
        return 0
    return search(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
