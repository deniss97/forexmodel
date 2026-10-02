"""Лаборатория тренд-фильтра: сравнение вариантов и подбор параметров (8 инструментов).

Вопросы 2026-10-03 (ноутбук notebooks/S_TrendFiter_dev (1).ipynb):
  * сброс тренда по EMA5 — тренд не может быть +1 при закрытии свечи ниже EMA и −1 при закрытии
    выше; как это меняет разметку и торговлю;
  * как подбирать параметры тренд-фильтра лучше, чем в разделе «Перебор данных для поиска лучшего
    фильтра» ноутбука.

`compare` — именованные варианты (текущий гейт C, сброс по EMA часа и 4ч, варианты ноутбука v2 /
v3, шок-слой) на 8 инструментах. Для каждого: метрики разметки (forexmodel/evaluation/
trend_quality.py) за 2015–2020 и 2021–2026 и торговля быстрым движком: вход по тренду на каждом
свободном часе и импульс 6 ч ≥ 3 ATR с подтверждением гейтом, выход 240 ч / 6 ATR и 10 ч / 2 ATR.

`tune` — подбор: случайный поиск по пространству параметров (оба режима фильтра, гистерезис,
шок-слой, сброс по EMA), затем доводка вокруг лучших. Цель — медиана по парам «инструмент × год»
периода выбора (2015–2020) результата следования метке после комиссии, в ATR на 1000 часов.
Медиана — чтобы не выиграл фильтр, удачный в одном году (2020). Выбор — «плато»: среди лучших
берётся тот, у кого и соседние наборы параметров дают хороший результат. Проверка — 2021–2026,
на ней ничего не выбирается.

    python scripts/trend_filter_lab.py compare
    python scripts/trend_filter_lab.py tune --n-random 300 --workers 3
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

from forexmodel.config import TrendConfig, load_config, validate_trend_layers
from forexmodel.evaluation.trend_quality import _atr, follow_pnl, trend_quality, zigzag_legs
from forexmodel.features.trend import add_trend_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "trend_lab"
INSTRUMENTS = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
SPLIT = 2021

# варианты для compare: поверх гейта C (configs/silver.yaml, trend_gate)
NB_V2 = dict(mode="confirm", ema_period=3, adx_period=14, adx_threshold=15, max_ffill_bars=5, slope_period=4,
             require_slope_agreement=True, update_every_bar=False, hold_bars=0)
NB_V3 = dict(mode="confirm", ema_period=3, adx_period=14, adx_threshold=15, slope_period=4,
             require_slope_agreement=False, hysteresis=True, update_every_bar=False, hold_bars=0)
VARIANTS = {
    "гейт C (сейчас)": {},
    "C + EMA5 часа": dict(ema_reset_period=5),
    "C + EMA5 часа, 2 свечи": dict(ema_reset_period=5, ema_reset_confirm_bars=2),
    "C + EMA10 часа": dict(ema_reset_period=10),
    "C + EMA5 4ч": dict(ema_reset_period=5, ema_reset_tf="htf"),
    "C + шок 2 ATR": dict(shock_atr=2.0),
    "ноутбук v2 (лучшие из перебора)": NB_V2,
    "ноутбук v3 (гистерезис)": NB_V3,
    "ноутбук v3 + EMA5 4ч": {**NB_V3, "ema_reset_period": 5, "ema_reset_tf": "htf"},
    "ноутбук v3 + EMA5 часа": {**NB_V3, "ema_reset_period": 5},
    "trend_4h модели (A)": dict(require_adx_rising=True, update_every_bar=False, hold_bars=0),
}
# наборы из подбора (tune): плато и №1 по периоду выбора — если подбор уже запускался
_TUNED = Path(__file__).resolve().parents[1] / "reports" / "trend_lab" / "tuned_sets.json"
if _TUNED.exists():
    _t = json.loads(_TUNED.read_text(encoding="utf-8"))
    VARIANTS["подбор: плато"] = _t["plateau"]
    VARIANTS["подбор: №1 по выбору"] = _t["top1"]


def base_config() -> TrendConfig:
    cfg = load_config(ROOT / "configs" / "silver.yaml")
    return cfg.trend_gate.resolve(cfg.trend)


def load_hourly(inst: str) -> pd.DataFrame:
    h = pd.read_pickle(ROOT / "reports" / "_cache" / f"hourly_{inst}.pkl")
    h["_atr"] = _atr(h)
    return h


def label(h: pd.DataFrame, tc: TrendConfig) -> np.ndarray:
    return add_trend_filter(h[["time", "open", "high", "low", "close"]], tc)["trend_4h"].to_numpy(dtype=float)


# ---------------------------------------------------------------- compare

def trading(inst: str, h: pd.DataFrame, labels: dict) -> list:
    """Вход по тренду на каждом свободном часе и импульс с подтверждением гейтом — быстрый движок."""
    from early_entry import simulate
    from forexmodel.data.loader import load_minute_compact
    from impulse_universe import hourly
    from pattern_lab import apply_exit, instrument_cfg

    cfg0 = instrument_cfg(inst)
    minute = load_minute_compact(ROOT / cfg0.data.csv_path, cfg0.data.time_col, ROOT / "reports" / "_cache")
    hh = hourly(minute, cfg0)                       # часы с ATR и t_close — как у импульса
    hh = hh.merge(h[["time"]].assign(_i=np.arange(len(h))), on="time", how="left")
    rows = []
    for name, t in labels.items():
        g = np.full(len(hh), np.nan)
        ok = hh["_i"].notna().to_numpy()
        g[ok] = t[hh.loc[ok, "_i"].astype(int).to_numpy()]
        g = np.nan_to_num(g)
        imp = np.where(hh["move_atr"] >= 3, 1, np.where(hh["move_atr"] <= -3, -1, 0))
        sigs = {
            "по тренду": np.where(g != 0, g, 0),
            "импульс + гейт": np.where((imp != 0) & (imp == g), imp, 0),
        }
        for sname, side in sigs.items():
            sel = side != 0
            sig = pd.DataFrame({"t_close": hh.loc[sel, "t_close"].to_numpy(), "side": side[sel],
                                "atr": hh.loc[sel, "atr"].to_numpy()})
            for ex in ("240:6:6:3", "10:2:1.5:1"):
                if sname == "импульс + гейт" and ex != "240:6:6:3":
                    continue
                cfg = apply_exit(instrument_cfg(inst), ex)
                tr = simulate(minute, sig, cfg, cfg.simulation.commission_pct)
                yr = pd.to_datetime(tr["time"]).dt.year
                for per, m in (("2015–20", yr < SPLIT), ("2021–26", yr >= SPLIT)):
                    x = tr.loc[m, "net"]
                    rows.append({"инструмент": inst, "вариант": name, "стратегия": sname, "выход": ex, "период": per,
                                 "сделок": len(x), "итог": x.sum(), "на_сделку": x.mean() if len(x) else np.nan,
                                 "t": x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan})
    del minute
    return rows


def compare(args) -> int:
    base = base_config()
    q_rows, t_rows = [], []
    for inst in INSTRUMENTS:
        h = load_hourly(inst)
        legs = zigzag_legs(h["close"].to_numpy(), h["_atr"].to_numpy(), args.k)
        yr = h["time"].dt.year.to_numpy()
        labels = {}
        for name, kw in VARIANTS.items():
            tc = dataclasses.replace(base, **kw)
            t = label(h, tc)
            labels[name] = t
            for per, m in (("2015–20", yr < SPLIT), ("2021–26", yr >= SPLIT)):
                q = trend_quality(h, t, k=args.k, legs=legs, mask=m)
                q_rows.append({"инструмент": inst, "вариант": name, "период": per, **q})
        labels["всегда лонг"] = np.ones(len(h))
        if not args.no_trading:
            t_rows += trading(inst, h, labels)
        print(f"[{inst}] готово", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    q = pd.DataFrame(q_rows)
    q.to_csv(OUT / "compare_quality.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["доля_тренда", "следование_до", "следование", "в_крупных_по_ходу", "в_крупных_против", "в_крупных_нейтраль",
            "захват_хода", "ход_против", "задержка_ч", "смен_на_100", "нарушений_EMA", "ошибка_против"]
    for per in ("2015–20", "2021–26"):
        print(f"\nРАЗМЕТКА, среднее по 8 инструментам, {per}")
        print(q[q["период"] == per].groupby("вариант", sort=False)[cols].mean().round(3).to_string())
        pos = q[q["период"] == per].assign(p=lambda d: d["следование"] > 0).groupby("вариант", sort=False)["p"].sum()
        print("инструментов с положительным следованием после комиссии:", pos.to_dict())
    if t_rows:
        t = pd.DataFrame(t_rows)
        t.to_csv(OUT / "compare_trading.csv", index=False)
        n = t["инструмент"].nunique()
        port = t.groupby(["стратегия", "выход", "вариант", "период"], sort=False).agg(
            итог=("итог", lambda x: x.sum() / n), сделок=("сделок", "sum"),
            в_плюсе=("итог", lambda x: int((x > 0).sum()))).reset_index()
        print("\nТОРГОВЛЯ, портфель 8 инструментов по 1/8, итог % после комиссии (в скобках — инструментов в плюсе)")
        port["v"] = port["итог"].round(1).astype(str) + " (" + port["в_плюсе"].astype(str) + ")"
        print(port.pivot_table(index="вариант", columns=["стратегия", "выход", "период"], values="v", aggfunc="first",
                               sort=False).to_string())
    return 0


# ---------------------------------------------------------------- tune

SPACE = {
    "mode": ["early", "confirm"],
    "timeframe": ["2h", "4h", "6h"],
    "update_every_bar": [True, False],
    "hold_bars": [0, 1, 2, 3, 4],
    "slope_period": [3, 4, 6, 8, 12],
    "adx_period": [5, 7, 9, 14, 20],
    "adx_min": [10, 15, 20, 25],
    "adx_max": [35, 40, 50, 100],
    "require_adx_rising": [False, True],
    "ema_period": [3, 5, 7, 10, 14, 21],
    "adx_threshold": [10, 15, 20, 25],
    "max_ffill_bars": [0, 1, 3, 5],
    "require_slope_agreement": [False, True],
    "hysteresis": [False, True],
    "shock_atr": [0.0, 0.0, 1.5, 2.0, 3.0],
    "shock_window": [2, 3, 6],
    "shock_mode": ["neutral", "flip"],
    "ema_reset_period": [0, 0, 3, 5, 8, 13],
    "ema_reset_tf": ["base", "htf"],
    "ema_reset_confirm_bars": [1, 2, 3],
}
_W = {}


def _init_worker(k: float, commission: float):
    """Данные и колена зигзага — один раз на процесс."""
    _W["data"] = {}
    for inst in INSTRUMENTS:
        h = load_hourly(inst)
        legs = zigzag_legs(h["close"].to_numpy(), h["_atr"].to_numpy(), k)
        _W["data"][inst] = (h, legs, h["time"].dt.year.to_numpy())
    _W["base"] = base_config()
    _W["k"], _W["commission"] = k, commission


def evaluate(params: dict) -> dict:
    """Следование метке после комиссии по каждой паре «инструмент × год» + сводные метрики."""
    try:
        return _evaluate(params)
    except Exception as e:  # noqa: BLE001 — один неудачный набор не должен ронять перебор
        return {"params": params, "error": f"{type(e).__name__}: {e}"}


def _evaluate(params: dict) -> dict:
    tc = dataclasses.replace(_W["base"], **params)
    try:
        validate_trend_layers(tc)
    except ValueError as e:
        return {"params": params, "error": str(e)}
    per_year, summ = [], {}
    for inst, (h, legs, yr) in _W["data"].items():
        t = label(h, tc)
        _, net = follow_pnl(h["close"].to_numpy(dtype=float), h["_atr"].to_numpy(), t, _W["commission"])
        ok = np.isfinite(net) & np.isfinite(t) & np.isfinite(legs["dir"])
        for y in np.unique(yr):
            m = ok & (yr == y)
            if m.sum() > 500:
                per_year.append((inst, int(y), float(net[m].sum() * 1000 / m.sum()), float(np.mean(t[m] != 0)),
                                 np.nan, np.nan))
        for per, m in (("sel", yr < SPLIT), ("val", yr >= SPLIT)):
            q = trend_quality(h, t, k=_W["k"], legs=legs, mask=m, commission_pct=_W["commission"])
            for key in ("следование", "в_крупных_по_ходу", "в_крупных_против", "доля_тренда", "смен_на_100",
                        "нарушений_EMA"):
                summ.setdefault(f"{per}_{key}", []).append(q.get(key, np.nan))
    py = pd.DataFrame(per_year, columns=["inst", "year", "f", "cov", "contra", "flips"])
    sel, val = py[py["year"] < SPLIT], py[py["year"] >= SPLIT]
    res = {"params": params,
           "sel_median": float(sel["f"].median()), "sel_mean": float(sel["f"].mean()),
           "sel_pos": float((sel["f"] > 0).mean()),
           "val_median": float(val["f"].median()), "val_mean": float(val["f"].mean()),
           "val_pos": float((val["f"] > 0).mean())}
    for key, v in summ.items():
        res[key] = float(np.nanmean(v))
    return res


def sample(rng: random.Random) -> dict:
    p = {k: rng.choice(v) for k, v in SPACE.items()}
    return _clean(p)


def _clean(p: dict) -> dict:
    """Убирает параметры, которые в выбранном режиме не действуют, — чтобы дубли не считались дважды."""
    p = dict(p)
    if p["mode"] == "early":
        for k in ("ema_period", "adx_threshold", "max_ffill_bars", "require_slope_agreement", "hysteresis"):
            p.pop(k, None)
    else:
        for k in ("adx_min", "adx_max", "require_adx_rising"):
            p.pop(k, None)
        if p.get("hysteresis"):
            p.pop("max_ffill_bars", None)
    if not p.get("shock_atr"):
        for k in ("shock_window", "shock_mode"):
            p.pop(k, None)
    if not p.get("ema_reset_period"):
        for k in ("ema_reset_tf", "ema_reset_confirm_bars"):
            p.pop(k, None)
    return p


def neighbours(p: dict, rng: random.Random, n: int) -> list:
    """Соседи: 1–2 параметра сдвинуты на соседнее значение сетки."""
    out = []
    keys = [k for k in p if k in SPACE and k != "mode" and len(SPACE[k]) > 1]
    for _ in range(n):
        q = dict(p)
        for k in rng.sample(keys, min(len(keys), rng.choice([1, 2]))):
            vals = sorted(set(SPACE[k]), key=lambda x: (str(type(x)), x))
            i = vals.index(q[k]) if q[k] in vals else 0
            j = min(len(vals) - 1, max(0, i + rng.choice([-1, 1])))
            q[k] = vals[j]
        out.append(_clean(q))
    return out


def tune(args) -> int:
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    named = {name: _clean({**{"mode": base_config().mode}, **kw}) if "mode" not in kw else kw
             for name, kw in VARIANTS.items() if not name.startswith("подбор")}
    cand = [sample(rng) for _ in range(args.n_random)]
    seen, uniq = set(), []
    for p in cand:
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.k, args.commission)) as ex:
        ref = dict(zip(named, ex.map(evaluate, list(named.values()))))
        print(f"эталоны готовы за {time.time() - t0:.0f} с", flush=True)
        res = []
        for i, r in enumerate(ex.map(evaluate, uniq, chunksize=4)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"  случайный поиск: {i + 1}/{len(uniq)} ({time.time() - t0:.0f} с)", flush=True)
        ok = [r for r in res if "error" not in r]
        ok.sort(key=lambda r: r["sel_median"], reverse=True)
        # доводка: соседи лучших
        refine = []
        for r in ok[: args.top]:
            refine += neighbours(r["params"], rng, args.n_neigh)
        refine = [p for p in refine if json.dumps(p, sort_keys=True) not in seen]
        for p in refine:
            seen.add(json.dumps(p, sort_keys=True))
        res2 = [r for r in ex.map(evaluate, refine, chunksize=4) if "error" not in r]
        print(f"доводка: {len(res2)} наборов ({time.time() - t0:.0f} с)", flush=True)
        allr = ok + res2
        allr.sort(key=lambda r: r["sel_median"], reverse=True)
        # плато: для лучших — средний результат соседей на периоде выбора
        top = allr[: args.top]
        plate = []
        for r in top:
            nb = list(ex.map(evaluate, neighbours(r["params"], rng, args.n_neigh)))
            nb = [x for x in nb if "error" not in x]
            plate.append(np.mean([x["sel_median"] for x in nb]) if nb else np.nan)
        for r, pv in zip(top, plate):
            r["sel_neigh_median"] = pv
    df = pd.DataFrame([{**{f"p_{k}": v for k, v in r["params"].items()}, **{k: v for k, v in r.items() if k != "params"}}
                       for r in allr])
    df.to_csv(OUT / "tune_results.csv", index=False)
    refdf = pd.DataFrame([{"вариант": k, **{kk: vv for kk, vv in v.items() if kk != "params"}} for k, v in ref.items()])
    refdf.to_csv(OUT / "tune_reference.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["sel_median", "sel_pos", "val_median", "val_pos", "sel_следование", "val_следование",
            "val_в_крупных_по_ходу", "val_в_крупных_против", "val_доля_тренда", "val_смен_на_100", "val_нарушений_EMA"]
    print("\nЭТАЛОНЫ (медиана по инструмент-годам, ATR на 1000 ч после комиссии; sel — 2015–20, val — 2021–26)")
    print(refdf.set_index("вариант")[show].round(2).to_string())
    print("\nЛУЧШИЕ ПО ПЕРИОДУ ВЫБОРА (2015–2020)")
    pcols = [c for c in df.columns if c.startswith("p_")]
    print(df.head(15)[pcols + show + (["sel_neigh_median"] if "sel_neigh_median" in df else [])].round(2).to_string())
    from scipy.stats import spearmanr
    rho = spearmanr(df["sel_median"], df["val_median"]).statistic
    print(f"\nранговая корреляция «выбор → проверка» по {len(df)} наборам: {rho:.2f}")
    best_i = int(np.nanargmax([r.get("sel_neigh_median", np.nan) for r in allr[: args.top]]))
    best = allr[best_i]
    print("\nВЫБРАН (плато среди лучших по выбору):", json.dumps(best["params"], ensure_ascii=False))
    print({k: round(v, 3) for k, v in best.items() if k != "params"})
    (OUT / "tune_best.json").write_text(json.dumps(best, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--k", type=float, default=6.0, help="разворот зигзага, ATR")
    c.add_argument("--no-trading", action="store_true")
    t = sub.add_parser("tune")
    t.add_argument("--k", type=float, default=6.0)
    t.add_argument("--commission", type=float, default=0.04)
    t.add_argument("--n-random", type=int, default=300)
    t.add_argument("--top", type=int, default=10)
    t.add_argument("--n-neigh", type=int, default=8)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    return compare(args) if args.cmd == "compare" else tune(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
