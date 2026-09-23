"""Лаборатория тренд-фильтров и правил входа на готовом прогоне (без переобучения).

Отвечает на вопросы из разбора сделок (docs/results/silver_1h_cb/trade_review.md):
почему фильтр стоит в нейтрали посреди явного тренда, помогает ли запрет «не
покупать после медвежьей свечи», отсев поздних входов, пауза после стопа.

Три таблицы:

  1. качество разметки тренда: какую долю «видимого глазом» тренда (ход > 3 ATR
     за ±12 часов, задним числом) фильтр ловит, как часто он показывает обратное
     направление и с какой задержкой включается;
  2. правила входа БЕЗ модели на обучающем периоде: вход по фильтру на каждом
     свободном баре — проверка идеи на годах истории, а не на 80 сделках;
  3. те же фильтры и правила на сигналах модели для test и sim — через штатный
     `simulate_trades`, число сделок и PnL совпадают с отчётом прогона.

Модель не переобучается: варианты фильтра подставляются только как гейт в
симуляции (`simulation.trend_col`), признаки модели остаются прежними.
Вариант «C каждый час, удержание 2 ч» встроен в пайплайн как секция `trend_gate`
(features/trend.py); здесь он повторён, чтобы сравнивать с вариантами, которых
в пайплайне нет.

    python scripts/entry_rules_lab.py --run silver_1h_cb
    python scripts/entry_rules_lab.py --run lkoh_2024_base_reg_cb --no-history
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (путь до пакета)
import numpy as np
import pandas as pd

from forexmodel.config import load_config
from forexmodel.data.loader import load_minute_csv, resample_ohlcv
from forexmodel.features import indicators as ind
from forexmodel.simulation.simulator import simulate_trades

ROOT = Path(__file__).resolve().parents[1]
LONG, SHORT = 2, 0


# ---------------------------------------------------------------- тренд-фильтры

def _htf(h: pd.DataFrame, offset_h: int, adx_p: int, slope_p: int) -> pd.DataFrame:
    x = (
        h.set_index("time")[["open", "high", "low", "close"]]
        .resample("4h", offset=f"{offset_h}h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    atr = ind.atr(x, adx_p, method="wilder").replace(0, np.nan)
    x["slope"] = ind.linreg_slope(x["close"], slope_p) / atr
    a = ind.adx(x, adx_p)
    x["adx"], x["adx_slope"] = a["adx"], a["adx"].diff()
    x["di"], x["spread"] = np.sign(a["dmp"] - a["dmn"]), a["dmp"] - a["dmn"]
    return x


def trend_variant(h: pd.DataFrame, kind: str, hourly: bool, cfg, hold_h: int = 0) -> np.ndarray:
    """Каузальный тренд на часовых барах.

    kind:
      A — как в пайплайне (mode: early): adx_min < ADX < adx_max и ADX растёт;
      C — без условия «ADX растёт»;
      D — только ADX > adx_min (без потолка и без роста);
      R — вместо «ADX растёт»: разрыв DI расширяется в сторону тренда. ADX при
          развороте сначала падает, а разрыв DI в новую сторону растёт сразу.
    Во всех вариантах направление = знак наклона регрессии, и он должен
    совпадать со знаком DI.

    hourly=True — те же 4h-бары, но построенные со сдвигом 0..3 часа: значение
    обновляется каждый час, а не раз в 4 часа (задержка до 1 часа вместо 4).
    Цена — мигание: соседние сдвиги иногда расходятся, и тренд включается на
    час-другой. hold_h > 0 сглаживает это: нейтраль не длиннее hold_h часов не
    сбрасывает направление (противоположный знак срабатывает сразу).
    """
    t = cfg.trend
    parts = []
    for off in (range(4) if hourly else [0]):
        x = _htf(h, off, t.adx_period, t.slope_period)
        d = np.sign(x["slope"])
        if kind == "A":
            ok = (x["adx"] > t.adx_min) & (x["adx"] < t.adx_max) & (x["adx_slope"] > 0)
        elif kind == "C":
            ok = (x["adx"] > t.adx_min) & (x["adx"] < t.adx_max)
        elif kind == "D":
            ok = x["adx"] > t.adx_min
        elif kind == "R":
            ok = (x["adx"] > t.adx_min) & ((d * x["spread"]).diff() > 0)
        else:
            raise ValueError(kind)
        tr = np.where(ok & (d == x["di"]), d, 0.0)
        parts.append(pd.DataFrame({"time": x["time"] + pd.Timedelta("4h"), "v": tr}))
    p = pd.concat(parts).sort_values("time").drop_duplicates("time", keep="last")
    out = pd.merge_asof(h[["time"]], p, on="time", direction="backward")["v"].fillna(0).to_numpy()
    return _hold(out, hold_h) if hold_h else out


def _hold(tr: np.ndarray, hold_h: int) -> np.ndarray:
    out, cur, gap = tr.copy(), 0.0, 0
    for i, v in enumerate(tr):
        if v != 0:
            cur, gap = v, 0
        elif cur != 0:
            gap += 1
            if gap <= hold_h:
                out[i] = cur
            else:
                cur = 0.0
    return out


VARIANTS = {
    "A (как сейчас)": ("A", False),
    "A каждый час": ("A", True),
    "C каждый час": ("C", True),
    "C каждый час, удержание 2 ч": ("C", True, 2),
    "D каждый час": ("D", True),
    "R каждый час": ("R", True),
}


# ------------------------------------------------- исходы сделки на каждом баре

def _trailing(h, l, c, entry, atr, sgn, sl, trail, act):
    """Векторная копия simulation.exits.trailing_exit (результат совпадает до знака)."""
    if sgn > 0:
        best = np.maximum.accumulate(np.concatenate([[entry], h[:-1]]))
        init = entry - sl * atr
        on = (best - entry) >= act * atr
        stop = np.where(on, np.maximum(init, best - trail * atr), init)
        hit = np.flatnonzero(l <= stop)
    else:
        best = np.minimum.accumulate(np.concatenate([[entry], l[:-1]]))
        init = entry + sl * atr
        on = (entry - best) >= act * atr
        stop = np.where(on, np.minimum(init, best + trail * atr), init)
        hit = np.flatnonzero(h >= stop)
    if hit.size:
        j = hit[0]
        return sgn * (stop[j] - entry) / entry * 100, j, not on[j]
    j = len(c) - 1
    return sgn * (c[j] - entry) / entry * 100, j, False


def forward_outcomes(minute: pd.DataFrame, cfg) -> pd.DataFrame:
    """Часовые бары + результат лонга/шорта, открытого по сигналу этого бара (до комиссии)."""
    sim = cfg.simulation
    h = resample_ohlcv(minute, cfg.data.base_timeframe)
    h["atr_14"] = ind.atr(h, cfg.features.atr_period)
    t = minute["time"].to_numpy()
    hi, lo, cl, op = (minute[c].to_numpy(dtype=float) for c in ("high", "low", "close", "open"))
    delay = sim.open_delay_minutes
    oi = np.searchsorted(t, (h["time"] + pd.Timedelta(minutes=delay)).to_numpy(), "left")
    li = np.searchsorted(t, (h["time"] + pd.Timedelta(minutes=delay + cfg.horizon_minutes)).to_numpy(), "right") - 1
    atr = h["atr_14"].to_numpy()
    res = {k: np.full(len(h), np.nan) for k in ("g_long", "g_short")}
    ex = {k: np.full(len(h), np.datetime64("NaT", "ns")) for k in ("exit_long", "exit_short")}
    stop = {k: np.zeros(len(h), bool) for k in ("stop_long", "stop_short")}
    for i in range(len(h)):
        a, b = oi[i], li[i]
        if a >= len(t) or b <= a or not np.isfinite(atr[i]):
            continue
        w = slice(a, b + 1)
        for sgn, k in ((1, "long"), (-1, "short")):
            g, j, st = _trailing(hi[w], lo[w], cl[w], op[a], atr[i], sgn, sim.sl_atr, sim.trail_atr, sim.activate_atr)
            res[f"g_{k}"][i], ex[f"exit_{k}"][i], stop[f"stop_{k}"][i] = g, t[a + j], st
    for d in (res, ex, stop):
        for k, v in d.items():
            h[k] = v
    return h


def sequential(h: pd.DataFrame, d: np.ndarray, allow: np.ndarray, commission: float, cooldown_h: int = 0) -> pd.DataFrame:
    """Сделки без перекрытия, как в симуляторе; cooldown_h — пауза после стопа в ту же сторону."""
    t = h["time"].to_numpy()
    last_exit, last_side, last_stop = np.datetime64("1970-01-01"), 0, False
    rows = []
    for i in np.flatnonzero((d != 0) & allow):
        side = d[i]
        k = "long" if side > 0 else "short"
        g = h[f"g_{k}"].iat[i]
        if t[i] <= last_exit or not np.isfinite(g):
            continue
        if cooldown_h and last_stop and side == last_side and (t[i] - last_exit) < np.timedelta64(cooldown_h, "h"):
            continue
        rows.append((t[i], g))
        last_exit, last_side, last_stop = h[f"exit_{k}"].iat[i], side, h[f"stop_{k}"].iat[i]
    r = pd.DataFrame(rows, columns=["time", "gross"])
    r["net"] = r["gross"] - commission
    return r


# ---------------------------------------------------------------- правила входа

def rule_masks(d: np.ndarray, bar_atr, r3_atr, ext_long, ext_short) -> dict[str, tuple[np.ndarray, int]]:
    """Имя правила -> (маска разрешённых баров, пауза после стопа в часах)."""
    ext = np.where(d > 0, ext_long, ext_short)
    ok = np.ones(len(d), bool)
    return {
        "базовый": (ok, 0),
        "запрет против свечи сигнала": (~(d * bar_atr < 0), 0),
        "нож: 3 бара против > 1 ATR": (~(d * r3_atr < -1), 0),
        "растяжение от экстремума 24ч ≤ 5 ATR": (~(ext > 5), 0),
        "растяжение ≤ 5 + нож": (~(ext > 5) & ~(d * r3_atr < -1), 0),
        "пауза 6 ч после стопа": (ok, 6),
    }


def _stats(net: pd.Series, gross: pd.Series) -> dict:
    if len(net) == 0:
        return {"сделок": 0}
    loss = -net[net < 0].sum()
    return {
        "сделок": len(net),
        "до_комиссии": round(gross.mean(), 3),
        "итог_%": round(net.sum(), 2),
        "PF": round(net[net > 0].sum() / loss, 2) if loss > 0 else np.inf,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="имя прогона: artifacts/<run>/config.yaml и reports/<run>/signals_*.csv")
    ap.add_argument("--signal-source", default="cb", help="источник сигнала, с которым строились signals_*.csv")
    ap.add_argument("--no-history", action="store_true", help="пропустить проверку без модели на обучающем периоде")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    cfg = load_config(ROOT / "artifacts" / args.run / "config.yaml")
    cfg.simulation.signal_source = args.signal_source
    reports = ROOT / "reports" / args.run
    cache = reports / "entry_rules_fwd.pkl"

    splits = cfg.data.splits if isinstance(cfg.data.splits, dict) else vars(cfg.data.splits)
    test_start, sim_start = pd.Timestamp(splits["test_start"]), pd.Timestamp(splits["sim_start"])

    minute = load_minute_csv(ROOT / cfg.data.csv_path, time_col=cfg.data.time_col, volume_candidates=cfg.data.volume_candidates)
    if cache.exists():
        h = pd.read_pickle(cache)
    else:
        print("Считаю исход сделки на каждом баре (один раз, ~1 мин)...")
        h = forward_outcomes(minute, cfg)
        h.to_pickle(cache)
    # дальше минуты нужны только симулятору на test/sim: вся история — это
    # миллионы строк, и держать их в памяти незачем
    minute = minute[minute["time"] >= test_start - pd.Timedelta(days=3)].reset_index(drop=True)
    period = np.where(h["time"] < test_start, "train", np.where(h["time"] < sim_start, "test", "sim"))
    trends = {name: trend_variant(h, spec[0], spec[1], cfg, *spec[2:]) for name, spec in VARIANTS.items()}

    atr = h["atr_14"]
    bar_atr = ((h["close"] - h["open"]) / atr).to_numpy()
    r3_atr = ((h["close"] - h["close"].shift(3)) / atr).to_numpy()
    ext_long = ((h["close"] - h["low"].rolling(24).min()) / atr).to_numpy()
    ext_short = ((h["high"].rolling(24).max() - h["close"]) / atr).to_numpy()

    # 1. качество разметки
    move = (h["close"].shift(-12) - h["close"].shift(12)) / atr
    truth = np.where(move > 3, 1, np.where(move < -3, -1, 0))
    rows = []
    for name, tr in trends.items():
        for p in ("train", "test", "sim"):
            m = period == p
            tm, tt = tr[m], truth[m]
            seen, lab = tt != 0, tm != 0
            lags, s = [], pd.Series(tt)
            for _, idx in s.groupby((s != s.shift()).cumsum()).groups.items():
                if s[idx[0]] == 0 or len(idx) < 6:
                    continue
                hit = np.flatnonzero(tm[idx] == s[idx[0]])
                lags.append(hit[0] if hit.size else len(idx))
            rows.append({
                "фильтр": name, "период": p,
                "покрытие": round(lab.mean(), 2),
                "ловит_видимый_тренд": round(((tm == tt) & seen).sum() / seen.sum(), 2),
                "показывает_обратное": round(((tm == -tt) & seen).sum() / seen.sum(), 2),
                "точность": round(((tm == tt) & lab & seen).sum() / max((lab & seen).sum(), 1), 2),
                "задержка_ч_медиана": float(np.median(lags)) if lags else np.nan,
            })
    q = pd.DataFrame(rows)
    print("\n1. КАЧЕСТВО РАЗМЕТКИ ТРЕНДА")
    print(q.pivot_table(index="фильтр", columns="период", values=["ловит_видимый_тренд", "показывает_обратное", "точность", "задержка_ч_медиана"], sort=False).to_string())
    q.to_csv(reports / "entry_rules_trend_quality.csv", index=False)

    # 2. без модели, обучающий период
    comm = cfg.simulation.commission_pct
    if not args.no_history:
        tr_mask = period == "train"
        ht = h[tr_mask].reset_index(drop=True)
        rows = []
        for fname, tr in trends.items():
            d = tr[tr_mask].astype(int)
            masks = rule_masks(d, bar_atr[tr_mask], r3_atr[tr_mask], ext_long[tr_mask], ext_short[tr_mask])
            base = None
            for rname, (allow, cd) in masks.items():
                r = sequential(ht, d, allow, comm, cd)
                yearly = r.groupby(r["time"].dt.year)["net"].mean()
                base = yearly if base is None else base
                rows.append({"фильтр": fname, "правило": rname, **_stats(r["net"], r["gross"]),
                             "лет_лучше_базового": int(((yearly - base).dropna() > 0).sum()), "лет": len(yearly)})
        hist = pd.DataFrame(rows)
        print("\n2. БЕЗ МОДЕЛИ, обучающий период: вход по фильтру на каждом свободном баре")
        print(hist.to_string(index=False))
        hist.to_csv(reports / "entry_rules_history.csv", index=False)

    # 3. сигналы модели, штатный симулятор
    rows = []
    hidx = pd.Index(h["time"])
    for split in ("test", "sim"):
        path = reports / f"signals_{split}.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path, parse_dates=["time"])
        idx = hidx.get_indexer(s["time"])
        px = minute
        sgn = np.where(s["final_class"] == LONG, 1, np.where(s["final_class"] == SHORT, -1, 0))
        variants = {"без тренд-фильтра": None, **trends}
        for fname, tr in variants.items():
            masks = rule_masks(sgn, bar_atr[idx], r3_atr[idx], s["ext_from_low_24"].to_numpy(), s["ext_from_high_24"].to_numpy())
            for rname, (allow, cd) in masks.items():
                if cd:  # пауза после стопа в штатном симуляторе не реализована
                    continue
                c = copy.deepcopy(cfg)
                c.simulation.use_trend_filter = tr is not None
                c.simulation.trend_col = "trend_gate"
                s2 = s.copy()
                s2["trend_gate"] = tr[idx] if tr is not None else 0.0
                s2["final_class"] = s["final_class"].where(allow)
                trades, _ = simulate_trades(s2, px, c)
                st = _stats(trades["profit_pct"], trades["gross_pct"]) if len(trades) else {"сделок": 0}
                rows.append({"выборка": split, "фильтр": fname, "правило": rname, **st})
    if rows:
        mt = pd.DataFrame(rows)
        print("\n3. СИГНАЛЫ МОДЕЛИ (штатный simulate_trades)")
        print(mt.pivot_table(index=["фильтр", "правило"], columns="выборка", values=["сделок", "до_комиссии", "итог_%", "PF"], sort=False).to_string())
        mt.to_csv(reports / "entry_rules_model.csv", index=False)
    print(f"\nТаблицы сохранены в {reports}/entry_rules_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
