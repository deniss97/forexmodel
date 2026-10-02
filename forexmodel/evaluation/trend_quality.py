"""Качество разметки тренд-фильтра: насколько метка совпадает с «видимым трендом» и что она даёт.

Используется для сравнения и подбора тренд-фильтров (scripts/trend_filter_lab.py) и годится для
ноутбука: на вход — часовые бары (time, high, low, close) и метка тренда −1 / 0 / +1 на каждом
баре, известная на его закрытии.

Эталон «видимого тренда» — зигзаг задним числом: колено заканчивается, когда цена откатила от
экстремума на k ATR (ATR(14) рабочего ТФ на экстремуме). Это постфактум-разметка только для
оценки: в сам фильтр будущее не попадает. Метка бара i оценивается по ходу i → i+1 и колену,
которому этот ход принадлежит.

Метрики:
  * доля баров крупных колен (≥ `big` ATR), где метка по ходу / против хода / нейтраль, и
    задержка включения — сколько часов от начала крупного колена до первой верной метки;
  * доля хода крупных колен, пройденного с верной меткой (capture) и с меткой против (contra_capture);
  * результат следования метке: Σ метка_i · (close_{i+1} − close_i) / ATR_i, в ATR на 1000 баров,
    до и после комиссии за каждую смену метки (`commission_pct` за круг: 0 → ±1 → 0 — один круг,
    разворот ±1 → ∓1 — два). Это непрерывный вариант двух ошибок из ноутбука (против хода и
    ложная нейтраль) без порогов в процентах, которые зависят от волатильности периода;
  * смены метки на 100 баров и короткие эпизоды (мигание);
  * нарушения правила EMA: метка +1 при закрытии свечи ниже EMA или −1 при закрытии выше;
  * ошибки из ноутбука в ATR: «против» — направленный бар, после которого за `h` баров цена ушла
    против метки на ≥ `move_atr` ATR; «ложная нейтраль» — нейтральный бар, после которого ушла
    на ≥ `move_atr` ATR в любую сторону.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["zigzag_legs", "trend_quality", "follow_pnl", "score_trend_filter", "search_trend_filter"]


def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """ATR(n) рабочего ТФ; нулевой (часы без движения у неликвидных акций) — NaN."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    a = pd.Series(tr).rolling(n).mean().to_numpy()
    return np.where(a > 0, a, np.nan)


def zigzag_legs(close: np.ndarray, atr: np.ndarray, k: float = 6.0) -> Dict[str, np.ndarray]:
    """Колена зигзага по закрытиям. Для каждого хода i → i+1: направление его колена, размер колена
    в ATR (на начале колена), номер колена и индекс начала колена. Последний бар — NaN."""
    n = len(close)
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    piv = []
    d, hi_i, lo_i, ext = 0, 0, 0, 0
    first = int(np.argmax(np.isfinite(atr)))
    hi_i = lo_i = first
    for i in range(first + 1, n):
        c = close[i]
        if d == 0:
            if c > close[hi_i]:
                hi_i = i
            if c < close[lo_i]:
                lo_i = i
            thr = k * np.nanmean([atr[hi_i], atr[lo_i]])
            if close[hi_i] - close[lo_i] >= thr:
                if hi_i > lo_i:
                    piv, d, ext = [lo_i], 1, hi_i
                else:
                    piv, d, ext = [hi_i], -1, lo_i
        elif d == 1:
            if c > close[ext]:
                ext = i
            elif close[ext] - c >= k * atr[ext]:
                piv.append(ext)
                d, ext = -1, i
        else:
            if c < close[ext]:
                ext = i
            elif c - close[ext] >= k * atr[ext]:
                piv.append(ext)
                d, ext = 1, i
    if piv:
        piv.append(ext)
    leg_dir = np.full(n, np.nan)
    leg_size = np.full(n, np.nan)
    leg_id = np.full(n, -1)
    leg_start = np.full(n, -1)
    for j in range(len(piv) - 1):
        a, b = piv[j], piv[j + 1]
        if b <= a:
            continue
        move = close[b] - close[a]
        leg_dir[a:b] = np.sign(move)
        leg_size[a:b] = abs(move) / atr[a] if np.isfinite(atr[a]) else np.nan
        leg_id[a:b] = j
        leg_start[a:b] = a
    return {"dir": leg_dir, "size_atr": leg_size, "id": leg_id, "start": leg_start}


def follow_pnl(close: np.ndarray, atr: np.ndarray, trend: np.ndarray, commission_pct: float = 0.04):
    """Результат следования метке по барам, в ATR: (до комиссии, после комиссии). Метка бара i
    действует на ход i → i+1; комиссия — за изменение метки на баре i (в долях круга)."""
    t = np.nan_to_num(trend.astype(float))
    atr = np.where(atr > 0, atr, np.nan)
    move = np.r_[np.diff(close), np.nan] / atr
    gross = t * move
    change = np.abs(np.diff(np.r_[0.0, t]))                      # 1 — вход или выход, 2 — разворот
    cost = change * (commission_pct / 100.0 / 2.0) * close / atr  # половина круга за единицу изменения
    return gross, gross - cost


def trend_quality(df: pd.DataFrame, trend, *, k: float = 6.0, big: float = 12.0, commission_pct: float = 0.04,
                  ema_period: int = 5, h: int = 5, move_atr: float = 1.0,
                  legs: Optional[Dict[str, np.ndarray]] = None, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Метрики разметки тренда (см. описание модуля). `mask` — какие бары учитывать (например, годы);
    колена зигзага считаются по всему ряду, чтобы обрезка периода не рвала их."""
    close = df["close"].to_numpy(dtype=float)
    atr = df["_atr"].to_numpy() if "_atr" in df.columns else _atr(df)
    atr = np.where(atr > 0, atr, np.nan)
    t = np.asarray(trend, dtype=float)
    legs = legs if legs is not None else zigzag_legs(close, atr, k)
    n = len(close)
    m = np.isfinite(t) & np.isfinite(atr) & np.isfinite(legs["dir"])
    m[-1] = False
    if mask is not None:
        m &= mask
    if not m.any():
        return {}
    d, size = legs["dir"], legs["size_atr"]
    gross, net = follow_pnl(close, atr, t, commission_pct)
    move = np.r_[np.diff(close), np.nan] / atr
    ideal = d * move

    out: Dict[str, float] = {"баров": float(m.sum()), "доля_тренда": float(np.mean(t[m] != 0))}
    k1000 = 1000.0 / m.sum()
    out["следование_до"] = float(np.nansum(gross[m]) * k1000)
    out["следование"] = float(np.nansum(net[m]) * k1000)
    out["идеал"] = float(np.nansum(ideal[m]) * k1000)
    out["эффективность"] = out["следование"] / out["идеал"] if out["идеал"] > 0 else np.nan

    bigm = m & (size >= big)
    if bigm.any():
        out["в_крупных_по_ходу"] = float(np.mean(t[bigm] == d[bigm]))
        out["в_крупных_против"] = float(np.mean(t[bigm] == -d[bigm]))
        out["в_крупных_нейтраль"] = float(np.mean(t[bigm] == 0))
        prog = d[bigm] * move[bigm]
        out["захват_хода"] = float(np.nansum(prog[t[bigm] == d[bigm]]) / np.nansum(prog))
        out["ход_против"] = float(np.nansum(prog[t[bigm] == -d[bigm]]) / np.nansum(prog))
        # задержка: часы от начала крупного колена до первой верной метки
        lags, missed = [], 0
        ids = np.unique(legs["id"][bigm])
        for j in ids:
            idx = np.flatnonzero((legs["id"] == j) & m)
            ok = np.flatnonzero(t[idx] == d[idx[0]])
            if ok.size:
                lags.append(ok[0])
            else:
                missed += 1
        out["задержка_ч"] = float(np.median(lags)) if lags else np.nan
        out["крупных_пропущено"] = missed / max(1, len(ids))
    dirm = m & (t != 0)
    out["против_колена"] = float(np.mean(t[dirm] == -d[dirm])) if dirm.any() else np.nan

    tv = t[m]
    out["смен_на_100"] = float(np.mean(np.diff(tv) != 0) * 100) if len(tv) > 1 else np.nan
    seg = np.diff(np.flatnonzero(np.r_[True, np.diff(tv) != 0, True]))
    out["коротких_на_100"] = float(np.sum(seg < 3) / len(tv) * 100)

    ema = pd.Series(close).ewm(span=ema_period, adjust=False).mean().to_numpy()
    side = np.sign(close - ema)
    out["нарушений_EMA"] = float(np.mean(((t > 0) & (side < 0) | (t < 0) & (side > 0))[m]))

    fwd = (np.r_[close[h:], np.full(h, np.nan)] - close) / atr
    fm = m & np.isfinite(fwd)
    dm, nm = fm & (t != 0), fm & (t == 0)
    out["ошибка_против"] = float(np.mean(t[dm] * fwd[dm] <= -move_atr)) if dm.any() else np.nan
    out["ошибка_нейтраль"] = float(np.mean(np.abs(fwd[nm]) >= move_atr)) if nm.any() else np.nan
    return out


# ---------------------------------------------------------------- подбор параметров (годится для ноутбука)

def score_trend_filter(frames: Dict[str, pd.DataFrame], make_trend, params: Optional[dict] = None, *,
                       split_year: int = 2021, commission_pct: float = 0.04, k: float = 6.0,
                       min_bars: int = 500) -> Dict[str, float]:
    """Оценка тренд-фильтра на нескольких инструментах: следование метке после комиссии по каждой паре
    «инструмент × год», ATR на 1000 баров.

    frames      — {имя: часовые бары time/open/high/low/close} (по инструменту);
    make_trend  — функция (df, **params) -> метка −1/0/+1 на каждом баре df (массив или Series той же
                  длины) или DataFrame с колонкой trend_4h, как у add_trend_filter_v3 из ноутбука;
    Возвращает медиану и среднее по инструмент-годам до `split_year` (sel) и после (val), долю
    положительных, а также сводные метрики разметки за оба периода (trend_quality).
    """
    params = params or {}
    per_year, summ = [], {}
    for name, df in frames.items():
        df = df.sort_values("time").reset_index(drop=True)
        if "_atr" not in df.columns:
            df = df.assign(_atr=_atr(df))
        out = make_trend(df, **params)
        if isinstance(out, pd.DataFrame):
            out = out["trend_4h"]
        t = np.asarray(out, dtype=float)
        if len(t) != len(df):
            raise ValueError(f"{name}: метка длины {len(t)}, а баров {len(df)} — фильтр должен вернуть метку на каждый бар")
        close, atr = df["close"].to_numpy(dtype=float), df["_atr"].to_numpy()
        legs = zigzag_legs(close, atr, k)
        _, net = follow_pnl(close, atr, t, commission_pct)
        yr = pd.to_datetime(df["time"]).dt.year.to_numpy()
        ok = np.isfinite(net) & np.isfinite(t) & np.isfinite(legs["dir"])
        for y in np.unique(yr):
            m = ok & (yr == y)
            if m.sum() >= min_bars:
                per_year.append((name, int(y), float(net[m].sum() * 1000 / m.sum())))
        for per, m in (("sel", yr < split_year), ("val", yr >= split_year)):
            q = trend_quality(df, t, k=k, legs=legs, mask=m, commission_pct=commission_pct)
            for key in ("следование", "в_крупных_по_ходу", "в_крупных_против", "доля_тренда", "смен_на_100",
                        "нарушений_EMA"):
                summ.setdefault(f"{per}_{key}", []).append(q.get(key, np.nan))
    py = pd.DataFrame(per_year, columns=["inst", "year", "f"])
    sel, val = py[py["year"] < split_year]["f"], py[py["year"] >= split_year]["f"]
    res = {"sel_median": float(sel.median()), "sel_mean": float(sel.mean()), "sel_pos": float((sel > 0).mean()),
           "val_median": float(val.median()), "val_mean": float(val.mean()), "val_pos": float((val > 0).mean())}
    res.update({key: float(np.nanmean(v)) for key, v in summ.items()})
    return res


def search_trend_filter(frames: Dict[str, pd.DataFrame], make_trend, space: Dict[str, list], *, n_random: int = 100,
                        top: int = 5, n_neigh: int = 6, seed: int = 42, verbose: bool = True, **score_kw) -> pd.DataFrame:
    """Случайный поиск + выбор по плато (без перебора всей сетки).

    1) n_random случайных наборов из `space` ({параметр: [значения]}), оценка score_trend_filter;
    2) для `top` лучших по медиане периода выбора — n_neigh соседей (1–2 параметра сдвинуты на соседнее
       значение сетки); «плато» = медиана соседей. Выбирать стоит набор с лучшим плато, а не с лучшим
       sel_median: одиночный пик в сетке — обычно случайность;
    3) val_* — проверка: по ней ничего не выбирается, только смотрим, держится ли результат.
    Возвращает таблицу, отсортированную по sel_median; колонка plateau — у `top` лучших.
    """
    import itertools
    import random

    rng = random.Random(seed)
    keys = list(space)
    combos = list(itertools.product(*space.values()))
    picks = rng.sample(combos, min(n_random, len(combos)))
    rows = []
    for i, values in enumerate(picks):
        p = dict(zip(keys, values))
        try:
            rows.append({**p, **score_trend_filter(frames, make_trend, p, **score_kw)})
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  ошибка на {p}: {e}")
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(picks)}")
    res = pd.DataFrame(rows).sort_values("sel_median", ascending=False).reset_index(drop=True)
    plateau = []
    for _, r in res.head(top).iterrows():
        base = {k: r[k] for k in keys}
        scores = []
        for _ in range(n_neigh):
            q = dict(base)
            for key in rng.sample(keys, min(len(keys), rng.choice([1, 2]))):
                vals = list(space[key])
                j = vals.index(q[key]) if q[key] in vals else 0
                q[key] = vals[min(len(vals) - 1, max(0, j + rng.choice([-1, 1])))]
            try:
                scores.append(score_trend_filter(frames, make_trend, q, **score_kw)["sel_median"])
            except Exception:  # noqa: BLE001
                pass
        plateau.append(float(np.median(scores)) if scores else np.nan)
    res["plateau"] = plateau + [np.nan] * (len(res) - len(plateau))
    if verbose and len(res):
        from scipy.stats import spearmanr

        print(f"ранговая корреляция выбор → проверка: {spearmanr(res['sel_median'], res['val_median']).statistic:.2f}")
    return res
