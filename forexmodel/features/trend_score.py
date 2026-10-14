"""Тренд-фильтры из ТЗ 15.10.2026: трёхслойный скор и классические индикаторы.

Оба режима — ещё два значения `trend.mode` (и `trend_gate.mode`) рядом с early/confirm, выдают ту же
колонку `trend_4h` со значениями −1 / 0 / +1 и считаются только по закрытым барам.

`score` — трёхслойный фильтр (гипотеза H44):

  1. медленный слой (режим): знак (close − EMA(score_regime_ema)) на таймфрейме score_regime_tf
     (по последнему ЗАКРЫТОМУ бину, как гейт C; «base» — на рабочем ТФ);
  2. быстрый слой: мультимасштабный скор как у CTA-фондов — разницы EMA на парах score_spans, каждая
     в «волатильностях» бара (c·σ доходности за score_vol_window), обрезка ±3 и деление на 3, среднее
     по парам → непрерывный скор в [−1, 1]. Вариант score_fast=kama — расстояние цены от адаптивной
     средней Кауфмана (KAMA) в тех же единицах; при направленном ходе ER → 1, и средняя ускоряется;
     score_vol_adapt — при высокой волатильности (ATR14/ATR100 > 1) вес коротких пар больше, при низкой —
     длинных (прогноз волатильности как параметр окон);
  3. шок: |доходность бара| > score_shock_z·σ или пробой канала Дончиана за score_shock_donchian баров —
     скор этого бара = знак хода, без ожидания медленного слоя. Вне шока скор, идущий против режима,
     ослабляется в score_against раз;
  4. гистерезис: вход в «вверх» при скоре > score_enter, выход только при скоре < score_exit (и
     симметрично вниз), не раньше чем через score_min_hold баров после входа (встречный шок —
     разворот сразу).

`classic` — один классический индикатор из спецификации `rule` (гипотеза H45), на таймфрейме
`timeframe` (как гейт C: с update_every_bar — пересчёт каждый час из только что закрытого окна):

  ema_price:N        знак (close − EMA N)
  ema_slope:N:K      знак (EMA N − EMA N K баров назад)
  ema_cross:F:S      знак (EMA F − EMA S)
  macd:F:S:G:вид     вид sig — MACD выше / ниже сигнальной, zero — знак MACD, hist — гистограмма растёт / падает
  supertrend:N:M     направление Supertrend (ATR Уайлдера N, множитель M)
  donchian:N:M       «Черепахи»: пробой максимума / минимума N прошлых баров — тренд, выход при пробое
                     M-барного канала в обратную сторону (нейтраль)
  keltner:N:A:K      закрытие за EMA N ± K·ATR A — тренд до возврата к EMA N
  bb:N:K             закрытие за полосой Боллинджера (N, K) — тренд до возврата к средней
  adx:N:T            ADX N > T — направление по DI, иначе нейтраль
  kama:N             знак наклона KAMA (ER за N баров, 2/30)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind

__all__ = ["classic_direction", "trend_score_components", "score_direction", "hysteresis_state",
           "kama", "efficiency_ratio", "CLASSIC_RULES"]

CLASSIC_RULES = ("ema_price", "ema_slope", "ema_cross", "macd", "supertrend", "donchian", "keltner", "bb", "adx",
                 "kama")


# ---------------------------------------------------------------- общие блоки

def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """ER Кауфмана: |изменение за n| / сумма |изменений| за n (1 — прямая линия, ~0 — пила)."""
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return change / path.replace(0, np.nan)


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Адаптивная средняя Кауфмана: сглаживание sc = (ER·(2/(f+1) − 2/(s+1)) + 2/(s+1))²."""
    er = efficiency_ratio(close, n).to_numpy(dtype=float)
    fa, sa = 2.0 / (fast + 1), 2.0 / (slow + 1)
    sc = (np.nan_to_num(er) * (fa - sa) + sa) ** 2
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c):
        out[0] = c[0]
        for i in range(1, len(c)):
            prev = out[i - 1] if np.isfinite(out[i - 1]) else c[i]
            out[i] = prev + sc[i] * (c[i] - prev) if np.isfinite(c[i]) else prev
    return pd.Series(out, index=close.index)


def _latch(enter_long: np.ndarray, enter_short: np.ndarray, exit_long: np.ndarray,
           exit_short: np.ndarray) -> np.ndarray:
    """Состояние по событиям: вход в ±1 по сигналу, выход в 0 по своему условию, разворот — сразу."""
    out = np.zeros(len(enter_long))
    state = 0.0
    for i in range(len(enter_long)):
        if enter_long[i] and state <= 0:
            state = 1.0
        elif enter_short[i] and state >= 0:
            state = -1.0
        elif state > 0 and exit_long[i]:
            state = 0.0
        elif state < 0 and exit_short[i]:
            state = 0.0
        out[i] = state
    return out


def _supertrend(b: pd.DataFrame, n: int, mult: float) -> np.ndarray:
    atr = ind.atr(b, n, method="wilder").to_numpy(dtype=float)
    hl2 = ((b["high"] + b["low"]) / 2).to_numpy(dtype=float)
    c = b["close"].to_numpy(dtype=float)
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    out = np.zeros(len(c))
    fu, fd, d = np.nan, np.nan, 0.0
    for i in range(len(c)):
        if not np.isfinite(atr[i]):
            continue
        if not np.isfinite(fu):
            fu, fd, d = up[i], dn[i], 1.0 if c[i] >= hl2[i] else -1.0
        else:
            # нижняя граница не опускается, пока закрытия выше неё (и наоборот для верхней)
            fu = max(up[i], fu) if c[i - 1] > fu else up[i]
            fd = min(dn[i], fd) if c[i - 1] < fd else dn[i]
            if d < 0 and c[i] > fd:
                d = 1.0
            elif d > 0 and c[i] < fu:
                d = -1.0
        out[i] = d
    return out


def classic_direction(b: pd.DataFrame, rule: str) -> np.ndarray:
    """Направление классического индикатора на барах b (time/open/high/low/close). Только прошлое:
    значение бара i зависит от баров <= i; каналы считаются по прошлым барам без текущего."""
    name, *p = rule.split(":")
    c = b["close"].astype(float)
    if name == "ema_price":
        d = np.sign(c - ind.ema(c, int(p[0])))
    elif name == "ema_slope":
        e = ind.ema(c, int(p[0]))
        d = np.sign(e - e.shift(int(p[1])))
    elif name == "ema_cross":
        d = np.sign(ind.ema(c, int(p[0])) - ind.ema(c, int(p[1])))
    elif name == "macd":
        line, sig, hist = ind.macd(c, int(p[0]), int(p[1]), int(p[2]))
        kind = p[3] if len(p) > 3 else "sig"
        d = {"sig": np.sign(line - sig), "zero": np.sign(line), "hist": np.sign(hist.diff())}[kind]
    elif name == "supertrend":
        return _supertrend(b, int(p[0]), float(p[1]))
    elif name == "donchian":
        n, m = int(p[0]), int(p[1])
        hi_n, lo_n = b["high"].rolling(n).max().shift(1), b["low"].rolling(n).min().shift(1)
        hi_m, lo_m = b["high"].rolling(m).max().shift(1), b["low"].rolling(m).min().shift(1)
        return _latch((c > hi_n).to_numpy(), (c < lo_n).to_numpy(), (c < lo_m).to_numpy(), (c > hi_m).to_numpy())
    elif name in ("keltner", "bb"):
        if name == "keltner":
            mid = ind.ema(c, int(p[0]))
            width = float(p[2]) * ind.atr(b, int(p[1]), method="wilder")
        else:
            mid = c.rolling(int(p[0])).mean()
            width = float(p[1]) * c.rolling(int(p[0])).std()
        return _latch((c > mid + width).to_numpy(), (c < mid - width).to_numpy(), (c < mid).to_numpy(),
                      (c > mid).to_numpy())
    elif name == "adx":
        a = ind.adx(b, int(p[0]))
        d = pd.Series(np.where(a["adx"] > float(p[1]), np.sign(a["dmp"] - a["dmn"]), 0.0), index=b.index)
    elif name == "kama":
        k = kama(c, int(p[0]))
        d = np.sign(k - k.shift(1))
    else:
        raise ValueError(f"trend.rule: неизвестный индикатор {name!r} (ожидалось одно из {CLASSIC_RULES})")
    return np.nan_to_num(np.asarray(d, dtype=float))


# ---------------------------------------------------------------- трёхслойный скор

def _htf_value(df: pd.DataFrame, tf: str, compute: Callable[[pd.DataFrame], pd.Series]) -> np.ndarray:
    """Значение на старшем ТФ по последнему ЗАКРЫТОМУ бину, пересчёт на каждом баре рабочего ТФ."""
    from dataclasses import replace

    from ..config import TrendConfig
    from .trend import _htf_grids, merge_htf_causal

    grids = _htf_grids(df[["time", "open", "high", "low", "close"]],
                       replace(TrendConfig(), timeframe=tf, update_every_bar=True))
    frames = [pd.DataFrame({"time": g["time"], "_v": compute(g).to_numpy(dtype=float)}) for g in grids]
    return merge_htf_causal(df[["time"]], pd.concat(frames, ignore_index=True), tf, ["_v"])["_v"].to_numpy(dtype=float)


def trend_score_components(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Компоненты трёхслойного фильтра на каждом баре рабочего ТФ (только прошлое): быстрый скор
    `ts_fast`, режим `ts_regime` (±1), шок `ts_shock` (±1 / 0), ER `ts_er`, итоговый скор `ts_score`.
    Все безразмерные — годятся и как признаки модели."""
    c = df["close"].astype(float).reset_index(drop=True)
    ret = c.pct_change()
    vol = ret.rolling(cfg.score_vol_window).std()
    unit = (c * vol).replace(0, np.nan)
    spans = [tuple(int(x) for x in s) for s in cfg.score_spans]
    if cfg.score_fast == "kama":
        parts = [((c - kama(c, cfg.score_er_period)) / unit).clip(-3, 3) / 3]
        weights = None
    else:
        parts = [((ind.ema(c, f) - ind.ema(c, s)) / unit).clip(-3, 3) / 3 for f, s in spans]
        weights = None
        if cfg.score_vol_adapt and len(parts) > 1:
            b = df[["high", "low", "close"]].reset_index(drop=True).astype(float)
            vr = (ind.atr(b, 14) / ind.atr(b, 100).replace(0, np.nan)).fillna(1.0)
            tilt = (vr - 1.0).clip(-0.5, 0.5).to_numpy()                      # >0 — волатильность выше обычной
            pos = np.linspace(1, -1, len(parts))                               # короткие пары — +1, длинные — −1
            weights = np.clip(1.0 + np.outer(tilt, pos), 0.1, None)
    if weights is None:
        fast = sum(parts) / len(parts)
    else:
        stack = np.column_stack([p.to_numpy() for p in parts])
        fast = pd.Series((stack * weights).sum(axis=1) / weights.sum(axis=1), index=c.index)

    if cfg.score_regime_tf in ("base", "", None):
        regime = np.sign(c - ind.ema(c, cfg.score_regime_ema)).to_numpy(dtype=float)
    else:
        period = cfg.score_regime_ema
        ema_htf = _htf_value(df.reset_index(drop=True), cfg.score_regime_tf, lambda g: ind.ema(g["close"], period))
        regime = np.sign(c.to_numpy() - ema_htf)
    regime = np.nan_to_num(regime)

    ret_z = (ret / vol).to_numpy(dtype=float)
    shock = np.where(np.abs(np.nan_to_num(ret_z)) > cfg.score_shock_z, np.sign(np.nan_to_num(ret_z)), 0.0)
    if cfg.score_shock_donchian > 0:
        n = cfg.score_shock_donchian
        hi = df["high"].astype(float).reset_index(drop=True).rolling(n).max().shift(1)
        lo = df["low"].astype(float).reset_index(drop=True).rolling(n).min().shift(1)
        brk = np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0))
        shock = np.where(shock != 0, shock, brk)

    f = fast.to_numpy(dtype=float)
    weak = np.where((np.sign(f) == regime) | (regime == 0), f, cfg.score_against * f)
    score = np.where(shock != 0, shock, weak)
    score[~np.isfinite(f)] = np.nan
    return pd.DataFrame({"ts_fast": f, "ts_regime": regime, "ts_shock": shock,
                         "ts_er": efficiency_ratio(c, cfg.score_er_period).to_numpy(dtype=float),
                         "ts_score": score}, index=df.index)


def hysteresis_state(score: np.ndarray, shock: np.ndarray, enter: float, exit_: float, min_hold: int) -> np.ndarray:
    """Состояние −1/0/+1 по скору с гистерезисом и минимальным удержанием.

    Из нейтрали — в +1 при скоре > enter (в −1 при < −enter). Из +1 — в нейтраль только при скоре < exit_
    и не раньше min_hold баров после входа; скор < −enter после удержания — сразу разворот. Встречный шок
    разворачивает без ожидания. NaN скора — нейтраль.
    """
    out = np.zeros(len(score))
    state, held = 0.0, 0
    for i in range(len(score)):
        s = score[i]
        if not np.isfinite(s):
            state, held = 0.0, 0
            out[i] = 0.0
            continue
        new = state
        if shock[i] != 0 and shock[i] != state:
            new = shock[i]
        elif state == 0:
            new = 1.0 if s > enter else (-1.0 if s < -enter else 0.0)
        elif held >= min_hold:
            if state * s < -enter:
                new = -state
            elif state * s < exit_:
                new = 0.0
        if new != state:
            state, held = new, 0
        held += 1
        out[i] = state
    return out


def score_direction(df: pd.DataFrame, cfg) -> np.ndarray:
    comp = trend_score_components(df, cfg)
    return hysteresis_state(comp["ts_score"].to_numpy(dtype=float), comp["ts_shock"].to_numpy(dtype=float),
                            cfg.score_enter, cfg.score_exit, cfg.score_min_hold)
