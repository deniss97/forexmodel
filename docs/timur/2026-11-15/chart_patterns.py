"""
chart_patterns.py — нечёткая разметка фигур технического анализа на OHLC-данных.

Идея
----
1. Экстремумы ищутся ZigZag'ом с порогом в единицах ATR на НЕСКОЛЬКИХ масштабах
   (например 1.5, 2.5, 4 и 6 ATR). Мелкий масштаб видит короткие фигуры, крупный —
   длинные, поэтому длительность фигуры заранее не задаётся.
2. Каждая фигура описывается набором условий («вершины примерно равны», «голова выше
   плеч», «линия шеи почти горизонтальна» ...). Каждое условие даёт оценку 0..1
   через гладкие функции принадлежности, а не через жёсткий порог. Итоговый score —
   √(среднее геометрическое × минимум) оценок. Так «размытые» фигуры получают средний балл,
   а не выпадают целиком. Общая мягкость регулируется параметром --tol.
3. Все допуски измеряются в долях высоты фигуры или в ATR, поэтому код одинаково
   работает на любом инструменте и таймфрейме.
4. Для каждой фигуры ищется пробой (линии шеи, границы треугольника, флага), цель
   (классическая проекция высоты) и уровень отмены. Есть простая проверка исхода:
   что случилось раньше — цель или стоп.
5. Для каждой фигуры хранится known_at — бар, на котором все её экстремумы уже
   подтверждены ZigZag'ом. Это важно для ML: признаки нужно строить на known_at
   (или signal_at = бар пробоя), иначе будет заглядывание в будущее.

Фигуры: двойная вершина/дно, тройная вершина/дно, голова и плечи (+ перевёрнутая),
восходящий/нисходящий/симметричный треугольник, бычий/медвежий флаг, high-tight flag.

Использование
-------------
    pip install pandas numpy plotly
    python chart_patterns.py data.csv --out patterns.html --csv patterns.csv
    python chart_patterns.py data.csv --min-score 0.6 --tol 1.3 --last 3000 --no-gaps
    python chart_patterns.py --demo          # синтетические данные с заложенными фигурами

CSV: нужны колонки open/high/low/close (регистр и <скобки> не важны) и время —
колонка datetime/time/date/timestamp либо пара date + time (формат Финама тоже читается).

Из Python:
    from chart_patterns import Config, detect_patterns, evaluate, plot_patterns
    pats = detect_patterns(df, Config(min_score=0.6))
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Настройки
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    atr_period: int = 14
    scales: tuple = (1.5, 2.5, 4.0, 6.0)  # пороги ZigZag в ATR: от коротких фигур к длинным
    min_score: float = 0.55               # минимальная оценка, чтобы фигура попала в результат
    tol: float = 1.0                      # мягкость допусков: >1 — размытее, <1 — строже
    min_height_atr: float = 3.0           # минимальная высота фигуры в ATR
    min_width_bars: int = 10              # минимальная ширина фигуры в барах (мягкое условие)
    max_wait_mult: float = 1.0            # сколько ждать пробоя: доля ширины фигуры
    eval_horizon_mult: float = 2.0        # горизонт проверки исхода: доля ширины фигуры
    htf_min_return: float = 0.9           # рост флагштока для high-tight flag (0.9 = +90%)
    dedup_iou: float = 0.4                # перекрытие по времени, при котором фигуры одной
                                          # группы считаются дублями (остаётся лучшая)
    exclusive: bool = False               # True — на одном участке остаётся одна фигура любого типа


NAMES_RU = {
    "double_top": "Двойная вершина", "double_bottom": "Двойное дно",
    "triple_top": "Тройная вершина", "triple_bottom": "Тройное дно",
    "head_shoulders": "Голова и плечи", "inv_head_shoulders": "Перевёрнутая Г-и-П",
    "asc_triangle": "Восходящий треугольник", "desc_triangle": "Нисходящий треугольник",
    "sym_triangle": "Симметричный треугольник",
    "bull_flag": "Бычий флаг", "bear_flag": "Медвежий флаг",
    "high_tight_flag": "High-tight flag",
}
# детекторы пишутся для «верхней»/бычьей версии; зеркальная получается инверсией цены
MIRROR_NAME = {
    "double_top": "double_bottom", "triple_top": "triple_bottom",
    "head_shoulders": "inv_head_shoulders", "asc_triangle": "desc_triangle",
    "bull_flag": "bear_flag", "sym_triangle": "sym_triangle",
}
# группы для удаления дублей (разные масштабы ZigZag находят одну и ту же фигуру)
GROUP = {
    "double_top": "top", "triple_top": "top", "head_shoulders": "top",
    "double_bottom": "bottom", "triple_bottom": "bottom", "inv_head_shoulders": "bottom",
    "asc_triangle": "triangle", "desc_triangle": "triangle", "sym_triangle": "triangle",
    "bull_flag": "bull_flag", "high_tight_flag": "bull_flag", "bear_flag": "bear_flag",
}


# ════════════════════════════════════════════════════════════════════════════
# Структуры
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Pivot:
    idx: int           # бар экстремума
    price: float
    kind: int          # +1 максимум, -1 минимум
    confirm_idx: int   # бар, на котором экстремум подтвердился (-1 = ещё не подтверждён)


@dataclass
class Pattern:
    name: str
    direction: int                 # +1 бычья, -1 медвежья, 0 — по направлению пробоя
    score: float
    scale: float                   # масштаб ZigZag (в ATR), на котором найдена
    pivot_idx: list
    pivot_price: list
    start_idx: int
    end_idx: int
    lines: list = field(default_factory=list)   # [(i0, p0, i1, p1), ...] — шея, границы
    status: str = "pending"        # confirmed / invalidated / expired / pending
    breakout_idx: Optional[int] = None
    target: Optional[float] = None
    stop: Optional[float] = None
    known_at: Optional[int] = None  # все экстремумы подтверждены (None — фигура ещё формируется)
    components: dict = field(default_factory=dict)  # оценки отдельных условий — для отладки
    outcome: Optional[str] = None   # target / stop / timeout
    ret: Optional[float] = None     # доходность в направлении сделки до выхода

    @property
    def name_ru(self) -> str:
        return NAMES_RU.get(self.name, self.name)


# ════════════════════════════════════════════════════════════════════════════
# Нечёткие функции принадлежности
# ════════════════════════════════════════════════════════════════════════════
def near(x: float, target: float, width: float) -> float:
    """1 при x == target, плавно падает (гаусс) по мере удаления."""
    return float(np.exp(-0.5 * ((x - target) / max(width, 1e-12)) ** 2))


def at_least(x: float, lo: float, soft: float) -> float:
    """1 при x >= lo, ниже — плавно падает."""
    return 1.0 if x >= lo else near(x, lo, soft)


def at_most(x: float, hi: float, soft: float) -> float:
    """1 при x <= hi, выше — плавно падает."""
    return 1.0 if x <= hi else near(x, hi, soft)


def combine(d: dict) -> float:
    """Итоговая оценка: среднее геометрическое (общее качество) × минимум (самое
    слабое условие). Одно сильно проваленное условие не «размывается» остальными."""
    v = np.clip(np.array(list(d.values()), dtype=float), 1e-9, 1.0)
    return float(np.sqrt(np.exp(np.log(v).mean()) * v.min()))


# ════════════════════════════════════════════════════════════════════════════
# ATR и многомасштабный ZigZag
# ════════════════════════════════════════════════════════════════════════════
def compute_atr(high, low, close, period=14):
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev), abs(low - prev)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    return atr


def zigzag(high, low, atr, k) -> list[Pivot]:
    """Причинный ZigZag: разворот фиксируется, когда цена отошла от экстремума на k*ATR.
    confirm_idx — бар, на котором это стало известно (без заглядывания вперёд)."""
    n = len(high)
    piv: list[Pivot] = []
    trend = 0
    hi, hi_i, lo, lo_i = high[0], 0, low[0], 0
    ext, ext_i = 0.0, 0
    for i in range(1, n):
        th = k * atr[i]
        if trend == 0:
            if high[i] > hi:
                hi, hi_i = high[i], i
            if low[i] < lo:
                lo, lo_i = low[i], i
            if hi - lo >= th:
                if hi_i > lo_i:
                    piv.append(Pivot(lo_i, lo, -1, i)); trend, ext, ext_i = 1, hi, hi_i
                else:
                    piv.append(Pivot(hi_i, hi, 1, i)); trend, ext, ext_i = -1, lo, lo_i
        elif trend == 1:
            if high[i] >= ext:
                ext, ext_i = high[i], i
            elif ext - low[i] >= th:
                piv.append(Pivot(ext_i, ext, 1, i)); trend, ext, ext_i = -1, low[i], i
        else:
            if low[i] <= ext:
                ext, ext_i = low[i], i
            elif high[i] - ext >= th:
                piv.append(Pivot(ext_i, ext, -1, i)); trend, ext, ext_i = 1, high[i], i
    if trend != 0:  # последний экстремум ещё не подтверждён
        piv.append(Pivot(ext_i, ext, trend, -1))
    return piv


# ════════════════════════════════════════════════════════════════════════════
# Контекст: позволяет писать детектор один раз, а зеркальную фигуру
# получать инверсией цены (вершина ↔ дно, восходящий ↔ нисходящий и т.д.)
# ════════════════════════════════════════════════════════════════════════════
class Ctx:
    def __init__(self, high, low, close, atr, sign, cfg):
        self.s = sign
        if sign == 1:
            self.high, self.low, self.close = high, low, close
        else:
            self.high, self.low, self.close = -low, -high, -close
        self.atr, self.n, self.cfg = atr, len(close), cfg

    def unit(self, i0, i1):
        return float(np.median(self.atr[max(i0, 0):i1 + 1]))


def scan(ctx: Ctx, start: int, wait: int,
         up: Optional[Callable] = None, down: Optional[Callable] = None):
    """Ищет первое закрытие выше линии up(j) или ниже down(j) после бара start."""
    end = min(ctx.n - 1, start + max(wait, 3))
    for j in range(start + 1, end + 1):
        c = ctx.close[j]
        if up is not None and c > up(j):
            return "up", j
        if down is not None and c < down(j):
            return "down", j
    return ("pending" if end >= ctx.n - 1 else "expired"), None


def _kinds(a):
    return [p.kind for p in a]


# ════════════════════════════════════════════════════════════════════════════
# Детекторы (в «рабочем» пространстве: вершины = максимумы, бычий флаг)
# Каждый возвращает список словарей-кандидатов.
# ════════════════════════════════════════════════════════════════════════════
def det_double_top(ctx: Ctx, P, i):
    a = P[i:i + 4]
    if len(a) < 4 or _kinds(a) != [-1, 1, -1, 1]:
        return []
    L0, H1, L1, H2 = a
    cfg, t = ctx.cfg, ctx.cfg.tol
    u = ctx.unit(H1.idx, H2.idx)
    h = min(H1.price, H2.price) - L1.price
    if h <= 0:
        return []
    c = {
        "height": at_least(h / u, cfg.min_height_atr, 1.5 * t),
        "equal_tops": near((H2.price - H1.price) / h, 0, 0.12 * t),
        "prior_trend": at_least((H1.price - L0.price) / h, 0.8, 0.4 * t),
        "symmetry": near(np.log(max(H2.idx - L1.idx, 1) / max(L1.idx - H1.idx, 1)), 0, 1.0 * t),
        "width": at_least(H2.idx - H1.idx, cfg.min_width_bars, 3 * t),
    }
    neck = L1.price
    top = max(H1.price, H2.price)
    width = H2.idx - H1.idx
    res, bo = scan(ctx, H2.idx, int(width * cfg.max_wait_mult), up=lambda j: top, down=lambda j: neck)
    return [dict(name="double_top", dir=-1, comp=c, piv=a, start=H1.idx, end=H2.idx,
                 lines=[(H1.idx, neck, bo or H2.idx + width // 2, neck)],
                 res=res, bo_dir="down", bo=bo, target=neck - h, stop=top)]


def det_triple_top(ctx: Ctx, P, i):
    a = P[i:i + 6]
    if len(a) < 6 or _kinds(a) != [-1, 1, -1, 1, -1, 1]:
        return []
    L0, H1, L1, H2, L2, H3 = a
    cfg, t = ctx.cfg, ctx.cfg.tol
    u = ctx.unit(H1.idx, H3.idx)
    tops = np.array([H1.price, H2.price, H3.price])
    neck = min(L1.price, L2.price)
    h = tops.mean() - neck
    if h <= 0:
        return []
    c = {
        "height": at_least(h / u, cfg.min_height_atr, 1.5 * t),
        "equal_tops": near((tops.max() - tops.min()) / h, 0, 0.15 * t),
        "equal_lows": near((L1.price - L2.price) / h, 0, 0.25 * t),
        "width": at_least(H3.idx - H1.idx, 1.5 * cfg.min_width_bars, 4 * t),
        "prior_trend": at_least((H1.price - L0.price) / h, 0.8, 0.4 * t),
    }
    width = H3.idx - H1.idx
    top = tops.max()
    res, bo = scan(ctx, H3.idx, int(width * cfg.max_wait_mult), up=lambda j: top, down=lambda j: neck)
    return [dict(name="triple_top", dir=-1, comp=c, piv=a, start=H1.idx, end=H3.idx,
                 lines=[(H1.idx, neck, bo or H3.idx + width // 3, neck)],
                 res=res, bo_dir="down", bo=bo, target=neck - h, stop=top)]


def det_head_shoulders(ctx: Ctx, P, i):
    a = P[i:i + 6]
    if len(a) < 6 or _kinds(a) != [-1, 1, -1, 1, -1, 1]:
        return []
    L0, H1, L1, H2, L2, H3 = a
    cfg, t = ctx.cfg, ctx.cfg.tol
    m = (L2.price - L1.price) / (L2.idx - L1.idx)

    def nl(j):
        return L1.price + m * (j - L1.idx)

    h = H2.price - nl(H2.idx)
    if h <= 0:
        return []
    u = ctx.unit(H1.idx, H3.idx)
    c = {
        "height": at_least(h / u, cfg.min_height_atr, 1.5 * t),
        "head_above": at_least((H2.price - max(H1.price, H3.price)) / h, 0.15, 0.08 * t),
        "equal_shoulders": near((H1.price - H3.price) / h, 0, 0.2 * t),
        "shoulders_high": at_least(min(H1.price - nl(H1.idx), H3.price - nl(H3.idx)) / h, 0.35, 0.15 * t),
        "flat_neck": near((L2.price - L1.price) / h, 0, 0.3 * t),
        "time_symmetry": near(np.log(max(H3.idx - H2.idx, 1) / max(H2.idx - H1.idx, 1)), 0, 0.7 * t),
        "prior_trend": at_least((H1.price - L0.price) / h, 0.6, 0.3 * t),
        "width": at_least(H3.idx - H1.idx, 1.5 * cfg.min_width_bars, 4 * t),
    }
    width = H3.idx - H1.idx
    res, bo = scan(ctx, H3.idx, int(width * cfg.max_wait_mult), up=lambda j: H2.price, down=nl)
    x_end = bo or H3.idx + width // 3
    return [dict(name="head_shoulders", dir=-1, comp=c, piv=a, start=H1.idx, end=H3.idx,
                 lines=[(H1.idx, nl(H1.idx), x_end, nl(x_end))],
                 res=res, bo_dir="down", bo=bo,
                 target=(nl(bo) if bo else nl(H3.idx)) - h, stop=H2.price)]


def det_triangles(ctx: Ctx, P, i, with_symmetric: bool):
    out = []
    cfg, t = ctx.cfg, ctx.cfg.tol
    for m in (5, 6, 7):
        a = P[i:i + m]
        if len(a) < m:
            break
        hs = [p for p in a if p.kind == 1]
        ls = [p for p in a if p.kind == -1]
        if len(hs) < 2 or len(ls) < 2:
            continue
        i0, i1 = a[0].idx, a[-1].idx
        W = i1 - i0
        if W < 8:
            continue
        mu, bu = np.polyfit([p.idx for p in hs], [p.price for p in hs], 1)
        ml, bl = np.polyfit([p.idx for p in ls], [p.price for p in ls], 1)

        def up(j, mu=mu, bu=bu):
            return mu * j + bu

        def lo(j, ml=ml, bl=bl):
            return ml * j + bl

        h0, h1 = up(i0) - lo(i0), up(i1) - lo(i1)
        if h0 <= 0 or h1 <= 0:
            continue
        u = ctx.unit(i0, i1)
        su, sl = mu * W / h0, ml * W / h0           # наклон границ за время фигуры, в долях высоты
        resid = [p.price - up(p.idx) for p in hs] + [p.price - lo(p.idx) for p in ls]
        rmse = float(np.sqrt(np.mean(np.square(resid)))) / h0
        jj = np.arange(i0, i1 + 1)
        cl = ctx.close[i0:i1 + 1]
        outside = np.mean((cl > up(jj) + 0.15 * h0) | (cl < lo(jj) - 0.15 * h0))
        legs = np.abs(np.diff([p.price for p in a]))
        common = {
            "height": at_least(h0 / u, cfg.min_height_atr, 1.5 * t),
            "fit": near(rmse, 0, 0.1 * t),
            "inside": near(float(outside), 0, 0.08 * t),
            "converging": at_most(h1 / h0, 0.7, 0.2 * t),
            "width": at_least(W, 1.5 * cfg.min_width_bars, 4 * t),
            # волны сужаются постепенно: первая не должна быть «трендом», к которому
            # прилеплен маленький треугольник
            "gradual": at_most(legs[0] / max(np.mean(legs[1:]), 1e-12), 1.8, 0.6 * t),
        }
        # время ожидания пробоя — не дальше вершины треугольника
        wait = int(W * cfg.max_wait_mult)
        if mu != ml:
            apex = (bl - bu) / (mu - ml)
            if apex > i1:
                wait = int(min(wait, apex - i1))
        res, bo = scan(ctx, i1, wait, up=up, down=lo)
        x_end = bo or i1 + W // 4
        lines = [(i0, up(i0), x_end, up(x_end)), (i0, lo(i0), x_end, lo(x_end))]
        bo_px = (up(bo) if res == "up" else lo(bo)) if bo else None

        kinds = [("asc_triangle", +1, {"flat_top": near(su, 0, 0.12 * t),
                                       "rising_lows": at_least(sl, 0.3, 0.12 * t)})]
        if with_symmetric:
            kinds.append(("sym_triangle", 0, {"falling_highs": at_most(su, -0.2, 0.1 * t),
                                              "rising_lows": at_least(sl, 0.2, 0.1 * t),
                                              "symmetric": near(su + sl, 0, 0.35 * t)}))
        for name, d, spec in kinds:
            c = {**common, **spec}
            if d == +1:
                tgt = (bo_px + h0) if res == "up" else None
                stop = lo(bo) if res == "up" else None
            elif bo:  # симметричный: направление задаёт пробой
                tgt = bo_px + h0 if res == "up" else bo_px - h0
                stop = lo(bo) if res == "up" else up(bo)
            else:
                tgt = stop = None
            out.append(dict(name=name, dir=d, comp=c, piv=a, start=i0, end=i1, lines=lines,
                            res=res, bo_dir="up" if d >= 0 else "down", bo=bo,
                            target=tgt, stop=stop, sym=(d == 0)))
    return out


def det_flag(ctx: Ctx, P, i):
    out = []
    if i + 3 >= len(P):
        return out
    L0, H1 = P[i], P[i + 1]
    if (L0.kind, H1.kind) != (-1, 1):
        return out
    cfg, t = ctx.cfg, ctx.cfg.tol
    M = H1.price - L0.price
    Tp = max(H1.idx - L0.idx, 1)
    u = ctx.unit(L0.idx, H1.idx)
    pole = {
        "pole_size": at_least(M / u, 5.0, 2.0 * t),
        "pole_speed": at_least(M / (u * Tp), 0.5, 0.25 * t),   # ATR за бар
    }
    for nc in (3, 4, 5):   # консолидация: минимум 2 минимума и 2 максимума (с вершиной флагштока)
        C = P[i + 2:i + 2 + nc]
        if len(C) < nc:
            break
        lows = [p for p in C if p.kind == -1]
        highs = [H1] + [p for p in C if p.kind == 1]
        Tc = C[-1].idx - H1.idx
        retr = (H1.price - min(p.price for p in lows)) / M
        over = (max(p.price for p in highs) - H1.price) / M
        mu, bu = np.polyfit([p.idx for p in highs], [p.price for p in highs], 1)
        ml, _ = np.polyfit([p.idx for p in lows], [p.price for p in lows], 1)
        su, sl = mu * Tc / M, ml * Tc / M

        def up(j, mu=mu, bu=bu):
            return mu * j + bu

        c = {**pole,
             "shallow_pullback": at_most(retr, 0.4, 0.12 * t),
             "no_new_high": at_most(over, 0.05, 0.05 * t),
             "short_vs_pole": at_most(Tc / Tp, 1.2, 0.6 * t),
             "min_length": at_least(Tc, cfg.min_width_bars / 2, 2 * t),
             "flat_or_down": at_most(su, 0.02, 0.1 * t),
             "channel": at_most(su - sl, 0.1, 0.1 * t)}   # параллельно или сходится (вымпел)
        low_min = min(p.price for p in lows)
        res, bo = scan(ctx, C[-1].idx, int(max(Tc, Tp) * cfg.max_wait_mult),
                       up=up, down=lambda j: low_min)
        x_end = bo or C[-1].idx + max(Tc // 3, 2)
        name = "bull_flag"
        if ctx.s == 1 and L0.price > 0 and M / L0.price >= cfg.htf_min_return:
            name = "high_tight_flag"
        out.append(dict(name=name, dir=+1, comp=c, piv=[L0, H1] + C, start=L0.idx, end=C[-1].idx,
                        lines=[(H1.idx, up(H1.idx), x_end, up(x_end))],
                        res=res, bo_dir="up", bo=bo,
                        target=(up(bo) + M) if bo else None, stop=low_min))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Основной поиск
# ════════════════════════════════════════════════════════════════════════════
def _finalize(r: dict, s: int, scale: float, min_score: float) -> Optional[Pattern]:
    score = combine(r["comp"])
    if score < min_score:
        return None
    name = r["name"] if s == 1 else MIRROR_NAME.get(r["name"], r["name"])
    d = r["dir"] * s
    if r.get("sym") and r["bo"] is not None:
        d = (1 if r["res"] == "up" else -1) * s
    # статус пробоя
    if r["res"] in ("pending", "expired"):
        status = r["res"]
    elif r.get("sym"):
        status = "confirmed"
    else:
        status = "confirmed" if r["res"] == r["bo_dir"] else "invalidated"
    if status != "confirmed":
        bo, tgt, stop = None, None, (r["stop"] * s if r["stop"] is not None else None)
        if r.get("sym"):
            stop = None
    else:
        bo = r["bo"]
        tgt = r["target"] * s if r["target"] is not None else None
        stop = r["stop"] * s if r["stop"] is not None else None
    piv = r["piv"]
    conf = [p.confirm_idx for p in piv]
    known = None if min(conf) < 0 else max(conf)
    return Pattern(
        name=name, direction=d, score=round(score, 3), scale=scale,
        pivot_idx=[p.idx for p in piv], pivot_price=[p.price * s for p in piv],
        start_idx=r["start"], end_idx=r["end"],
        lines=[(i0, p0 * s, i1, p1 * s) for (i0, p0, i1, p1) in r["lines"]],
        status=status, breakout_idx=bo, target=tgt, stop=stop, known_at=known,
        components={k: round(v, 3) for k, v in r["comp"].items()},
    )


def _iou(a: Pattern, b: Pattern) -> float:
    inter = min(a.end_idx, b.end_idx) - max(a.start_idx, b.start_idx)
    if inter <= 0:
        return 0.0
    union = max(a.end_idx, b.end_idx) - min(a.start_idx, b.start_idx)
    return inter / max(union, 1)


def dedup(pats: list[Pattern], iou: float, exclusive: bool = False) -> list[Pattern]:
    kept: list[Pattern] = []
    for p in sorted(pats, key=lambda x: -x.score):
        if all((not exclusive and GROUP[p.name] != GROUP[k.name]) or _iou(p, k) < iou for k in kept):
            kept.append(p)
    return sorted(kept, key=lambda x: x.start_idx)


def detect_patterns(df: pd.DataFrame, cfg: Config = Config()) -> list[Pattern]:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = compute_atr(high, low, close, cfg.atr_period)
    found: list[Pattern] = []
    for k in cfg.scales:
        piv = zigzag(high, low, atr, k)
        for s in (1, -1):
            ctx = Ctx(high, low, close, atr, s, cfg)
            P = [Pivot(p.idx, p.price * s, p.kind * s, p.confirm_idx) for p in piv]
            for i in range(len(P)):
                cands = (det_double_top(ctx, P, i) + det_triple_top(ctx, P, i)
                         + det_head_shoulders(ctx, P, i)
                         + det_triangles(ctx, P, i, with_symmetric=(s == 1))
                         + det_flag(ctx, P, i))
                for r in cands:
                    p = _finalize(r, s, k, cfg.min_score)
                    if p is not None:
                        found.append(p)
    return dedup(found, cfg.dedup_iou, cfg.exclusive)


# ════════════════════════════════════════════════════════════════════════════
# Грубая проверка исхода: что раньше — цель или стоп (без комиссий!)
# ════════════════════════════════════════════════════════════════════════════
def evaluate(df: pd.DataFrame, pats: list[Pattern], cfg: Config = Config()) -> pd.DataFrame:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(close)
    for p in pats:
        if p.status != "confirmed" or p.target is None or p.stop is None or p.direction == 0:
            continue
        bo, d = p.breakout_idx, p.direction
        entry = close[bo]
        horizon = int(max(p.end_idx - p.start_idx, 5) * cfg.eval_horizon_mult)
        p.outcome, exit_px = "timeout", close[min(bo + horizon, n - 1)]
        for j in range(bo + 1, min(bo + horizon, n - 1) + 1):
            hit_stop = low[j] <= p.stop if d > 0 else high[j] >= p.stop
            hit_tgt = high[j] >= p.target if d > 0 else low[j] <= p.target
            if hit_stop:          # если в одном баре и то и другое — считаем стоп
                p.outcome, exit_px = "stop", p.stop
                break
            if hit_tgt:
                p.outcome, exit_px = "target", p.target
                break
        p.ret = float(d * (exit_px / entry - 1))
    rows = [p for p in pats if p.outcome]
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame({"pattern": [p.name_ru for p in rows], "outcome": [p.outcome for p in rows],
                      "ret": [p.ret for p in rows]})
    g = t.groupby("pattern")
    return pd.DataFrame({
        "сделок": g.size(),
        "цель, %": g["outcome"].apply(lambda s: 100 * (s == "target").mean()).round(1),
        "стоп, %": g["outcome"].apply(lambda s: 100 * (s == "stop").mean()).round(1),
        "ср. доходн., %": (100 * g["ret"].mean()).round(3),
        "медиана, %": (100 * g["ret"].median()).round(3),
    }).sort_values("сделок", ascending=False)


def patterns_to_frame(df: pd.DataFrame, pats: list[Pattern]) -> pd.DataFrame:
    ts = df.index

    def tt(i):
        return ts[i] if i is not None and 0 <= i < len(ts) else None

    return pd.DataFrame([{
        "pattern": p.name, "pattern_ru": p.name_ru, "direction": p.direction, "score": p.score,
        "zigzag_atr": p.scale, "start": tt(p.start_idx), "end": tt(p.end_idx),
        "known_at": tt(p.known_at), "status": p.status, "breakout": tt(p.breakout_idx),
        "target": p.target, "stop": p.stop, "outcome": p.outcome, "ret": p.ret,
        "components": p.components,
    } for p in pats])


# ════════════════════════════════════════════════════════════════════════════
# График
# ════════════════════════════════════════════════════════════════════════════
COLORS = {1: "#1b9e5a", -1: "#d63c3c", 0: "#7a7a7a"}


def plot_patterns(df: pd.DataFrame, pats: list[Pattern], out_html: str,
                  last: Optional[int] = None, no_gaps: bool = False,
                  plotlyjs: str | bool = True, title: str = "Фигуры теханализа"):
    import plotly.graph_objects as go

    lo_i = 0 if not last else max(0, len(df) - last)
    d = df.iloc[lo_i:]
    t = df.index
    fig = go.Figure(go.Candlestick(
        x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="Цена", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350", showlegend=False))
    shown = set()
    for p in pats:
        if p.end_idx < lo_i:
            continue
        col = COLORS[p.direction]
        dash = None if p.status == "confirmed" else "dot"
        hover = (f"<b>{p.name_ru}</b><br>score {p.score} · ZigZag {p.scale} ATR<br>"
                 f"статус: {p.status}" + (f"<br>исход: {p.outcome}" if p.outcome else "")
                 + "<br>" + "<br>".join(f"{k}: {v}" for k, v in p.components.items()))
        fig.add_trace(go.Scatter(
            x=[t[i] for i in p.pivot_idx], y=p.pivot_price, mode="lines+markers",
            line=dict(color=col, width=2, dash=dash), marker=dict(size=6),
            name=p.name_ru, legendgroup=p.name, showlegend=p.name not in shown,
            hovertext=hover, hoverinfo="text"))
        shown.add(p.name)
        for (i0, p0, i1, p1) in p.lines:
            i1 = min(i1, len(t) - 1)
            fig.add_trace(go.Scatter(x=[t[i0], t[i1]], y=[p0, p1], mode="lines",
                                     line=dict(color=col, width=1.2, dash="dash"),
                                     legendgroup=p.name, showlegend=False, hoverinfo="skip"))
        if p.breakout_idx is not None:
            b = p.breakout_idx
            fig.add_trace(go.Scatter(x=[t[b]], y=[df["close"].iloc[b]], mode="markers",
                                     marker=dict(symbol="star", size=11, color=col,
                                                 line=dict(width=1, color="black")),
                                     legendgroup=p.name, showlegend=False,
                                     hovertext=f"Пробой: {p.name_ru}", hoverinfo="text"))
            if p.target is not None:
                b2 = min(b + max((p.end_idx - p.start_idx) // 2, 3), len(t) - 1)
                fig.add_trace(go.Scatter(x=[t[b], t[b2]], y=[p.target, p.target], mode="lines",
                                         line=dict(color=col, width=1, dash="dot"),
                                         legendgroup=p.name, showlegend=False,
                                         hovertext=f"Цель {p.target:.4g}", hoverinfo="text"))
        top_i = int(np.argmax(p.pivot_price)) if p.direction <= 0 else int(np.argmin(p.pivot_price))
        fig.add_trace(go.Scatter(
            x=[t[p.pivot_idx[top_i]]], y=[p.pivot_price[top_i]], mode="text",
            text=[f"{p.name_ru} {p.score:.2f}"],
            textposition="top center" if p.direction <= 0 else "bottom center",
            textfont=dict(size=10, color=col), legendgroup=p.name, showlegend=False,
            hoverinfo="skip"))
    fig.update_layout(title=title, template="plotly_white", height=750,
                      xaxis_rangeslider_visible=False, hovermode="closest",
                      legend=dict(orientation="h", y=1.02, x=0, title_text="Клик — скрыть/показать тип: "))
    if no_gaps:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.write_html(out_html, include_plotlyjs=plotlyjs)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Загрузка данных и демо
# ════════════════════════════════════════════════════════════════════════════
def load_ohlc(path: str, time_col: Optional[str] = None, sep: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
    alias = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vol": "volume"}
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    if time_col:
        ts = pd.to_datetime(df[time_col.lower()])
    elif "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str).str.zfill(6),
                            errors="coerce")
        if ts.isna().mean() > 0.5:
            ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        cand = [c for c in ("datetime", "timestamp", "time", "date") if c in df.columns]
        if not cand:
            raise ValueError("Не нашёл колонку времени — укажите --time-col")
        ts = pd.to_datetime(df[cand[0]])
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Нет колонок: {missing}")
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = pd.DatetimeIndex(ts)
    return out.sort_index()


def make_demo(seed: int = 7) -> pd.DataFrame:
    """Случайное блуждание с вклеенными фигурами — для проверки разметки."""
    rng = np.random.default_rng(seed)
    S = 0.6  # размах «свинга» в цене
    templates = {
        "hs": [(15, 3), (25, 2), (40, 4.5), (55, 2.1), (68, 3.1), (85, 0.2)],
        "dbl_bottom": [(15, -3), (30, -1.3), (46, -3.05), (66, 0.6)],
        "asc_tri": [(10, 2.5), (18, 0.6), (28, 2.5), (34, 1.2), (42, 2.5), (46, 1.8), (55, 3.8)],
        "flag": [(8, 4.5), (14, 3.6), (19, 4.1), (25, 3.3), (29, 3.8), (38, 6.5)],
        "dbl_top": [(15, 3), (30, 1.4), (46, 3.05), (66, -0.6)],
        "sym_tri": [(10, 2.6), (18, -0.4), (26, 2.0), (33, 0.3), (39, 1.5), (44, 0.8), (52, -1.4)],
        "inv_hs": [(15, -3), (25, -2), (40, -4.5), (55, -2.0), (68, -3.0), (85, -0.3)],
    }
    anchors = [(0, 25.0)]
    planted = []
    t, p = 0, 25.0
    for name in list(templates) * 2:
        for _ in range(rng.integers(3, 6)):        # случайный участок между фигурами
            t += int(rng.integers(8, 20)); p += rng.normal(0, 1.2) * S
            anchors.append((t, p))
        base_t, base_p = t, p
        for dt, dp in templates[name]:
            anchors.append((base_t + dt, base_p + dp * S))
        planted.append((name, base_t, anchors[-1][0]))
        t, p = anchors[-1]
    ts_, ps_ = zip(*anchors)
    n = ts_[-1] + 1
    close = np.interp(np.arange(n), ts_, ps_)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 0.035)
    close = close + noise
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.01, n)
    wick = np.abs(rng.normal(0, 0.04, (2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    idx = pd.date_range("2026-01-05", periods=n, freq="h")
    out = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    out.attrs["planted"] = planted   # (шаблон, бар начала, бар конца) — для проверки
    return out


# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Нечёткая разметка фигур теханализа")
    ap.add_argument("csv", nargs="?", help="CSV с OHLC")
    ap.add_argument("--demo", action="store_true", help="запустить на синтетических данных")
    ap.add_argument("--time-col")
    ap.add_argument("--sep")
    ap.add_argument("--out", default="patterns.html")
    ap.add_argument("--csv-out", default="patterns.csv")
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--tol", type=float, default=1.0, help="мягкость допусков (1.3 — размытее, 0.7 — строже)")
    ap.add_argument("--scales", default="1.5,2.5,4,6", help="пороги ZigZag в ATR через запятую")
    ap.add_argument("--atr", type=int, default=14)
    ap.add_argument("--htf", type=float, default=0.9, help="рост флагштока для high-tight flag")
    ap.add_argument("--last", type=int, help="на графике показать только последние N баров")
    ap.add_argument("--exclusive", action="store_true", help="одна фигура на участок (лучшая по score)")
    ap.add_argument("--no-gaps", action="store_true", help="убрать выходные с оси времени")
    ap.add_argument("--cdn", action="store_true", help="plotly.js с CDN (файл меньше, нужен интернет)")
    a = ap.parse_args()

    if a.demo:
        df = make_demo()
    elif a.csv:
        df = load_ohlc(a.csv, a.time_col, a.sep)
    else:
        ap.error("укажите CSV или --demo")

    cfg = Config(min_score=a.min_score, tol=a.tol, atr_period=a.atr, htf_min_return=a.htf,
                 scales=tuple(float(x) for x in a.scales.split(",")), exclusive=a.exclusive)
    pats = detect_patterns(df, cfg)
    stats = evaluate(df, pats, cfg)
    patterns_to_frame(df, pats).to_csv(a.csv_out, index=False)
    plot_patterns(df, pats, a.out, last=a.last, no_gaps=a.no_gaps,
                  plotlyjs="cdn" if a.cdn else True)

    print(f"Баров: {len(df)}, найдено фигур: {len(pats)}")
    print(pd.Series([p.name_ru for p in pats]).value_counts().to_string())
    if not stats.empty:
        print("\nИсходы после пробоя (без комиссий, грубая оценка):")
        print(stats.to_string())
    print(f"\nГрафик: {a.out}\nТаблица: {a.csv_out}")


if __name__ == "__main__":
    main()
