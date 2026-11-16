"""Разметка фигур теханализа на свечах 19 акций Мосбиржи (H74, план подтверждён в банке 15.11).

Фигуры из гипотезы Тагира. Код разметки Тагира на сервер не попал (вложение к гипотезе недоступно, автор написал
«код передал Денису»), поэтому правила написаны по определениям из самой гипотезы — это классические описания
«Энциклопедии графических моделей» Булковски:

  htf  — high and tight flag: резкий рост за 1–2 месяца, затем узкая консолидация и пробой вверх;
  ihs  — перевёрнутая «голова и плечи»: плечо — голова — плечо, пробой линии шеи;
  db   — двойное дно (вариант «Eve & Eve» — оба минимума округлые — помечается отдельно);
  tb   — тройное дно;
  barr — bump-and-run reversal (нижний): спокойный наклон, резкий провал («горб»), возврат выше линии наклона;
  tri  — восходящий треугольник: плоская вершина, растущие минимумы, пробой вверх.

Фигуры ищутся на дневных свечах (на них они и описаны), ход цены после пробоя считается по часовым.
Пробой фиксируется по закрытию той свечи, на которой он случился: заглядывания вперёд в разметке нет.
Протокол журнала экспериментов: выбор — пробои до 2021 года, проверка — с 2021 года.

    .venv/bin/py scripts/h74_patterns.py            # разметка, подбор порогов, таблицы, графики
    .venv/bin/py scripts/h74_patterns.py --quick    # три бумаги (отладка)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HOURLY = ROOT / "data" / "derived" / "hourly"
REP = ROOT / "reports" / "h74"
OUT = ROOT / "docs" / "results" / "h74_patterns"

TICKERS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
           "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS"]
MAIN = ["SBER", "GAZP", "LKOH"]
SRC = {"SBER": RAW / "SBER_1m_15-26.csv", "GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv"}
SPLIT = pd.Timestamp("2021-01-01")       # протокол журнала: выбор / проверка
MIN_EVENTS = 50                          # критерий плана: не меньше 50 событий на выборе
PATS = {"htf": "high and tight flag", "ihs": "перевёрнутая «голова и плечи»", "db": "двойное дно",
        "tb": "тройное дно", "barr": "bump-and-run reversal", "tri": "восходящий треугольник"}

# ─────────────────────────────── данные ───────────────────────────────


def hourly(t: str) -> pd.DataFrame:
    """Часовые свечи бумаги (кеш собирается scripts/viz_examples.py, иначе — из минуток)."""
    f = HOURLY / f"{t}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
    else:
        src = SRC.get(t, RAW / "moex" / f"{t}_1m.csv")
        parts = []
        for ch in pd.read_csv(src, usecols=["open", "close", "high", "low", "volume", "begin"], chunksize=500_000):
            ch["h"] = pd.to_datetime(ch["begin"]).dt.floor("h")
            parts.append(ch.groupby("h").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                             close=("close", "last"), volume=("volume", "sum")))
        d = pd.concat(parts).groupby(level=0).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                 close=("close", "last"), volume=("volume", "sum"))
        HOURLY.mkdir(parents=True, exist_ok=True)
        d.to_parquet(f)
    return d[(d.index.hour >= 10) & (d.index.hour <= 18)]


def to_daily(h: pd.DataFrame) -> pd.DataFrame:
    d = h.groupby(h.index.normalize()).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                           close=("close", "last"), volume=("volume", "sum"))
    return d[d["high"] > 0].dropna()


def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    c = d["close"]
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(3, n // 2)).mean()


def zigzag(h: np.ndarray, l: np.ndarray, pct: float) -> list[tuple[int, str, float]]:
    """Опорные вершины и впадины: точка подтверждается, когда цена развернулась от неё на pct."""
    piv: list[tuple[int, str, float]] = []
    direction = 0
    hi_i, hi, lo_i, lo = 0, h[0], 0, l[0]
    for i in range(1, len(h)):
        if direction >= 0:
            if h[i] > hi:
                hi_i, hi = i, h[i]
            if l[i] <= hi * (1 - pct):
                piv.append((hi_i, "H", hi))
                direction, lo_i, lo = -1, i, l[i]
                continue
        if direction <= 0:
            if l[i] < lo:
                lo_i, lo = i, l[i]
            if h[i] >= lo * (1 + pct):
                piv.append((lo_i, "L", lo))
                direction, hi_i, hi = 1, i, h[i]
    return piv


class Bars:
    """Дневные свечи одной бумаги в массивах + опорные точки зигзага для нескольких порогов."""

    def __init__(self, t: str, d: pd.DataFrame):
        self.t = t
        self.dt = d.index.to_numpy()
        self.o, self.h, self.l, self.c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.atr = atr(d).to_numpy(float)
        self.n = len(d)
        self.piv = {p: zigzag(self.h, self.l, p) for p in (0.03, 0.05, 0.08)}

    def date(self, i: int) -> pd.Timestamp:
        return pd.Timestamp(self.dt[i])


def ev(b: Bars, pat: str, i0: int, brk: int, pts: list, lines: list, note: str = "") -> dict:
    return {"ticker": b.t, "pat": pat, "i0": int(i0), "brk": int(brk), "start": b.date(i0), "date": b.date(brk),
            "price": float(b.c[brk]), "pts": [[int(i), float(p)] for i, p in pts],
            "lines": [[int(a), float(pa), int(z), float(pz)] for a, pa, z, pz in lines], "note": note,
            "span": int(brk - i0)}


def roundness(b: Bars, i: int, tol: float = 0.03, win: int = 12) -> int:
    """Сколько дней рядом с минимумом держатся в пределах tol от него: «Eve» (округлое дно) — широкое, «Adam» — узкое."""
    a, z = max(0, i - win), min(b.n, i + win + 1)
    return int(np.sum(b.l[a:z] <= b.l[i] * (1 + tol)))


def dedupe(events: list[dict], gap: int = 5) -> list[dict]:
    """Одна фигура на бумагу и пробой: из близких пробоев остаётся самая длинная фигура."""
    out: list[dict] = []
    for e in sorted(events, key=lambda x: (x["ticker"], x["brk"], -x["span"])):
        if out and out[-1]["ticker"] == e["ticker"] and e["brk"] - out[-1]["brk"] <= gap:
            if e["span"] > out[-1]["span"]:
                out[-1] = e
            continue
        out.append(e)
    return out


def breakout(b: Bars, after: int, level, limit: int, invalid: float) -> int | None:
    """Первое закрытие выше уровня (числа или линии) в пределах limit дней; падение ниже invalid отменяет фигуру."""
    for j in range(after + 1, min(b.n, after + limit + 1)):
        lv = level(j) if callable(level) else level
        if b.l[j] < invalid:
            return None
        if b.c[j] > lv:
            return j
    return None


# ─────────────────────────────── фигуры ───────────────────────────────


def find_db(b: Bars, p: dict) -> list[dict]:
    """Двойное дно: два близких минимума, между ними вершина, пробой выше вершины."""
    out = []
    piv = b.piv[p["zz"]]
    for k in range(len(piv) - 2):
        (i1, k1, l1), (ih, kh, ph), (i2, k2, l2) = piv[k], piv[k + 1], piv[k + 2]
        if (k1, kh, k2) != ("L", "H", "L"):
            continue
        if abs(l2 - l1) / min(l1, l2) > p["eq"]:
            continue
        if (ph - max(l1, l2)) / max(l1, l2) < p["depth"]:
            continue
        if not p["gap"][0] <= i2 - i1 <= p["gap"][1]:
            continue
        brk = breakout(b, i2, ph, p["confirm"], min(l1, l2) * (1 - p["eq"]))
        if brk is None:
            continue
        r1, r2 = roundness(b, i1), roundness(b, i2)
        note = "Eve & Eve" if min(r1, r2) >= p["round"] else ("Adam & Adam" if max(r1, r2) < p["round"] else "смешанное")
        note += f", дно ±{abs(l2 - l1) / min(l1, l2) * 100:.1f}%, глубина {(ph - max(l1, l2)) / max(l1, l2) * 100:.0f}%"
        out.append(ev(b, "db", i1, brk, [(i1, l1), (ih, ph), (i2, l2)], [(i1, ph, brk, ph)], note))
    return dedupe(out)


def find_tb(b: Bars, p: dict) -> list[dict]:
    """Тройное дно: три близких минимума, пробой выше верхней из двух промежуточных вершин."""
    out = []
    piv = b.piv[p["zz"]]
    for k in range(len(piv) - 4):
        seq = piv[k:k + 5]
        if "".join(x[1] for x in seq) != "LHLHL":
            continue
        lows = [seq[0][2], seq[2][2], seq[4][2]]
        peaks = [seq[1][2], seq[3][2]]
        if (max(lows) - min(lows)) / min(lows) > p["eq"]:
            continue
        if (min(peaks) - max(lows)) / max(lows) < p["depth"]:
            continue
        if not p["gap"][0] <= seq[4][0] - seq[0][0] <= p["gap"][1]:
            continue
        lvl = max(peaks)
        brk = breakout(b, seq[4][0], lvl, p["confirm"], min(lows) * (1 - p["eq"]))
        if brk is None:
            continue
        out.append(ev(b, "tb", seq[0][0], brk, [(x[0], x[2]) for x in seq], [(seq[0][0], lvl, brk, lvl)]))
    return dedupe(out)


def find_ihs(b: Bars, p: dict) -> list[dict]:
    """Перевёрнутая «голова и плечи»: средний минимум ниже плеч, пробой линии шеи через две вершины."""
    out = []
    piv = b.piv[p["zz"]]
    for k in range(len(piv) - 4):
        seq = piv[k:k + 5]
        if "".join(x[1] for x in seq) != "LHLHL":
            continue
        (is1, _, s1), (ih1, _, h1), (ihd, _, hd), (ih2, _, h2), (is2, _, s2) = seq
        if hd > min(s1, s2) * (1 - p["head"]):                     # голова должна быть ниже плеч
            continue
        if abs(s1 - s2) / min(s1, s2) > p["sh_tol"]:               # плечи примерно равны
            continue
        d1, d2 = ihd - is1, is2 - ihd
        if min(d1, d2) < 5 or max(d1, d2) / max(1, min(d1, d2)) > p["sym"]:
            continue
        if abs(h2 - h1) / min(h1, h2) > p["slope"]:                # линия шеи не слишком кривая
            continue
        if not p["gap"][0] <= is2 - is1 <= p["gap"][1]:
            continue
        sl = (h2 - h1) / max(1, ih2 - ih1)

        def line(j, _h1=h1, _sl=sl, _i=ih1):
            return _h1 + _sl * (j - _i)

        brk = breakout(b, is2, line, p["confirm"], hd * 0.99)
        if brk is None:
            continue
        out.append(ev(b, "ihs", is1, brk, [(is1, s1), (ih1, h1), (ihd, hd), (ih2, h2), (is2, s2)],
                      [(ih1, h1, brk, float(line(brk)))],
                      f"плечи ±{abs(s1 - s2) / min(s1, s2) * 100:.0f}%, голова ниже на "
                      f"{(min(s1, s2) - hd) / min(s1, s2) * 100:.0f}%"))
    return dedupe(out)


def find_htf(b: Bars, p: dict) -> list[dict]:
    """High and tight flag: рост не меньше gain за 1–2 месяца, узкая консолидация и пробой её вершины."""
    out = []
    lo_d, hi_d = p["pole"]
    for top in range(lo_d, b.n):
        if b.h[top] < np.max(b.h[max(0, top - 5):top + 1]):        # вершина древка — локальный максимум
            continue
        w0, w1 = max(0, top - hi_d), top - lo_d + 1
        if w1 <= w0:
            continue
        base = int(np.argmin(b.l[w0:w1])) + w0
        gain = (b.h[top] - b.l[base]) / b.l[base]
        if gain < p["gain"]:
            continue
        pole = b.h[top] - b.l[base]
        for j in range(top + p["flag"][0], min(b.n, top + p["flag"][1] + 1)):
            if b.c[j] > b.h[top]:                                  # пробой вершины древка закрытием
                flag_low = float(np.min(b.l[top + 1:j])) if j > top + 1 else b.l[top]
                if (b.h[top] - flag_low) / pole > p["retrace"]:
                    break
                out.append(ev(b, "htf", base, j, [(base, b.l[base]), (top, b.h[top])],
                              [(top, b.h[top], j, b.h[top]), (base, b.l[base], top, b.h[top])],
                              f"рост {gain * 100:.0f}% за {top - base} дней"))
                break
            if b.l[j] < b.h[top] - p["retrace"] * pole:            # консолидация оказалась глубокой — не флаг
                break
    return dedupe(out, gap=10)


def find_tri(b: Bars, p: dict) -> list[dict]:
    """Восходящий треугольник: две и больше вершины на одном уровне, растущие минимумы, пробой уровня."""
    out = []
    piv = b.piv[p["zz"]]
    highs = [(i, v) for i, k, v in piv if k == "H"]
    lows = [(i, v) for i, k, v in piv if k == "L"]
    for a in range(len(highs) - 1):
        for z in range(a + 1, len(highs)):
            ia, va = highs[a]
            iz, vz = highs[z]
            if not p["dur"][0] <= iz - ia <= p["dur"][1]:
                continue
            if abs(vz - va) / va > p["top"]:
                continue
            tops = [(i, v) for i, v in highs[a:z + 1] if abs(v - va) / va <= p["top"]]
            if len(tops) < p["touch"]:
                continue
            lvl = float(np.mean([v for _, v in tops]))
            ins = [(i, v) for i, v in lows if ia <= i <= iz]
            if len(ins) < p["touch"]:
                continue
            if any(ins[k + 1][1] <= ins[k][1] * (1 + p["rise"]) for k in range(len(ins) - 1)):
                continue                                           # минимумы должны расти
            if ins[-1][1] > lvl * (1 - p["rise"]):
                continue
            brk = breakout(b, iz, lvl * 1.002, p["confirm"], ins[0][1])
            if brk is None:
                continue
            spread = (max(v for _, v in tops) - min(v for _, v in tops)) / lvl * 100
            out.append(ev(b, "tri", ia, brk, tops + ins, [(ia, lvl, brk, lvl), (ins[0][0], ins[0][1], ins[-1][0], ins[-1][1])],
                          f"касаний сверху {len(tops)}, снизу {len(ins)}, вершина ±{spread:.1f}%"))
    return dedupe(out, gap=10)


def find_barr(b: Bars, p: dict) -> list[dict]:
    """Bump-and-run reversal: спокойный наклон по вершинам, резкий провал глубже 2 «высот» наклона, возврат выше линии."""
    out = []
    lead, step = p["lead"], 5
    for s in range(0, max(0, b.n - lead - 20), step):
        x = np.arange(lead, dtype=float)
        y = b.h[s:s + lead]
        xm, ym = float(np.mean(x)), float(np.mean(y))              # прямая по вершинам, метод наименьших квадратов
        sl = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
        ic = ym - sl * xm
        base = ym
        if abs(sl * lead) / base > p["drift"]:                     # наклон должен быть спокойным
            continue
        if p.get("down") and sl >= 0:                              # «разворот вниз → вверх»: наклон только вниз
            continue

        def line(j, _ic=ic, _sl=sl, _s=s):
            return _ic + _sl * (j - _s)

        hgt = float(np.mean(np.clip(line(np.arange(s, s + lead)) - b.l[s:s + lead], 0, None)))
        if hgt <= 0:
            continue
        deep, deep_i = 0.0, -1
        for j in range(s + lead, min(b.n, s + lead + p["bump"])):
            gap = float(line(j)) - b.l[j]
            if gap > deep and j - (s + lead) <= p["fast"]:         # провал должен быть и глубоким, и быстрым
                deep, deep_i = gap, j
            if deep >= p["mult"] * hgt and b.c[j] > line(j) and j > deep_i:
                out.append(ev(b, "barr", s, j, [(s, float(b.h[s])), (deep_i, float(b.l[deep_i]))],
                              [(s, float(line(s)), j, float(line(j)))],
                              f"провал {deep / hgt:.1f} высоты наклона"))
                break
            if gap < -p["mult"] * hgt:                             # ушли вверх — это уже другая картина
                break
    return dedupe(out, gap=20)


FIND = {"db": find_db, "tb": find_tb, "ihs": find_ihs, "htf": find_htf, "tri": find_tri, "barr": find_barr}

# Пороги: «строгий» — как в гипотезе и у Булковски, дальше ослабления (шаг 4 плана: фигур должно хватать).
TIERS: dict[str, list[tuple[str, dict]]] = {
    "htf": [("строгий: рост 90% за 1–2 месяца", dict(gain=0.90, pole=(20, 45), flag=(5, 25), retrace=0.25)),
            ("рост 70%", dict(gain=0.70, pole=(20, 45), flag=(5, 25), retrace=0.25)),
            ("рост 60%", dict(gain=0.60, pole=(20, 45), flag=(5, 25), retrace=0.25)),
            ("рост 50%", dict(gain=0.50, pole=(20, 50), flag=(5, 30), retrace=0.25)),
            ("рост 40%", dict(gain=0.40, pole=(20, 50), flag=(5, 30), retrace=0.30)),
            ("рост 25%", dict(gain=0.25, pole=(15, 50), flag=(4, 30), retrace=0.35))],
    "ihs": [("строгий: плечи ±8%, голова ниже на 3%", dict(zz=0.05, head=0.03, sh_tol=0.08, sym=2.0, slope=0.05, gap=(20, 250), confirm=60)),
            ("плечи ±15%", dict(zz=0.05, head=0.02, sh_tol=0.15, sym=3.0, slope=0.08, gap=(15, 300), confirm=90)),
            ("мягкий: плечи ±25%, зигзаг 3%", dict(zz=0.03, head=0.015, sh_tol=0.25, sym=4.0, slope=0.12, gap=(10, 400), confirm=120))],
    "db": [("строгий: дно ±3%, глубина 15%", dict(zz=0.05, eq=0.03, depth=0.15, gap=(15, 250), confirm=60, round=5)),
           ("дно ±3%, глубина 10%", dict(zz=0.05, eq=0.03, depth=0.10, gap=(15, 300), confirm=90, round=5)),
           ("дно ±4%, глубина 10%", dict(zz=0.05, eq=0.04, depth=0.10, gap=(10, 300), confirm=90, round=5)),
           ("дно ±5%, глубина 10%", dict(zz=0.05, eq=0.05, depth=0.10, gap=(10, 300), confirm=90, round=5)),
           ("мягкий: дно ±8%, глубина 7%", dict(zz=0.03, eq=0.08, depth=0.07, gap=(8, 400), confirm=120, round=5))],
    "tb": [("строгий: дно ±4%, глубина 12%", dict(zz=0.05, eq=0.04, depth=0.12, gap=(25, 300), confirm=60)),
           ("дно ±7%, глубина 8%", dict(zz=0.05, eq=0.07, depth=0.08, gap=(20, 350), confirm=90)),
           ("мягкий: дно ±10%, глубина 6%", dict(zz=0.03, eq=0.10, depth=0.06, gap=(15, 450), confirm=120))],
    "barr": [("строгий: наклон вниз, провал 3 высоты", dict(lead=40, drift=0.07, down=True, bump=60, mult=3.0, fast=30)),
             ("наклон вниз, провал 2.5 высоты", dict(lead=40, drift=0.07, down=True, bump=80, mult=2.5, fast=40)),
             ("провал 3 высоты за 30 дней", dict(lead=40, drift=0.07, bump=60, mult=3.0, fast=30)),
             ("провал 2.5 высоты за 40 дней", dict(lead=40, drift=0.07, bump=80, mult=2.5, fast=40)),
             ("провал 2 высоты за 60 дней", dict(lead=40, drift=0.07, bump=90, mult=2.0, fast=60)),
             ("мягкий: провал 1.7 высоты", dict(lead=40, drift=0.12, bump=120, mult=1.7, fast=90))],
    "tri": [("строгий: вершина ±2%, 3 касания", dict(zz=0.05, top=0.02, rise=0.02, dur=(25, 120), touch=3, confirm=30)),
            ("вершина ±3%, 3 касания", dict(zz=0.03, top=0.03, rise=0.015, dur=(20, 150), touch=3, confirm=40)),
            ("вершина ±3%, 2 касания", dict(zz=0.03, top=0.03, rise=0.015, dur=(20, 150), touch=2, confirm=40)),
            ("вершина ±4%, 2 касания", dict(zz=0.03, top=0.04, rise=0.01, dur=(20, 180), touch=2, confirm=50)),
            ("мягкий: вершина ±5%", dict(zz=0.03, top=0.05, rise=0.01, dur=(15, 200), touch=2, confirm=60))],
}


# ─────────────────────────── разметка и отбор ───────────────────────────


def mark(bars: dict[str, Bars], pat: str, params: dict) -> pd.DataFrame:
    rows = []
    for b in bars.values():
        rows += FIND[pat](b, params)
    d = pd.DataFrame(rows)
    if not d.empty:
        d["year"] = d["date"].dt.year
        d["part"] = np.where(d["date"] < SPLIT, "выбор", "проверка")
    return d


def pick_tier(bars: dict[str, Bars]) -> tuple[dict, pd.DataFrame]:
    """Шаг 4 плана: по выбору (до 2021) берём самый строгий порог, на котором событий не меньше 50."""
    chosen, rows = {}, []
    for pat, tiers in TIERS.items():
        marks = []
        for name, params in tiers:
            d = mark(bars, pat, params)
            n_sel = 0 if d.empty else int((d["part"] == "выбор").sum())
            n_all = 0 if d.empty else len(d)
            marks.append((name, params, n_sel, n_all, d))
            rows.append({"pat": pat, "tier": name, "выбор": n_sel, "всего": n_all})
        ok = [m for m in marks if m[2] >= MIN_EVENTS]
        name, params, n_sel, n_all, d = ok[0] if ok else marks[-1]
        chosen[pat] = {"tier": name, "params": params, "n_sel": n_sel, "n_all": n_all, "enough": bool(ok), "events": d}
        print(f"{pat}: {name} — выбор {n_sel}, всего {n_all}{'' if ok else '  (50 событий не набралось ни на одном пороге)'}",
              flush=True)
    return chosen, pd.DataFrame(rows)


# ─────────────────────── ход цены после пробоя ───────────────────────


def forward(events: pd.DataFrame, hours: dict[str, pd.DataFrame], horizons=(6, 24, 72)) -> pd.DataFrame:
    """Ход цены в ATR от первого часа после дня пробоя; рядом — тот же ход от случайного («среднего») часа."""
    rows = []
    for t, g in events.groupby("ticker"):
        h = hours[t]
        c = h["close"].to_numpy(float)
        tr = pd.concat([h["high"] - h["low"], (h["high"] - h["close"].shift()).abs(),
                        (h["low"] - h["close"].shift()).abs()], axis=1).max(axis=1)
        a = tr.rolling(24, min_periods=12).mean().to_numpy(float)
        idx = h.index
        for e in g.itertuples():
            k = int(idx.searchsorted(e.date + pd.Timedelta(days=1)))      # первый час следующего дня
            if k >= len(idx) - max(horizons) or not np.isfinite(a[k]) or a[k] <= 0:
                continue
            r = {"ticker": t, "pat": e.pat, "date": e.date, "part": e.part, "year": e.year}
            for q in horizons:
                r[f"h{q}"] = (c[k + q] - c[k]) / a[k]
            rows.append(r)
    return pd.DataFrame(rows)


def baseline(hours: dict[str, pd.DataFrame], horizons=(6, 24, 72)) -> pd.DataFrame:
    """«Средний час»: тот же ход в ATR из каждого часа истории — к чему сравнивать события."""
    rows = []
    for t, h in hours.items():
        c = h["close"].to_numpy(float)
        tr = pd.concat([h["high"] - h["low"], (h["high"] - h["close"].shift()).abs(),
                        (h["low"] - h["close"].shift()).abs()], axis=1).max(axis=1)
        a = tr.rolling(24, min_periods=12).mean().to_numpy(float)
        n = len(c) - max(horizons) - 1
        ok = np.isfinite(a[:n]) & (a[:n] > 0)
        part = np.where(h.index[:n] < SPLIT, "выбор", "проверка")
        for q in horizons:
            mv = (c[np.arange(n) + q] - c[:n]) / a[:n]
            for pt in ("выбор", "проверка"):
                m = ok & (part == pt)
                rows.append({"ticker": t, "part": pt, "h": q, "mean": float(np.mean(mv[m])), "n": int(m.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = MAIN if a.quick else TICKERS

    hours, bars = {}, {}
    for t in tickers:
        h = hourly(t)
        hours[t] = h
        d = to_daily(h)
        bars[t] = Bars(t, d)
        print(f"{t}: часов {len(h)}, дней {len(d)} ({d.index[0].date()} … {d.index[-1].date()})", flush=True)

    chosen, relax = pick_tier(bars)
    relax.to_csv(REP / "relax.csv", index=False)

    evs = pd.concat([c["events"].assign(tier=c["tier"]) for c in chosen.values() if not c["events"].empty])
    evs = evs.sort_values(["pat", "ticker", "date"]).reset_index(drop=True)
    evs.drop(columns=["pts", "lines"]).to_csv(REP / "events.csv", index=False)
    print(f"событий всего {len(evs)}: выбор {int((evs['part'] == 'выбор').sum())}, "
          f"проверка {int((evs['part'] == 'проверка').sum())}", flush=True)

    cnt = evs.pivot_table(index="pat", columns="part", values="date", aggfunc="count").fillna(0).astype(int)
    cnt.to_csv(REP / "counts.csv")
    by_year = evs.pivot_table(index="year", columns="pat", values="date", aggfunc="count").fillna(0).astype(int)
    by_year.to_csv(REP / "counts_year.csv")
    by_tick = evs.pivot_table(index="ticker", columns="pat", values="date", aggfunc="count").fillna(0).astype(int)
    by_tick.to_csv(REP / "counts_ticker.csv")

    fwd = forward(evs, hours)
    fwd.to_csv(REP / "forward.csv", index=False)
    base = baseline(hours)
    base.to_csv(REP / "baseline.csv", index=False)

    # наложения: как часто фигуры разных видов дают пробой в одни и те же дни
    ov = []
    for (t, _), g in evs.groupby(["ticker", "part"]):
        for e in g.itertuples():
            near = g[(g["pat"] != e.pat) & ((g["date"] - e.date).abs() <= pd.Timedelta(days=7))]
            ov.append({"ticker": t, "pat": e.pat, "date": e.date, "others": len(near)})
    ovd = pd.DataFrame(ov)
    ovd.to_csv(REP / "overlap.csv", index=False)

    meta = {"tickers": tickers, "min_events": MIN_EVENTS, "split": str(SPLIT.date()),
            "chosen": {p: {"tier": c["tier"], "params": {k: list(v) if isinstance(v, tuple) else v
                                                         for k, v in c["params"].items()},
                           "n_sel": c["n_sel"], "n_all": c["n_all"], "enough": c["enough"]}
                       for p, c in chosen.items()},
            "overlap_share": {p: round(float((g["others"] > 0).mean()), 3) for p, g in ovd.groupby("pat")} if not ovd.empty else {}}
    (REP / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    evs.to_pickle(REP / "events.pkl")
    print("готово: reports/h74/{events.csv,counts.csv,forward.csv,relax.csv,meta.json}", flush=True)


if __name__ == "__main__":
    main()
