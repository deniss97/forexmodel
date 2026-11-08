"""Скептик к H40 и H42 (docs/results/skeptic_2026-11-08.md): варианты, предложенные скептиком по ленте сделок. Отчёт —
docs/results/skeptic_orderflow.md.

H40, исправленный (признак объявлен заранее — «расстояние до ближайшего HVN сверху», основной профиль — затухание 120 ч):
  * LKOH: профиль сдвигается вниз на дивиденд в момент отсечки (data/dividends/moex_dividends.csv), часы, в чьи 10 ч
    попадает отсечка, из цели исключены;
  * серебро: профиль сбрасывается при смене основного контракта (не тянет уровни старого);
  * нет узла сверху — расстояние до верхнего края профиля (не больше 5 ATR), час не выпадает.
  Прошло, если признак отрицателен в обеих половинах на обоих инструментах, общий t ≤ −2.5 на LKOH и ≤ −1.5 на серебре.
H42, «остаточная дельта»: крупная дельта за 6 и 24 ч (крупная секунда — объём выше среднего 95-го процентиля прошлых 5 дней)
против хода через 10 и 24 ч при контроле хода цены за то же окно (частная ранговая корреляция); серебро — по цене своего
основного контракта, часы со сменой контракта в окне исключены. Прошло, если на LKOH знак один в обеих половинах и общий
|t| ≥ 2.5 хотя бы в одной из 4 комбинаций, а на серебре в той же комбинации знак тот же и |t| ≥ 1.5.
t = IC·√(n / (h + w − 1)) — с поправкой на перекрытие окон.

    python scripts/skeptic_orderflow.py
"""
from __future__ import annotations

import gc

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

import h40_profile as P
from experiments import ROOT

DOC = ROOT / "docs" / "results" / "skeptic_orderflow.md"
DIV = ROOT / "data" / "dividends" / "moex_dividends.csv"
CAP = 5.0


def pic(x, y, ctrl=None) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    if ctrl is not None:
        m &= np.isfinite(ctrl)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if ctrl is not None:
        C = np.column_stack([np.ones(m.sum()), rankdata(ctrl[m])])
        rx = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
        ry = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def atr14(H: pd.DataFrame) -> np.ndarray:
    tr = np.maximum(H["h"] - H["l"], np.maximum((H["h"] - H["c"].shift()).abs(), (H["l"] - H["c"].shift()).abs()))
    return tr.rolling(14).mean().to_numpy()


def bad_target(H: pd.DataFrame, inst: str, h: int) -> np.ndarray:
    """Часы, чья цель (ход за h ч) проходит через отсечку дивиденда (LKOH) или смену контракта (серебро)."""
    n = len(H)
    bad = np.zeros(n, bool)
    if inst == "silver":
        tk = H["tk"].to_numpy()
        bad[:-h] = tk[h:] != tk[:-h]
        bad[-h:] = True
    else:
        d = pd.read_csv(DIV, parse_dates=["ex_time"])
        ex = d.loc[d["ticker"] == "LKOH", "ex_time"].to_numpy()
        idx = H.index.to_numpy()
        for e in ex:
            k = np.searchsorted(idx, e)                       # первый час после отсечки
            bad[max(0, k - h):k] = True
    return bad


# ---------------- H40 ----------------
def h40(inst: str) -> list[dict]:
    H, L = P.load(inst)
    M, lo = P.matrix(H, L)
    del L
    gc.collect()
    n, nb = M.shape
    c = H["c"].to_numpy(float)
    atr = atr14(H)
    atr_b = atr / c / P.STEP
    cb = (np.floor(np.log(c) / P.STEP).astype(np.int64) - lo).clip(0, nb - 1)
    shift = np.zeros(n, dtype=np.int64)                         # сдвиг профиля вниз в этот час (отсечка), в уровнях
    reset = np.zeros(n, bool)
    if inst == "lkoh":
        d = pd.read_csv(DIV, parse_dates=["ex_time"])
        d = d[d["ticker"] == "LKOH"]
        idx = H.index.to_numpy()
        for e, amt in zip(d["ex_time"].to_numpy(), d["amount"].to_numpy(float)):
            k = np.searchsorted(idx, e)
            if 0 < k < n:
                shift[k] = int(round(-np.log(1 - amt / c[k - 1]) / P.STEP))
    else:
        tk = H["tk"].to_numpy()
        reset[1:] = tk[1:] != tk[:-1]
    fwd = (H["c"].shift(-10) - H["c"]).to_numpy() / atr
    fwd[bad_target(H, inst, 10)] = np.nan
    out = {}
    for name, half in (("затухание 120 ч", 120), ("затухание 24 ч", 24), ("окно 70 ч", None)):
        prof = np.zeros(nb)
        rows = []                                                 # для окна: последние 70 строк с их сдвигами
        feat = np.full(n, np.nan)
        lam = 0.5 ** (1 / half) if half else None
        for t in range(n):
            if reset[t]:
                prof[:] = 0
                rows = []
            if shift[t]:
                s = shift[t]
                prof = np.concatenate([prof[s:], np.zeros(s)])
                rows = [np.concatenate([r[s:], np.zeros(s)]) for r in rows]
            if half:
                prof = prof * lam + M[t]
                cur = prof
            else:
                rows.append(M[t].astype(np.float64))
                if len(rows) > 70:
                    rows.pop(0)
                cur = np.sum(rows, axis=0)
            if t < 70 or not np.isfinite(atr_b[t]):
                continue
            S = np.convolve(cur, np.ones(3) / 3, mode="same")
            if S.max() <= 0:
                continue
            pk = np.flatnonzero((S[1:-1] >= S[:-2]) & (S[1:-1] >= S[2:]) & (S[1:-1] >= 0.25 * S.max())) + 1
            up = pk[pk > cb[t]]
            if len(up):
                dist = (up.min() - cb[t]) / atr_b[t]
            else:
                nz = np.flatnonzero(cur > 0)
                dist = (nz.max() - cb[t]) / atr_b[t] if len(nz) else np.nan
            feat[t] = min(dist, CAP) if np.isfinite(dist) else np.nan
        out[name] = feat
    del M
    gc.collect()
    cut = H.index[len(H) // 2]
    res = []
    for name, f in out.items():
        for half_name, m in (("1-я половина", H.index < cut), ("2-я половина", H.index >= cut), ("все", np.ones(n, bool))):
            ic, k = pic(f[m], fwd[m])
            res.append({"инструмент": inst, "профиль": name, "часть": half_name, "IC": ic, "часов": k, "t": ic * np.sqrt(k / 10)})
    return res


# ---------------- H42 ----------------
def hours_large(inst: str) -> pd.DataFrame:
    pf = pq.ParquetFile(ROOT / "data" / "raw" / P.SRC[inst])
    front = None
    if inst == "silver":
        parts = []
        for i in range(pf.num_row_groups):
            d = pf.read_row_group(i, columns=["TICKER_CC", "time", "volume"]).to_pandas()
            d["day"] = (pd.to_datetime(d["time"]).dt.tz_convert(None) + pd.Timedelta(hours=3)).dt.normalize()
            parts.append(d.groupby(["day", "TICKER_CC"])["volume"].sum())
        dv = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
        front = dv.sort_values("volume").groupby("day").tail(1).set_index("day")["TICKER_CC"]
    def chunk(i):
        d = pf.read_row_group(i, columns=["TICKER_CC", "time", "price", "volume", "signed_vol"]).to_pandas()
        d["time"] = pd.to_datetime(d["time"]).dt.tz_convert(None) + pd.Timedelta(hours=3)
        d = d[d["volume"] > 0]
        d["day"] = d["time"].dt.normalize()
        if front is not None:
            d = d[d["TICKER_CC"].to_numpy() == d["day"].map(front).to_numpy()]
        return d

    vols: dict = {}                                               # проход 1: объёмы секунд по дням → 95-й процентиль дня
    for i in range(pf.num_row_groups):
        d = chunk(i)
        for day, v in d.groupby("day")["volume"]:
            vols.setdefault(day, []).append(v.to_numpy(np.int32))
        del d
    q = pd.Series({k: float(np.quantile(np.concatenate(v), 0.95)) for k, v in vols.items()}).sort_index()
    del vols
    thr_by_day = q.shift(1).rolling(5, min_periods=1).mean()
    parts = []                                                    # проход 2: крупная дельта и часы
    for i in range(pf.num_row_groups):
        d = chunk(i).sort_values("time")
        d["ld"] = np.where(d["volume"] >= d["day"].map(thr_by_day), d["signed_vol"], 0)
        g = d.groupby(d["time"].dt.floor("1h"))
        parts.append(pd.DataFrame({"t0": g["time"].min(), "o": g["price"].first(), "h": g["price"].max(), "l": g["price"].min(),
                                   "c": g["price"].last(), "tk": g["TICKER_CC"].last(), "vol": g["volume"].sum(), "ld": g["ld"].sum()}))
        del d, g
        gc.collect()
    H = pd.concat(parts).rename_axis("hour").reset_index().sort_values(["hour", "t0"])
    H = H.groupby("hour").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), tk=("tk", "last"),
                              vol=("vol", "sum"), ld=("ld", "sum"))
    return H


def h42(inst: str) -> list[dict]:
    H = hours_large(inst)
    atr = atr14(H)
    res = []
    cut = H.index[len(H) // 2]
    tk = H["tk"].to_numpy()
    for w in (6, 24):
        ld = (H["ld"].rolling(w).sum() / H["vol"].rolling(w).sum()).to_numpy().astype(float).copy()
        r = ((H["c"] - H["c"].shift(w)).to_numpy()) / atr
        if inst == "silver":                                       # окно признака через смену контракта — не считаем
            same = np.r_[np.zeros(w, bool), tk[w:] == tk[:-w]]
            ld[~same] = np.nan
        for h in (10, 24):
            fwd = (H["c"].shift(-h) - H["c"]).to_numpy() / atr
            fwd[bad_target(H, inst, h)] = np.nan
            for half_name, m in (("1-я половина", H.index < cut), ("2-я половина", H.index >= cut), ("все", np.ones(len(H), bool))):
                ic, k = pic(ld[m], fwd[m], r[m])
                res.append({"инструмент": inst, "окно, ч": w, "горизонт, ч": h, "часть": half_name, "IC": ic, "часов": k,
                            "t": ic * np.sqrt(k / (h + w - 1))})
    return res


def main() -> int:
    A = pd.DataFrame([r for i in ("lkoh", "silver") for r in h40(i)])
    gc.collect()
    B = pd.DataFrame([r for i in ("lkoh", "silver") for r in h42(i)])
    pd.set_option("display.width", 220)
    print(A.round(3).to_string(), "\n", B.round(3).to_string())
    a = A[A["профиль"] == "затухание 120 ч"].set_index(["инструмент", "часть"])
    ok40 = bool(all(a.loc[(i, hh), "IC"] < 0 for i in ("lkoh", "silver") for hh in ("1-я половина", "2-я половина"))
                and a.loc[("lkoh", "все"), "t"] <= -2.5 and a.loc[("silver", "все"), "t"] <= -1.5)
    b = B.set_index(["инструмент", "окно, ч", "горизонт, ч", "часть"])
    ok42 = []
    for w in (6, 24):
        for h in (10, 24):
            l1, l2 = b.loc[("lkoh", w, h, "1-я половина"), "IC"], b.loc[("lkoh", w, h, "2-я половина"), "IC"]
            lt, st = b.loc[("lkoh", w, h, "все"), "t"], b.loc[("silver", w, h, "все"), "t"]
            if np.sign(l1) == np.sign(l2) and abs(lt) >= 2.5 and np.sign(st) == np.sign(lt) and abs(st) >= 1.5:
                ok42.append(f"окно {w} ч, горизонт {h} ч")
    md = ["# Скептик к ленте сделок: H40 и H42 в вариантах скептика", "",
          "Правило и разбор — `docs/results/skeptic_2026-11-08.md`. Расчёт — `scripts/skeptic_orderflow.py`. IC × 100; t — с поправкой на "
          "перекрытие окон.", "",
          "## H40: «до ближайшего HVN сверху», исправленный профиль", "",
          "LKOH — профиль сдвигается на дивиденд в момент отсечки, серебро — профиль сбрасывается при смене контракта; нет узла сверху — "
          f"расстояние до края профиля (не больше {CAP:g} ATR). Основной профиль (объявлен заранее) — затухание 120 ч.", "",
          "| инструмент | профиль | 1-я половина | 2-я половина | все | t (все) |", "|---|---|---|---|---|---|"]
    for (i, pr), g in A.groupby(["инструмент", "профиль"], sort=False):
        v = g.set_index("часть")
        md.append(f"| {i} | {pr} | {v.loc['1-я половина', 'IC'] * 100:+.1f} | {v.loc['2-я половина', 'IC'] * 100:+.1f} | "
                  f"{v.loc['все', 'IC'] * 100:+.1f} | {v.loc['все', 't']:+.1f} |")
    md += ["", f"Критерий H40 (затухание 120 ч): {'выполнен' if ok40 else 'не выполнен'}.", "",
           "## H42: «остаточная дельта» при контроле хода цены за то же окно", "",
           "| инструмент | окно, ч | горизонт, ч | 1-я половина | 2-я половина | все | t (все) |", "|---|---|---|---|---|---|---|"]
    for (i, w, h), g in B.groupby(["инструмент", "окно, ч", "горизонт, ч"], sort=False):
        v = g.set_index("часть")
        md.append(f"| {i} | {w} | {h} | {v.loc['1-я половина', 'IC'] * 100:+.1f} | {v.loc['2-я половина', 'IC'] * 100:+.1f} | "
                  f"{v.loc['все', 'IC'] * 100:+.1f} | {v.loc['все', 't']:+.1f} |")
    md += ["", f"Критерий H42: {'выполнен — ' + ', '.join(ok42) if ok42 else 'не выполнен'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерии:", ok40, ok42, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
