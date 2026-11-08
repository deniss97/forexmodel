"""H40 по коду Тимура (S_HorizonVolume.ipynb, прикреплён к H40 в банке): профиль объёма по уровням цены с затуханием старых
сделок и его отметки как признаки. Отчёт — docs/results/h40_profile.md.

Данные — секундные агрегаты ленты LKOH (TQBR) и фьючерсов серебра SV* (в каждый день — самый торгуемый контракт), 2024–2026,
время МСК. Уровни — логарифмическая сетка с шагом 0.1% цены (у LKOH ≈ 6 ₽). Профиль на конец каждого часа:
  * с затуханием (как halflife у Тимура): вес объёма падает вдвое за 24 или 120 торговых часов;
  * без затухания — объём последних 70 часов (как профиль в признаках ленты L02).
Отметки — как build_volume_profile / nearest_levels у Тимура: POC, зона стоимости 70% (расширение от POC в сторону большего
соседнего уровня), HVN — локальные максимумы сглаженного профиля не меньше 25% POC, LVN — цена в локальном минимуме.
Признаки (в ATR(14 ч)): расстояние от цены до POC, место в зоне стоимости (0 — низ, 1 — верх), расстояние до ближайшего HVN
сверху и снизу, флаг «цена в LVN». Цель — ход за 10 ч в ATR (для LVN — его модуль). У серебра часы, после которых в
пределах 10 ч сменился основной контракт, не учитываются.
Критерий (банк, до расчёта): признак профиля с затуханием даёт |IC| ≥ 0.03 одного знака в обеих половинах данных на обоих
инструментах и больше по модулю, чем тот же признак профиля без затухания.

    python scripts/h40_profile.py
"""
from __future__ import annotations

import gc

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from experiments import ROOT

DOC = ROOT / "docs" / "results" / "h40_profile.md"
SRC = {"lkoh": "orderflow_lkoh_1s.parquet", "silver": "orderflow_spbfut_1s.parquet"}
STEP = np.log(1.001)
H = 10
HALF = {"затухание 24 ч": 24, "затухание 120 ч": 120}
ROLL = 70
FEATS = {"poc": "до POC", "va": "место в зоне стоимости", "hvn_up": "до HVN сверху", "hvn_dn": "до HVN снизу", "lvn": "цена в LVN"}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def load(inst: str):
    """Часы: OHLC, контракт; объём по уровням (час × уровень)."""
    pf = pq.ParquetFile(ROOT / "data" / "raw" / SRC[inst])
    front = None
    if inst == "silver":
        parts = []
        for i in range(pf.num_row_groups):
            d = pf.read_row_group(i, columns=["TICKER_CC", "time", "volume"]).to_pandas()
            d["day"] = (pd.to_datetime(d["time"]).dt.tz_convert(None) + pd.Timedelta(hours=3)).dt.normalize()
            parts.append(d.groupby(["day", "TICKER_CC"])["volume"].sum())
        dv = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
        front = dv.sort_values("volume").groupby("day").tail(1).set_index("day")["TICKER_CC"]
    hours, lv = [], []
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["TICKER_CC", "time", "price", "vwap", "volume"]).to_pandas()
        d["time"] = pd.to_datetime(d["time"]).dt.tz_convert(None) + pd.Timedelta(hours=3)
        d = d[d["volume"] > 0]
        if front is not None:
            d = d[d["TICKER_CC"].to_numpy() == d["time"].dt.normalize().map(front).to_numpy()]
        d = d.sort_values("time")
        d["hour"] = d["time"].dt.floor("1h")
        g = d.groupby("hour")
        hours.append(pd.DataFrame({"t0": g["time"].min(), "o": g["price"].first(), "h": g["price"].max(), "l": g["price"].min(),
                                   "c": g["price"].last(), "tk": g["TICKER_CC"].last()}))
        px = d["vwap"].where(d["vwap"] > 0, d["price"]).astype(float)
        d["b"] = np.floor(np.log(px) / STEP).astype(np.int64)
        lv.append(d.groupby(["hour", "b"])["volume"].sum())
        del d, g, px
        gc.collect()
    Hh = pd.concat(hours).rename_axis("hour").reset_index().sort_values(["hour", "t0"])
    Hh = Hh.groupby("hour").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), tk=("tk", "last"))
    L = pd.concat(lv).groupby(level=[0, 1]).sum()
    return Hh, L


def matrix(Hh: pd.DataFrame, L: pd.Series) -> tuple[np.ndarray, int]:
    b = L.index.get_level_values(1)
    lo, hi = int(b.min()), int(b.max())
    M = np.zeros((len(Hh), hi - lo + 1), dtype=np.float32)
    pos = pd.Series(np.arange(len(Hh)), index=Hh.index)
    M[pos.loc[L.index.get_level_values(0)].to_numpy(), (b - lo).to_numpy()] = L.to_numpy()
    return M, lo


def marks(P: np.ndarray, cbin: int) -> tuple[float, float, float, float, float]:
    """Отметки профиля для цены в уровне cbin: (POC−цена, место в VA, HVN сверху, HVN снизу, цена в LVN) в уровнях."""
    tot = P.sum()
    if tot <= 0:
        return (np.nan,) * 5
    pi = int(P.argmax())
    lo = hi = pi
    acc, n = P[pi], len(P)
    while acc < 0.7 * tot and (lo > 0 or hi < n - 1):
        dn = P[lo - 1] if lo > 0 else -1.0
        up = P[hi + 1] if hi < n - 1 else -1.0
        if dn >= up:
            lo -= 1; acc += P[lo]
        else:
            hi += 1; acc += P[hi]
    S = np.convolve(P, np.ones(3) / 3, mode="same")
    pk = np.flatnonzero((S[1:-1] >= S[:-2]) & (S[1:-1] >= S[2:]) & (S[1:-1] >= 0.25 * S.max())) + 1
    up_ = pk[pk > cbin]
    dn_ = pk[pk < cbin]
    c = min(max(cbin, 1), n - 2)
    lvn = float(S[c] < S[c - 1] and S[c] < S[c + 1])
    return (pi - cbin, (cbin - lo) / (hi - lo) if hi > lo else np.nan,
            (up_.min() - cbin) if len(up_) else np.nan, (cbin - dn_.max()) if len(dn_) else np.nan, lvn)


def features(Hh: pd.DataFrame, M: np.ndarray, lo: int, inst: str) -> pd.DataFrame:
    c = Hh["c"].to_numpy(float)
    tr = np.maximum(Hh["h"] - Hh["l"], np.maximum((Hh["h"] - Hh["c"].shift()).abs(), (Hh["l"] - Hh["c"].shift()).abs()))
    atr = tr.rolling(14).mean().to_numpy()
    atr_bins = atr / c / STEP                                  # ATR в уровнях сетки (уровень = 0.1% цены)
    cbin = (np.floor(np.log(c) / STEP).astype(np.int64) - lo).clip(0, M.shape[1] - 1)
    out = pd.DataFrame(index=Hh.index)
    out["fwd"] = (Hh["c"].shift(-H) - Hh["c"]).to_numpy() / atr
    if inst == "silver":                                       # смена основного контракта в пределах 10 ч — не считаем
        same = (Hh["tk"].shift(-H) == Hh["tk"]).to_numpy()
        out.loc[~same, "fwd"] = np.nan
    C = np.cumsum(M, axis=0, dtype=np.float64)
    for name in [*HALF, "без затухания (70 ч)"]:
        rows = np.full((len(Hh), 5), np.nan)
        if name in HALF:
            lam = 0.5 ** (1.0 / HALF[name])
            P = np.zeros(M.shape[1], dtype=np.float64)
        for t in range(len(Hh)):
            if name in HALF:
                P = P * lam + M[t]
                prof = P
            else:
                prof = C[t] - (C[t - ROLL] if t >= ROLL else 0.0)
            if t < ROLL:
                continue
            rows[t] = marks(prof, int(cbin[t]))
        for k, col in enumerate(FEATS):
            v = rows[:, k]
            out[f"{col}|{name}"] = v if col in ("va", "lvn") else v / atr_bins
    del C
    return out.iloc[ROLL:]


def main() -> int:
    R, info = [], {}
    for inst in SRC:
        Hh, L = load(inst)
        M, lo = matrix(Hh, L)
        del L
        gc.collect()
        F = features(Hh, M, lo, inst)
        del M
        cut = F.index[len(F) // 2]
        info[inst] = {"с": F.index.min().date(), "по": F.index.max().date(), "часов": len(F), "раздел": cut.date()}
        for half, part in (("1-я половина", F[F.index < cut]), ("2-я половина", F[F.index >= cut]), ("все", F)):
            for col in [c for c in F.columns if "|" in c]:
                feat, prof = col.split("|")
                y = part["fwd"].abs() if feat == "lvn" else part["fwd"]
                m = part[col].notna() & y.notna() & np.isfinite(part[col])
                r = spearmanr(part.loc[m, col], y[m]).correlation if m.sum() > 200 else np.nan
                R.append({"инструмент": inst, "часть": half, "признак": FEATS[feat], "профиль": prof, "IC": float(r), "часов": int(m.sum())})
        del F, Hh
        gc.collect()
    T = pd.DataFrame(R)
    pd.set_option("display.width", 250)
    print(info, "\n", T[T["часть"] != "все"].round(3).to_string())
    halves = ("1-я половина", "2-я половина")
    g = T.set_index(["инструмент", "часть", "признак", "профиль"])["IC"]
    passed = []
    for feat in FEATS.values():
        for prof in HALF:
            v = [g[(i, h, feat, prof)] for i in SRC for h in halves]
            base = [g[(i, h, feat, "без затухания (70 ч)")] for i in SRC for h in halves]
            if all(abs(x) >= 0.03 for x in v) and len({np.sign(x) for x in v}) == 1 and all(abs(x) > abs(b) for x, b in zip(v, base)):
                passed.append(f"{feat} ({prof})")
    ok = bool(passed)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Профиль объёма по уровням с затуханием, по коду Тимура (H40)", "",
          "Код Тимура — S_HorizonVolume.ipynb (прикреплён к H40 в банке): профиль объёма по уровням цены, где старые сделки весят "
          "меньше (halflife — «время жизни позиции»), POC, зона стоимости 70%, узлы HVN и провалы LVN и расстояния до ближайших "
          "как признаки для модели. Данные — лента LKOH (TQBR) и фьючерсов серебра (основной контракт дня), 2024–2026, уровни — "
          "сетка 0.1% цены. Расчёт — `scripts/h40_profile.py`.", "",
          "**Критерий (банк, до расчёта):** признак профиля с затуханием (24 или 120 ч) даёт |IC| ≥ 0.03 одного знака в обеих "
          "половинах данных на обоих инструментах и больше по модулю, чем у профиля без затухания (70 ч, как в L02); только тогда — "
          "модель LKOH.", "", verdict, "",
          "Данные: " + "; ".join(f"{i}: {v['с']} — {v['по']}, {v['часов']} часов, раздел половин {v['раздел']}" for i, v in info.items()) + ".", "",
          "## IC с ходом за 10 часов (× 100; для «цена в LVN» — с модулем хода)", "",
          "| признак | профиль | LKOH 1-я | LKOH 2-я | серебро 1-я | серебро 2-я |", "|---|---|---|---|---|---|"]
    for feat in FEATS.values():
        for prof in [*HALF, "без затухания (70 ч)"]:
            vals = [g.get((i, h, feat, prof), np.nan) for i in SRC for h in halves]
            md.append(f"| {feat} | {prof} | " + " | ".join(f"{x * 100:+.1f}" for x in vals) + " |")
    md += ["", f"Прошли критерий: {', '.join(passed) if passed else 'нет'}.", "", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, passed, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
