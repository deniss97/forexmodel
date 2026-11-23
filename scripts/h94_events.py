"""H94 (подготовка, план ждёт подтверждения автора): найти все резкие ходы — цена за 60 или 120 минут ушла на
1.5 / 2 / 3 ATR — на первом эшелоне и серебре, сразу для двух вариантов ATR (вопрос 1 плана: дневной или часовой).

Здесь только поиск случаев и их число по годам — без выводов о продолжении (их считаем после подтверждения плана).

* Цена — закрытие минуты; ход за w минут = закрытие сейчас − последнее закрытие не позже чем w минут назад.
  Если та минута — в прошлом торговом дне, случай помечается cross (ход через ночной разрыв, вопрос 4 плана).
* ATR дневной — средний истинный размах 14 прошлых дней (сегодняшний не входит); часовой — 14 прошлых закрытых часов.
* Эпизод: подряд идущие минуты, где |ход| ≥ 1.5 ATR в одну сторону, с разрывом меньше w минут — один случай; у него
  время начала, пик (наибольший |ход| в ATR) и время пика.

    python scripts/h94_events.py            # → data/derived/h94_events.parquet, docs/results/h94/counts.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "results" / "h94"
DER = ROOT / "data" / "derived"
SRC = {"SBER": RAW / "SBER_1m_15-26.csv", "GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv",
       "MTSS": RAW / "MTSS_1m_15-26.csv", "MOEX": RAW / "MOEX_1m_15-26.csv",
       **{k: RAW / "moex" / f"{k}_1m.csv" for k in ("ROSN", "GMKN", "NVTK", "TATN", "VTBR")},
       "SILV": RAW / "moex" / "SILV_1m_fut.csv"}
WINDOWS = (60, 120)
KS = (1.5, 2.0, 3.0)


def load(tk: str) -> pd.DataFrame:
    src = SRC[tk]
    head = pd.read_csv(src, nrows=1).columns
    tcol = "begin" if "begin" in head else "time"
    d = pd.read_csv(src, usecols=[tcol, "high", "low", "close"], dtype={"high": "float64", "low": "float64", "close": "float64"})
    d[tcol] = pd.to_datetime(d[tcol])
    d = d.drop_duplicates(tcol).set_index(tcol).sort_index()
    d.index.name = "t"
    return d[d.index >= "2015-06-01"]


def atr_daily(d: pd.DataFrame) -> pd.Series:
    day = d.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = day["close"].shift()
    tr = pd.concat([day["high"] - day["low"], (day["high"] - pc).abs(), (day["low"] - pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(14).mean().shift()            # только прошлые дни
    return a.reindex(d.index.normalize()).set_axis(d.index)


def atr_hourly(d: pd.DataFrame) -> pd.Series:
    h = d.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = h["close"].shift()
    tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(), (h["low"] - pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(14).mean()
    a.index = a.index + pd.Timedelta(hours=1)    # ATR часа известен после его закрытия
    return a.reindex(d.index, method="ffill")


def episodes(tk: str, d: pd.DataFrame, atr: pd.Series, kind: str, w: int) -> list[dict]:
    c = d["close"]
    prev_t = d.index - pd.Timedelta(minutes=w)
    pos = c.index.searchsorted(prev_t, side="right") - 1
    ok = pos >= 0
    p0 = np.where(ok, c.to_numpy()[np.clip(pos, 0, None)], np.nan)
    t0 = c.index.to_numpy()[np.clip(pos, 0, None)]
    move = c.to_numpy() - p0
    a = atr.to_numpy()
    r = np.where(a > 0, move / np.where(a > 0, a, 1.0), np.nan)   # нулевой ATR (часы без сделок) — не случай
    cross = pd.DatetimeIndex(t0).normalize() != c.index.normalize()
    out = []
    for sgn in (1, -1):
        hit = np.flatnonzero(np.nan_to_num(r * sgn) >= KS[0])
        if not len(hit):
            continue
        ts = c.index[hit]
        # новый эпизод, если от прошлого срабатывания прошло не меньше w минут
        new = np.r_[True, np.diff(ts.values).astype("timedelta64[m]").astype(np.int64) >= w]   # единица времени у pandas бывает разной
        gid = np.cumsum(new)
        for g in np.unique(gid):
            idx = hit[gid == g]
            k = idx[np.argmax(r[idx] * sgn)]
            out.append({"tk": tk, "atr": kind, "w": w, "dir": sgn, "start": c.index[idx[0]], "peak_t": c.index[k],
                        "peak": float(abs(r[k])), "move_pct": float(move[k] / p0[k] * 100), "atr_pct": float(atr.iloc[k] / c.iloc[k] * 100),
                        "cross": bool(cross[idx[0]])})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tk in (sys.argv[1:] or SRC):
        d = load(tk)
        for kind, atr in (("день", atr_daily(d)), ("час", atr_hourly(d))):
            for w in WINDOWS:
                rows += episodes(tk, d, atr, kind, w)
        print(tk, len(d), "минут, эпизодов всего", sum(1 for x in rows if x["tk"] == tk), flush=True)
        del d
    ev = pd.DataFrame(rows)
    ev["year"] = ev["start"].dt.year
    ev = ev[ev["year"] >= 2016]
    ev.to_parquet(DER / "h94_events.parquet", index=False)
    # число случаев: по порогам, ATR, окну, с ночным разрывом и без
    cnt = []
    for (kind, w), g in ev.groupby(["atr", "w"]):
        for k in KS:
            x = g[g["peak"] >= k]
            years = x["year"].nunique() or 1
            cnt.append({"ATR": kind, "окно мин": w, "порог ATR": k, "случаев": len(x),
                        "из них через ночь": int(x["cross"].sum()), "в год на бумагу": round(len(x) / years / ev["tk"].nunique(), 1),
                        "вверх": int((x["dir"] > 0).sum()), "вниз": int((x["dir"] < 0).sum()),
                        "средний ход %": round(x["move_pct"].abs().mean(), 2),
                        "2016–2020": int((x["year"] <= 2020).sum()), "2021–2026": int((x["year"] >= 2021).sum())})
    cnt = pd.DataFrame(cnt)
    cnt.to_csv(OUT / "counts.csv", index=False)
    by_tk = ev[ev["peak"] >= 2].groupby(["tk", "atr", "w"]).size().unstack(["atr", "w"]).fillna(0).astype(int)
    by_tk.to_csv(OUT / "counts_by_ticker_k2.csv")
    print(cnt.to_string(index=False))
    print(by_tk.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
