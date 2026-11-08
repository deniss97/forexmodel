"""H42 (Тимур): память о крупных объёмах — «непогашенная дельта» крупных сделок против часовой дельты. Отчёт —
docs/results/h42_trace.md.

Данные — секундные агрегаты ленты: LKOH (data/raw/orderflow_lkoh_1s.parquet, время → МСК) и фьючерсы серебра SV* на
Мосбирже (orderflow_spbfut_1s.parquet, время → нью-йоркское, как котировки XAGUSD). Часовые суммы — штатным
`aggregate_orderflow`: «крупная» секунда — объём выше 95-го процентиля прошлых 5 дней (каузально).

Признаки (банк, до расчёта):
  * ld_1 — крупная дельта часа / объём часа (база сравнения), d_1 — вся дельта часа / объём;
  * ld_w — крупная дельта за w часов / объём за w часов, w = 6, 24, 72, 168;
  * trace — «след»: час-всплеск (|z| ≥ 2, z крупной дельты по прошлым 300 часам) добавляет свою крупную дельту в память;
    обычный час её только гасит — если его крупная дельта против знака памяти; память обнуляется, когда погашена
    («дельта пришла и ушла — сценарий отменён»); в долях среднего часового объёма прошлых 300 часов.
Цель — ход за 10 ч в ATR(14 ч) по котировкам (LKOH; XAGUSD для серебра). IC Спирмена по половинам данных.
Подтвердилась, если |IC| trace больше |IC| ld_1 и знак один — в обеих половинах на обоих инструментах.

    python scripts/h42_trace.py
"""
from __future__ import annotations

import gc

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forexmodel.config import FeatureConfig

from experiments import ROOT

DOC = ROOT / "docs" / "results" / "h42_trace.md"
FEAT = ROOT / "reports" / "night_search" / "feat"
SRC = {"lkoh": ("orderflow_lkoh_1s.parquet", dict(tz_shift_hours=3)),
       "silver": ("orderflow_spbfut_1s.parquet", dict(tz="America/New_York"))}
WINDOWS = (6, 24, 72, 168)
H = 10
HIST = 300
Z_SPIKE = 2.0
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def hourly(inst: str) -> pd.DataFrame:
    """Часовые суммы ленты с той же «крупной» секундой, что в aggregate_orderflow (объём ≥ среднего 95-го процентиля
    прошлых 5 дней), но читаются только три колонки — весь файл с ценами не помещается в память машины."""
    import pyarrow.parquet as pq

    name, kw = SRC[inst]
    cfg = FeatureConfig()
    tbl = pq.read_table(ROOT / "data" / "raw" / name, columns=["time", "volume", "signed_vol"])
    t = pd.Series(tbl.column("time").to_numpy())
    t = (t.dt.tz_localize("UTC") if getattr(t.dt, "tz", None) is None else t)
    t = t.dt.tz_convert(kw["tz"]).dt.tz_localize(None) if "tz" in kw else t.dt.tz_convert(None) + pd.Timedelta(hours=kw["tz_shift_hours"])
    of = pd.DataFrame({"time": t, "volume": tbl.column("volume").to_numpy().astype("int32"),
                       "signed_vol": tbl.column("signed_vol").to_numpy().astype("int32")})
    del tbl, t
    day = of["time"].dt.normalize()
    q = of.groupby(day)["volume"].quantile(cfg.orderflow_large_quantile)
    thr = day.map(q.shift(1).rolling(cfg.orderflow_large_lookback_days, min_periods=1).mean())
    large = (of["volume"] >= thr).to_numpy()
    of["large_delta"] = np.where(large, of["signed_vol"], 0).astype("int32")
    del day, thr, large
    g = of.groupby(of["time"].dt.floor("1h"))
    h = pd.DataFrame({"of_volume": g["volume"].sum(), "of_delta": g["signed_vol"].sum(), "of_large_delta": g["large_delta"].sum()})
    del of, g
    gc.collect()
    d = pd.read_pickle(FEAT / f"{inst}.pkl")
    b = d["bars"].assign(atr=d["f"]["atr14"].astype(float))
    b["time"] = pd.to_datetime(b["time"])
    b = b.set_index("time")[["close", "atr"]]
    b["fwd"] = (b["close"].shift(-H) - b["close"]) / b["atr"].replace(0, np.nan)
    h.index = pd.to_datetime(h.index)
    return b.join(h, how="inner").dropna(subset=["of_volume"])


def trace(ld: np.ndarray, z: np.ndarray) -> np.ndarray:
    u, out = 0.0, np.zeros(len(ld))
    for i in range(len(ld)):
        x = ld[i]
        if np.isfinite(z[i]) and abs(z[i]) >= Z_SPIKE:
            u += x
        elif u != 0 and x * u < 0:
            nu = u + x
            u = 0.0 if nu * u <= 0 else nu
        out[i] = u
    return out


def features(x: pd.DataFrame) -> pd.DataFrame:
    v = x["of_volume"].replace(0, np.nan)
    f = pd.DataFrame(index=x.index)
    f["d_1"] = x["of_delta"] / v
    f["ld_1"] = x["of_large_delta"] / v
    for w in WINDOWS:
        f[f"ld_{w}"] = x["of_large_delta"].rolling(w).sum() / x["of_volume"].rolling(w).sum().replace(0, np.nan)
    m = x["of_large_delta"].rolling(HIST).mean().shift(1)
    s = x["of_large_delta"].rolling(HIST).std().shift(1)
    z = ((x["of_large_delta"] - m) / s.replace(0, np.nan)).to_numpy()
    vavg = x["of_volume"].rolling(HIST).mean().shift(1)
    f["trace"] = trace(x["of_large_delta"].to_numpy(float), z) / vavg.to_numpy()
    f["trace_on"] = (f["trace"] != 0) & f["trace"].notna()
    f["fwd"] = x["fwd"]
    f = f.iloc[HIST:]
    return f


def ic(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan, int(m.sum())
    return float(spearmanr(a[m], b[m]).correlation), int(m.sum())


def main() -> int:
    rows, info = [], {}
    cols = ["d_1", "ld_1"] + [f"ld_{w}" for w in WINDOWS] + ["trace"]
    for inst in SRC:
        f = features(hourly(inst))
        cut = f.index[len(f) // 2]
        info[inst] = {"с": f.index.min().date(), "по": f.index.max().date(), "часов": len(f), "раздел": cut.date(),
                      "часов со следом %": float(f["trace_on"].mean() * 100)}
        for half, part in (("1-я половина", f[f.index < cut]), ("2-я половина", f[f.index >= cut]), ("все", f)):
            for c in cols:
                r, n = ic(part[c], part["fwd"])
                rows.append({"инструмент": inst, "часть": half, "признак": c, "IC": r, "часов": n, "t": r * np.sqrt(n / H)})
            on = part[part["trace_on"]]
            r, n = ic(on["trace"], on["fwd"])
            rows.append({"инструмент": inst, "часть": half, "признак": "trace, только часы со следом", "IC": r, "часов": n,
                         "t": r * np.sqrt(n / H) if n else np.nan})
        del f
        gc.collect()
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(info, "\n", R.round(3).to_string())
    g = R.set_index(["инструмент", "часть", "признак"])["IC"]
    halves = ("1-я половина", "2-я половина")
    signs = [np.sign(g[(i, h, "trace")]) for i in SRC for h in halves]
    ok = all(abs(g[(i, h, "trace")]) > abs(g[(i, h, "ld_1")]) for i in SRC for h in halves) and len(set(signs)) == 1

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Память о крупных объёмах: «непогашенная дельта» (H42)", "",
          "Идея Тимура: всплеск дельты крупных сделок помнить днями, а не фиксированный горизонт; если дельта пришла и ушла "
          "(погашена встречной) — сценарий отменён. Код Тимура не пришёл — своя версия. Данные — секундные агрегаты ленты "
          "LKOH и фьючерсов серебра SV* на Мосбирже; «крупная» секунда — объём выше 95-го процентиля прошлых 5 дней. "
          "Расчёт — `scripts/h42_trace.py`.", "",
          "**Критерий (банк, до расчёта):** |IC| «следа» с ходом за 10 ч больше, чем у часовой крупной дельты, и знак один — "
          "в обеих половинах данных на обоих инструментах; только тогда — модель LKOH против L01.", "", verdict, "",
          "Данные: " + "; ".join(f"{i}: {v['с']} — {v['по']}, {v['часов']} часов, раздел половин {v['раздел']}, след ненулевой "
                                   f"{v['часов со следом %']:.0f}% часов" for i, v in info.items()) + ".", "",
          "## IC с ходом за 10 часов (× 100)", "", "| инструмент | признак | 1-я половина | 2-я половина | все | t (все) |", "|---|---|---|---|---|---|"]
    for inst in SRC:
        for c in cols + ["trace, только часы со следом"]:
            a1, a2, a3 = (g.get((inst, h, c), np.nan) for h in ("1-я половина", "2-я половина", "все"))
            t = R[(R["инструмент"] == inst) & (R["часть"] == "все") & (R["признак"] == c)]["t"].iloc[0]
            md.append(f"| {inst} | {c} | {a1 * 100:+.1f} | {a2 * 100:+.1f} | {a3 * 100:+.1f} | {t:+.1f} |")
    md += ["", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
