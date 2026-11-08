"""H42 по коду Тимура (S_InsideDeals.ipynb, прикреплён к H42 в банке): «информированная торговля» и «топ-20 минут дня» на
нашей ленте. Отчёт — docs/results/h42_timur.md.

Данные — секундные агрегаты ленты: LKOH (только основной режим TQBR; у Тимура в данные попали ещё режимы РЕПО EQRP и
PSRP, где огромные «дисбалансы» — это сделки РЕПО, а не направленные) и фьючерсы серебра SV* (в каждый день — самый
торгуемый контракт). Время — МСК. Секунды → минуты: цена закрытия, объём, покупки, продажи, ofi.

(1) «Информированная торговля» без взгляда вперёд (как флаг imbalance_anomaly у Тимура): z дисбаланса ofi минуты по
    60 минутам (включая текущую), аномалия — |z| ≥ 2.5. Доля минут, после которых цена через 15 минут и к закрытию
    следующего торгового дня ушла в сторону дисбаланса, и средний ход в его сторону — против всех минут с ofi ≠ 0.
(2) «Топ-20 минут дня»: доля покупок в 20 самых крупных по объёму минутах дня → ход от закрытия дня до закрытия через
    2 торговых дня; IC Спирмена против доли покупок за весь день.
Критерий (банк, до расчёта): (1) в обеих половинах данных на обоих инструментах доля «в сторону дисбаланса» у аномалий
выше, чем у всех минут, и средний ход в сторону дисбаланса > 0; или (2) IC одного знака в обеих половинах на обоих
инструментах, |IC| ≥ 0.05 и больше, чем у доли покупок за день.

    python scripts/h42_timur.py
"""
from __future__ import annotations

import gc

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from experiments import ROOT

DOC = ROOT / "docs" / "results" / "h42_timur.md"
SRC = {"lkoh": "orderflow_lkoh_1s.parquet", "silver": "orderflow_spbfut_1s.parquet"}
WIN, Z, AHEAD, TOP = 60, 2.5, 15, 20
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def _chunk(pf: pq.ParquetFile, i: int, cols: list[str]) -> pd.DataFrame:
    df = pf.read_row_group(i, columns=cols).to_pandas()
    t = pd.to_datetime(df["time"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    df["time"] = t + pd.Timedelta(hours=3)                          # UTC → МСК
    df["day"] = df["time"].dt.normalize()
    return df


def minutes(inst: str) -> pd.DataFrame:
    """Секунды → минуты по частям файла (весь файл в память машины не помещается)."""
    pf = pq.ParquetFile(ROOT / "data" / "raw" / SRC[inst])
    front = None
    if inst == "silver":                                             # самый торгуемый контракт дня
        dv = pd.concat([_chunk(pf, i, ["TICKER_CC", "time", "volume"]).groupby(["day", "TICKER_CC"])["volume"].sum()
                        for i in range(pf.num_row_groups)]).groupby(level=[0, 1]).sum().reset_index()
        front = dv.sort_values("volume").groupby("day").tail(1).set_index("day")["TICKER_CC"]
    parts = []
    for i in range(pf.num_row_groups):
        df = _chunk(pf, i, ["TICKER_CC", "time", "price", "volume", "buy_volume", "sell_volume", "ofi"])
        if front is not None:
            df = df[df["TICKER_CC"].to_numpy() == df["day"].map(front).to_numpy()]
        df = df.sort_values("time")
        g = df.groupby(df["time"].dt.floor("1min"))
        parts.append(pd.DataFrame({"t_last": g["time"].max(), "close": g["price"].last(), "volume": g["volume"].sum(),
                                   "buy": g["buy_volume"].sum(), "sell": g["sell_volume"].sum(), "ofi": g["ofi"].sum()}))
        del df, g
        gc.collect()
    p = pd.concat(parts).sort_values("t_last")                       # минута на стыке частей собирается заново
    g = p.groupby(level=0)
    m = pd.DataFrame({"close": g["close"].last(), "volume": g["volume"].sum(), "buy": g["buy"].sum(), "sell": g["sell"].sum(),
                      "ofi": g["ofi"].sum()}).sort_index()
    m = m[m["volume"] > 0]
    m["day"] = m.index.normalize()
    return m


def informed(m: pd.DataFrame) -> pd.DataFrame:
    mu = m["ofi"].rolling(WIN, min_periods=WIN // 3).mean()
    sd = m["ofi"].rolling(WIN, min_periods=WIN // 3).std()
    m = m.assign(z=(m["ofi"] - mu) / sd.replace(0, np.nan))
    m["fwd15"] = m["close"].shift(-AHEAD) / m["close"] - 1
    dclose = m.groupby("day")["close"].last()
    nxt = pd.Series(dclose.shift(-1).to_numpy(), index=dclose.index)  # закрытие следующего торгового дня
    m["fwd_next"] = m["day"].map(nxt) / m["close"] - 1
    return m


def daily_top(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in m.groupby("day"):
        top = g.nlargest(TOP, "volume")
        v = top["volume"].sum()
        rows.append({"day": d, "close": g["close"].iloc[-1], "top_buy_share": top["buy"].sum() / v if v else np.nan,
                     "day_buy_share": g["buy"].sum() / g["volume"].sum() if g["volume"].sum() else np.nan, "minutes": len(g)})
    d = pd.DataFrame(rows).set_index("day").sort_index()
    d = d[d["minutes"] >= 60]                                        # без коротких выходных сессий
    d["fwd2"] = d["close"].shift(-2) / d["close"] - 1
    return d


def main() -> int:
    R1, R2, info = [], [], {}
    for inst in SRC:
        m = informed(minutes(inst))
        cut = m["day"].iloc[len(m) // 2]
        info[inst] = {"с": m.index.min().date(), "по": m.index.max().date(), "минут": len(m), "раздел": cut.date(),
                      "аномалий %": float((m["z"].abs() >= Z).mean() * 100)}
        for half, part in (("1-я половина", m[m["day"] < cut]), ("2-я половина", m[m["day"] >= cut]), ("все", m)):
            nz = part[part["ofi"] != 0]
            an = nz[nz["z"].abs() >= Z]
            for h in ("fwd15", "fwd_next"):
                s_all = np.sign(nz["ofi"]) * nz[h]
                s_an = np.sign(an["ofi"]) * an[h]
                R1.append({"инструмент": inst, "часть": half, "горизонт": "15 мин" if h == "fwd15" else "до закрытия след. дня",
                           "аномалий": int(s_an.notna().sum()), "в сторону дисбаланса: аномалии %": float((s_an > 0).mean() * 100),
                           "все минуты %": float((s_all > 0).mean() * 100), "ход аномалий, б.п.": float(s_an.mean() * 1e4),
                           "ход всех, б.п.": float(s_all.mean() * 1e4)})
        d = daily_top(m)
        dcut = d.index[len(d) // 2]
        for half, part in (("1-я половина", d[d.index < dcut]), ("2-я половина", d[d.index >= dcut]), ("все", d)):
            p = part.dropna(subset=["fwd2"])
            R2.append({"инструмент": inst, "часть": half, "дней": len(p),
                       "IC топ-20": float(spearmanr(p["top_buy_share"], p["fwd2"]).correlation),
                       "IC весь день": float(spearmanr(p["day_buy_share"], p["fwd2"]).correlation)})
        del m, d
        gc.collect()
    A, B = pd.DataFrame(R1), pd.DataFrame(R2)
    pd.set_option("display.width", 250)
    print(info, "\n", A.round(2).to_string(), "\n", B.round(3).to_string())
    halves = ("1-я половина", "2-я половина")
    a = A[A["часть"].isin(halves)]
    ok1 = bool(((a["в сторону дисбаланса: аномалии %"] > a["все минуты %"]) & (a["ход аномалий, б.п."] > 0)).all())
    b = B[B["часть"].isin(halves)]
    ok2 = bool((np.sign(b["IC топ-20"]).nunique() == 1) and (b["IC топ-20"].abs() >= 0.05).all()
               and (b["IC топ-20"].abs() > b["IC весь день"].abs()).all())

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# «След» крупных сделок по коду Тимура: информированная торговля и топ-20 минут дня (H42)", "",
          "Код Тимура — S_InsideDeals.ipynb (прикреплён к H42 в банке): детектор «информированной торговли» (аномальный дисбаланс "
          "в минуте, после которого цена идёт в ту же сторону) и дневной отчёт «топ-20 минут по объёму». Детектор Тимура "
          "отбирает минуты с учётом будущего хода (forward_return_zscore) — для торговли так нельзя, поэтому здесь проверяется "
          "только то, что известно в момент минуты. Данные — наша лента: LKOH только в режиме TQBR (у Тимура в данных были и "
          "режимы РЕПО EQRP/PSRP — их «дисбалансы» в сотни тысяч лотов не направленные), фьючерсы серебра — самый торгуемый "
          "контракт дня. Расчёт — `scripts/h42_timur.py`.", "",
          "**Критерий (банк, до расчёта):** (1) после аномального дисбаланса цена чаще, чем после обычной минуты, идёт в его "
          "сторону, и средний ход в его сторону > 0 — в обеих половинах данных на обоих инструментах; или (2) доля покупок в "
          "топ-20 минутах дня связана с ходом за 2 дня: IC одного знака в обеих половинах на обоих инструментах, |IC| ≥ 0.05 и "
          "больше, чем у доли покупок за весь день.", "", verdict, "",
          "Данные: " + "; ".join(f"{i}: {v['с']} — {v['по']}, {v['минут']} минут, раздел половин {v['раздел']}, аномалий "
                                   f"{v['аномалий %']:.1f}% минут" for i, v in info.items()) + ".", "",
          "## 1. После аномального дисбаланса", "",
          "| инструмент | часть | горизонт | аномалий | в сторону дисбаланса: аномалии | все минуты | ход аномалий, б. п. | ход всех, б. п. |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in A.iterrows():
        md.append(f"| {r['инструмент']} | {r['часть']} | {r['горизонт']} | {r['аномалий']} | {r['в сторону дисбаланса: аномалии %']:.1f}% | "
                  f"{r['все минуты %']:.1f}% | {r['ход аномалий, б.п.']:+.1f} | {r['ход всех, б.п.']:+.1f} |")
    md += ["", "Доля «в сторону» ниже 50%, потому что цена через 15 минут часто не меняется (такие минуты — не в сторону).", "",
           "## 2. Доля покупок в топ-20 минутах дня → ход за 2 дня", "", "| инструмент | часть | дней | IC топ-20 | IC весь день |", "|---|---|---|---|---|"]
    for _, r in B.iterrows():
        md.append(f"| {r['инструмент']} | {r['часть']} | {r['дней']} | {r['IC топ-20']:+.3f} | {r['IC весь день']:+.3f} |")
    md += ["", f"Критерий (1): {'да' if ok1 else 'нет'}; критерий (2): {'да' if ok2 else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok1, ok2, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
