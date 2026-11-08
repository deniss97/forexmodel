"""H80 (Тимур): данные moexalgo как признаки — позиции юрлиц и физлиц по фьючерсам на акции (FUTOI) и S42. Отчёт —
docs/results/h80_futoi.md.

Что из moexalgo доступно бесплатно: свечи и FUTOI (позиции юрлиц и физлиц по фьючерсам, каждые 5 мин с 03.01.2020,
задержка 14 дней); Super Candles (tradestats, orderstats, obstats, hi2) — только подписчикам. Данные —
`python scripts/moex_data.py futoi sr gz lk --from 2020-01-03` → data/raw/moex/futoi_{sr,gz,lk}.csv.

Признак (банк, до расчёта): изменение чистой позиции юрлиц за k торговых дней по закрытию дня (последний срез дня), в
долях открытого интереса (OI = длинные позиции юрлиц + физлиц); основной k = 5, k = 1 и 20 — для устойчивости.
(1) IC Спирмена с ходом акции за следующие 5 дней (от закрытия дня признака) — по бумагам и годам.
(2) Фильтр к S42 (GAZP, SBER, LKOH в лонг, издержки биржи, без наложений): вход, только если признак прошлого торгового
дня > 0. Подтвердилась, если PF и итог к просадке выше S42 на 2021–2026 и не хуже на 2020. Отдельно — тот же фильтр по
данным с задержкой 14 дней (последний день FUTOI не позже даты входа − 14 дней).

    python scripts/h80_futoi.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "h80_futoi.md"
FUT = ROOT / "data" / "raw" / "moex"
FEAT = ROOT / "reports" / "night_search" / "feat"
WF = ROOT / "reports" / "walk_forward"
CODE = {"sber": "sr", "gazp": "gz", "lkoh": "lk"}
KS = (1, 5, 20)
MAIN = 5
DELAY = 14
SLOTS = 3
START = pd.Timestamp("2020-01-03")
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x) -> float:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def account(t: pd.DataFrame) -> dict:
    x = t.sort_values("close_dt")
    eq = (x["profit_pct"] / SLOTS).cumsum().to_numpy()
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    end = float(eq[-1]) if len(eq) else 0.0
    return {"сделок": len(x), "PF": pf(x["profit_pct"]), "средняя %": float(x["profit_pct"].mean()) if len(x) else np.nan,
            "итог % депозита": end, "просадка %": dd, "итог / просадка": end / abs(dd) if dd else float("nan")}


def futoi_daily(inst: str) -> pd.DataFrame:
    """Последний срез каждого торгового дня: чистая позиция юрлиц, OI, признаки d_k."""
    f = pd.read_csv(FUT / f"futoi_{CODE[inst]}.csv")
    f["ts"] = pd.to_datetime(f["tradedate"] + " " + f["tradetime"])
    last = f.sort_values("ts").groupby(["tradedate", "clgroup"]).tail(1)
    p = last.pivot(index="tradedate", columns="clgroup", values=["pos", "pos_long"])
    d = pd.DataFrame({"yur": p[("pos", "YUR")], "oi": p[("pos_long", "YUR")] + p[("pos_long", "FIZ")]}).dropna()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    for k in KS:
        d[f"d{k}"] = (d["yur"] - d["yur"].shift(k)) / d["oi"]
    return d


def daily_close(inst: str) -> pd.Series:
    b = pd.read_pickle(FEAT / f"{inst}.pkl")["bars"]
    b = b.assign(time=pd.to_datetime(b["time"]))
    c = b.groupby(b["time"].dt.normalize())["close"].last()
    return c[c.index >= START - pd.Timedelta(days=40)]


def ic_table(fd: dict) -> pd.DataFrame:
    rows = []
    for inst, d in fd.items():
        c = daily_close(inst)
        x = d.join(c.rename("close"), how="inner")
        x["fwd"] = x["close"].shift(-5) / x["close"] - 1
        x["год"] = x.index.year
        for k in KS:
            for g, z in list(x.groupby("год")) + [("все", x)]:
                z = z[[f"d{k}", "fwd"]].dropna()
                if len(z) < 30:
                    continue
                r = spearmanr(z[f"d{k}"], z["fwd"]).correlation
                rows.append({"бумага": inst, "окно, дней": k, "год": str(g), "IC": float(r), "дней": len(z),
                             "t": float(r * np.sqrt(len(z) / 5))})
    return pd.DataFrame(rows)


def feature_at(d: pd.DataFrame, when: pd.Series, k: int, delay_days: int = 0) -> np.ndarray:
    """Значение d_k по последнему дню FUTOI строго до даты входа (или не позже даты входа − delay_days)."""
    days = d.index.to_numpy()
    cut = (when.dt.normalize() - pd.Timedelta(days=delay_days)).to_numpy()
    side = "right" if delay_days else "left"
    j = np.searchsorted(days, cut, side=side) - 1
    v = d[f"d{k}"].to_numpy()
    return np.where(j >= 0, v[j.clip(0)], np.nan)


def strategy(fd: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    t = no_overlaps(t[t["instrument"].isin(CODE) & (t["open_dt"] >= START)]).reset_index(drop=True)
    for k in KS:
        t[f"d{k}"] = np.nan
        t[f"d{k}_late"] = np.nan
        for inst, d in fd.items():
            m = t["instrument"] == inst
            t.loc[m, f"d{k}"] = feature_at(d, t.loc[m, "open_dt"], k)
            t.loc[m, f"d{k}_late"] = feature_at(d, t.loc[m, "open_dt"], k, DELAY)
    rows = []
    for per, pm in (("2020", t["open_dt"] < TEST_START), ("проверка 2021–2026", t["open_dt"] >= TEST_START)):
        x = t[pm]
        rows.append({"вариант": "S42", "период": per, **account(x)})
        for k in KS:
            rows.append({"вариант": f"юрлица нарастили за {k} дн." + (" (основной)" if k == MAIN else ""), "период": per,
                         **account(x[x[f"d{k}"] > 0])})
            rows.append({"вариант": f"юрлица сократили за {k} дн. (обратное)", "период": per, **account(x[x[f"d{k}"] < 0])})
        rows.append({"вариант": f"за {MAIN} дн., данные с задержкой {DELAY} дн. (бесплатно)", "период": per,
                     **account(x[x[f"d{MAIN}_late"] > 0])})
    return pd.DataFrame(rows), t


def write_filtered(t: pd.DataFrame) -> None:
    """Сделки S42 с фильтром основного признака — файлы для журнала (как S64, S66)."""
    for inst in CODE:
        raw = pd.read_csv(WF / f"{inst}_lo__impulse_6_3_long_trades.csv", parse_dates=["open_dt"])
        raw = raw[raw["open_dt"] >= START]
        keep = t.loc[(t["instrument"] == inst) & (t[f"d{MAIN}"] > 0), "open_dt"]
        raw[raw["open_dt"].isin(keep)].to_csv(WF / f"{inst}_lo_futoi__impulse_6_3_long_trades.csv", index=False)
        raw.to_csv(WF / f"{inst}_lo_from2020__impulse_6_3_long_trades.csv", index=False)


def main() -> int:
    fd = {inst: futoi_daily(inst) for inst in CODE}
    for inst, d in fd.items():
        print(inst, d.index.min().date(), "…", d.index.max().date(), len(d), "дней; доля дней с ростом позиции юрлиц за 5 дн.:",
              round(float((d["d5"] > 0).mean()), 2), "; чистая позиция юрлиц, медиана в долях OI:", round(float((d["yur"] / d["oi"]).median()), 2))
    IC = ic_table(fd)
    S, t = strategy(fd)
    write_filtered(t)
    pd.set_option("display.width", 250)
    print(IC[IC["год"] == "все"].round(3).to_string(), "\n", S.round(2).to_string())
    s = S.set_index(["вариант", "период"])
    v = f"юрлица нарастили за {MAIN} дн. (основной)"
    P, T = "проверка 2021–2026", "2020"
    ok = (s.loc[(v, P), "PF"] > s.loc[("S42", P), "PF"] and s.loc[(v, P), "итог / просадка"] > s.loc[("S42", P), "итог / просадка"]
          and s.loc[(v, T), "PF"] >= s.loc[("S42", T), "PF"])
    allic = IC[(IC["год"] == "все") & (IC["окно, дней"] == MAIN)]["IC"]
    ok_ic = bool((allic > 0).all() or (allic < 0).all())

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Позиции юрлиц по фьючерсам на акции (FUTOI, moexalgo) и S42 (H80)", "",
          "Гипотеза Тимура: проверить данные библиотеки moexalgo для самостоятельных стратегий и признаков, в первую очередь по "
          "акциям нашей стратегии. Бесплатно из moexalgo доступны свечи (уже есть) и FUTOI — позиции юрлиц и физлиц по "
          "фьючерсам каждые 5 минут с 03.01.2020 (с задержкой 14 дней); Super Candles (tradestats, orderstats, obstats, hi2) — "
          "только по подписке (проверено запросом). Фьючерсы: SR — SBER, GZ — GAZP, LK — LKOH. Расчёт — `scripts/h80_futoi.py`.", "",
          "**Критерий (банк, до расчёта):** признак — изменение чистой позиции юрлиц за 5 торговых дней по закрытию прошлого "
          "дня в долях открытого интереса. (1) IC с ходом акции за 5 дней — один знак на всех трёх бумагах в 2020–2026. "
          "(2) S42 только при росте позиции юрлиц: PF и итог к просадке выше S42 на 2021–2026 и не хуже на 2020. Окна 1 и 20 "
          "дней — для устойчивости; отдельно — данные с задержкой 14 дней.", "", verdict, "",
          "## 1. Связь с ходом акции за 5 дней (IC Спирмена, все дни 2020–2026)", "",
          "| бумага | окно, дней | IC | t | дней |", "|---|---|---|---|---|"]
    for _, r in IC[IC["год"] == "все"].iterrows():
        md.append(f"| {r['бумага']} | {r['окно, дней']} | {r['IC']:+.3f} | {r['t']:+.1f} | {r['дней']} |")
    md += ["", "### По годам (окно 5 дней)", "", "| бумага | " + " | ".join(sorted(IC.loc[IC['год'] != 'все', 'год'].unique())) + " |",
           "|---|" + "---|" * IC.loc[IC['год'] != 'все', 'год'].nunique()]
    for inst in CODE:
        z = IC[(IC["бумага"] == inst) & (IC["окно, дней"] == MAIN) & (IC["год"] != "все")].set_index("год")["IC"]
        md.append(f"| {inst} | " + " | ".join(f"{z.get(y, np.nan):+.2f}" for y in sorted(IC.loc[IC['год'] != 'все', 'год'].unique())) + " |")
    md += ["", "## 2. S42 с фильтром по позиции юрлиц", "",
           "Сделка = 1/3 депозита, без лимита одновременных сделок. Сделки до 03.01.2020 не входят (нет данных).", "",
           "| вариант | период | сделок | PF | средняя % | итог, % депозита | просадка, % | итог / просадка |", "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} | {r['итог % депозита']:+.1f} | "
                  f"{r['просадка %']:.1f} | {r['итог / просадка']:.2f} |")
    md += ["", f"Критерий (1) — один знак IC на трёх бумагах: {'да' if ok_ic else 'нет'}; критерий (2): {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok_ic, ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
