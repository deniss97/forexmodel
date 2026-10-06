"""История дивидендов акций MOEX → data/dividends/moex_dividends.csv, с проверкой по гэпам в данных.

Источники (открытые страницы; ISS MOEX бесплатно дивидендную историю не отдаёт):
  * dohod.ru/ik/analytics/dividend/<тикер> — дата закрытия реестра и дивиденд с 2000-х (основной);
  * smart-lab.ru/q/<ТИКЕР>/dividend/ — последний день с дивидендом («дата T-1») с 2017 (сверка дат).

Последний день покупки с дивидендом (last_with) считается по режиму расчётов: до 31.07.2023 T+2,
после — T+1: рабочий день за `lag` рабочих дней до последнего рабочего дня не позже даты реестра.
Рабочие дни — будни с основной сессией (праздничные и субботние сессии 2025–2026 годов, где торги
идут только днём, рабочими не считаются: расчёты по ним проходят в следующий рабочий день).

Момент отсечки (ex_time) — начало сессии, в которой цена отдаёт дивиденд: первая сессия после
last_with (утро следующего дня, праздничная или субботняя сессия) — из сессий в ближайшие 5 дней
берётся та, что открылась с самым большим гэпом вниз. Проверка: этот гэп в долях дивиденда (≈ 1,
если дата и сумма верны; разброс — движение рынка за ночь). Дивиденды с одной отсечкой суммируются;
выплаты после конца минутных данных не пишутся.

    python scripts/fetch_dividends.py            # скачать страницы и пересобрать таблицу
    python scripts/fetch_dividends.py --offline  # только из сохранённых страниц reports/_cache/dividends
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "reports" / "_cache" / "dividends"
OUT = ROOT / "data" / "dividends" / "moex_dividends.csv"
MINUTES = {"LKOH": "LKOH_1m_15-26", "GAZP": "GAZP_1m_15-26_upd", "SBER": "SBER_1m_15-26",
           "MOEX": "MOEX_1m_15-26", "MTSS": "MTSS_1m_15-26"}
T1_FROM = pd.Timestamp("2023-07-31")          # переход акций MOEX на T+1
SINCE = pd.Timestamp("2015-01-01")


def _rows(path: Path) -> list[list[str]]:
    s = path.read_text(encoding="utf-8", errors="ignore")
    out = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S):
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)]
        if cells:
            out.append(cells)
    return out


def download(ticker: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for site, url in (("dohod", f"https://www.dohod.ru/ik/analytics/dividend/{ticker.lower()}"),
                      ("smartlab", f"https://smart-lab.ru/q/{ticker}/dividend/")):
        subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "--max-time", "30", url,
                        "-o", str(CACHE / f"{site}_{ticker}.html")], check=True)


def parse(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = [r for r in _rows(CACHE / f"dohod_{ticker}.html")
         if len(r) == 4 and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[1])]          # без «(прогноз)»
    dohod = pd.DataFrame({"record_date": pd.to_datetime([r[1] for r in d], dayfirst=True),
                          "amount": [float(r[3].replace(",", ".")) for r in d]})
    s = [r for r in _rows(CACHE / f"smartlab_{ticker}.html")
         if len(r) >= 5 and r[0] == ticker and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[2])]
    sl = pd.DataFrame({"last_with": pd.to_datetime([r[1] for r in s], dayfirst=True),
                       "record_date": pd.to_datetime([r[2] for r in s], dayfirst=True),
                       "amount_sl": [float(re.sub(r"[^\d,.]", "", r[4]).replace(",", ".")) for r in s]})
    return dohod, sl


def business_days(m: pd.DataFrame) -> np.ndarray:
    """Будни с основной сессией: с 2025 года праздничные сессии идут только днём (без вечерней)."""
    g = m.groupby(m["time"].dt.normalize())["time"].agg(["min", "max"])
    d = g.index
    ok = (d.dayofweek < 5) & ((d.year < 2025) | (g["max"].dt.hour >= 20))
    return d[ok].to_numpy()


def last_with_day(record: pd.Timestamp, bdays: np.ndarray) -> pd.Timestamp | None:
    """Последний рабочий день покупки с дивидендом по режиму расчётов T+2 / T+1."""
    lag = 1 if record >= T1_FROM else 2
    k = int(np.searchsorted(bdays, np.datetime64(record), side="right")) - 1   # последний рабочий день <= реестра
    return pd.Timestamp(bdays[k - lag]) if k - lag >= 0 else None


def ex_moment(last_with: pd.Timestamp, t: np.ndarray, o: np.ndarray, c: np.ndarray) -> int | None:
    """Индекс бара, открывшего первую сессию без дивиденда: из начал сессий в (last_with, +5 дней] —
    с самым большим гэпом вниз к предыдущему закрытию."""
    lo = int(np.searchsorted(t, np.datetime64(last_with + pd.Timedelta("1D")), side="left"))
    hi = int(np.searchsorted(t, np.datetime64(last_with + pd.Timedelta("6D")), side="left"))
    if lo <= 0 or lo >= len(t):
        return None
    days = t[lo:hi].astype("datetime64[D]")
    first_two = np.unique(days)[:2]                       # первые две даты с торгами после last_with
    starts = [i for i in range(lo, min(hi, len(t)))
              if (i == lo or t[i] - t[i - 1] >= np.timedelta64(10, "m")) and days[i - lo] in first_two]
    return min(starts, key=lambda i: o[i] / c[i - 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    rows = []
    for ticker, fname in MINUTES.items():
        if not args.offline:
            download(ticker)
        dohod, sl = parse(ticker)
        m = pd.read_pickle(ROOT / "reports" / "_cache" / f"{fname}_1m.pkl").sort_values("time")
        t = m["time"].to_numpy()
        bdays = business_days(m)
        o, c = m["open"].to_numpy(dtype=float), m["close"].to_numpy(dtype=float)
        # дивиденды с одной датой реестра суммируются (LKOH 21.12.2022: 537 + 256)
        dohod = dohod.groupby("record_date", as_index=False)["amount"].sum()
        for rec, amount in zip(dohod["record_date"], dohod["amount"]):
            if rec < SINCE or amount <= 0 or rec > pd.Timestamp(bdays[-1]):
                continue
            note = "dohod.ru"
            hit = sl[(sl["record_date"] == rec) & np.isclose(sl["amount_sl"], amount, rtol=0.02)]
            cands = [(rec, hit, note)]
            if not len(hit):
                # та же сумма в том же году с другой датой реестра (MOEX 2024: dohod 13.05, верно 14.06;
                # у smart-lab бывают и свои ошибки — MTSS 2016) — берём дату, при которой гэп ближе к дивиденду
                alt = sl[(sl["record_date"].dt.year == rec.year) & np.isclose(sl["amount_sl"], amount, rtol=0.02)]
                if len(alt):
                    cands.append((alt["record_date"].iloc[0], alt.iloc[:1], "дата реестра по smart-lab"))
            best = None
            for r, h, n in cands:
                lw_ = last_with_day(r, bdays)
                i_ = ex_moment(lw_, t, o, c) if lw_ is not None else None
                if i_ is None:
                    continue
                err = abs(-(o[i_] - c[i_ - 1]) / amount - 1)
                if best is None or err < best[0]:
                    best = (err, r, h, n, lw_, i_)
            if best is None:
                continue
            _, rec, hit, note, lw, i = best
            lw_sl = hit["last_with"].iloc[0].date() if len(hit) else None
            rows.append({"ticker": ticker, "record_date": rec.date(), "last_with": lw.date(),
                         "last_with_smartlab": lw_sl, "ex_time": pd.Timestamp(t[i]), "ex_date": pd.Timestamp(t[i]).date(),
                         "amount": round(amount, 4), "dividend_pct": round(amount / c[i - 1] * 100, 2),
                         "gap_open_pct": round((o[i] / c[i - 1] - 1) * 100, 2),
                         "gap_to_dividend": round(-(o[i] - c[i - 1]) / amount, 2),
                         "source": note + (" + smart-lab" if len(hit) and note == "dohod.ru" else "")})
    d = pd.DataFrame(rows).sort_values(["ticker", "ex_time"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.drop(columns=["ex_date"]).to_string(index=False))
    bad = d[d["last_with_smartlab"].notna() & (d["last_with_smartlab"] != d["last_with"])]
    print(f"\nзаписано {len(d)} отсечек → {OUT.relative_to(ROOT)}; расхождений последнего дня с дивидендом "
          f"со smart-lab: {len(bad)}")
    if len(bad):
        print(bad.drop(columns=["ex_date"]).to_string(index=False))
    print("гэп открытия / дивиденд: медиана", round(float(d["gap_to_dividend"].median()), 2),
          "| доля выплат с гэпом 0.5–1.5 дивиденда:", round(float(d["gap_to_dividend"].between(0.5, 1.5).mean()), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
