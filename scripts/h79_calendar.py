"""H79 (созвон 05.11): экономический календарь — решения ЦБ по ключевой ставке и S42 на акциях. Отчёт —
docs/results/h79_calendar.md.

Календарь — data/calendar/cbr_key_rate_meetings.csv: плановые заседания 2016–2026 из архивных снимков
cbr.ru/dkp/cal_mp (web.archive.org), внеочередные и направление решения — из data/calendar/cbr_key_rate_changes.csv.
Решение публикуется в 13:30 МСК.

Критерий (банк, до расчёта): S42 на GAZP, SBER, LKOH (лонг, издержки биржи, без наложений) без входов за 24 ч до 13:30
дня решения — PF и итог к просадке выше, чем у S42, и на 2016–2020, и на 2021–2026. Описательно: ход акций в часы после
решения против тех же часов обычных дней; сделки S42, державшиеся через решение.

    python scripts/h79_calendar.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "h79_calendar.md"
CAL = ROOT / "data" / "calendar" / "cbr_key_rate_meetings.csv"
FEAT = ROOT / "reports" / "night_search" / "feat"
STOCKS = ["gazp", "sber", "lkoh"]
SLOTS = 3
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
PER = ("выбор", "проверка")


def pf(x) -> float:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def in_per(ts: pd.Series, per: str) -> pd.Series:
    return (ts < TEST_START) if per == "выбор" else (ts >= TEST_START)


def account(t: pd.DataFrame) -> dict:
    x = t.sort_values("close_dt")
    eq = (x["profit_pct"] / SLOTS).cumsum().to_numpy()
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    end = float(eq[-1]) if len(eq) else 0.0
    return {"сделок": len(x), "PF": pf(x["profit_pct"]), "средняя %": float(x["profit_pct"].mean()) if len(x) else np.nan,
            "итог % депозита": end, "просадка %": dd, "итог / просадка": end / abs(dd) if dd else float("nan")}


def events() -> pd.DataFrame:
    c = pd.read_csv(CAL, comment="#", parse_dates=["date"], dtype={"time_msk": str})
    # только плановые заседания (скептик 08.11): внеочередные объявлялись меньше чем за сутки, а решение выходило около 10:30 —
    # заранее их не знать, и фильтр по ним убирал сделки, открытые уже после решения
    c = c[c["scheduled"] == 1]
    c["event"] = c["date"] + pd.to_timedelta(c["time_msk"].fillna("13:30") + ":00")
    return c[c["date"] >= "2016-01-01"].reset_index(drop=True)


def strategy(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    t = no_overlaps(t[t["instrument"].isin(STOCKS)]).reset_index(drop=True)
    e = ev["event"].to_numpy()
    # ближайшее решение после входа и до выхода
    k = np.searchsorted(e, t["open_dt"].to_numpy(), side="left")
    nxt = pd.Series(np.where(k < len(e), e[k.clip(max=len(e) - 1)], np.datetime64("NaT")), index=t.index)
    t["часов до решения"] = (nxt - t["open_dt"]).dt.total_seconds() / 3600
    t["через решение"] = nxt < t["close_dt"]
    t["решение"] = np.where(k < len(e), ev["decision"].to_numpy()[k.clip(max=len(e) - 1)], "")
    rows = []
    for per in PER:
        x = t[in_per(t["open_dt"], per)]
        for name, m in (("S42", np.ones(len(x), bool)),
                        ("S42 без входов за 24 ч до решения", ~(x["часов до решения"] <= 24).to_numpy()),
                        ("для сравнения: без входов за 48 ч", ~(x["часов до решения"] <= 48).to_numpy())):
            rows.append({"вариант": name, "период": per, **account(x[m])})
    return pd.DataFrame(rows), t


def held(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for per in PER:
        x = t[in_per(t["open_dt"], per)]
        for name, m in (("держались через решение", x["через решение"]), ("остальные", ~x["через решение"]),
                        ("вход за ≤ 24 ч до решения", x["часов до решения"] <= 24)):
            y = x[m]
            rows.append({"сделки": name, "период": per, "сделок": len(y), "PF": pf(y["profit_pct"]),
                         "средняя %": float(y["profit_pct"].mean()) if len(y) else np.nan})
        for d in ("up", "down", "hold"):
            y = x[x["через решение"] & (x["решение"] == d)]
            rows.append({"сделки": f"через решение: {dict(up='повышение', down='снижение', hold='без изменения')[d]}",
                         "период": per, "сделок": len(y), "PF": pf(y["profit_pct"]),
                         "средняя %": float(y["profit_pct"].mean()) if len(y) else np.nan})
    return pd.DataFrame(rows)


def reaction(ev: pd.DataFrame) -> pd.DataFrame:
    """|ход| за час 13:00–14:00 и 13:00–16:00 в ATR(14 ч): дни решений против остальных будней (те же часы)."""
    rows = []
    days = set(ev["date"].dt.normalize())
    dec = dict(zip(ev["date"].dt.normalize(), ev["decision"]))
    for inst in STOCKS:
        d = pd.read_pickle(FEAT / f"{inst}.pkl")
        b = d["bars"].copy()
        b["time"] = pd.to_datetime(b["time"])
        b["atr"] = d["f"]["atr14"].astype(float)
        b["day"] = b["time"].dt.normalize()
        b["hour"] = b["time"].dt.hour
        g = b.set_index(["day", "hour"])
        for day, z in b[b["hour"] == 13].groupby("day"):
            if z.empty:
                continue
            o, atr = float(z["open"].iloc[0]), float(z["atr"].iloc[0])
            if not np.isfinite(atr) or atr <= 0:
                continue
            c14 = g["close"].get((day, 13))
            c16 = g["close"].get((day, 15))
            rows.append({"бумага": inst, "день": day, "решение": dec.get(day, "") if day in days else "обычный день",
                         "1 ч": abs(c14 - o) / atr if c14 is not None else np.nan,
                         "3 ч": abs(c16 - o) / atr if c16 is not None else np.nan})
    r = pd.DataFrame(rows)
    r["день решения"] = r["решение"] != "обычный день"
    r["период"] = np.where(r["день"] < TEST_START, "выбор", "проверка")
    out = r.groupby(["период", "день решения"])[["1 ч", "3 ч"]].agg(["mean", "count"])
    return out


def write_filtered(ev: pd.DataFrame, hours: float = 24) -> None:
    """Сделки S42 без входов за `hours` до решения — файлы для журнала (S66), как у S64 (h75_regimes.py)."""
    wf = ROOT / "reports" / "walk_forward"
    e = ev["event"].to_numpy()
    for inst in STOCKS:
        raw = pd.read_csv(wf / f"{inst}_lo__impulse_6_3_long_trades.csv")
        od = pd.to_datetime(raw["open_dt"]).to_numpy()
        k = np.searchsorted(e, od, side="left").clip(max=len(e) - 1)
        gap = (e[k] - od) / np.timedelta64(1, "h")
        raw[~((gap >= 0) & (gap <= hours))].to_csv(wf / f"{inst}_lo_cbr__impulse_6_3_long_trades.csv", index=False)


def main() -> int:
    ev = events()
    write_filtered(ev)
    S, t = strategy(ev)
    Hd = held(t)
    R = reaction(ev)
    pd.set_option("display.width", 250)
    print(ev.groupby(ev["date"].dt.year).size().to_dict())
    print(S.round(2).to_string(), "\n", Hd.round(2).to_string(), "\n", R.round(3).to_string())
    s = S.set_index(["вариант", "период"])
    v = "S42 без входов за 24 ч до решения"
    ok = all(s.loc[(v, p), "PF"] > s.loc[("S42", p), "PF"] and s.loc[(v, p), "итог / просадка"] > s.loc[("S42", p), "итог / просадка"]
             for p in PER)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    n_sched, n_extra = int(ev["scheduled"].sum()), int((~ev["scheduled"].astype(bool)).sum())
    md = ["# Экономический календарь: решения ЦБ по ставке и S42 на акциях (H79)", "",
          "Идея созвона 05.11: акции ведут себя иначе вокруг заранее известных событий. Первое событие — решение Банка России "
          f"по ключевой ставке (13:30 МСК): {n_sched} плановых заседаний 2016–2026 и {n_extra} внеочередных. Календарь — "
          "`data/calendar/cbr_key_rate_meetings.csv` (плановые даты — архивные снимки cbr.ru/dkp/cal_mp на web.archive.org, "
          "решения — `cbr_key_rate_changes.csv`). Отчётности эмитентов и инфляции здесь нет. Расчёт — `scripts/h79_calendar.py`.", "",
          "**Критерий (банк, до расчёта):** S42 на GAZP, SBER, LKOH (лонг, издержки биржи) без входов за 24 ч до 13:30 дня "
          "решения — PF и итог к просадке выше, чем у S42, и на 2016–2020, и на 2021–2026.", "", verdict, "",
          "## S42 с запретом входа перед решением", "",
          "Сделка = 1/3 депозита, без лимита одновременных сделок (как в H70). Запрет убирает сделку целиком (без замены).", "",
          "| вариант | период | сделок | PF | средняя % | итог, % депозита | просадка, % | итог / просадка |", "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} | {r['итог % депозита']:+.1f} | "
                  f"{r['просадка %']:.1f} | {r['итог / просадка']:.2f} |")
    md += ["", "## Сделки S42 вокруг решений", "", "| сделки | период | сделок | PF | средняя % |", "|---|---|---|---|---|"]
    for _, r in Hd.iterrows():
        md.append(f"| {r['сделки']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} |")
    md += ["", "## Ход акций после решения", "",
           "Модуль хода от открытия часа 13:00 до закрытия часа 13:00 (1 ч) и 15:00 (3 ч), в ATR(14 ч); GAZP, SBER, LKOH вместе.", "",
           "| период | дни | 1 ч, ATR | 3 ч, ATR | дней × бумаг |", "|---|---|---|---|---|"]
    for (per, dd), r in R.iterrows():
        md.append(f"| {per} | {'решения ЦБ' if dd else 'обычные'} | {r[('1 ч', 'mean')]:.2f} | {r[('3 ч', 'mean')]:.2f} | {int(r[('1 ч', 'count')])} |")
    md += ["", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
