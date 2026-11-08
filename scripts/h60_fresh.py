"""H60 (S50) на свежих месяцах — вопрос Тимура на созвоне 05.11: тренд-фильтр по смене направления на GAZP и SBER был в плюсе
в обоих периодах — почему не развили? H60 развита 19.10, но журнал считал до 28.05.2026. Здесь — то же правило на живых
минутках (data/raw/live, Т-Инвестиции) с 01.06.2026 по последнюю минуту: лонг, пока трёхслойный фильтр (режим по
дневной EMA200) не покажет вниз, нейтраль не закрывает; вход и выход по открытию часа после смены; издержки биржи.
Фильтр считается по всей истории (2015 — сейчас), сделки берутся с входом с 01.06.2026; открытая позиция — по последней
цене. Для сравнения — «купить и держать» за тот же отрезок. Отчёт — docs/results/h60_fresh.md.

    python scripts/h60_fresh.py
"""
from __future__ import annotations

import logging

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import trend_grid as tg
from forexmodel.data.dividends import adjust_for_dividends, load_dividends
from forexmodel.data.loader import load_minute_compact, resample_ohlcv
from forexmodel.simulation.costs import profile_entry, recost
from trend_follow import TL, follow

from experiments import ROOT

DOC = ROOT / "docs" / "results" / "h60_fresh.md"
STOCKS = {"gazp": "GAZP", "sber": "SBER", "lkoh": "LKOH"}
START = pd.Timestamp("2026-06-01")
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def bars(ticker: str) -> pd.DataFrame:
    m = load_minute_compact(ROOT / "data" / "raw" / "live" / f"{ticker}_1m_live.csv", "begin", ROOT / "reports" / "_cache", ("volume",))
    m = adjust_for_dividends(m, load_dividends(ROOT / "data" / "dividends" / "moex_dividends.csv", ticker))
    return resample_ohlcv(m, "1h")[["time", "open", "high", "low", "close"]].reset_index(drop=True)


def main() -> int:
    logging.disable(logging.WARNING)
    tg._init()
    rows, trades_all = [], []
    for inst, tk in STOCKS.items():
        b = bars(tk)
        g = tg.gate_values(b, TL)
        t = follow(b, g, long_only=True, hold=True)
        t = recost(t, profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))
        fresh = t[t["open_dt"] >= START]
        w = b[b["time"] >= START]
        bh = (w["close"].iloc[-1] / w["open"].iloc[0] - 1) * 100
        inpos = float(np.mean(np.nan_to_num(g[b["time"].to_numpy() >= np.datetime64(START)]) > 0))
        rows.append({"бумага": tk, "сделок": len(fresh), "итог %": fresh["profit_pct"].sum(), "в плюсе": int((fresh["profit_pct"] > 0).sum()),
                     "купить и держать %": bh, "фильтр вверх, доля часов": inpos,
                     "данные до": f"{b['time'].iloc[-1]:%d.%m.%Y %H:%M}"})
        trades_all.append(fresh.assign(бумага=tk))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    T = pd.concat(trades_all, ignore_index=True)
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# H60 (S50) на свежих месяцах: акции в лонг по тренд-фильтру", "",
          "Вопрос Тимура (созвон 05.11): по смене направления тренд-фильтра GAZP и SBER были в плюсе и на выборе, и на проверке "
          "(PF около 1.3–1.4) — почему не развили? Развили 19.10 в H60 (S50): против «купить и держать» лучше на единицу риска, "
          "но с акциями S42 связан на 0.75–0.79 и портфель с ним не лучше. Журнал считал до 28.05.2026; здесь — то же правило "
          f"на свежих минутках Т-Инвестиций с {START:%d.%m.%Y}. Расчёт — `scripts/h60_fresh.py`.", "", verdict, "",
          "| бумага | сделок | итог, % | в плюсе | «купить и держать», % | фильтр вверх, доля часов | данные до |",
          "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['бумага']} | {r['сделок']} | {r['итог %']:+.1f} | {r['в плюсе']} | {r['купить и держать %']:+.1f} | "
                  f"{r['фильтр вверх, доля часов']:.0%} | {r['данные до']} |")
    md += ["", "## Сделки", "", "| бумага | вход | выход | результат после издержек, % |", "|---|---|---|---|"]
    for _, r in T.sort_values("open_dt").iterrows():
        md.append(f"| {r['бумага']} | {r['open_dt']:%d.%m %H:%M} | {r['close_dt']:%d.%m %H:%M} | {r['profit_pct']:+.2f} |")
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
