"""Лучшие стратегии на свежих данных 12.06–21.10.2026: S42 (импульс), стратегии итератора S45–S48, RSI(2) Коннорса (22.10).

Тот же код, что посчитал их раньше, на минутках data/raw/live (история + свежие минуты ISS и Т-Инвестиций), акции
SBER, GAZP, LKOH на бирже только в лонг, издержки биржи:
  * S42 — пайплайн walk-forward (`walk_forward._rule_trades`, как scripts/fresh_check.py);
  * S45–S48 — часовой движок перебора (`iterator_walkthrough.trades_frame`); признаки и фильтры перебора пересобраны
    функциями night_search в reports/fresh_feat (старый кэш reports/night_search/feat не трогается);
  * RSI(2) — `known_strategies.trades_for` на дневках из тех же признаков.
Свежий период — после конца истории всех трёх бумаг: ни один его день не участвовал в выборе. Обычный ли это
результат — итог свежего периода против всех окон той же длины на проверке 2021 — май 2026 (сдвиг по дню).
На истории сделки S45–S48 сверяются с журналом: совпадают — значит, код и данные согласованы.

    python scripts/fresh_strategies.py
"""
from __future__ import annotations

import gc
import logging
import sys

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import iterator_walkthrough as iw  # noqa: E402  (подключает second_search: акции только в лонг)
import known_strategies as ks  # noqa: E402
import live_signals as ls  # noqa: E402
import strategy_search as ss  # noqa: E402
from experiments import BG, GRID, INK, INK2, ROOT, WF, load_catalog, load_trades  # noqa: E402
from forexmodel.data.dividends import adjust_for_dividends, load_dividends  # noqa: E402
from forexmodel.data.loader import load_minute_compact, resample_ohlcv  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from night_search import build_features, build_gates  # noqa: E402
from trend_grid import instrument_config  # noqa: E402
from walk_forward import _rule_trades  # noqa: E402

FEAT = ROOT / "reports" / "fresh_feat"
DOC = ROOT / "docs" / "results" / "fresh_strategies.md"
PNG = ROOT / "docs" / "results" / "fresh_strategies" / "equity.png"
STOCKS = ["sber", "gazp", "lkoh"]
FRESH = pd.Timestamp("2026-06-12")
TEST = pd.Timestamp("2021-01-01")
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
COLORS = ["#16202b", "#2459b3", "#1d7a55", "#c8641e", "#8a5cc4", "#b4462a"]


def build_feat(inst: str, live: dict) -> pd.Timestamp:
    """Часовые свечи, признаки и фильтры перебора по живому файлу (история + свежие минуты), с поправкой на дивиденды."""
    spec = live["instruments"][inst]
    fp = FEAT / f"{inst}.pkl"
    last = pd.Timestamp(pd.read_csv(ROOT / spec["csv"], usecols=["begin"]).iloc[-1, 0])
    if fp.exists() and pd.read_pickle(fp)["bars"]["time"].iloc[-1] >= last.floor("h"):
        return last
    cfg = instrument_config(inst)
    m = load_minute_compact(ROOT / spec["csv"], cfg.data.time_col, ROOT / "reports" / "_cache", cfg.data.volume_candidates)
    m = adjust_for_dividends(m, load_dividends(ROOT / cfg.data.dividends_path, cfg.data.dividend_ticker))
    h = resample_ohlcv(m, cfg.data.base_timeframe).reset_index(drop=True)
    del m
    pd.to_pickle({"bars": h[["time", "open", "high", "low", "close"]], "f": build_features(h, inst), "gates": build_gates(h)}, fp)
    print(f"[{inst}] признаки перебора по свежему файлу: {len(h)} часов до {h['time'].iloc[-1]}", flush=True)
    return last


def s42(inst: str, live: dict, last: pd.Timestamp, journal: pd.DataFrame) -> pd.DataFrame:
    """История — сделки журнала S42 (акции), свежие — тот же пайплайн на окне с начала 2026 года (как fresh_check):
    окно с 2020 года не помещается в память рядом с остальным."""
    spec = live["instruments"][inst]
    cfg = ls.instrument_cfg(spec, live, pd.Timestamp("2026-01-01"), last + pd.Timedelta(days=1))
    t = _rule_trades(ls.build_dataset(cfg), cfg, "impulse:6:3:long")
    t = recost(t[t["open_dt"] >= FRESH], profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))
    j = journal[(journal["instrument"] == inst) & (journal["open_dt"] < FRESH)]
    return pd.concat([j, t], ignore_index=True)


def stats(t: pd.DataFrame) -> dict:
    p = t["profit_pct"] / 3                         # три бумаги поровну
    loss = -p[p < 0].sum()
    return {"сделок": len(t), "итог": p.sum(), "PF": p[p > 0].sum() / loss if loss > 0 else np.nan,
            "средняя": t["profit_pct"].mean() if len(t) else np.nan}


def window_sums(t: pd.DataFrame, days: int, end: pd.Timestamp) -> np.ndarray:
    """Итог (3 бумаги поровну) по сделкам, открытым в каждом окне [d, d + days) на проверке 2021 — конец истории."""
    starts = pd.date_range(TEST, end - pd.Timedelta(days=days), freq="D")
    o = t["open_dt"].to_numpy("datetime64[ns]")
    order = np.argsort(o)
    o, p = o[order], (t["profit_pct"].to_numpy() / 3)[order]
    cs = np.concatenate([[0], np.cumsum(p)])
    a = np.searchsorted(o, starts.to_numpy("datetime64[ns]"))
    b = np.searchsorted(o, (starts + pd.Timedelta(days=days)).to_numpy("datetime64[ns]"))
    return cs[b] - cs[a]


def main(argv=None) -> int:
    logging.disable(logging.WARNING)
    FEAT.mkdir(parents=True, exist_ok=True)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    lasts = {i: build_feat(i, live) for i in STOCKS}
    gc.collect()
    end = min(lasts.values())
    hist_end = min(pd.Timestamp(pd.read_csv(ROOT / "data" / "raw" / f, usecols=["begin"]).iloc[-1, 0])
                   for f in ("SBER_1m_15-26.csv", "GAZP_1m_15-26_upd.csv", "LKOH_1m_15-26.csv"))
    ss.FEAT, ks.FEAT = FEAT, FEAT                     # движок перебора и дневки — по свежим признакам
    strategies = {"S42 импульс": {}, **{f"{iw.EXP_ID[k]} итератор: {iw.STRATS[k]['entry']}": {} for k in iw.STRATS}, "RSI(2) Коннорса (H56)": {}}
    check = []
    _, exps = load_catalog()
    j42 = load_trades(exps["S42"])
    for inst in STOCKS:
        strategies["S42 импульс"][inst] = s42(inst, live, lasts[inst], j42)
        gc.collect()
        ins = iw.load_inst(inst)
        for k, prm in iw.STRATS.items():
            t = iw.trades_frame(ins, prm)
            strategies[f"{iw.EXP_ID[k]} итератор: {prm['entry']}"][inst] = t
            j = pd.read_csv(WF / f"{inst}_wt{k}__eng_trades.csv", parse_dates=["open_dt"])
            cut = hist_end - pd.Timedelta(days=30)         # сделки, закрытые задолго до конца истории
            a, b = t[t["open_dt"] < cut], j[j["open_dt"] < cut]
            check.append({"стратегия": iw.EXP_ID[k], "бумага": inst, "сделок: заново": len(a), "сделок: журнал": len(b),
                          "итог: заново": a["profit_pct"].sum(), "итог: журнал": b["profit_pct"].sum()})
        del ins
        gc.collect()
        strategies["RSI(2) Коннорса (H56)"][inst] = ks.trades_for(ks.connors_rsi2, inst)
        print(f"[{inst}] сделки посчитаны", flush=True)
    days = (end - FRESH).days
    rows, curves = [], {}
    for name, per in strategies.items():
        t = pd.concat([x.assign(instrument=i) for i, x in per.items() if len(x)], ignore_index=True)
        t = t[pd.to_datetime(t["close_dt"]) <= end]
        fresh, hist = t[t["open_dt"] >= FRESH], t[(t["open_dt"] >= TEST) & (t["open_dt"] < hist_end)]
        w = window_sums(hist, days, hist_end)
        fs, hs = stats(fresh), stats(hist)
        rows.append({"стратегия": name, **{f"свежие: {k}": v for k, v in fs.items()},
                     "проверка: сделок в год": hs["сделок"] / ((hist_end - TEST).days / 365.25), "проверка: PF": hs["PF"],
                     "проверка: средняя": hs["средняя"], "окон того же размера с итогом не выше, %": (w <= fs["итог"]).mean() * 100,
                     "медиана окон": np.median(w)})
        curves[name] = fresh.sort_values("close_dt")
    write_doc(pd.DataFrame(rows), pd.DataFrame(check), curves, end, hist_end, days)
    return 0


def write_doc(res: pd.DataFrame, check: pd.DataFrame, curves: dict, end, hist_end, days) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    for (name, t), c in zip(curves.items(), COLORS):
        ax.step(pd.to_datetime(t["close_dt"]), (t["profit_pct"] / 3).cumsum(), where="post", color=c, linewidth=2.4 if c == COLORS[0] else 1.5, label=name)
    ax.axhline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, % (3 бумаги поровну)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110, facecolor=BG)
    plt.close(fig)
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    f = lambda v, d=1: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{d}f}"  # noqa: E731
    md = ["# Лучшие стратегии на свежих данных", "",
          f"SBER, GAZP, LKOH на бирже только в лонг, сделки, открытые {FRESH:%d.%m.%Y} — {end:%d.%m.%Y} ({days} дней) и закрытые до конца "
          f"данных. Конец истории — {hist_end:%d.%m.%Y}: свежие дни не участвовали ни в выборе, ни в проверке. Итог — на капитал трёх бумаг "
          "поровну, после издержек биржи (0.05% комиссии и 0.05% проскальзывания на сторону). Расчёт — `scripts/fresh_strategies.py`.", "",
          "![Накопленный итог на свежих данных](fresh_strategies/equity.png)", "", verdict, "",
          "## Свежие данные против проверки 2021 — май 2026", "",
          "| стратегия | сделок | итог % | PF | средняя сделка % | проверка: сделок в год | проверка: PF | проверка: средняя % | "
          f"окон по {days} дней на проверке с итогом не выше, % | медиана окон, % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples(index=False):
        md.append(f"| {r[0]} | {r[1]} | {f(r[2])} | {r[3]:.2f} | {f(r[4], 2)} | {r[5]:.0f} | {r[6]:.2f} | {f(r[7], 2)} | {r[8]:.0f} | {f(r[9])} |")
    md += ["", "## Сверка с журналом на истории (S45–S48, сделки до конца истории минус 30 дней)", "",
           "| стратегия | бумага | сделок: заново | в журнале | итог %: заново | в журнале |", "|---|---|---|---|---|---|"]
    for r in check.itertuples(index=False):
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {f(r[4])} | {f(r[5])} |")
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
