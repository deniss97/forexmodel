"""Найденные стратегии на других акциях Мосбиржи — проверка вне выборки по инструментам (гипотеза H66, T41, 23.10).

Стратегии выбраны на SBER, GAZP, LKOH; здесь они без изменений идут на бумаги, которые в выборе не участвовали:
S42 (импульс, набор для запуска), S45–S48 (финалисты итератора), H60 (лонг по трёхслойному фильтру до сигнала вниз).
Отбор бумаг — до расчёта, по правилу `universe`. MOEX и MTSS — отдельной группой: их результат видели 29.09, когда
выбирали набор. Всё только в лонг, издержки биржи, цены с поправкой на дивиденды.

Шаги — по одной бумаге за раз (память); scripts/queues/new_stocks_process.sh делает их по мере выгрузки:
    python scripts/new_stocks.py universe                 # отбор бумаг → docs/results/new_stocks/universe.csv
    scripts/queues/new_stocks_download.sh                 # минутки ISS с 2015 года → data/raw/moex/<БУМАГА>_1m.csv
    python scripts/new_stocks.py prepare [БУМАГИ]         # проверка минуток, объём после сплитов, обрезка до консолидации
    python scripts/fetch_dividends.py --tickers БУМАГИ    # дивиденды в data/dividends/moex_dividends.csv
    python scripts/new_stocks.py bars [БУМАГИ]            # часовые свечи, признаки перебора, трёхслойный фильтр
    python scripts/new_stocks.py trades [БУМАГИ]          # S45–S48 и H60 → reports/walk_forward (как в журнале)
    scripts/queues/new_stocks_s42.sh [БУМАГИ]             # S42 — штатный walk-forward
    python scripts/new_stocks.py report                   # метрики, случайные входы → docs/results/new_stocks.md

Случайность: у каждой сделки стратегии — случайный вход в лонг на той же бумаге в том же периоде с тем же числом
часов в сделке (от открытия часа до открытия часа, издержки биржи). Лонг по акциям в плюсе и от роста рынка — сравнение
с такими входами отделяет умение выбрать момент от роста рынка. Результат сделки — на единицу позиции (у S45–S47
размер по риску; для сравнения моментов входа он не нужен).
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import sys

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import moex_data as md  # noqa: E402
import trend_grid as tg  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, NEW_STOCKS, ROOT, WF  # noqa: E402

NEW = [i.upper() for i in NEW_STOCKS]
SEEN = ["MOEX", "MTSS"]                       # видели 29.09 при выборе набора — отдельная группа
FILES = {**{t: f"moex/{t}_1m" for t in NEW}, "MOEX": "MOEX_1m_15-26", "MTSS": "MTSS_1m_15-26"}
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "new_stocks"
FEAT = OUT / "feat"
DOCDIR = ROOT / "docs" / "results" / "new_stocks"
DOC = ROOT / "docs" / "results" / "new_stocks.md"
EXCHANGE = ROOT / "configs" / "costs" / "exchange.yaml"
PERIODS = {"выбор": ("2016-01-01", "2021-01-01"), "проверка": ("2021-01-01", None)}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
DRAWS = 2000
YELLOW = "#f2b705"


def register() -> None:
    """Новые бумаги — в словари модулей, которые считают стратегии (сами модули не меняются, как в fresh_strategies)."""
    import forexmodel.simulation.costs as costs
    import second_search as s2
    import strategy_search as ss

    names = {t.lower() for t in FILES}
    for t, f in FILES.items():
        tg.INSTRUMENTS.setdefault(t.lower(), ("default.yaml", f, 2016))
    s2.STOCKS.update(names)                   # акции только в лонг и издержки биржи (iterator_walkthrough.load_inst)
    ss.FEAT = FEAT
    pe = costs.profile_entry
    if getattr(pe, "_new_stocks", False):
        return

    def profile_entry(path, instrument, overrides=None):
        # конструктор движка перебора берёт базовый профиль (CFD), где новых бумаг нет; load_inst сразу заменяет
        # его биржевым — для новых бумаг сразу биржевой
        if instrument in names and str(path).endswith("base.yaml"):
            path = EXCHANGE
        return pe(path, instrument, overrides)

    profile_entry._new_stocks = True
    costs.profile_entry = profile_entry


def config(inst: str):
    cfg = tg.instrument_config(inst)
    cfg.data.dividend_ticker = inst.upper()    # у instrument_config тикер — из имени файла («moex/TATN»)
    return cfg


# ---------------------------------------------------------------- отбор бумаг

def universe() -> pd.DataFrame:
    """Состав IMOEX и дневной оборот с 2014 года (ISS). Правило отбора — в NEW: история на TQBR с начала 2015 года,
    медианный оборот 2016–2020 ≥ 100 млн ₽ в день, одна бумага на эмитента, без бумаг выбора и SBERP."""
    j = md.get_json("statistics/engines/stock/markets/index/analytics/IMOEX", limit=100)
    a = pd.DataFrame(j["analytics"]["data"], columns=j["analytics"]["columns"])[["ticker", "shortnames", "weight"]]
    rows = []
    for t in a["ticker"]:
        d = md.candles(t, "2014-06-01", None, 24, pause=0.1)
        if d.empty:
            rows.append({"ticker": t})
            continue
        d["begin"] = pd.to_datetime(d["begin"])
        v = [d[(d["begin"] >= x) & (d["begin"] < y)]["value"].median() / 1e6 for x, y in (("2016", "2021"), ("2021", "2027"))]
        rows.append({"ticker": t, "первый день": d["begin"].min().date(), "оборот 2016–2020, млн ₽": v[0],
                     "оборот 2021–2026, млн ₽": v[1]})
    u = a.merge(pd.DataFrame(rows), on="ticker").sort_values("weight", ascending=False)
    u["группа"] = u["ticker"].map(lambda t: "новая" if t in NEW else "видели при выборе" if t in SEEN
                                  else "выбор стратегий" if t in ("SBER", "GAZP", "LKOH") else "")
    DOCDIR.mkdir(parents=True, exist_ok=True)
    u.to_csv(DOCDIR / "universe.csv", index=False)
    print(u.to_string(index=False))
    return u


# ---------------------------------------------------------------- минутки

def prepare(t: str) -> dict:
    """Проверка файла ISS и две правки на месте (повторный запуск ничего не меняет):
    * консолидация без поправки цены (IRAO 20.01.2015, цена ×118) — история до неё отрезается (только прогрев, до 2016);
    * объём: цены ISS уже с поправкой на сплиты (GMKN, VTBR, PLZL, TRNFP), объём в штуках — нет. Множитель дня —
      оборот / (объём × цена), округлённый до одной значащей цифры (100, 0.0002, 1)."""
    p = RAW / f"{FILES[t]}.csv"
    d = pd.read_csv(p)
    n0 = len(d)
    d = d.drop_duplicates("begin").sort_values("begin", kind="stable").reset_index(drop=True)
    day = d["begin"].str[:10]
    g = d.groupby(day, sort=True)
    jump = (g["open"].first() / g["close"].last().shift(1)).dropna()
    bad = jump[(jump > 1.8) | (jump < 0.55)]
    cut = None
    if len(bad):
        cut = bad.index[-1]
        if cut >= "2016-01-01":
            raise ValueError(f"{t}: скачок цены без поправки {bad.to_dict()} — после начала проверки, нужен разбор")
        d = d[d["begin"] >= cut].reset_index(drop=True)
        day = d["begin"].str[:10]
    ok = (d["volume"] > 0) & (d["close"] > 0)
    f = (d["value"] / (d["volume"] * d["close"])).where(ok)
    k = f.groupby(day).transform("median").fillna(1.0).map(lambda x: float(f"{x:.1g}"))
    changed = int((k != 1).sum())
    if changed or cut or len(d) != n0:
        d["volume"] = (d["volume"] * k).round().astype("int64")
        d.to_csv(p, index=False)
    r = (d["close"] / d["open"]).to_numpy()
    info = {**md.check(d), "отрезано до": cut, "минут с объёмом ×": changed,
            "множители": sorted(set(k.unique()) - {1.0}), "макс. ход за минуту, %": round(float(np.abs(r - 1).max() * 100), 1)}
    print(t, info, flush=True)
    return info


# ---------------------------------------------------------------- часовые свечи и признаки

def bars(inst: str) -> None:
    """Как fresh_strategies.build_feat: часовые свечи с поправкой на дивиденды, признаки и фильтры перебора; плюс
    трёхслойный фильтр (режим по дневной EMA200, H55) для H60."""
    from forexmodel.data.dividends import adjust_for_dividends, load_dividends
    from forexmodel.data.loader import load_minute_compact, resample_ohlcv
    from night_search import build_features, build_gates
    from trend_filter_lab import base_config
    from trend_follow import TL

    FEAT.mkdir(parents=True, exist_ok=True)
    cfg = config(inst)
    divs = load_dividends(ROOT / cfg.data.dividends_path, cfg.data.dividend_ticker)
    if divs.empty:
        raise ValueError(f"{inst}: нет дивидендов — сначала scripts/fetch_dividends.py --tickers {inst.upper()}")
    m = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache", cfg.data.volume_candidates)
    m = adjust_for_dividends(m, divs)
    h = resample_ohlcv(m, cfg.data.base_timeframe).reset_index(drop=True)
    del m
    tg._W["base"] = dataclasses.replace(base_config(), **tg.GATE_C)
    b = h[["time", "open", "high", "low", "close"]]
    pd.to_pickle({"bars": b, "f": build_features(h, inst), "gates": build_gates(h), "tl": tg.gate_values(b, TL)},
                 FEAT / f"{inst}.pkl")
    print(f"[{inst}] {len(h)} часов {h['time'].iloc[0]:%Y-%m-%d} — {h['time'].iloc[-1]:%Y-%m-%d}, отсечек {len(divs)}", flush=True)


def trades(inst: str) -> None:
    """S45–S48 — движок перебора (iterator_walkthrough), H60 — trend_follow.follow; файлы как у журнала."""
    import iterator_walkthrough as iw
    from trend_follow import follow

    d = pd.read_pickle(FEAT / f"{inst}.pkl")
    h = follow(d["bars"], d["tl"], long_only=True, hold=True)
    h.to_csv(WF / f"{inst}_tf3hold__follow_trades.csv", index=False)
    ins = iw.load_inst(inst)
    out = [f"H60 {len(h)}"]
    for k, prm in iw.STRATS.items():
        t = iw.trades_frame(ins, prm)
        t = t[t["open_dt"] >= "2016-01-01"] if len(t) else t
        t.to_csv(WF / f"{inst}_wt{k}__eng_trades.csv", index=False)
        out.append(f"{iw.EXP_ID[k]} {len(t)}")
    print(f"[{inst}] сделок: " + ", ".join(out), flush=True)


# ---------------------------------------------------------------- итог

STRATEGIES = {"S42": "{i}_lo__impulse_6_3_long", "S45": "{i}_wt1__eng", "S46": "{i}_wt2__eng", "S47": "{i}_wt3__eng",
              "S48": "{i}_wt4__eng", "H60": "{i}_tf3hold__follow"}
TITLES = {"S42": "S42 импульс (запуск)", "S45": "S45 NR7 → пробой", "S46": "S46 пробой 55 дней", "S47": "S47 канал Кельтнера",
          "S48": "S48 пробой 20 дней + 0.25 ATR", "H60": "H60 лонг по трёхслойному фильтру"}
JOURNAL_ID = {"H60": "S50"}
GROUPS = {"новые 16": [t.lower() for t in NEW], "MOEX, MTSS": [t.lower() for t in SEEN], "SBER, GAZP, LKOH": ["sber", "gazp", "lkoh"]}
M0 = np.datetime64("2016-01", "M")
_bars: dict = {}


def load(code: str, insts: list[str]) -> pd.DataFrame:
    """Сделки стратегии на бумагах: новые — из reports/walk_forward, бумаги выбора — из журнала."""
    from experiments import load_catalog, load_trades

    if insts == GROUPS["SBER, GAZP, LKOH"]:
        _, exps = load_catalog()
        t = load_trades(exps[JOURNAL_ID.get(code, code)])
        return t[t["instrument"].isin(insts)]
    src = {i: [STRATEGIES[code].format(i=i)] for i in insts if (WF / f"{STRATEGIES[code].format(i=i)}_trades.csv").exists()}
    return load_trades({"id": code, "sources": src, "costs": {"profile": "exchange"}}) if src else pd.DataFrame()


def unit_net(t: pd.DataFrame) -> np.ndarray:
    s = t["size"].to_numpy(float) if "size" in t else np.ones(len(t))
    return t["profit_pct"].to_numpy(float) / np.where(s > 0, s, 1.0)


def month(t) -> np.ndarray:
    return (np.asarray(t, dtype="datetime64[M]") - M0).astype(int)


def in_period(t: pd.DataFrame, per: str) -> np.ndarray:
    a, b = PERIODS[per]
    return ((t["open_dt"] >= a) & ((t["open_dt"] < b) if b else True)).to_numpy()


def random_draws(t: pd.DataFrame, inst: str, rng: np.random.Generator, cost: float) -> dict:
    """По периодам: (результат, месяц выхода) случайных входов в лонг, матрицы DRAWS × сделок периода. Вход — открытие
    случайного часа периода, выход — открытие часа через столько же баров, сколько длилась сделка; минус издержки за круг."""
    if inst not in _bars:
        b = pd.read_pickle(FEAT / f"{inst}.pkl")["bars"]
        _bars[inst] = (b["time"].to_numpy(), b["open"].to_numpy(float))
    tt, o = _bars[inst]
    mon = month(tt)
    e = np.searchsorted(tt, t["open_dt"].dt.floor("h").to_numpy(), "left")
    x = np.searchsorted(tt, t["close_dt"].dt.floor("h").to_numpy(), "left")
    span = np.maximum(x - e, 1)
    out = {}
    for per, (a, b) in PERIODS.items():
        s = span[in_period(t, per)]
        lo = int(np.searchsorted(tt, np.datetime64(a)))
        hi = int(np.searchsorted(tt, np.datetime64(b))) if b else len(tt)
        r = lo + (rng.random((DRAWS, len(s))) * np.maximum(hi - lo - s - 1, 1)).astype(np.int64)
        j = np.minimum(r + s, len(o) - 1)
        out[per] = (((o[j] / o[r] - 1) * 100 - cost).astype(np.float32), mon[j].astype(np.int32))
    return out


def pf(v: np.ndarray, axis=None) -> np.ndarray:
    gain = np.where(v > 0, v, 0).sum(axis=axis)
    loss = -np.where(v < 0, v, 0).sum(axis=axis)
    return np.where(loss > 0, gain / np.where(loss > 0, loss, 1), np.inf)


def report() -> int:
    from forexmodel.evaluation.experiments import metrics
    from forexmodel.simulation.costs import profile_entry

    logging.disable(logging.WARNING)
    rng = np.random.default_rng(66)
    cost = profile_entry(EXCHANGE, "sber")["spread_pct"]
    n_months = int(month(np.datetime64("today")) + 1)
    summary, per_inst, curves = [], [], {}
    for code in STRATEGIES:
        for g, insts in GROUPS.items():
            t = load(code, insts)
            if t.empty:
                continue
            have = sorted(t["instrument"].unique())
            row = {"стратегия": code, "группа": g, "бумаг": len(have)}
            for per in PERIODS:
                m = metrics(t, have, per)
                row.update({f"{per}: {k}": m.get(k) for k in ("сделок", "итог", "PF", "без_2_лучших", "t")})
            works = {}
            for i in have:
                r = {"стратегия": code, "группа": g, "бумага": i.upper()}
                for per in PERIODS:
                    m = metrics(t, [i], per)
                    r.update({f"{per}: {k}": m.get(k) for k in ("сделок", "итог", "PF", "без_2_лучших")})
                works[i] = bool((r["выбор: PF"] or 0) > 1.2 and (r["проверка: PF"] or 0) > 1.2)
                per_inst.append({**r, "работает": works[i]})
            row["в плюсе на проверке"] = sum(1 for r in per_inst[-len(have):] if (r["проверка: итог"] or 0) > 0)
            row["работает (PF > 1.2 в обоих)"] = sum(works.values())
            if g != "SBER, GAZP, LKOH":
                draws = {i: random_draws(t[t["instrument"] == i], i, rng, cost) for i in have}
                for per in PERIODS:
                    real = unit_net(t[in_period(t, per)]).mean()
                    rand = np.concatenate([draws[i][per][0] for i in have], axis=1).mean(axis=1)
                    row[f"{per}: средняя сделка, %"] = real
                    row[f"{per}: случайные, %"] = float(rand.mean())
                    row[f"{per}: p"] = (1 + int((rand >= real).sum())) / (1 + DRAWS)
                ok = sum(((pf(draws[i]["выбор"][0], 1) > 1.2) & (pf(draws[i]["проверка"][0], 1) > 1.2)).astype(int) for i in have)
                row["работает: случайно в среднем"] = float(ok.mean())
                row["работает: p"] = (1 + int((ok >= row["работает (PF > 1.2 в обоих)"]).sum())) / (1 + DRAWS)
                if g == "новые 16":
                    curves[code] = curve(t, draws, have, n_months)
                del draws
            summary.append(row)
            gc.collect()
    s, pi = pd.DataFrame(summary), pd.DataFrame(per_inst)
    hold = hold_vs_rule()
    DOCDIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(DOCDIR / "summary.csv", index=False)
    pi.to_csv(DOCDIR / "per_instrument.csv", index=False)
    hold.to_csv(DOCDIR / "h60_vs_hold.csv", index=False)
    plot_curves(curves, s)
    plot_map(pi)
    write_doc(s, pi, hold)
    return 0


def curve(t: pd.DataFrame, draws: dict, have: list[str], n: int) -> dict:
    """Накопленный результат по месяцам выхода (бумаги поровну, на единицу позиции): стратегия и коридор случайных."""
    real = np.bincount(month(t["close_dt"].to_numpy()).clip(0, n - 1), weights=unit_net(t) / len(have), minlength=n)[:n]
    ret = np.concatenate([draws[i][per][0] for i in have for per in PERIODS], axis=1)
    mon = np.concatenate([draws[i][per][1] for i in have for per in PERIODS], axis=1).clip(0, n - 1)
    sims = np.stack([np.bincount(mon[k], weights=ret[k], minlength=n)[:n] for k in range(DRAWS)]) / len(have)
    q = np.percentile(np.cumsum(sims, axis=1), [5, 50, 95], axis=0)
    return {"real": np.cumsum(real), "p5": q[0], "p50": q[1], "p95": q[2], "n": len(have)}


def hold_vs_rule() -> pd.DataFrame:
    """H60 против «купить и держать» на капитале с реинвестированием (как trend_hold): по бумаге и бумаги поровну."""
    import trend_hold as th
    from forexmodel.simulation.costs import profile_entry

    cost = profile_entry(EXCHANGE, "sber")["spread_pct"]
    rows = []
    for g in ("новые 16", "MOEX, MTSS"):
        for per, (a, b) in PERIODS.items():
            curves = {}
            for i in [i for i in GROUPS[g] if (FEAT / f"{i}.pkl").exists()]:
                d = pd.read_pickle(FEAT / f"{i}.pkl")
                curves[i] = th.marked(d["bars"], d["tl"], cost, a, b)
                rr, hh = th.risk(th.daily(curves[i]["rule"])), th.risk(th.daily(curves[i]["hold"]))
                rows.append({"группа": g, "период": per, "бумага": i.upper(), **{f"правило: {k}": v for k, v in rr.items()},
                             **{f"держать: {k}": v for k, v in hh.items()}})
            day = {k: pd.concat([th.daily(c[k]) for c in curves.values()], axis=1).ffill().fillna(1.0).mean(axis=1)
                   for k in ("rule", "hold")}
            rr, hh = th.risk(day["rule"]), th.risk(day["hold"])
            rows.append({"группа": g, "период": per, "бумага": "поровну", **{f"правило: {k}": v for k, v in rr.items()},
                         **{f"держать: {k}": v for k, v in hh.items()}})
    return pd.DataFrame(rows)


def _axes(ax) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_curves(curves: dict, s: pd.DataFrame) -> None:
    """Панель на стратегию: накопленный результат на 16 новых бумагах против коридора случайных входов."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), facecolor=BG, sharex=True)
    for ax, code in zip(axes.flat, STRATEGIES):
        _axes(ax)
        c = curves.get(code)
        if c is None:
            ax.set_visible(False)
            continue
        x = (M0 + np.arange(len(c["real"]))).astype("datetime64[D]")
        ax.fill_between(x, c["p5"], c["p95"], color=GRAY, alpha=0.28, linewidth=0, label="случайные входы, 5–95%")
        ax.plot(x, c["p50"], color=GRAY, linewidth=1.2, linestyle="--", label="случайные входы, медиана")
        ax.plot(x, c["real"], color=BLUE, linewidth=2, label="стратегия")
        ax.axvline(np.datetime64("2021-01-01"), color=INK2, linewidth=0.8)
        ax.axhline(0, color=INK2, linewidth=0.6)
        r = s[(s["стратегия"] == code) & (s["группа"] == "новые 16")].iloc[0]
        ax.set_title(f"{TITLES[code]}\np на проверке {r['проверка: p']:.3f}, PF {r['проверка: PF']:.2f}", color=INK, fontsize=9, loc="left")
    axes.flat[0].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    n = max(c["n"] for c in curves.values()) if curves else 0
    for ax in axes[:, 0]:
        ax.set_ylabel(f"накоплено, % ({n} бумаг поровну)", color=INK2, fontsize=8)
    for ax in axes[1]:
        ax.text(np.datetime64("2021-02-01"), ax.get_ylim()[1], "проверка →", va="top", color=INK2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DOCDIR / "random.png", dpi=110, facecolor=BG)
    plt.close(fig)


def plot_map(pi: pd.DataFrame) -> None:
    """PF на проверке по бумагам и стратегиям; рамка — PF > 1.2 и на выборе, и на проверке."""
    order = [(g, i.upper()) for g in ("SBER, GAZP, LKOH", "MOEX, MTSS", "новые 16") for i in GROUPS[g]]
    codes = list(STRATEGIES)
    v = np.full((len(order), len(codes)), np.nan)
    w = np.zeros_like(v, dtype=bool)
    for r in pi.itertuples(index=False):
        if (r.группа, r.бумага) in order:
            k, c = order.index((r.группа, r.бумага)), codes.index(r.стратегия)
            v[k, c] = r[pi.columns.get_loc("проверка: PF")]
            w[k, c] = r.работает
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(order) + 1.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(np.clip(v, 0.5, 1.5), cmap="RdBu", vmin=0.5, vmax=1.5, aspect="auto")
    for k in range(len(order)):
        for c in range(len(codes)):
            if np.isfinite(v[k, c]):
                dark = abs(min(max(v[k, c], 0.5), 1.5) - 1) > 0.3          # на насыщенной клетке — светлый текст
                ax.text(c, k, f"{v[k, c]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if dark else INK,
                        fontweight="bold" if w[k, c] else "normal")
            if w[k, c]:                                                   # «работает» — жёлтым (Денис, созвон 23.10)
                ax.add_patch(plt.Rectangle((c - 0.44, k - 0.41), 0.88, 0.82, fill=False, edgecolor=YELLOW, linewidth=2.2))
    ax.set_xticks(range(len(codes)), codes)
    ax.set_yticks(range(len(order)), [b for _, b in order])
    ax.xaxis.tick_top()
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for y in (2.5, 4.5):
        ax.axhline(y, color=BG, linewidth=4)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("PF на проверке 2021–2026 (жёлтая рамка — PF > 1.2 и на выборе 2016–2020)", color=INK, fontsize=9, pad=24)
    fig.tight_layout()
    fig.savefig(DOCDIR / "map.png", dpi=110, facecolor=BG)
    plt.close(fig)


def _f(v, d=2, sign=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def verdicts(s: pd.DataFrame, hold: pd.DataFrame) -> dict:
    """Критерий H66 (записан в банк до расчёта) — по каждой стратегии на 16 новых бумагах."""
    out = {}
    for code in STRATEGIES:
        r = s[(s["стратегия"] == code) & (s["группа"] == "новые 16")]
        if r.empty:
            continue
        r = r.iloc[0]
        c = {"проверка PF > 1.1": r["проверка: PF"] > 1.1, "без 2 лучших > 0": r["проверка: без_2_лучших"] > 0,
             "выбор PF > 1.0": r["выбор: PF"] > 1.0, "в плюсе ≥ 8 из 16": r["в плюсе на проверке"] >= 8,
             "лучше случайных (p < 0.05)": r["проверка: p"] < 0.05}
        if code == "H60":
            h = hold[hold["группа"] == "новые 16"]
            c = {"итог/просадка лучше держания: поровну, оба периода":
                 bool((h[h["бумага"] == "поровну"]["правило: и/п"] > h[h["бумага"] == "поровну"]["держать: и/п"]).all())}
            for per in PERIODS:
                x = h[(h["период"] == per) & (h["бумага"] != "поровну")]
                c[f"на большинстве бумаг: {per}"] = bool((x["правило: и/п"] > x["держать: и/п"]).sum() > len(x) / 2)
        out[code] = c
    return out


def write_doc(s: pd.DataFrame, pi: pd.DataFrame, hold: pd.DataFrame) -> None:
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    v = verdicts(s, hold)
    new = s[s["группа"] == "новые 16"]
    md_ = ["# Найденные стратегии на других акциях Мосбиржи (H66)", "",
           "Стратегии, выбранные на SBER, GAZP, LKOH, — без изменений на 16 акциях индекса Мосбиржи, которые ни разу не "
           "участвовали в выборе: TATN, NVTK, GMKN, ROSN, VTBR, PLZL, SNGSP, IRAO, CHMF, NLMK, MAGN, RTKM, TRNFP, AFLT, ALRS, AFKS. "
           "Отбор — до расчёта: в индексе на 23.10, история на TQBR с 2015 года, медианный оборот 2016–2020 ≥ 100 млн ₽ в день, "
           "одна бумага на эмитента ([universe.csv](new_stocks/universe.csv)). MOEX и MTSS — отдельно: их видели 29.09, когда "
           "выбирали набор. Только лонг, издержки биржи (0.1% комиссии и 0.05% проскальзывания на сторону), цены с поправкой "
           "на дивиденды; выбор 2016–2020, проверка 2021 — октябрь 2026. Бумаги поровну. Критерий записан в банк (H66) до расчёта. "
           "Расчёт — `scripts/new_stocks.py`.", "", verdict, "",
           "## Критерий H66 на 16 новых бумагах", "",
           "| стратегия | проверка: сделок | итог % | PF | без 2 лучших | t | выбор: PF | итог % | в плюсе на проверке | "
           "средняя сделка, % | у случайных входов, % | p | критерий |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in new.to_dict("records"):
        c = v.get(g["стратегия"], {})
        ok = "✅ подтвердилась" if c and all(c.values()) else "❌ " + "; ".join(k for k, val in c.items() if not val)
        md_.append(f"| {TITLES[g['стратегия']]} | {g['проверка: сделок']} | {_f(g['проверка: итог'], 1, True)} | {_f(g['проверка: PF'])} | "
                   f"{_f(g['проверка: без_2_лучших'], 1, True)} | {_f(g['проверка: t'])} | {_f(g['выбор: PF'])} | {_f(g['выбор: итог'], 1, True)} | "
                   f"{g['в плюсе на проверке']} из {g['бумаг']} | {_f(g['проверка: средняя сделка, %'], 3, True)} | "
                   f"{_f(g['проверка: случайные, %'], 3, True)} | {g['проверка: p']:.3f} | {ok} |")
    md_ +=["", "Средняя сделка — на единицу позиции, после издержек. Случайные входы — у каждой сделки случайный вход в лонг на "
            "той же бумаге в том же периоде с той же длительностью (2000 повторов); p — доля повторов, где случайные входы "
            "дали не меньше. Рост рынка лонгу помогает и без стратегии, поэтому сравнение с нулём здесь недостаточно.", "",
            "![Стратегии на 16 новых бумагах против случайных входов](new_stocks/random.png)", "",
            "## Где работает: по бумагам", "",
            "PF на проверке; рамка и 🟨 — PF > 1.2 и на выборе, и на проверке. Строка «случайно» — сколько таких бумаг в "
            "среднем дали бы случайные входы с теми же числом и длительностью сделок, и p — как часто случайно столько же или больше.", "",
            "![PF по бумагам](new_stocks/map.png)", "",
            "| бумага | " + " | ".join(STRATEGIES) + " |", "|---|" + "---|" * len(STRATEGIES)]
    for g in ("новые 16", "MOEX, MTSS", "SBER, GAZP, LKOH"):
        md_.append(f"| **{g}** |" + " |" * len(STRATEGIES))
        for i in GROUPS[g]:
            cells = []
            for code in STRATEGIES:
                x = pi[(pi["стратегия"] == code) & (pi["бумага"] == i.upper()) & (pi["группа"] == g)]
                if x.empty:
                    cells.append("—")
                    continue
                x = x.iloc[0]
                cells.append(f"{'🟨 ' if x['работает'] else ''}{_f(x['проверка: PF'])} ({_f(x['выбор: PF'])})")
            md_.append(f"| {i.upper()} | " + " | ".join(cells) + " |")
        if g != "SBER, GAZP, LKOH":
            row = []
            for code in STRATEGIES:
                x = s[(s["стратегия"] == code) & (s["группа"] == g)]
                row.append("—" if x.empty else f"{int(x.iloc[0]['работает (PF > 1.2 в обоих)'])}; случайно "
                           f"{x.iloc[0]['работает: случайно в среднем']:.1f}, p {x.iloc[0]['работает: p']:.2f}")
            md_.append("| 🟨 бумаг | " + " | ".join(row) + " |")
    md_ += ["", "В скобках — PF на выборе 2016–2020.", "", "## H60 против «купить и держать»", "",
            "Капитал с реинвестированием, стоимость позиции каждый час (как в [trend_hold.md](trend_hold.md)); и/п — итог к просадке.", "",
            "| группа | период | бумаг, где правило лучше по и/п | правило: итог % | просадка % | и/п | держать: итог % | просадка % | и/п |",
            "|---|---|---|---|---|---|---|---|---|"]
    for (g, per), x in hold.groupby(["группа", "период"], sort=False):
        p = x[x["бумага"] == "поровну"].iloc[0]
        each = x[x["бумага"] != "поровну"]
        md_.append(f"| {g} | {per} | {int((each['правило: и/п'] > each['держать: и/п']).sum())} из {len(each)} | "
                   f"{_f(p['правило: итог'], 1, True)} | {_f(p['правило: просадка'], 1)} | {_f(p['правило: и/п'])} | "
                   f"{_f(p['держать: итог'], 1, True)} | {_f(p['держать: просадка'], 1)} | {_f(p['держать: и/п'])} |")
    md_ += ["", "## Для сравнения: MOEX, MTSS и бумаги выбора", "",
            "| группа | стратегия | проверка: итог % | PF | без 2 лучших | выбор: итог % | PF | p на проверке |", "|---|---|---|---|---|---|---|---|"]
    for g in s[s["группа"] != "новые 16"].to_dict("records"):
        p = g.get("проверка: p")
        md_.append(f"| {g['группа']} | {TITLES[g['стратегия']]} | {_f(g['проверка: итог'], 1, True)} | {_f(g['проверка: PF'])} | "
                   f"{_f(g['проверка: без_2_лучших'], 1, True)} | {_f(g['выбор: итог'], 1, True)} | {_f(g['выбор: PF'])} | "
                   f"{'—' if p is None or not np.isfinite(p) else f'{p:.3f}'} |")
    md_ += ["", "На бумагах выбора — сделки из журнала (S42, S45–S48, S50), тот же протокол.", ""]
    DOC.write_text("\n".join(md_), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["universe", "prepare", "bars", "trades", "report"])
    ap.add_argument("tickers", nargs="*", help="по умолчанию — все новые и MOEX, MTSS (prepare — только новые)")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    if a.step == "universe":
        universe()
        return 0
    register()
    if a.step == "report":
        return report()
    todo = [t.upper() for t in a.tickers] or (NEW if a.step == "prepare" else NEW + SEEN)
    for t in todo:
        {"prepare": prepare, "bars": lambda x: bars(x.lower()), "trades": lambda x: trades(x.lower())}[a.step](t)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
