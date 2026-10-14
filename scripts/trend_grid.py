"""Перебор тренд-фильтров импульса по ТЗ 15.10 (гипотезы H44 — трёхслойный фильтр, H45 — классические
индикаторы EMA / MACD / Supertrend / Дончиан / Кельтнер / Боллинджер / ADX / KAMA на 1 ч, 4 ч и днях).

Как сделано быстро и при этом точно (как штатный walk-forward базы B01):
  1. `candidates`: на каждом инструменте все сигналы импульса 6 ч / 3 ATR БЕЗ тренд-фильтра прогоняются
     штатным симулятором с разрешённым перекрытием — у каждого сигнала своя сделка (дивиденды акций,
     стоп по гэпу, выход 240 ч / стоп и трейлинг 6 ATR / активация 3), затем издержки базы B01
     (configs/costs/base.yaml: комиссия, проскальзывание 0.05% на сторону, свопы). Окна — годы, как в
     walk-forward: сигналы считаются внутри года, позиция не переходит через границу окна в цепочке.
  2. Тренд-фильтр только решает, брать ли сигнал (сторона сигнала = знак фильтра на баре сигнала), а
     сделки без перекрытия отбираются жадной цепочкой по временам выхода — ровно правило симулятора.
  3. Метрики — те же функции, что у журнала экспериментов (forexmodel/evaluation/experiments.py):
     «выбор» — сделки до 2021 года, «проверка» — с 2021-го. Выбор финалистов — только по выбору.

    python scripts/trend_grid.py candidates          # ~10–15 мин, reports/trend_grid/candidates.pkl
    python scripts/trend_grid.py grid --workers 3    # таблица всех фильтров, reports/trend_grid/grid.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.config import load_config
from forexmodel.evaluation.experiments import metrics, scopes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "trend_grid"
CAND = OUT / "candidates.pkl"
# инструмент -> (конфиг, минутный файл, первый год окна) — как в scripts/queues/costs.sh (база B01)
INSTRUMENTS = {
    "silver": ("silver.yaml", "XAGUSD_1m_full", 2016), "gold": ("silver.yaml", "XAUUSD_1m_full", 2016),
    "lkoh": ("default.yaml", "LKOH_1m_15-26", 2016), "gazp": ("default.yaml", "GAZP_1m_15-26_upd", 2016),
    "sber": ("default.yaml", "SBER_1m_15-26", 2016), "moex": ("default.yaml", "MOEX_1m_15-26", 2016),
    "mtss": ("default.yaml", "MTSS_1m_15-26", 2016), "eth": ("silver.yaml", "ETHUSDT_1m_clean", 2021),
}
LAST_YEAR = 2026
GATE_C = dict(mode="early", require_adx_rising=False, update_every_bar=True, hold_bars=2)


def instrument_config(inst: str):
    conf, csv, _ = INSTRUMENTS[inst]
    cfg = load_config(ROOT / "configs" / conf)
    cfg.data.csv_path = f"data/raw/{csv}.csv"
    if conf == "default.yaml":
        cfg.data.dividends_path = "data/dividends/moex_dividends.csv"
        cfg.data.dividend_ticker = csv.split("_")[0]
    s = cfg.simulation
    s.horizon_minutes, s.sl_atr, s.trail_atr, s.activate_atr, s.stop_fill = 14400, 6.0, 6.0, 3.0, "gap"
    s.signal_source, s.impulse_bars, s.impulse_atr = "impulse", 6, 3.0
    s.use_expected_value_filter, s.use_trend_filter, s.allow_overlapping_positions = False, False, True
    return cfg


# ---------------------------------------------------------------- 1. сделки всех сигналов

def build_candidates(only: list[str] | None) -> int:
    from forexmodel.data.dividends import adjust_for_dividends, load_dividends
    from forexmodel.data.loader import load_minute_compact, resample_ohlcv
    from forexmodel.features import indicators as ind
    from forexmodel.simulation.costs import profile_entry, recost
    from forexmodel.simulation.signals import build_signal_column
    from forexmodel.simulation.simulator import simulate_trades

    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(CAND) if CAND.exists() else {}
    for inst in only or list(INSTRUMENTS):
        t0 = time.time()
        cfg = instrument_config(inst)
        minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache",
                                     cfg.data.volume_candidates)
        if cfg.data.dividends_path:
            minute = adjust_for_dividends(minute, load_dividends(ROOT / cfg.data.dividends_path, cfg.data.dividend_ticker))
        h = resample_ohlcv(minute, cfg.data.base_timeframe)
        h[cfg.atr_col] = ind.atr(h, cfg.features.atr_period)
        sigs = []
        for y in range(INSTRUMENTS[inst][2], LAST_YEAR + 1):
            w = h[(h["time"] >= f"{y}-01-01") & (h["time"] < f"{y + 1}-01-01")]
            if len(w):
                s = build_signal_column(w.copy(), cfg)
                sigs.append(s[s["final_class"].notna()].assign(window=y))
        sig = pd.concat(sigs, ignore_index=True)
        trades, _ = simulate_trades(sig, minute, cfg)
        trades = recost(trades, profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst))
        trades = trades.merge(sig[["time", "window"]].rename(columns={"time": "signal_dt"}), on="signal_dt")
        data[inst] = {"bars": h[["time", "open", "high", "low", "close"]].reset_index(drop=True), "trades": trades}
        print(f"[{inst}] сигналов {len(sig)}, сделок {len(trades)} ({time.time() - t0:.0f} с)", flush=True)
        del minute
        pd.to_pickle(data, CAND)
    return 0


# ---------------------------------------------------------------- 2. отбор сделок по фильтру

def chain(sig_t: np.ndarray, close_t: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Жадная цепочка без перекрытия, как в симуляторе: сигнал пропускается, если он не позже выхода
    предыдущей сделки. Возвращает индексы взятых сделок."""
    idx = np.flatnonzero(elig)
    out, last = [], None
    for i in idx:
        if last is not None and sig_t[i] <= last:
            continue
        out.append(i)
        last = close_t[i]
    return np.asarray(out, dtype=int)


def select(ev: dict, gate: np.ndarray | None) -> pd.DataFrame:
    t = ev["trades"]
    side = np.where(t["side"].to_numpy() == "buy", 1.0, -1.0)
    elig = np.ones(len(t), bool)
    if gate is not None:
        g = pd.Series(gate, index=ev["bars"]["time"]).reindex(t["signal_dt"]).to_numpy(dtype=float)
        elig = ~np.isfinite(g) | (g == side)          # NaN гейта пропускает сигнал — как в симуляторе
    keep = []
    for _, grp in t.groupby("window", sort=True):
        ii = grp.index.to_numpy()
        keep.append(ii[chain(grp["signal_dt"].to_numpy(), grp["close_dt"].to_numpy(), elig[ii])])
    return t.loc[np.concatenate(keep)] if keep else t.iloc[:0]


def standard(t: pd.DataFrame, inst: str) -> pd.DataFrame:
    return pd.DataFrame({"instrument": inst, "run": "grid", "side": np.where(t["side"] == "buy", 1, -1),
                         "open_dt": pd.to_datetime(t["open_dt"]), "close_dt": pd.to_datetime(t["close_dt"]),
                         "profit_pct": t["profit_pct"].astype(float)})


# ---------------------------------------------------------------- 3. варианты фильтров

def classic_rules() -> list[str]:
    r = [f"ema_price:{n}" for n in (9, 21, 50, 100, 200)]
    r += ["ema_slope:21:3", "ema_slope:50:5", "ema_slope:200:10"]
    r += [f"ema_cross:{a}:{b}" for a, b in ((9, 21), (12, 26), (20, 50), (50, 200), (8, 21), (13, 34), (21, 55), (34, 89))]
    r += [f"macd:{m}:{k}" for m in ("12:26:9", "8:17:9", "5:35:5", "19:39:9") for k in ("sig", "zero", "hist")]
    r += ["supertrend:10:3", "supertrend:7:3", "supertrend:10:2"]
    r += ["donchian:20:10", "donchian:55:20", "keltner:20:10:2", "bb:20:2", "adx:14:20", "adx:14:25", "kama:10"]
    return r


SCORE_VARIANTS = {
    "ТЗ как есть": {},
    "режим по 4ч EMA200": {"score_regime_tf": "4h"},
    "режим по дневной EMA200": {"score_regime_tf": "24h"},
    "удержание 6 ч": {"score_min_hold": 6},
    "удержание 24 ч": {"score_min_hold": 24},
    "KAMA вместо пар EMA": {"score_fast": "kama"},
    "окна от волатильности": {"score_vol_adapt": True},
    "шок + Дончиан 48 ч": {"score_shock_donchian": 48},
    "без шока": {"score_shock_z": 99.0},
    "пары ×4 (как на 4ч)": {"score_spans": [[32, 128], [64, 256], [128, 512]], "score_vol_window": 192},
    "пары ×24 (как на днях)": {"score_spans": [[192, 768], [384, 1536], [768, 3072]], "score_vol_window": 1152},
    "полный: дневной режим, удержание 12, Дончиан 48, окна от волат.": {
        "score_regime_tf": "24h", "score_min_hold": 12, "score_shock_donchian": 48, "score_vol_adapt": True},
    "порог входа 0.5": {"score_enter": 0.5},
    "порог входа 0.15": {"score_enter": 0.15},
}


def gate_items() -> list[tuple[str, str, dict | None]]:
    """(группа, имя, параметры TrendConfig поверх гейта C или None — без фильтра / «всегда лонг»)."""
    items = [("эталон", "без фильтра", None), ("эталон", "всегда лонг", {"always": 1}), ("эталон", "гейт C (база)", {})]
    items += [("H44 трёхслойный", f"скор: {k}", {"mode": "score", **v}) for k, v in SCORE_VARIANTS.items()]
    for tf in ("1h", "4h", "24h"):
        items += [("H45 " + r.split(":")[0], f"{r} @{tf}", {"mode": "classic", "rule": r, "timeframe": tf,
                                                             "update_every_bar": True, "hold_bars": 0})
                  for r in classic_rules()]
    return items


_W: dict = {}


def _init():
    from trend_filter_lab import base_config

    _W["data"] = pd.read_pickle(CAND)
    _W["base"] = dataclasses.replace(base_config(), **GATE_C)


def gate_values(bars: pd.DataFrame, params: dict) -> np.ndarray:
    from forexmodel.features.trend import add_trend_filter

    if "always" in params:
        return np.full(len(bars), float(params["always"]))
    tc = dataclasses.replace(_W["base"], **params)
    return add_trend_filter(bars, tc)["trend_4h"].to_numpy(dtype=float)


def evaluate(item):
    group, name, params = item
    try:
        parts, share = [], {}
        for inst, ev in _W["data"].items():
            gate = None if params is None else gate_values(ev["bars"], params)
            if gate is not None:
                share[inst] = float(np.mean(gate[np.isfinite(gate)] != 0))
            parts.append(standard(select(ev, gate), inst))
        t = pd.concat(parts, ignore_index=True)
        rows = []
        for scope, members in scopes(list(_W["data"])).items():
            if scope not in ("все", "портфель 4", "металлы"):
                continue
            for period in ("выбор", "проверка"):
                m = metrics(t, members, period)
                rows.append({"группа": group, "фильтр": name, "срез": scope, "период": period,
                             **{k: m.get(k) for k in ("сделок", "итог", "PF", "без_2_лучших", "просадка", "t",
                                                      "лет_в_плюсе")}})
        years = t.assign(y=t["open_dt"].dt.year).groupby(["y", "instrument"])["profit_pct"].sum().unstack(fill_value=0)
        return rows, {"фильтр": name, "доля_тренда": float(np.mean(list(share.values()))) if share else 1.0,
                      "по_годам": (years.sum(axis=1) / len(_W["data"])).round(2).to_dict()}
    except Exception as e:  # noqa: BLE001
        return None, f"{name}: {e!r}"


def grid(workers: int) -> int:
    items = gate_items()
    t0 = time.time()
    rows, extra = [], []
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for i, (r, x) in enumerate(ex.map(evaluate, items)):
            if r is None:
                print("  ошибка", x, flush=True)
                continue
            rows += r
            extra.append(x)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f} с)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "grid.csv", index=False)
    (OUT / "grid_extra.json").write_text(json.dumps(extra, ensure_ascii=False, default=str), encoding="utf-8")
    summary(df, pd.DataFrame(extra))
    return 0


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """Строка на фильтр: итог и PF портфеля всех инструментов, портфеля 4 и металлов на выборе и проверке."""
    w = df.pivot_table(index=["группа", "фильтр"], columns=["срез", "период"], values=["итог", "PF", "без_2_лучших"])
    w.columns = [f"{s}|{p}|{v}" for v, s, p in w.columns]
    return w.reset_index()


def summary(df: pd.DataFrame, extra: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    w = wide(df).merge(extra[["фильтр", "доля_тренда"]], on="фильтр")
    cols = ["группа", "фильтр", "доля_тренда", "все|выбор|итог", "все|проверка|итог", "все|выбор|PF", "все|проверка|PF",
            "портфель 4|проверка|итог", "металлы|проверка|итог", "все|проверка|без_2_лучших"]
    w = w.sort_values("все|выбор|итог", ascending=False)
    lines = [w[cols].round(2).to_string(index=False)]
    c = w[~w["группа"].eq("эталон")]
    rho = spearmanr(c["все|выбор|итог"], c["все|проверка|итог"]).statistic
    rho_pf = spearmanr(c["все|выбор|PF"], c["все|проверка|PF"]).statistic
    base = w[w["фильтр"] == "гейт C (база)"].iloc[0]
    lines.append(f"\nранговая корреляция выбор → проверка по {len(c)} фильтрам: итог {rho:.2f}, PF {rho_pf:.2f}")
    lines.append(f"гейт C: выбор {base['все|выбор|итог']:+.1f}%, проверка {base['все|проверка|итог']:+.1f}% "
                 f"(место на выборе {int((c['все|выбор|итог'] > base['все|выбор|итог']).sum()) + 1} из {len(c) + 1}, "
                 f"на проверке {int((c['все|проверка|итог'] > base['все|проверка|итог']).sum()) + 1})")
    lines.append(f"медиана фильтров на проверке: итог {c['все|проверка|итог'].median():+.1f}%, PF {c['все|проверка|PF'].median():.2f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "grid.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- 4. качество фильтра (метрики ТЗ)

ZZ_ATR = 6.0     # разворот эталона — ход от экстремума не меньше 6 ATR часа (масштаб импульса и стопа)


def zigzag(close: np.ndarray, atr: np.ndarray, k: float) -> list[tuple[int, int, int]]:
    """Ноги ZigZag задним числом: (начало-экстремум, конец-экстремум, направление +1/−1)."""
    legs, piv, ext, d = [], 0, 0, 0
    for i in range(1, len(close)):
        a = atr[ext]
        if not np.isfinite(a) or not np.isfinite(close[i]):
            if d == 0:
                piv = ext = i
            continue
        if d == 0:
            if abs(close[i] - close[piv]) >= k * a:
                d, ext = (1 if close[i] > close[piv] else -1), i
        elif d * (close[i] - close[ext]) > 0:
            ext = i                                        # нога продолжается
        elif d * (close[ext] - close[i]) >= k * a:          # разворот от экстремума
            legs.append((piv, ext, d))
            piv, ext, d = ext, i, -d
    return legs


def quality_one(ev: dict, gate: np.ndarray) -> pd.DataFrame:
    """По каждой ноге эталона: задержка (часов от начала ноги до первого бара фильтра в её сторону;
    нет — длина ноги) и ложные переключения (входы фильтра в состояние против ноги внутри неё)."""
    b = ev["bars"]
    c = b["close"].to_numpy(dtype=float)
    atr = ((b["high"] - b["low"]).rolling(14).mean()).to_numpy()
    g = np.nan_to_num(gate)
    enter = np.r_[False, (g[1:] != g[:-1]) & (g[1:] != 0)]
    rows = []
    for s0, e0, d in zigzag(c, atr, ZZ_ATR):
        seg = g[s0 + 1:e0 + 1]
        hit = np.flatnonzero(seg == d)
        rows.append({"year": b["time"].iloc[s0].year, "hours": e0 - s0,
                     "lag": int(hit[0]) + 1 if hit.size else e0 - s0, "caught": bool(hit.size),
                     "false": int((enter[s0 + 1:e0 + 1] & (g[s0 + 1:e0 + 1] == -d)).sum()),
                     "switches": int(enter[s0 + 1:e0 + 1].sum())})
    return pd.DataFrame(rows)


def _quality(item):
    group, name, params = item
    if params is None:
        return None
    out = []
    for inst, ev in _W["data"].items():
        q = quality_one(ev, gate_values(ev["bars"], params))
        out.append(q.assign(inst=inst))
    q = pd.concat(out)
    res = {"группа": group, "фильтр": name}
    for per, m in (("выбор", q["year"] < 2021), ("проверка", q["year"] >= 2021)):
        x = q[m]
        res[f"{per}|задержка_медиана_ч"] = float(x["lag"].median())
        res[f"{per}|доля_задержки"] = float((x["lag"] / x["hours"]).median())
        res[f"{per}|пойманных_ног"] = float(x["caught"].mean())
        res[f"{per}|ложных_на_ногу"] = float(x["false"].mean())
        res[f"{per}|ложных_из_переключений"] = float(x["false"].sum() / max(x["switches"].sum(), 1))
    return res


def quality(workers: int) -> int:
    items = [it for it in gate_items() if it[2] is not None and "always" not in it[2]]
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        res = [r for r in ex.map(_quality, items) if r]
    q = pd.DataFrame(res)
    q.to_csv(OUT / "quality.csv", index=False)
    pd.set_option("display.width", 250)
    print(q.sort_values("проверка|задержка_медиана_ч").round(2).to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--only", nargs="*")
    g = sub.add_parser("grid")
    g.add_argument("--workers", type=int, default=3)
    sub.add_parser("summary")
    qq = sub.add_parser("quality")
    qq.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    if args.cmd == "candidates":
        return build_candidates(args.only)
    if args.cmd == "summary":
        summary(pd.read_csv(OUT / "grid.csv"), pd.DataFrame(json.loads((OUT / "grid_extra.json").read_text())))
        return 0
    if args.cmd == "quality":
        return quality(args.workers)
    return grid(args.workers)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
