"""Сигналы импульсной стратегии на сейчас: что держим, где стоп, что открыть или закрыть.

Тот же код, что у walk-forward (сборка данных, сигнал импульса, штатный симулятор), но окно проверки —
последние недели до свежей минуты в данных. Поэтому решения совпадают с бэктестом по построению.
Инструменты, площадки и стороны — configs/live.yaml (металлы CFD в обе стороны, акции на бирже только
в лонг). По каждому инструменту:
  * открытая позиция — сторона, вход, текущий стоп (с трейлингом), уровень включения трейлинга, срок;
  * сигнал на последней ЗАКРЫТОЙ часовой свече, по которому вход ещё впереди — «ОТКРЫТЬ»;
  * сделка, закрытая за последние часы, — для сверки с брокером.
Отчёт — reports/live/<время>.md и строка в reports/live/journal.csv (лог решений для сверки с демо-счётом).

    python scripts/fetch_moex_iss.py && python scripts/live_signals.py
    python scripts/live_signals.py --now "2026-05-20 12:05"      # как выглядело бы на момент в истории
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from forexmodel.config import load_config
from forexmodel.logging_utils import setup_logging
from forexmodel.pipelines.dataset import build_dataset
from forexmodel.pipelines.sweep import set_by_path
from forexmodel.simulation.signals import build_signal_column
from forexmodel.simulation.simulator import simulate_trades

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live"
LONG, SHORT, FLAT = 2, 0, 1
WINDOW_DAYS = 40            # окно проверки: больше срока сделки (240 ч) с запасом
# вход контура не старше стольких минут отдаётся как «открыть» по рынку: с минутками без задержки к запуску
# минута входа HH:01 уже в данных (бумажный прогон 22.10, docs/results/paper_replay.md: без окна 0 входов из 33)
ENTRY_GRACE = 30


def instrument_cfg(spec: dict, live: dict, start: pd.Timestamp, end: pd.Timestamp):
    cfg = load_config(ROOT / spec["config"])
    sets = dict(live["rule"]["set"])
    sets["data.csv_path"] = spec["csv"]
    if spec.get("ticker"):
        sets.update(live["stock_set"])
        sets["data.dividend_ticker"] = spec["ticker"]
    for k, v in sets.items():
        set_by_path(cfg, k, v)
    cfg.paths.run_name = "live"
    cfg.data.splits.update({"train_end": f"{start:%Y-%m-%d %H:%M:%S}", "test_start": f"{start:%Y-%m-%d %H:%M:%S}",
                            "test_end": f"{end:%Y-%m-%d %H:%M:%S}", "sim_start": f"{end:%Y-%m-%d %H:%M:%S}", "sim_end": None})
    parts = live["rule"]["variant"].split(":")
    c = copy.deepcopy(cfg)
    c.simulation.use_expected_value_filter = False
    c.simulation.signal_source = "impulse"
    c.simulation.impulse_bars, c.simulation.impulse_atr = int(parts[1]), float(parts[2])
    c.simulation.use_trend_filter = True
    return c


def stop_state(minutes: pd.DataFrame, side: str, entry: float, atr: float, sim) -> dict:
    """Стоп и трейлинг по минутам от входа до последней — логика forexmodel/simulation/exits.trailing_exit."""
    sgn = 1.0 if side == "buy" else -1.0
    stop, best, on = entry - sgn * sim.sl_atr * atr, entry, None
    for t, h, lo in zip(minutes.iloc[:, 0], minutes["high"].to_numpy(float), minutes["low"].to_numpy(float)):
        ext = h if sgn > 0 else lo
        if (ext - best) * sgn > 0:
            best = ext
            if (best - entry) * sgn >= sim.activate_atr * atr:
                on = on or t
                ns = best - sgn * sim.trail_atr * atr
                stop = max(stop, ns) if sgn > 0 else min(stop, ns)
    return {"stop": stop, "best": best, "trail_on": on, "activate": entry + sgn * sim.activate_atr * atr}


def analyse(name: str, spec: dict, live: dict, now: pd.Timestamp | None, grace_minutes: int = ENTRY_GRACE) -> dict:
    """grace_minutes — вход контура, случившийся не раньше чем столько минут назад, ещё отдаётся как «открыть»
    (по рынку сейчас): с минутками без задержки к запуску в :02 минута входа HH:01 уже в данных."""
    probe = load_config(ROOT / spec["config"])
    head = pd.read_csv(ROOT / spec["csv"], usecols=[probe.data.time_col]).iloc[-1, 0]
    tcol = "time"                       # внутри пайплайна время всегда в колонке time
    last = pd.Timestamp(head) if now is None else min(pd.Timestamp(head), now)
    now = now or last + pd.Timedelta(minutes=1)
    cfg = instrument_cfg(spec, live, last - pd.Timedelta(days=WINDOW_DAYS), last + pd.Timedelta(days=1))
    ds = build_dataset(cfg)
    sig = build_signal_column(ds.test.copy(), cfg)
    minute = ds.minute_slice("test")
    minute = minute[minute[tcol] <= last]
    sig = sig[sig[tcol] + pd.Timedelta(minutes=cfg.data.signal_tf_minutes) <= now]     # только закрытые свечи
    if spec.get("sides") == "long":
        sig.loc[pd.to_numeric(sig["final_class"], errors="coerce") == SHORT, "final_class"] = FLAT
    trades, _ = simulate_trades(sig, minute, cfg)
    sim = cfg.simulation
    res = {"инструмент": name, "площадка": spec["venue"], "стороны": "только лонг" if spec.get("sides") == "long" else "обе",
           "данные до": f"{last:%Y-%m-%d %H:%M}", "позиция": "нет", "действие": "ничего", "детали": "",
           # машинные поля решения — для исполнителя (scripts/live_trade.py); в отчёт и журнал не пишутся
           "_м": {"решение": "ничего", "данные_до": last, "цена": float(minute["close"].iloc[-1]) if len(minute) else None}}
    horizon = pd.Timedelta(minutes=sim.horizon_minutes)
    if len(trades):
        t = trades.iloc[-1]
        still_open = t["exit_reason"] == "timeout" and t["close_dt"] >= minute[tcol].iloc[-1] and t["close_dt"] - t["open_dt"] < horizon
        if still_open:
            m = minute[minute[tcol] >= t["open_dt"]][[tcol, "high", "low"]]
            s = stop_state(m, t["side"], t["open_price"], t["atr_at_entry"], sim)
            px = float(minute["close"].iloc[-1])
            d = 1 if t["side"] == "buy" else -1
            res.update({"позиция": f"{'лонг' if d > 0 else 'шорт'} с {t['open_dt']:%Y-%m-%d %H:%M} по {t['open_price']:.4g}",
                        "действие": f"держать; стоп {s['stop']:.4g}",
                        "детали": (f"цена {px:.4g} ({d * (px / t['open_price'] - 1) * 100:+.2f}%); ATR на входе {t['atr_at_entry']:.4g}; "
                                   + (f"трейлинг включён {s['trail_on']:%m-%d %H:%M}" if s["trail_on"] is not None
                                      else f"трейлинг включится на {s['activate']:.4g}")
                                   + f"; закрыть не позже {t['open_dt'] + horizon:%Y-%m-%d %H:%M}")})
            res["_м"].update({"решение": "держать", "сторона": d, "вход": t["open_dt"], "цена_входа": float(t["open_price"]),
                              "atr": float(t["atr_at_entry"]), "стоп": float(s["stop"]), "активация": float(s["activate"]),
                              "отступ": sim.trail_atr * float(t["atr_at_entry"]), "трейлинг_вкл": s["trail_on"] is not None,
                              "срок": t["open_dt"] + horizon})
            if grace_minutes and last - t["open_dt"] <= pd.Timedelta(minutes=grace_minutes) and s["trail_on"] is None:
                atr = float(t["atr_at_entry"])
                res["действие"] = f"ОТКРЫТЬ {'лонг' if d > 0 else 'шорт'} по рынку (вход контура {t['open_dt']:%m-%d %H:%M} по {t['open_price']:.4g})"
                res["_м"].update({"решение": "открыть", "свежий_вход": True, "вход_в": t["open_dt"], "цена": px, "atr": atr,
                                  "срок_часов": (t["open_dt"] + horizon - last).total_seconds() / 3600,
                                  "стоп_atr": sim.sl_atr, "активация_atr": sim.activate_atr})
        elif t["close_dt"] >= last - pd.Timedelta(hours=6):
            res.update({"действие": f"закрыто {t['close_dt']:%m-%d %H:%M} ({t['exit_reason']})",
                        "детали": f"{'лонг' if t['side'] == 'buy' else 'шорт'} с {t['open_dt']:%m-%d %H:%M}: {t['open_price']:.4g} → {t['exit_price']:.4g}, {t['gross_unit_pct']:+.2f}% до издержек"})
            res["_м"].update({"решение": "закрыто", "вход": t["open_dt"], "выход": t["close_dt"], "причина": t["exit_reason"]})
    # сигнал на последней закрытой свече, вход по которому ещё впереди (в минутах его нет)
    act = sig[pd.to_numeric(sig["final_class"], errors="coerce").isin([LONG, SHORT])]
    if len(act) and res["позиция"] == "нет":
        a = act.iloc[-1]
        entry_at = a[tcol] + pd.Timedelta(minutes=sim.open_delay_minutes)
        trend = a.get(sim.trend_col, np.nan)
        side = LONG if a["final_class"] == LONG else SHORT
        ok_trend = not np.isfinite(trend) or (trend > 0 if side == LONG else trend < 0)
        if entry_at > last and ok_trend:
            atr = float(a[sim.atr_col]); px = float(a["close"])
            d = 1 if side == LONG else -1
            res.update({"действие": f"ОТКРЫТЬ {'лонг' if d > 0 else 'шорт'} в {entry_at:%m-%d %H:%M} по рынку",
                        "детали": f"цена ≈ {px:.4g}; ATR {atr:.4g}; начальный стоп ≈ {px - d * sim.sl_atr * atr:.4g}; "
                                  f"трейлинг с ≈ {px + d * sim.activate_atr * atr:.4g}"})
            res["_м"].update({"решение": "открыть", "сторона": d, "вход_в": entry_at, "цена": px, "atr": atr,
                              "стоп": px - d * sim.sl_atr * atr, "активация": px + d * sim.activate_atr * atr,
                              "отступ": sim.trail_atr * atr, "срок_часов": sim.horizon_minutes / 60,
                              "стоп_atr": sim.sl_atr, "активация_atr": sim.activate_atr})
    return res


def decisions(only: list[str] | None = None, now: pd.Timestamp | None = None, config: Path = ROOT / "configs" / "live.yaml") -> list[dict]:
    """Решения на сейчас по инструментам с машинными полями `_м` — для исполнителя (scripts/live_trade.py)."""
    setup_logging(logging.WARNING)
    logging.disable(logging.WARNING)
    live = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out = []
    for name, spec in live["instruments"].items():
        if (only and name not in only) or not (ROOT / spec["csv"]).exists():
            continue
        out.append(analyse(name, spec, live, now))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "live.yaml"))
    ap.add_argument("--now", default=None, help="момент «сейчас» (время данных); по умолчанию — последняя минута + 1")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    setup_logging(logging.WARNING)
    logging.disable(logging.WARNING)
    live = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    now = pd.Timestamp(args.now) if args.now else None
    rows = []
    for name, spec in live["instruments"].items():
        if args.only and name not in args.only:
            continue
        if not (ROOT / spec["csv"]).exists():
            rows.append({"инструмент": name, "площадка": spec["venue"], "действие": "нет данных", "детали": spec["csv"]})
            continue
        rows.append(analyse(name, spec, live, now))
        print(f"[{name}] {rows[-1]['действие']} · {rows[-1].get('позиция', '')}", flush=True)
    R = pd.DataFrame([{k: v for k, v in r.items() if k != "_м"} for r in rows])
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    md = ["# Сигналы импульса", "", f"Запуск {stamp}" + (f", момент {now}" if now is not None else ""), "",
          "| " + " | ".join(R.columns) + " |", "|" + "---|" * len(R.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in R.fillna("").itertuples(index=False)]
    (OUT / f"{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    j = OUT / "journal.csv"
    R.assign(запуск=stamp, момент=str(now or "")).to_csv(j, mode="a", header=not j.exists(), index=False)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
