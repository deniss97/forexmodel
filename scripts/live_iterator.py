"""Сигналы S46 и S48 на 19 акциях каждый час (H67; ТЗ инференса 26.10): свежие минутки, часовые свечи дописываются
(scripts/live_bars.py), сделки — тем же движком перебора, что журнал. Заявок нет — решения берёт счёт
(scripts/live_account.py), сделки делаем руками по уведомлениям.

По каждой стратегии и бумаге:
  * сделки движка с 12.06 (свежие данные) — открытая позиция: вход, стоп, срок; закрытая — причина выхода;
  * сигнал на последней ЗАКРЫТОЙ часовой свече — «открыть» по открытию следующего часа. Движок перебора не берёт
    сигналы с двух последних свечей (entries_for), поэтому сигнал на последней закрытой проверяется здесь тем же
    правилом: вход, оба фильтра, ATR, нет открытой сделки (chain: свеча сигнала позже свечи выхода прошлой сделки).
Журнал reports/live/iterator_journal.csv — сделки, открытые с начала режима (24.10), и когда контур впервые увидел
вход и выход; состояние для счёта — reports/live/iterator_state.json; отчёт — reports/live/iterator_<дата>_<время>.md,
когда были события (и раз в сутки — в 23 ч по Москве).
Открытая сделка: движок закрывает её на последней свече «по сроку», а срок 240 ч ещё не вышел.

    python scripts/live_iterator.py             # каждый час — scripts/queues/live_hourly.sh
    python scripts/live_iterator.py --no-fetch  # без загрузки минуток
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import live_bars as lb
import new_stocks as ns
from experiments import NEW_STOCKS, ROOT

LAUNCH = lb.LAUNCH
STOCKS = LAUNCH + NEW_STOCKS
START = pd.Timestamp("2026-10-24")            # начало бумажного режима: сделки, открытые с этого дня
FRESH = pd.Timestamp("2026-06-12")            # свежие данные (после конца истории) — для сравнения в отчёте
CODES = {2: "S46", 4: "S48"}
GRACE = pd.Timedelta(minutes=30)              # вход не старше этого — ещё «открыть по рынку» (как у S42)
OUT = ROOT / "reports" / "live"
JOURNAL = OUT / "iterator_journal.csv"
STATE = OUT / "iterator_state.json"
KEY = ["стратегия", "бумага", "вход"]


def fetch() -> None:
    from fetch_tinvest import update
    from forexmodel.broker.tinvest import TInvest

    api = TInvest(sandbox=False)
    for i in NEW_STOCKS:
        update(api, i.upper(), out=lb.minutes_csv(i))


def signal_at(ins, prm: dict, j: int) -> bool:
    """Тот же вход, что entries_for (лонг: акции только в лонг), на свече j — без запрета двух последних свечей."""
    import strategy_search as ss

    if not ins.ok[j] or ins.entries[prm["entry"]][j] != 1:
        return False
    for f in ("f1", "f2"):
        if not ins.filters[prm[f]][0][j]:
            return False
    a = ins.atr[ss.EXITS[prm["exit"]].get("atr", prm["atr"])][j]
    return bool(np.isfinite(a) and a > 0)


def strategy_state(inst: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """Сделки движка с 12.06, свежие сигналы на последней закрытой свече, данные бумаги."""
    import iterator_walkthrough as iw
    import strategy_search as ss
    from moex_data import last_begin

    ss.FEAT = lb.features(inst)
    ins = iw.load_inst(inst)
    until = last_begin(lb.minutes_csv(inst))
    hour = np.timedelta64(60, "m")
    closed = np.flatnonzero(ins.t + hour <= np.datetime64(until + pd.Timedelta(minutes=1)))
    j = int(closed[-1])
    parts, fresh = [], []
    for k, code in CODES.items():
        prm, ex = iw.STRATS[k], ss.EXITS[iw.STRATS[k]["exit"]]
        p = ex.get("atr", prm["atr"])
        t = iw.trades_frame(ins, prm)
        if len(t):
            horizon = t["open_dt"] + pd.Timedelta(hours=ex["H"])
            t["открыта"] = (t["exit_reason"] == "timeout") & (horizon > until + pd.Timedelta(hours=1))
            t["стоп"] = t["open_price"] - ex["sl"] * t["atr_at_entry"]
            t["закрыть не позже"] = horizon
            t["стратегия"], t["бумага"] = code, inst.upper()
            last_exit = int(np.searchsorted(ins.t, t["close_dt"].iloc[-1].to_datetime64(), "right") - 1)
            parts.append(t[t["open_dt"] >= FRESH])
        else:
            last_exit = -1
        if j > last_exit and signal_at(ins, prm, j):
            e = j + 1
            at = (pd.Timestamp(ins.t[e]) if e < ins.n else pd.Timestamp(ins.t[j]) + pd.Timedelta(hours=1)) + pd.Timedelta(minutes=1)
            px = float(ins.o[e]) if e < ins.n else float(ins.c[j])
            a = float(ins.atr[p][j])
            size = float(np.clip(ins.stop_med[p] / (a / px), 0.25, 4.0)) if prm["size"] == "по риску" else 1.0
            fresh.append({"стратегия": code, "бумага": inst.upper(), "решение": "открыть", "вход_в": at, "цена": px, "atr": a,
                          "стоп": px - ex["sl"] * a, "размер": size, "срок": at + pd.Timedelta(hours=ex["H"]),
                          "сигнал": pd.Timestamp(ins.t[j])})
    info = {"бумага": inst.upper(), "данные до": until, "цена": float(ins.c[-1])}
    del ins
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), fresh, info


def merge(trades: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сделки режима против прошлого журнала: когда впервые увидели вход и выход; новые события этого запуска."""
    j = trades.rename(columns={"open_dt": "вход", "open_price": "цена входа", "close_dt": "выход", "exit_price": "цена выхода",
                               "exit_reason": "причина", "profit_pct": "результат %", "size": "размер"})
    j = j[["стратегия", "бумага", "вход", "цена входа", "стоп", "размер", "закрыть не позже", "открыта", "выход", "цена выхода",
           "причина", "результат %"]].copy()
    j.loc[j["открыта"], ["выход", "цена выхода", "причина", "результат %"]] = [pd.NaT, np.nan, "", np.nan]
    old = pd.read_csv(JOURNAL, parse_dates=["вход", "увидели вход", "увидели выход"]) if JOURNAL.exists() else pd.DataFrame(columns=KEY)
    j = j.merge(old[[c for c in KEY + ["увидели вход", "увидели выход"] if c in old]], on=KEY, how="left") if len(old) else \
        j.assign(**{"увидели вход": pd.NaT, "увидели выход": pd.NaT})
    new_in = j["увидели вход"].isna()
    j.loc[new_in, "увидели вход"] = now
    new_out = ~j["открыта"] & j["увидели выход"].isna()
    j.loc[new_out, "увидели выход"] = now
    events = pd.concat([j[new_in].assign(событие="вход"), j[new_out & ~new_in].assign(событие="выход")], ignore_index=True)
    gone = old.merge(j[KEY], on=KEY, how="left", indicator=True) if len(old) else old
    if len(gone) and (gone["_merge"] == "left_only").any():   # сделка пропала после пересчёта — данные поменялись задним числом
        events = pd.concat([events, gone[gone["_merge"] == "left_only"][KEY].assign(событие="пропала")], ignore_index=True)
    return j.sort_values(["вход", "стратегия", "бумага"]), events


def report(j: pd.DataFrame, events: pd.DataFrame, info: pd.DataFrame, fresh: pd.DataFrame, now: pd.Timestamp) -> str:
    f = lambda v, d=2: "—" if pd.isna(v) else f"{v:.{d}f}"  # noqa: E731
    dt = lambda v: "" if pd.isna(v) else f"{pd.Timestamp(v):%d.%m %H:%M}"  # noqa: E731
    price = dict(zip(info["бумага"], info["цена"]))
    md = ["# Сигналы S46 и S48", "", f"Запуск {now:%Y-%m-%d %H:%M} (Москва). Сделки, открытые с {START:%d.%m.%Y}; "
          f"заявок нет. Данные до: {info['данные до'].min():%d.%m %H:%M} — {info['данные до'].max():%d.%m %H:%M}.", ""]
    md += ["## События с прошлого запуска", ""]
    if events.empty:
        md.append("нет")
    else:
        md += ["| событие | стратегия | бумага | вход | цена входа | стоп | выход | причина | результат % |", "|---|---|---|---|---|---|---|---|---|"]
        for r in events.to_dict("records"):
            md.append(f"| {r['событие']} | {r['стратегия']} | {r['бумага']} | {dt(r['вход'])} | {f(r.get('цена входа'))} | "
                      f"{f(r.get('стоп'))} | {dt(r.get('выход'))} | {r.get('причина') or ''} | {f(r.get('результат %'))} |")
    op = j[j["открыта"]]
    md += ["", f"## Открытые позиции ({len(op)})", ""]
    if op.empty:
        md.append("нет")
    else:
        md += ["| стратегия | бумага | вход | цена входа | размер | стоп | цена сейчас | сейчас % | закрыть не позже |", "|---|---|---|---|---|---|---|---|---|"]
        for r in op.to_dict("records"):
            p = price.get(r["бумага"], np.nan)
            md.append(f"| {r['стратегия']} | {r['бумага']} | {r['вход']:%d.%m %H:%M} | {f(r['цена входа'])} | {f(r['размер'])} | {f(r['стоп'])} | "
                      f"{f(p)} | {f((p / r['цена входа'] - 1) * 100)} | {r['закрыть не позже']:%d.%m %H:%M} |")
    closed = j[~j["открыта"]]
    md += ["", "## Итог режима", ""]
    for code in CODES.values():
        c, fr = closed[closed["стратегия"] == code], fresh[fresh["стратегия"] == code]
        md.append(f"* **{code}** — закрыто {len(c)} сделок, сумма {f(c['результат %'].sum(), 1)}% (на бумагу; портфель — сумма / 19); "
                  f"для сравнения, свежие данные {FRESH:%d.%m} — {START:%d.%m}: {len(fr)} сделок, сумма {f(fr['profit_pct'].sum(), 1)}%.")
    md.append("")
    return "\n".join(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    now = pd.Timestamp.now("Europe/Moscow").tz_localize(None).floor("min")
    ns.register()
    if not a.no_fetch:
        fetch()
    parts, info, fresh = [], [], []
    for inst in STOCKS:                          # по одной бумаге — память
        t, sig, i = strategy_state(inst)
        parts.append(t)
        fresh += sig
        info.append(i)
        gc.collect()
    allt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    trades, before = allt[allt["open_dt"] >= START], allt[(allt["open_dt"] < START) & ~allt["открыта"]]
    j, events = merge(trades, now)
    OUT.mkdir(parents=True, exist_ok=True)
    j.to_csv(JOURNAL, index=False)
    # состояние для счёта: открытые позиции стратегий, сделки, закрытые за последние сутки, свежие сигналы
    recent = allt[~allt["открыта"] & (allt["close_dt"] >= now - pd.Timedelta(days=1))]
    cols = ["стратегия", "бумага", "open_dt", "open_price", "стоп", "size", "закрыть не позже", "close_dt", "exit_price", "exit_reason"]
    state = {"запуск": now, "данные": {i["бумага"]: {"до": i["данные до"], "цена": i["цена"]} for i in info},
             "позиции": allt[allt["открыта"]][cols].to_dict("records"), "закрыто": recent[cols].to_dict("records"),
             "сигналы": [s for s in fresh if now - s["вход_в"] <= GRACE or s["вход_в"] > now]}
    STATE.write_text(json.dumps(state, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    md = report(j, events, pd.DataFrame(info), before, now)
    if len(events) or state["сигналы"] or now.hour == 23:
        path = OUT / f"iterator_{now:%Y-%m-%d_%H%M}.md"
        path.write_text(md, encoding="utf-8")
        print(f"→ {path.relative_to(ROOT)}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
