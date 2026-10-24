"""Бумажный режим S46 и S48 на 19 акциях (H67, 24.10): раз в сутки — свежие минутки, пересчёт стратегий тем же
движком перебора, журнал сделок и отчёт. Заявок нет — только сигналы, как у этапа 0 импульса.

S46 «пробой 55 дней» и S48 «пробой 20 дней + 0.25 ATR» — дневные по смыслу стратегии (пробои 55 и 20 дней, держат до
240 ч), поэтому пересчёт раз в сутки после вечерней сессии: сделки те же, что дал бы ежечасный контур, только видны
с задержкой до суток. Минутки: SBER, GAZP, LKOH — живые файлы контура (data/raw/live, докачиваются каждый час),
16 новых бумаг H66 — data/raw/moex, сюда докачиваются из Т-Инвестиций. Часовые свечи и признаки пересобираются по
всей истории, когда файл минуток новее (тем же кодом, что журнал: fresh_strategies.build_feat, new_stocks.bars).

Журнал reports/live/iterator_journal.csv — сделки, открытые с начала режима: вход, стоп, выход, результат, когда
контур впервые увидел вход и выход. Отчёт — reports/live/iterator_<дата>_<время>.md.
Открытая сделка: движок закрывает её на последней свече «по сроку», а срок 240 ч ещё не вышел.

    python scripts/live_iterator.py             # cron: scripts/queues/live_iterator.sh, после вечерней сессии
    python scripts/live_iterator.py --no-fetch  # без загрузки минуток
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

import new_stocks as ns
from experiments import NEW_STOCKS, ROOT

LAUNCH = ["sber", "gazp", "lkoh"]
STOCKS = LAUNCH + NEW_STOCKS
START = pd.Timestamp("2026-10-24")            # начало бумажного режима: сделки, открытые с этого дня
FRESH = pd.Timestamp("2026-06-12")            # свежие данные (после конца истории) — для сравнения в отчёте
CODES = {2: "S46", 4: "S48"}
OUT = ROOT / "reports" / "live"
JOURNAL = OUT / "iterator_journal.csv"
KEY = ["стратегия", "бумага", "вход"]


def fetch() -> None:
    from fetch_tinvest import update
    from forexmodel.broker.tinvest import TInvest

    api = TInvest(sandbox=False)
    for i in NEW_STOCKS:
        update(api, i.upper(), out=ns.RAW / f"{ns.FILES[i.upper()]}.csv")


def features(inst: str, live: dict):
    """Свежие признаки бумаги; возвращает папку, где они лежат (движок перебора читает FEAT/<бумага>.pkl)."""
    import fresh_strategies as fs

    if inst in LAUNCH:
        fs.FEAT.mkdir(parents=True, exist_ok=True)
        fs.build_feat(inst, live)                # сам проверяет, покрывает ли готовый файл последние минуты
        return fs.FEAT
    csv, pkl = ns.RAW / f"{ns.FILES[inst.upper()]}.csv", ns.FEAT / f"{inst}.pkl"
    if not pkl.exists() or csv.stat().st_mtime > pkl.stat().st_mtime:
        ns.bars(inst)
    return ns.FEAT


def strategy_trades(inst: str, feat) -> tuple[pd.DataFrame, dict]:
    import iterator_walkthrough as iw
    import strategy_search as ss

    ss.FEAT = feat
    ins = iw.load_inst(inst)
    end = pd.Timestamp(ins.t[-1])
    parts = []
    for k, code in CODES.items():
        prm = iw.STRATS[k]
        ex = ss.EXITS[prm["exit"]]
        t = iw.trades_frame(ins, prm)
        if t.empty:
            continue
        t = t[t["open_dt"] >= FRESH].copy()
        horizon = t["open_dt"] + pd.Timedelta(hours=ex["H"])
        t["открыта"] = (t["exit_reason"] == "timeout") & (horizon > end + pd.Timedelta(hours=1))
        t["стоп"] = t["open_price"] - ex["sl"] * t["atr_at_entry"]
        t["закрыть не позже"] = horizon
        t["стратегия"], t["бумага"] = code, inst.upper()
        parts.append(t)
    info = {"бумага": inst.upper(), "данные до": end, "цена": float(ins.c[-1])}
    del ins
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), info


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
    md = ["# Бумажный режим S46 и S48", "", f"Запуск {now:%Y-%m-%d %H:%M} (Москва). Сделки, открытые с {START:%d.%m.%Y}; "
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
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    parts, info = [], []
    for inst in STOCKS:                          # по одной бумаге — память
        t, i = strategy_trades(inst, features(inst, live))
        parts.append(t)
        info.append(i)
        gc.collect()
    allt = pd.concat([p for p in parts if len(p)], ignore_index=True)
    trades, fresh = allt[allt["open_dt"] >= START], allt[(allt["open_dt"] < START) & ~allt["открыта"]]
    j, events = merge(trades, now)
    OUT.mkdir(parents=True, exist_ok=True)
    j.to_csv(JOURNAL, index=False)
    md = report(j, events, pd.DataFrame(info), fresh, now)
    path = OUT / f"iterator_{now:%Y-%m-%d_%H%M}.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
