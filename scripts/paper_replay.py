"""Бумажный прогон живого контура на свежих данных — как он торговал бы по cron, без денег и без брокера (22.10).

Тот же код, что в бою, кроме брокера:
  * решения — live_signals.analyse (окно 40 дней) по минуткам, завершённым к моменту запуска: запуск в HH:02
    видит минуты до HH:01 включительно — как fetch_tinvest.py без задержки;
  * действия и заявки — live_trade.plan и live_trade.execute;
  * брокер — PaperBroker: рыночная заявка исполняется по открытию первой минуты после запуска, стоп-лосс и
    трейлинг-стоп — как стоп-заявки брокера по минуткам (сработал — по уровню, при гэпе — по открытию минуты);
    цены сырые, без поправки на дивиденды (как у брокера); комиссия 0.05% на сторону.
Эталон — сделки S42 на тех же минутах тем же пайплайном одним прогоном (как scripts/fresh_check.py).

Период — после конца истории всех трёх бумаг (GAZP — 11.06.2026): ни один день не участвовал в выборе стратегии.
Варианты: расписание cron (сейчас — будни 10–23 ч; все торговые часы, включая утро и выходные) × приём входа
(сейчас — только если минута входа ещё впереди; «окно 30 минут» — вход контура не старше 30 минут берётся по рынку).
analyse зовётся только там, где эталон открывал сделку, и в случайной выборке остальных часов (проверка, что там
контур молчит): один вызов — 15–20 с. Решения кэшируются в reports/paper_replay/decisions.pkl (продолжение после обрыва).

    python scripts/paper_replay.py
"""
from __future__ import annotations

import logging
import sys
from decimal import Decimal

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import live_signals as ls  # noqa: E402
import live_trade as lt  # noqa: E402
from experiments import BG, BLUE, GRAY, GRID, INK, INK2, ROOT  # noqa: E402
from forexmodel.broker.tinvest import Share, d2q  # noqa: E402
from forexmodel.data.loader import load_minute_compact  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from walk_forward import _rule_trades  # noqa: E402

OUT = ROOT / "reports" / "paper_replay"
DOC = ROOT / "docs" / "results" / "paper_replay.md"
PNG = ROOT / "docs" / "results" / "paper_replay" / "equity.png"
FRESH = pd.Timestamp("2026-06-12")            # после конца истории всех трёх бумаг
WIN = pd.Timestamp("2026-01-01")              # окно эталона — как fresh_check
STOCKS = ["sber", "gazp", "lkoh"]
# лот и шаг цены — из scripts/tinvest_readonly_check.py (22.10)
SHARES = {"sber": Share("SBER", "SBER", "", 1, Decimal("0.01"), True), "gazp": Share("GAZP", "GAZP", "", 10, Decimal("0.01"), True),
          "lkoh": Share("LKOH", "LKOH", "", 1, Decimal("0.5"), True)}
COMMISSION = 0.0005                           # на сторону: брокер + биржа (configs/costs/exchange.yaml)
CAPITAL = 100_000                             # на бумагу, как live_trade.py по умолчанию
GRACE = 30
RUN_MINUTE = 5                                # запуск в HH:05: в HH:02 минута входа HH:01 — последняя в данных, и контур её ещё не видит
SCHEDULES = {"cron сейчас: будни 10–23 ч": lambda t: t.weekday() < 5 and 10 <= t.hour <= 23,
             "все торговые часы": lambda t: True}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


class PaperBroker:
    """Брокер на бумаге с теми же методами, что forexmodel.broker.tinvest.TInvest, которые зовёт live_trade.execute."""

    def __init__(self, minutes: dict[str, pd.DataFrame]):
        self.m = {SHARES[i].uid: (m["time"].to_numpy("datetime64[ns]"), m["open"].to_numpy(float), m["high"].to_numpy(float),
                                  m["low"].to_numpy(float), m["close"].to_numpy(float)) for i, m in minutes.items()}
        self.now = None
        self.pos: dict[str, int] = {}
        self.orders: dict[str, dict] = {}
        self.fills: list[dict] = []
        self.events: list[dict] = []
        self._n = 0

    def _i(self, uid: str, t) -> int:
        return int(np.searchsorted(self.m[uid][0], np.datetime64(t), "left"))

    def _fill(self, uid: str, lots: int, buy: bool, px: float, t, why: str) -> float:
        lot = next(s.lot for s in SHARES.values() if s.uid == uid)
        self.pos[uid] = self.pos.get(uid, 0) + (lots if buy else -lots)
        comm = px * lots * lot * COMMISSION
        self.fills.append({"uid": uid, "time": pd.Timestamp(t), "buy": buy, "lots": lots, "price": px, "commission": comm, "why": why})
        return comm

    def market(self, account, uid, lots, buy):
        t, o = self.m[uid][0], self.m[uid][1]
        i = min(self._i(uid, self.now), len(t) - 1)                # первая минута после запуска
        comm = self._fill(uid, int(lots), buy, float(o[i]), t[i], "рынок")
        return {"executedOrderPrice": d2q(o[i]), "executedCommission": d2q(comm)}

    def last_price(self, uid):
        i = max(self._i(uid, self.now) - 1, 0)
        return Decimal(str(self.m[uid][4][i]))

    def _order(self, uid, lots, **kw) -> str:
        self._n += 1
        oid = f"paper-{self._n}"
        self.orders[oid] = {"uid": uid, "lots": int(lots), "placed": self.now, **kw}
        return oid

    def stop_loss(self, account, share, lots, stop_price):
        return self._order(share.uid, lots, kind="стоп", price=float(share.round_price(stop_price, down=True)))

    def trailing(self, account, share, lots, activation_price, indent):
        return self._order(share.uid, lots, kind="трейлинг", activation=float(share.round_price(activation_price, down=False)),
                           indent=float(indent), active=False, best=None)

    def stop_orders(self, account):
        return [{"instrumentUid": o["uid"], "stopOrderId": k} for k, o in self.orders.items()]

    def cancel_stop(self, account, stop_order_id):
        self.orders.pop(stop_order_id, None)

    def advance(self, t) -> None:
        """Стоп-заявки по минуткам [now, t): стоп — по минимуму минуты; трейлинг включается, когда максимум дошёл до
        уровня активации, и дальше держит стоп на отступ ниже лучшей цены (сначала проверка, потом новый максимум)."""
        if self.now is None:
            self.now = t
            return
        for uid in {o["uid"] for o in self.orders.values()}:
            tt, o, h, lo, _ = self.m[uid]
            for j in range(self._i(uid, self.now), self._i(uid, t)):
                for oid, od in sorted(((k, v) for k, v in self.orders.items() if v["uid"] == uid), key=lambda kv: int(kv[0][6:])):
                    if od["kind"] == "стоп":
                        level = od["price"] if lo[j] <= od["price"] else None
                    elif od["active"]:
                        stop = od["best"] - od["indent"]
                        level = stop if lo[j] <= stop else None
                        od["best"] = max(od["best"], h[j]) if level is None else od["best"]
                    else:
                        level = None
                        if h[j] >= od["activation"]:
                            od["active"], od["best"] = True, h[j]
                    if level is None:
                        continue
                    px = min(level, o[j])
                    self.orders.pop(oid)
                    if self.pos.get(uid, 0) >= od["lots"]:
                        self._fill(uid, od["lots"], False, px, tt[j], od["kind"])
                    else:
                        self.events.append({"uid": uid, "time": pd.Timestamp(tt[j]), "kind": od["kind"],
                                            "what": "стоп-заявка сработала без позиции: без маржи — отказ, с маржой — открылся бы шорт"})
        self.now = t


def reference(inst: str, live: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Эталон: сделки S42 одним прогоном пайплайна на окне с начала года — как fresh_check."""
    spec = live["instruments"][inst]
    last = pd.Timestamp(pd.read_csv(ROOT / spec["csv"], usecols=["begin"]).iloc[-1, 0])
    cfg = ls.instrument_cfg(spec, live, WIN, last + pd.Timedelta(days=1))
    t = _rule_trades(ls.build_dataset(cfg), cfg, "impulse:6:3:long")
    t = t[(t["open_dt"] >= FRESH) & (t["open_dt"] + pd.Timedelta(minutes=1) <= last)].copy()
    t = recost(t, profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))
    t["открыта"] = (t["exit_reason"] == "timeout") & (t["close_dt"] >= last - pd.Timedelta(minutes=1))
    t["instrument"] = inst
    return t.reset_index(drop=True), last


def ticks(minutes: pd.DataFrame, rule) -> list[pd.Timestamp]:
    """Моменты запуска: HH:<RUN_MINUTE> после каждого часа, в котором были минуты, если час подходит под расписание."""
    hours = pd.DatetimeIndex(minutes["time"]).floor("h").unique()
    return [t for t in (hours + pd.Timedelta(minutes=60 + RUN_MINUTE)) if t >= FRESH and rule(t)]


def decide(cache: dict, inst: str, t: pd.Timestamp, live: dict) -> dict:
    key = (inst, t)
    if key not in cache:
        r = ls.analyse(inst, live["instruments"][inst], live, t - pd.Timedelta(minutes=1), grace_minutes=GRACE)
        cache[key] = {k: r[k] for k in ("действие", "позиция")} | {"_м": r["_м"]}
        pd.to_pickle(cache, OUT / "decisions.pkl")
        print(f"   [{inst}] {t:%m-%d %H:%M} {r['действие']}", flush=True)
    return cache[key]


def run(variant_grace: bool, rule, cand: dict, cache: dict, minutes: dict, live: dict, all_ticks: list):
    broker, state, log = PaperBroker(minutes), {}, []
    lt.log_order = lambda row: log.append(row)                    # журнал заявок — в память, не в reports/live
    for t in all_ticks:
        if not rule(t):
            continue
        broker.advance(t)
        dec = {}
        for inst in STOCKS:
            d = cache[(inst, t)]["_м"] if (inst, t) in cand[inst] else {"решение": "ничего"}
            if d.get("свежий_вход") and not variant_grace:
                d = {**d, "решение": "держать"}                    # без окна: вход уже в данных — контур «держит»
            dec[inst] = d
        held = {i: broker.pos.get(SHARES[i].uid, 0) for i in STOCKS}
        stops = {i: sum(o["uid"] == SHARES[i].uid for o in broker.orders.values()) for i in STOCKS}
        prices = {i: float(broker.last_price(SHARES[i].uid)) for i in STOCKS}
        acts = lt.plan(dec, held, state, stops, prices, {i: s.lot for i, s in SHARES.items()}, CAPITAL, t)
        for line in lt.execute(broker, "paper", SHARES, acts, state, True, t):
            log.append({"время": t, "строка": line})
    broker.advance(all_ticks[-1] + pd.Timedelta(minutes=1))
    return broker, log


def paper_trades(broker: PaperBroker) -> pd.DataFrame:
    rows, open_ = [], {}
    for f in broker.fills:
        inst = next(i for i, s in SHARES.items() if s.uid == f["uid"])
        if f["buy"]:
            open_[inst] = f
        elif inst in open_:
            b = open_.pop(inst)
            gross = (f["price"] / b["price"] - 1) * 100
            rows.append({"instrument": inst, "open_dt": b["time"], "close_dt": f["time"], "open_price": b["price"], "exit_price": f["price"],
                         "lots": b["lots"], "gross_pct": gross, "net_pct": gross - 2 * COMMISSION * 100, "exit_reason": f["why"]})
    for inst, b in open_.items():  # noqa: B007
        rows.append({"instrument": inst, "open_dt": b["time"], "close_dt": pd.NaT, "open_price": b["price"], "exit_price": np.nan,
                     "lots": b["lots"], "gross_pct": np.nan, "net_pct": np.nan, "exit_reason": "открыта"})
    cols = ["instrument", "open_dt", "close_dt", "open_price", "exit_price", "lots", "gross_pct", "net_pct", "exit_reason"]
    return pd.DataFrame(rows, columns=cols)


def match(ref: pd.DataFrame, pap: pd.DataFrame) -> pd.DataFrame:
    """Сделка эталона ↔ бумажная сделка той же бумаги, открытая в пределах 3 суток после входа эталона."""
    rows = []
    for r in ref.itertuples():
        p = pap[(pap["instrument"] == r.instrument) & (pap["open_dt"] >= r.open_dt - pd.Timedelta(minutes=5))
                & (pap["open_dt"] <= r.open_dt + pd.Timedelta(days=3))]
        p = p.iloc[0] if len(p) else None
        rows.append({"бумага": r.instrument, "вход эталона": r.open_dt, "выход эталона": r.close_dt, "причина эталона": r.exit_reason,
                     "эталон, % до издержек": r.gross_unit_pct, "эталон, % после": r.profit_pct,
                     "вход на бумаге": p["open_dt"] if p is not None else pd.NaT,
                     "опоздание входа, мин": (p["open_dt"] - r.open_dt).total_seconds() / 60 if p is not None else np.nan,
                     "цена входа хуже, %": (p["open_price"] / r.open_price - 1) * 100 if p is not None else np.nan,
                     "выход на бумаге": p["close_dt"] if p is not None else pd.NaT, "причина на бумаге": p["exit_reason"] if p is not None else "пропущена",
                     "бумага, % до издержек": p["gross_pct"] if p is not None else np.nan,
                     "бумага, % после": p["net_pct"] if p is not None else np.nan})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    cache = pd.read_pickle(OUT / "decisions.pkl") if (OUT / "decisions.pkl").exists() else {}
    refs, minutes, cand, lasts = {}, {}, {}, {}
    for inst in STOCKS:
        spec = live["instruments"][inst]
        refs[inst], lasts[inst] = reference(inst, live)
        m = load_minute_compact(ROOT / spec["csv"], "begin", ROOT / "reports" / "_cache")
        minutes[inst] = m[m["time"] >= FRESH - pd.Timedelta(days=1)].reset_index(drop=True)
        print(f"[{inst}] эталон: {len(refs[inst])} сделок с {FRESH:%d.%m}, данные до {lasts[inst]}", flush=True)
    end = min(lasts.values())
    all_ticks = sorted({t for inst in STOCKS for t in ticks(minutes[inst], SCHEDULES["все торговые часы"]) if t <= end + pd.Timedelta(minutes=2)})
    rng = np.random.default_rng(7)
    for inst in STOCKS:
        want = set()
        for r in refs[inst].itertuples():
            bar_close = r.open_dt - pd.Timedelta(minutes=1)           # вход эталона — минута HH:01 после закрытия часа
            for rule in SCHEDULES.values():
                nxt = [t for t in all_ticks if t >= bar_close + pd.Timedelta(minutes=RUN_MINUTE) and rule(t)][:1]
                want.update(nxt)
        others = [t for t in all_ticks if t not in want]
        want.update(others[k] for k in rng.choice(len(others), size=min(12, len(others)), replace=False))
        cand[inst] = {(inst, pd.Timestamp(t)) for t in want}
        print(f"[{inst}] запусков analyse: {len(want)} (из них в кэше {sum((inst, pd.Timestamp(t)) in cache for t in want)})", flush=True)
        for t in sorted(want):
            decide(cache, inst, pd.Timestamp(t), live)
    ref_all = pd.concat(refs.values(), ignore_index=True)
    results, matches = {}, {}
    for sname, rule in SCHEDULES.items():
        for g in (False, True):
            name = f"{sname}; {'окно 30 мин' if g else 'вход только если впереди'}"
            broker, log = run(g, rule, cand, cache, minutes, live, all_ticks)
            pap = paper_trades(broker)
            results[name] = (pap, broker, log)
            matches[name] = match(ref_all, pap)
            print(f"== {name}: бумажных сделок {len(pap)}, поймано {matches[name]['вход на бумаге'].notna().sum()} из {len(ref_all)}", flush=True)
    # молчание вне сигналов: среди случайных часов analyse не должен открывать
    silent = [(i, t, cache[(i, t)]["действие"]) for i in STOCKS for (_, t) in cand[i]
              if cache[(i, t)]["_м"].get("решение") == "открыть" and not any(abs((r.open_dt - t).total_seconds()) < 3 * 3600 for r in refs[i].itertuples())]
    write_doc(ref_all, results, matches, silent, end)
    return 0


def write_doc(ref_all, results, matches, silent, end) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    r = ref_all.sort_values("close_dt")
    ax.step(r["close_dt"], (r["profit_pct"] / 3).cumsum(), where="post", color=GRAY, linewidth=2.2, label="эталон S42 (после издержек журнала)")
    for (name, (pap, _, _)), c in zip(results.items(), ("#c8641e", "#8a5cc4", "#5b9bd5", BLUE)):
        p = pap.dropna(subset=["net_pct"]).sort_values("close_dt")
        ax.step(p["close_dt"], (p["net_pct"] / 3).cumsum(), where="post", color=c, linewidth=2.4 if c == BLUE else 1.4, label=f"бумага: {name}")
    ax.axhline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, % капитала (3 бумаги поровну)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110, facecolor=BG)
    plt.close(fig)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    f = lambda v, d=2: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{d}f}"  # noqa: E731
    md = ["# Бумажный прогон живого контура на свежих данных", "",
          f"SBER, GAZP, LKOH на бирже только в лонг, {FRESH:%d.%m.%Y} — {end:%d.%m.%Y %H:%M}: после конца истории всех трёх бумаг, "
          "ни один день не участвовал в выборе стратегии. Решения — `live_signals.analyse`, заявки — `live_trade.plan` и "
          "`live_trade.execute`, брокер — бумажный (минутки биржи, комиссия 0.05% на сторону, без проскальзывания сверх "
          "того, что даёт время исполнения). Эталон — сделки S42 тем же пайплайном одним прогоном. Расчёт — `scripts/paper_replay.py`.", "",
          "![Накопленный итог](paper_replay/equity.png)", "", verdict, "", "## Варианты", "",
          "| вариант | бумажных сделок | поймано входов эталона | опоздание входа, мин (медиана) | цена входа хуже, % (медиана) | "
          "итог бумаги, % после комиссии | итог эталона по тем же сделкам, % | стоп без позиции |", "|---|---|---|---|---|---|---|---|"]
    ref_sum = ref_all["profit_pct"].sum() / 3
    for name, (pap, broker, _) in results.items():
        m = matches[name]
        got = m["вход на бумаге"].notna()
        md.append(f"| {name} | {len(pap)} | {got.sum()} из {len(m)} | {m.loc[got, 'опоздание входа, мин'].median():.0f} | "
                  f"{f(m.loc[got, 'цена входа хуже, %'].median(), 3)} | {f(pap['net_pct'].sum() / 3, 1)} | {f(m.loc[got, 'эталон, % после'].sum() / 3, 1)} | "
                  f"{len(broker.events)} |")
    md += ["", f"Эталон целиком: {len(ref_all)} сделок, {f(ref_sum, 1)}% после издержек журнала (0.05% комиссии и 0.05% "
           "проскальзывания на сторону), на капитал трёх бумаг поровну.", ""]
    best = list(results)[-1]
    md += [f"## Сделки: эталон против бумаги («{best}»)", "",
           "| бумага | вход эталона | выход эталона | причина | эталон % до / после | вход на бумаге | опоздание, мин | цена входа хуже, % | "
           "выход на бумаге | причина | бумага % до / после |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in matches[best].itertuples(index=False):
        md.append(f"| {r[0]} | {r[1]:%d.%m %H:%M} | {r[2]:%d.%m %H:%M} | {r[3]} | {f(r[4])} / {f(r[5])} | "
                  f"{'' if pd.isna(r[6]) else f'{r[6]:%d.%m %H:%M}'} | {'' if pd.isna(r[7]) else f'{r[7]:.0f}'} | {f(r[8], 3)} | "
                  f"{'' if pd.isna(r[9]) else f'{r[9]:%d.%m %H:%M}'} | {r[10]} | {f(r[11])} / {f(r[12])} |")
    md += ["", "## Проверки", "",
           f"* Случайные часы вне сигналов эталона: контур открыл бы сделку {len(silent)} раз" + (": " + "; ".join(f"{i} {t:%d.%m %H:%M}" for i, t, _ in silent) if silent else "."),
           "* Стоп-заявки без позиции (вторая заявка срабатывает после того, как первая закрыла позицию, до снятия раз в час): "
           + ", ".join(f"{n}: {len(b.events)}" for n, (_, b, _) in results.items()), ""]
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
