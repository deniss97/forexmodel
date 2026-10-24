"""Металлы на фьючерсах Мосбиржи вместо CFD (гипотеза H68, 24.10): S42 по серебру и золоту на SILV и GOLD.

S42 по металлам посчитан на спот-ценах XAG/XAU под CFD Альфа-Форекс (свопы по серебру съедают результат, T01). Здесь
то же правило на непрерывном ряде фьючерса Мосбиржи — ближний контракт, переход за 5 дней до экспирации, прошлое
поправлено на разрыв цены при переходе (moex_data.continuous) — с издержками биржи без свопов (configs/costs/exchange.yaml:
silvfut, goldfut). Чтобы отделить «другой инструмент» от «нет свопов», сделки S42 на ценах CFD пересчитаны ещё и с
издержками фьючерса. Для портфеля — S46 и S48 на тех же рядах (движок перебора).

    scripts/queues/futures_download.sh             # минутки контрактов → data/raw/moex/<серия>/<контракт>_1m.csv
    python scripts/metals_futures.py prepare       # склейка в формате минуток XAG/XAU → data/raw/moex/<серия>_1m_fut.csv
    scripts/queues/metals_futures_s42.sh           # S42 — штатный walk-forward, как металлы в queues/costs.sh
    python scripts/metals_futures.py trades        # S46, S48 — движок перебора → reports/walk_forward
    python scripts/metals_futures.py report        # → docs/results/metals_futures.md
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

import trend_grid as tg  # noqa: E402
from experiments import BG, BLUE, FUTURES, GRAY, GRID, INK, INK2, ROOT, WF, load_catalog, load_trades  # noqa: E402

FUT = FUTURES
CFD = {"silvfut": "silver", "goldfut": "gold"}
RAW = ROOT / "data" / "raw" / "moex"
FEAT = ROOT / "reports" / "metals_futures" / "feat"
OUT = ROOT / "docs" / "results" / "metals_futures"
DOC = ROOT / "docs" / "results" / "metals_futures.md"
EXCHANGE = ROOT / "configs" / "costs" / "exchange.yaml"
PERIODS = ("выбор", "проверка")
COMMON_END = pd.Timestamp("2026-05-01")          # конец истории CFD-минуток: сравнение — по сделкам, открытым раньше
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def register() -> None:
    """Фьючерсы — в словари модулей движка (как new_stocks.register): конфиг серебра, минутки фьючерса, издержки биржи."""
    import forexmodel.simulation.costs as costs
    import strategy_search as ss

    for inst, ser in FUT.items():
        tg.INSTRUMENTS.setdefault(inst, ("silver.yaml", f"moex/{ser}_1m_fut", 2016))
    ss.FEAT = FEAT
    pe = costs.profile_entry
    if getattr(pe, "_futures", False):
        return

    def profile_entry(path, instrument, overrides=None):
        if instrument in FUT and str(path).endswith("base.yaml"):   # конструктор движка берёт базовый профиль CFD
            path = EXCHANGE
        return pe(path, instrument, overrides)

    profile_entry._futures = True
    costs.profile_entry = profile_entry


def prepare(inst: str) -> None:
    """Контракты → непрерывный ряд (переход за 5 дней до экспирации, прошлое поправлено на разрыв; moex_data.continuous_csv —
    потоком, в памяти один контракт) в колонках минуток XAG/XAU: time, open, high, low, close, volume + контракт."""
    import moex_data as md

    dst = RAW / f"{FUT[inst]}_1m_fut.csv"
    n = md.continuous_csv(RAW / FUT[inst], FUT[inst], dst, 1, 5, rename={"begin": "time"},
                          columns=["time", "open", "high", "low", "close", "volume", "contract"])
    print(f"[{inst}] {n} минут → {dst.relative_to(ROOT)}", flush=True)


def bars(inst: str) -> None:
    from forexmodel.data.loader import load_minute_compact, resample_ohlcv
    from night_search import build_features, build_gates
    from trend_filter_lab import base_config
    from trend_follow import TL

    FEAT.mkdir(parents=True, exist_ok=True)
    cfg = tg.instrument_config(inst)
    m = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, None, cfg.data.volume_candidates)   # без кэша — место на диске
    h = resample_ohlcv(m, cfg.data.base_timeframe).reset_index(drop=True)
    del m
    tg._W["base"] = dataclasses.replace(base_config(), **tg.GATE_C)
    b = h[["time", "open", "high", "low", "close"]]
    pd.to_pickle({"bars": b, "f": build_features(h, inst), "gates": build_gates(h), "tl": tg.gate_values(b, TL)}, FEAT / f"{inst}.pkl")
    print(f"[{inst}] {len(h)} часов {h['time'].iloc[0]:%Y-%m-%d} — {h['time'].iloc[-1]:%Y-%m-%d}", flush=True)


def trades(inst: str) -> None:
    import iterator_walkthrough as iw

    bars(inst)
    ins = iw.load_inst(inst)
    for k in (2, 4):
        t = iw.trades_frame(ins, iw.STRATS[k])
        t = t[t["open_dt"] >= "2016-01-01"] if len(t) else t
        t.to_csv(WF / f"{inst}_wt{k}__eng_trades.csv", index=False)
        print(f"[{inst}] {iw.EXP_ID[k]}: {len(t)} сделок", flush=True)


def coincide(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Доля сделок a, у которых в b есть сделка в ту же сторону с входом не дальше суток."""
    if a.empty or b.empty:
        return np.nan
    hits = 0
    for s in (1, -1):
        x, y = a[a["side"] == s]["open_dt"].to_numpy(), np.sort(b[b["side"] == s]["open_dt"].to_numpy())
        if not len(x) or not len(y):
            continue
        j = np.clip(np.searchsorted(y, x), 1, len(y) - 1)
        near = np.minimum(np.abs(x - y[j - 1]), np.abs(y[j] - x)) <= np.timedelta64(1, "D")
        hits += int(near.sum())
    return hits / len(a)


def report() -> int:
    from forexmodel.evaluation.experiments import metrics
    from forexmodel.simulation.costs import profile_entry, recost

    logging.disable(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    _, exps = load_catalog()
    fut = {"S42": load_trades({"id": "S42f", "sources": {i: [f"{i}_costs__impulse_6_3_h240_tr6"] for i in FUT}, "costs": {"profile": "exchange"}})}
    for code, k in (("S46", 2), ("S48", 4)):
        fut[code] = load_trades({"id": code, "sources": {i: [f"{i}_wt{k}__eng"] for i in FUT}, "costs": {"profile": "exchange"}})
    cfd = {c: load_trades(exps[c]) for c in ("S42", "S46", "S48")}
    cfd = {c: t[t["instrument"].isin(CFD.values())].replace({"instrument": {v: k for k, v in CFD.items()}}) for c, t in cfd.items()}
    # S42 на ценах CFD с издержками фьючерса — без свопов
    raw42 = load_trades({**exps["S42"], "sources": {v: exps["S42"]["sources"][v] for v in CFD.values()}, "costs": None})
    nos = []
    for inst, v in CFD.items():
        x = raw42[raw42["instrument"] == v]
        x = x.assign(gross_unit_pct=x["gross_pct"], size=1.0)
        nos.append(recost(x, profile_entry(EXCHANGE, inst)).assign(instrument=inst))
    cfd_nos = pd.concat(nos, ignore_index=True)
    insts = list(FUT)
    rows = []
    variants = [("S42 фьючерсы", fut["S42"], None), ("S42 CFD (журнал, свопы Альфа-Форекс)", cfd["S42"], COMMON_END),
                ("S42 CFD с издержками фьючерса", cfd_nos, COMMON_END), ("S42 фьючерсы до мая 2026", fut["S42"], COMMON_END),
                ("S46 фьючерсы", fut["S46"], None), ("S46 CFD (журнал)", cfd["S46"], COMMON_END),
                ("S48 фьючерсы", fut["S48"], None), ("S48 CFD (журнал)", cfd["S48"], COMMON_END)]
    for name, t, end in variants:
        t = t[t["open_dt"] < end] if end is not None else t
        for scope in (insts, ["silvfut"], ["goldfut"]):
            r = {"вариант": name, "срез": "серебро + золото" if len(scope) == 2 else {"silvfut": "серебро", "goldfut": "золото"}[scope[0]]}
            for per in PERIODS:
                m = metrics(t, scope, per)
                r.update({f"{per}: {k}": m.get(k) for k in ("сделок", "итог", "PF", "без_2_лучших", "t")})
            rows.append(r)
    res = pd.DataFrame(rows)
    same = {i: coincide(fut["S42"][(fut["S42"]["instrument"] == i) & (fut["S42"]["open_dt"] < COMMON_END)],
                        cfd["S42"][cfd["S42"]["instrument"] == i]) for i in insts}
    res.to_csv(OUT / "variants.csv", index=False)
    plot(fut["S42"], cfd["S42"], cfd_nos)
    write_doc(res, same)
    return 0


def plot(f: pd.DataFrame, c: pd.DataFrame, n: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for t, color, ls, label in ((c, GRAY, "-", "CFD (журнал S42, свопы Альфа-Форекс)"), (n, GRAY, "--", "CFD с издержками фьючерса"),
                                (f, BLUE, "-", "фьючерсы Мосбиржи")):
        t = t.sort_values("close_dt")
        ax.step(t["close_dt"], (t["profit_pct"] / 2).cumsum(), where="post", color=color, linestyle=ls, linewidth=2 if color == BLUE else 1.6, label=label)
    ax.axvline(pd.Timestamp("2021-01-01"), color=INK2, linewidth=0.8)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("накоплено, % (серебро и золото поровну)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)


def write_doc(res: pd.DataFrame, same: dict) -> None:
    f = lambda v, d=2, s=False: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.{d}f}" if s else f"{v:.{d}f}")  # noqa: E731
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    main = res[(res["вариант"] == "S42 фьючерсы") & (res["срез"] == "серебро + золото")].iloc[0]
    crit = {"проверка: PF > 1.1": main["проверка: PF"] > 1.1, "выбор: PF > 1.0": main["выбор: PF"] > 1.0,
            "без 2 лучших > 0 в обоих периодах": main["выбор: без_2_лучших"] > 0 and main["проверка: без_2_лучших"] > 0}
    md = ["# Металлы на фьючерсах Мосбиржи вместо CFD (H68)", "",
          "Импульс S42 без изменений по серебру и золоту в обе стороны на непрерывном ряде фьючерсов Мосбиржи SILV и GOLD "
          "(ближний контракт, переход за 5 дней до экспирации, прошлое поправлено на разрыв при переходе). Издержки — 0.04% "
          "комиссии и 0.05% проскальзывания на сторону, свопов нет. Сравнение — с S42 на спот-ценах CFD (журнал) и с теми же "
          "сделками CFD, пересчитанными с издержками фьючерса (без свопов); общий период для сравнения — сделки до мая 2026 "
          "(конец CFD-минуток). Критерий записан в банк (H68) до расчёта. Расчёт — `scripts/metals_futures.py`.", "",
          "![S42 по металлам: фьючерсы и CFD](metals_futures/equity.png)", "", verdict, "",
          "## Критерий H68", "", "| пункт | выполнен |", "|---|---|"] + [f"| {k} | {'✅' if v else '❌'} |" for k, v in crit.items()]
    md += ["", "## Варианты", "", "| вариант | срез | выбор: сделок | итог % | PF | без 2 лучших | проверка: сделок | итог % | PF | без 2 лучших | t |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        md.append(f"| {r['вариант']} | {r['срез']} | {r['выбор: сделок']} | {f(r['выбор: итог'], 1, True)} | {f(r['выбор: PF'])} | "
                  f"{f(r['выбор: без_2_лучших'], 1, True)} | {r['проверка: сделок']} | {f(r['проверка: итог'], 1, True)} | {f(r['проверка: PF'])} | "
                  f"{f(r['проверка: без_2_лучших'], 1, True)} | {f(r['проверка: t'])} |")
    md += ["", "Совпадение сделок S42 (вход в ту же сторону не дальше суток, до мая 2026): "
           + ", ".join(f"{'серебро' if i == 'silvfut' else 'золото'} {f(v * 100, 0)}% сделок фьючерса есть и на CFD" for i, v in same.items()) + ".", ""]
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    pd.set_option("display.width", 250)
    print(res.round(2).to_string(index=False))
    print(same)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["prepare", "trades", "report"])
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    register()
    if a.step == "report":
        return report()
    for inst in FUT:
        {"prepare": prepare, "trades": trades}[a.step](inst)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
