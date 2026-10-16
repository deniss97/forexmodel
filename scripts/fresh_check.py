"""Стратегии на свежих данных июнь–октябрь 2026 — акции на бирже, только лонг (задача T28).

Обобщение scripts/forward_test.py на любое правило walk-forward: тот же код сделок (walk_forward._rule_trades —
сигнал, гейт C, штатный минутный симулятор), свежие минуты Мосбиржи (data/raw/live), издержки биржи
(configs/costs/exchange.yaml). Окно — с 2026-01-01; «свежие» — сделки, открытые после конца истории (ни один
день не участвовал ни в выборе, ни в проверке). На истории сделки сверяются с прогоном журнала.

    python scripts/fresh_check.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from forexmodel.pipelines.dataset import build_dataset  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from experiments import BG, BLUE, GRID, INK, INK2, ROOT  # noqa: E402
from live_signals import instrument_cfg  # noqa: E402
from walk_forward import _rule_trades  # noqa: E402

OUT = ROOT / "docs" / "results" / "fresh_check"
DOC = ROOT / "docs" / "results" / "fresh_check.md"
WF = ROOT / "reports" / "walk_forward"
COSTS = ROOT / "configs" / "costs" / "exchange.yaml"
STOCKS = {"sber": "SBER_1m_15-26.csv", "gazp": "GAZP_1m_15-26_upd.csv", "lkoh": "LKOH_1m_15-26.csv"}
START = pd.Timestamp("2026-01-01")
# стратегия → (правило walk-forward, прогон журнала для сверки на истории)
RULES = {"S42 импульс": ("impulse:6:3:long", "{inst}_lo__impulse_6_3_long"),
         "S31 пробой 55 дней": ("pattern:donch:55:long", "{inst}_tz__pattern_donch_55")}
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def run(inst: str, live: dict, rule: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    spec = live["instruments"][inst]
    hist_end = pd.Timestamp(pd.read_csv(ROOT / "data" / "raw" / STOCKS[inst], usecols=["begin"]).iloc[-1, 0])
    last = pd.Timestamp(pd.read_csv(ROOT / spec["csv"], usecols=["begin"]).iloc[-1, 0])
    cfg = instrument_cfg(spec, live, START, last + pd.Timedelta(days=1))
    t = _rule_trades(build_dataset(cfg), cfg, rule)
    t = t[t["open_dt"] + pd.Timedelta(minutes=1) <= last]
    t = recost(t, profile_entry(COSTS, inst))
    t["instrument"] = inst
    t["открыта"] = (t["exit_reason"] == "timeout") & (t["close_dt"] >= last - pd.Timedelta(minutes=1)) & \
                   (t["close_dt"] - t["open_dt"] < pd.Timedelta(minutes=cfg.simulation.horizon_minutes))
    t["период"] = np.where(t["open_dt"] > hist_end, "свежий", "история")
    return t, hist_end, last


def journal(run_name: str, inst: str, hist_end) -> pd.DataFrame:
    b = pd.read_csv(WF / f"{run_name.format(inst=inst)}_trades.csv", parse_dates=["open_dt", "close_dt"])
    b = b[(b["side"] == "buy") & (b["open_dt"] >= START) & (b["close_dt"] <= hist_end)]
    return recost(b, profile_entry(COSTS, inst))


def main() -> int:
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    checks, fresh, ends = [], {}, {}
    for name, (rule, run_name) in RULES.items():
        parts = []
        for inst in STOCKS:
            t, hist_end, last = run(inst, live, rule)
            ends[inst] = (hist_end, last)
            parts.append(t)
            b = journal(run_name, inst, hist_end)
            h = t[(t["период"] == "история") & (t["close_dt"] <= hist_end)]
            m = h.merge(b, on="open_dt", how="outer", suffixes=("", "_j"), indicator=True)
            same = m[m["_merge"] == "both"]
            checks.append({"стратегия": name, "бумага": inst.upper(), "сделок скрипт": len(h), "сделок журнал": len(b),
                           "совпали по входу": len(same),
                           "расхождение результата, п.п.": float((same["profit_pct"] - same["profit_pct_j"]).abs().max()) if len(same) else np.nan})
            print(name, inst, len(h), len(b), len(same), flush=True)
        T = pd.concat(parts, ignore_index=True)
        T.to_csv(OUT / f"{name.split()[0]}_trades.csv", index=False)
        fresh[name] = T[T["период"] == "свежий"].sort_values("close_dt")

    # таблицы
    rows = []
    for name, F in fresh.items():
        for inst in STOCKS:
            p = F.loc[F["instrument"] == inst, "profit_pct"]
            rows.append({"стратегия": name, "бумага": inst.upper(), "сделок": len(p), "итог %": p.sum(),
                         "средняя %": p.mean() if len(p) else np.nan, "выигрышей": (p > 0).mean() if len(p) else np.nan})
        p = F["profit_pct"] / len(STOCKS)
        rows.append({"стратегия": name, "бумага": "3 бумаги поровну", "сделок": len(F), "итог %": p.sum(),
                     "средняя %": F["profit_pct"].mean() if len(F) else np.nan, "выигрышей": (F["profit_pct"] > 0).mean() if len(F) else np.nan})
    S = pd.DataFrame(rows)
    both = pd.concat([f.assign(w=0.5 / len(STOCKS)) for f in fresh.values()])
    port = float((both["w"] * both["profit_pct"]).sum())

    # график
    fig, ax = plt.subplots(figsize=(9.5, 3.8), facecolor=BG)
    ax.set_facecolor(BG)
    for (name, F), color in zip(fresh.items(), (BLUE, "#c8641e")):
        ax.step(F["close_dt"], (F["profit_pct"] / len(STOCKS)).cumsum(), where="post", color=color, linewidth=2.2,
                label=f"{name}: {(F['profit_pct'] / len(STOCKS)).sum():+.1f}%")
    g = both.sort_values("close_dt")
    ax.step(g["close_dt"], (g["w"] * g["profit_pct"]).cumsum(), where="post", color=INK, linewidth=2.6, label=f"S42 + S31 поровну: {port:+.1f}%")
    ax.axhline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, % (3 бумаги поровну)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fresh_equity.png", dpi=110, facecolor=BG)
    plt.close(fig)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    period = ", ".join(f"{i.upper()} {ends[i][0]:%d.%m}–{ends[i][1]:%d.%m.%Y}" for i in STOCKS)
    md = ["# Свежие данные июнь–октябрь 2026: импульс и пробой 55 дней на акциях", "",
          f"Задача **T28** (гипотезы H47, H53). Акции GAZP, SBER, LKOH на Мосбирже только в лонг, издержки биржи; свежий "
          f"период — после конца истории ({period}): ни один день не участвовал ни в выборе, ни в проверке. Код сделок — "
          "тот же, что у walk-forward и журнала (`scripts/fresh_check.py`). Металлы не считались: свежего спота нет, а "
          "фьючерс меняет почти половину сделок ([forward_test.md](forward_test.md)). Открытые сделки — по последней цене.", "",
          "![Свежий период](fresh_check/fresh_equity.png)", "", verdict, "",
          "## Итог на свежем периоде", "", "| стратегия | бумага | сделок | итог % | средняя сделка % | выигрышей |", "|---|---|---|---|---|---|"]
    md += [f"| {r['стратегия']} | {r['бумага']} | {r['сделок']} | {r['итог %']:+.1f} | "
           + ("—" if not np.isfinite(r["средняя %"]) else f"{r['средняя %']:+.2f}") + " | "
           + ("—" if not np.isfinite(r["выигрышей"]) else f"{r['выигрышей']:.0%}") + " |" for _, r in S.iterrows()]
    md += ["", f"Портфель S42 + S31 поровну (капитал пополам): **{port:+.1f}%**.", "",
           "## Сверка с журналом на истории (январь — конец истории 2026)", "",
           "| стратегия | бумага | сделок скрипт | сделок журнал | совпали по входу | расхождение результата, п.п. |", "|---|---|---|---|---|---|"]
    md += [f"| {c['стратегия']} | {c['бумага']} | {c['сделок скрипт']} | {c['сделок журнал']} | {c['совпали по входу']} | "
           f"{c['расхождение результата, п.п.']:.3f} |" for c in checks]
    md += ["", "## Сделки свежего периода", ""]
    for name, F in fresh.items():
        md += [f"### {name}", "", "| бумага | вход | выход | цена входа | цена выхода | причина | итог % |", "|---|---|---|---|---|---|---|"]
        md += [f"| {r['instrument'].upper()} | {r['open_dt']:%d.%m %H:%M} | {r['close_dt']:%d.%m %H:%M}{' (открыта)' if r['открыта'] else ''} | "
               f"{r['open_price']:.2f} | {r['exit_price']:.2f} | {r['exit_reason']} | {r['profit_pct']:+.2f} |" for _, r in F.sort_values("open_dt").iterrows()]
        md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
