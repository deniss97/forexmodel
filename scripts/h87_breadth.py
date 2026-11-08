"""H87 (скептик 08.11 к H09): «ширина импульса» — вход S42 лучше, когда импульс одновременно на нескольких бумагах. Отчёт —
docs/results/h87_breadth.md.

Сделки — правило S42 (импульс 6 ч ≥ 3 ATR, гейт C, лонг) на 21 акции: GAZP, SBER, LKOH и 18 других (<бумага>_lo__impulse_6_3_long),
издержки биржи, без наложений по бумаге. «Ширина» входа — сколько других бумаг дали вход по тому же правилу за 24 ч до него
(считаются все сигналы, до снятия наложений). Порог k (вход «широкий», если ширина ≥ k) выбирается только по 2016–2020 по
21 бумаге — наибольший PF «широких» при не меньше 20% сделок. Критерий (банк, до расчёта): на 2021–2026 PF «широких» выше,
чем «узких», на 18 других бумагах, а перестановка меток «широкий» по дням даёт такую же или большую разницу не чаще 5%.

    python scripts/h87_breadth.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "h87_breadth.md"
MAIN = ["gazp", "sber", "lkoh"]
OTHERS = ["afks", "aflt", "alrs", "chmf", "gmkn", "irao", "magn", "moex", "mtss", "nlmk", "nvtk", "plzl", "rosn", "rtkm",
          "sngsp", "tatn", "trnfp", "vtbr"]
KS = (1, 2, 3, 4, 5)
REPS = 2000
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x) -> float:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def main() -> int:
    raw, kept = [], []
    for i in MAIN + OTHERS:
        run = f"{i}_lo__impulse_6_3_long"
        t = load_trades({"id": run, "sources": {i: [run]}, "costs": {"profile": "exchange"}, "dividends": True})
        raw.append(t[["instrument", "open_dt"]])
        kept.append(no_overlaps(t))
    R, T = pd.concat(raw, ignore_index=True), pd.concat(kept, ignore_index=True).reset_index(drop=True)
    R = R.sort_values("open_dt")
    times, insts = R["open_dt"].to_numpy(), R["instrument"].to_numpy()
    width = np.zeros(len(T), dtype=int)
    for k, (od, ins) in enumerate(zip(T["open_dt"].to_numpy(), T["instrument"].to_numpy())):
        a, b = np.searchsorted(times, od - np.timedelta64(24, "h"), "left"), np.searchsorted(times, od, "right")
        width[k] = len(set(insts[a:b]) - {ins})
    T["ширина"] = width
    sel = T["open_dt"] < TEST_START
    # порог k — по 2016–2020, все 21 бумага
    best = None
    for k in KS:
        w = T[sel & (T["ширина"] >= k)]
        if len(w) >= 0.2 * sel.sum() and (best is None or pf(w["profit_pct"]) > best[1]):
            best = (k, pf(w["profit_pct"]))
    k = best[0]
    rows = []
    for name, insts_ in (("GAZP, SBER, LKOH", MAIN), ("18 других", OTHERS), ("все 21", MAIN + OTHERS)):
        for per, pm in (("выбор 2016–2020", sel), ("проверка 2021–2026", ~sel)):
            x = T[pm & T["instrument"].isin(insts_)]
            wide = x["ширина"] >= k
            rows.append({"бумаги": name, "период": per, "сделок": len(x), "широких": int(wide.sum()),
                         "PF широких": pf(x.loc[wide, "profit_pct"]), "PF узких": pf(x.loc[~wide, "profit_pct"]),
                         "средняя широких %": float(x.loc[wide, "profit_pct"].mean()), "средняя узких %": float(x.loc[~wide, "profit_pct"].mean())})
    S = pd.DataFrame(rows)
    # перестановка по дням на 18 других, проверка: метка «широкий день» переставляется между днями
    x = T[~sel & T["instrument"].isin(OTHERS)].copy()
    x["day"] = x["open_dt"].dt.normalize()
    real = pf(x.loc[x["ширина"] >= k, "profit_pct"]) - pf(x.loc[x["ширина"] < k, "profit_pct"])
    dayw = x.groupby("day")["ширина"].max() >= k
    rng = np.random.default_rng(87)
    sims = []
    for _ in range(REPS):
        lab = pd.Series(rng.permutation(dayw.to_numpy()), index=dayw.index)
        m = x["day"].map(lab).to_numpy()
        sims.append(pf(x.loc[m, "profit_pct"]) - pf(x.loc[~m, "profit_pct"]))
    p = float((np.array(sims) >= real).mean())
    pd.set_option("display.width", 220)
    print("k =", k, "\n", S.round(3).to_string(), "\nразница PF на 18 других, проверка:", round(real, 3), "p =", p)
    o = S[(S["бумаги"] == "18 других") & (S["период"] == "проверка 2021–2026")].iloc[0]
    ok = bool(o["PF широких"] > o["PF узких"] and p <= 0.05)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# «Ширина импульса»: вход S42 при импульсе сразу на нескольких бумагах (H87)", "",
          "Гипотеза из скептика к H09: ограничение «не больше 2 позиций» отрезало прибыльные третьи одновременные сделки — может быть, "
          "входы, когда импульс идёт сразу по нескольким бумагам, лучше одиночных. Правило S42 на 21 акции, «ширина» — сколько других "
          "бумаг дали вход за 24 ч до него. Расчёт — `scripts/h87_breadth.py`.", "",
          "**Критерий (банк, до расчёта):** порог k — только по 2016–2020; на 2021–2026 PF «широких» входов выше, чем «узких», на 18 "
          "других бумагах, и перестановка меток по дням даёт такую же разницу не чаще 5%.", "", verdict, "",
          f"Порог, выбранный по 2016–2020: вход «широкий», если ещё хотя бы {k} бумаг(и) дали вход за 24 ч.", "",
          "| бумаги | период | сделок | широких | PF широких | PF узких | средняя широких, % | средняя узких, % |", "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        md.append(f"| {r['бумаги']} | {r['период']} | {r['сделок']} | {r['широких']} | {r['PF широких']:.2f} | {r['PF узких']:.2f} | "
                  f"{r['средняя широких %']:+.2f} | {r['средняя узких %']:+.2f} |")
    md += ["", f"Перестановка по дням (18 других, проверка): разница PF {real:+.2f}, p = {p:.3f}.", "", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
