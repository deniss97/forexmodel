"""H85 (скептик 08.11 к H70): размер позиции S42 по режиму рынка без плеча — больший слот в «боковике», меньший вне его.
Отчёт — docs/results/h85_regime_size.md.

Режимы — карта H75 (KMeans по неделе часовых свечей), обученная на GAZP, SBER, LKOH за 2016–2020, без переобучения. Сделки —
правило S42 на 18 других акциях (<бумага>_lo__impulse_6_3_long), издержки биржи, без наложений; бары — reports/new_stocks/feat.
Деньги: % депозита = результат сделки × слот, без лимита одновременных сделок (как в H70); итог к просадке не зависит от
масштаба, поэтому «ровный слот» — это вес 1, а перевеска — 1 в «боковике» и 1/2 вне его.
Критерий (банк, до расчёта): в обоих периодах PF «боковика» выше, чем в остальных режимах, и итог к просадке с перевеской выше,
чем с ровным слотом.

    python scripts/h85_regime_size.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import h75_regimes as RG
from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "h85_regime_size.md"
NEW = ROOT / "reports" / "new_stocks" / "feat"
OTHERS = ["afks", "aflt", "alrs", "chmf", "gmkn", "irao", "magn", "moex", "mtss", "nlmk", "nvtk", "plzl", "rosn", "rtkm",
          "sngsp", "tatn", "trnfp", "vtbr"]
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x) -> float:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def acc(t: pd.DataFrame, w: np.ndarray) -> dict:
    x = t.assign(dep=t["profit_pct"].to_numpy() * w).sort_values("close_dt")
    eq = x["dep"].cumsum().to_numpy()
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    return {"итог": float(eq[-1]) if len(eq) else 0.0, "просадка": dd, "итог / просадка": float(eq[-1]) / abs(dd) if dd else np.nan}


def main() -> int:
    # карта режимов — как в H70: дни GAZP, SBER, LKOH 2016–2020
    b_main = {i: RG.bars(i) for i in RG.STOCKS}
    days = pd.concat([RG.daily_windows(i, b_main[i]) for i in RG.STOCKS], ignore_index=True)
    days = days[(days["time"] >= "2016-01-01") & (days["time"] < TEST_START)]
    rg = RG.Regimes(days, RG.K)
    names = RG.describe(rg)
    side = [c for c, n in enumerate(names) if n.startswith("боковик")]
    parts = []
    for i in OTHERS:
        run = f"{i}_lo__impulse_6_3_long"
        t = no_overlaps(load_trades({"id": run, "sources": {i: [run]}, "costs": {"profile": "exchange"}, "dividends": True}))
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    b = {i: pd.read_pickle(NEW / f"{i}.pkl")["bars"][["time", "open", "high", "low", "close"]].reset_index(drop=True) for i in OTHERS}
    for v in b.values():
        v["time"] = pd.to_datetime(v["time"])
    tw = RG.trade_windows(T, b)
    ok = tw.notna().all(axis=1).to_numpy()
    T, tw = T[ok].reset_index(drop=True), tw[ok].reset_index(drop=True)
    lab = rg.label(tw)
    T["боковик"] = np.isin(lab, side)
    rows, res = [], {}
    for per, pm in (("выбор 2016–2020", T["open_dt"] < TEST_START), ("проверка 2021–2026", T["open_dt"] >= TEST_START)):
        x = T[pm]
        s = x["боковик"].to_numpy()
        flat = acc(x, np.ones(len(x)))
        tilt = acc(x, np.where(s, 1.0, 0.5))
        r = {"период": per, "сделок": len(x), "в боковике": int(s.sum()), "PF боковика": pf(x.loc[s, "profit_pct"]),
             "PF остальных": pf(x.loc[~s, "profit_pct"]), "итог/просадка ровный": flat["итог / просадка"],
             "итог/просадка перевеска": tilt["итог / просадка"], "просадка ровный": flat["просадка"], "просадка перевеска": tilt["просадка"]}
        rows.append(r)
        res[per] = r["PF боковика"] > r["PF остальных"] and r["итог/просадка перевеска"] > r["итог/просадка ровный"]
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("режимы:", names, "\n", R.round(3).to_string())
    okc = all(res.values())
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Размер позиции S42 по режиму рынка без плеча (H85)", "",
          "Гипотеза из скептика к H70: та же перевеска, что плечо в «боковике», но без заёма — больший слот в «боковике», меньший вне "
          "его. Карта режимов H75, обученная на GAZP, SBER, LKOH за 2016–2020; проверка — на 18 других акциях (правило S42). Расчёт — "
          "`scripts/h85_regime_size.py`.", "",
          "**Критерий (банк, до расчёта):** в обоих периодах PF «боковика» выше, чем в остальных режимах, и итог к просадке с перевеской "
          "(слот 1 в «боковике», 1/2 вне) выше, чем с ровным слотом.", "", verdict, "",
          "| период | сделок | в боковике | PF боковика | PF остальных | итог/просадка: ровный слот | итог/просадка: перевеска |",
          "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['период']} | {r['сделок']} | {r['в боковике']} | {r['PF боковика']:.2f} | {r['PF остальных']:.2f} | "
                  f"{r['итог/просадка ровный']:.2f} | {r['итог/просадка перевеска']:.2f} |")
    md += ["", f"Критерий выполнен: {'да' if okc else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", okc, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
