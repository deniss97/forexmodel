"""H70 (Тимур): где большое плечо оправдано — плечо только в «хороших» ситуациях S42, а не на всё. Отчёт —
docs/results/h70_leverage.md.

Сделки — S42 на GAZP, SBER, LKOH из журнала (без наложений годовых прогонов, как в launch_rules.no_overlaps). Деньги:
депозит, сделка = 1/3 депозита × плечо, % депозита = результат сделки × эта доля; без платы за плечо и без лимита
денег на одновременные сделки (приближение — сколько их бывало одновременно, см. launch_rules).

Кандидаты «ситуации» (банк, до расчёта): режим «боковик» карты H75; стоп 6 ATR ближе медианы выбора (в % цены); оба.
Выбор — только по 2016–2020: наибольший PF при ≥ 10 сделках в год. Критерий: на проверке 2021–2026 «плечо 2 только в
ситуации, остальное без плеча» — итог к просадке выше, чем «плечо 2 на всё» и «без плеча».

    python scripts/h70_leverage.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import h75_regimes as RG
from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "h70_leverage.md"
STOCKS = RG.STOCKS
SLOTS = 3
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
PER = {"выбор": (pd.Timestamp("2016-01-01"), TEST_START), "проверка": (TEST_START, pd.Timestamp("2026-07-01"))}


def pf(x):
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


BORROW = 20.0       # % годовых за заёмную часть позиции (консервативно: ключевая ставка 2021–2026 + надбавка брокера)


def account(t: pd.DataFrame, lev: np.ndarray, per: str, borrow: float = BORROW) -> dict:
    a, b = PER[per]
    m = ((t["open_dt"] >= a) & (t["open_dt"] < b)).to_numpy()
    days = ((t.loc[m, "close_dt"] - t.loc[m, "open_dt"]).dt.total_seconds() / 86400).to_numpy()
    fee = (lev[m] - 1) / SLOTS * borrow / 100 * days / 365 * 100          # % депозита за заём на время сделки
    x = t[m].assign(dep=t.loc[m, "profit_pct"].to_numpy() * lev[m] / SLOTS - fee).sort_values("close_dt")
    eq = x["dep"].cumsum().to_numpy()
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    yrs = (b - a).days / 365.25
    return {"сделок": len(x), "итог % депозита": float(eq[-1]) if len(eq) else 0.0, "в год %": float(eq[-1]) / yrs if len(eq) else 0.0,
            "просадка %": dd, "итог / просадка": float(eq[-1]) / abs(dd) if dd else float("nan"),
            "худшая сделка % депозита": float(x["dep"].min()) if len(x) else 0.0}


def main() -> int:
    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    t = no_overlaps(t[t["instrument"].isin(STOCKS)]).reset_index(drop=True)
    b = {i: RG.bars(i) for i in STOCKS}
    days = pd.concat([RG.daily_windows(i, b[i]) for i in STOCKS], ignore_index=True)
    days = days[days["time"] >= "2016-01-01"]
    rg = RG.Regimes(days[days["time"] < TEST_START], RG.K)
    names = RG.describe(rg)
    tw = RG.trade_windows(t, b)
    ok = tw.notna().all(axis=1).to_numpy()
    t, tw = t[ok].reset_index(drop=True), tw[ok].reset_index(drop=True)
    lab = rg.label(tw)
    side = [c for c, n in enumerate(names) if n.startswith("боковик")]
    t["режим"] = [names[c] for c in lab]
    t["стоп %"] = 6 * t["atr_at_entry"] / t["open_price"] * 100
    sel = (t["open_dt"] < TEST_START).to_numpy()
    med = float(t.loc[sel, "стоп %"].median())
    cands = {"боковик (H75)": np.isin(lab, side), f"стоп ближе {med:.2f}% (медиана выбора)": (t["стоп %"] < med).to_numpy()}
    cands["боковик и узкий стоп"] = cands["боковик (H75)"] & cands[f"стоп ближе {med:.2f}% (медиана выбора)"]
    yrs_sel = (PER["выбор"][1] - PER["выбор"][0]).days / 365.25
    crow = []
    for k, m in cands.items():
        r = {"ситуация": k}
        for per, pm in (("выбор", sel), ("проверка", ~sel)):
            x = t.loc[m & pm, "profit_pct"]
            r.update({f"{per}: сделок": len(x), f"{per}: PF": pf(x), f"{per}: средняя %": x.mean(), f"{per}: худшая %": x.min()})
        r["допустима"] = r["выбор: сделок"] / yrs_sel >= 10
        crow.append(r)
    C = pd.DataFrame(crow)
    pick = C[C["допустима"]].sort_values("выбор: PF", ascending=False).iloc[0]["ситуация"]
    good = cands[pick]
    plans = {"без плеча": np.ones(len(t)), "плечо 2 на всё": np.full(len(t), 2.0),
             f"плечо 2 только «{pick}»": np.where(good, 2.0, 1.0), "плечо 3 на всё": np.full(len(t), 3.0),
             f"плечо 3 только «{pick}»": np.where(good, 3.0, 1.0)}
    rows = [{"вариант": k, "период": per, **account(t, v, per)} for k, v in plans.items() for per in PER]
    rows += [{"вариант": k + " (без платы за плечо)", "период": per, **account(t, v, per, borrow=0.0)}
             for k, v in plans.items() if k != "без плеча" and k.startswith("плечо 2") for per in PER]
    R = pd.DataFrame(rows)
    print(C.round(2).to_string(), "\nвыбрано:", pick, "\n", R.round(2).to_string())
    q = R[R["период"] == "проверка"].set_index("вариант")["итог / просадка"]
    ok2 = q[f"плечо 2 только «{pick}»"] > q["плечо 2 на всё"] and q[f"плечо 2 только «{pick}»"] > q["без плеча"]

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Где оправдано большое плечо: плечо только в «хороших» ситуациях S42 (H70)", "",
          "Гипотеза Тимура: плечо даёт преимущество, если брать его там, где соотношение вероятности, выигрыша и убытка лучше. "
          "Сделки — S42 на GAZP, SBER, LKOH (журнал, издержки биржи). Деньги: сделка = 1/3 депозита × плечо, % депозита, без "
          f"лимита денег на одновременные сделки; плата за плечо — {BORROW:g}% годовых на заёмную часть за дни сделки (для сравнения — "
          "строки «без платы»). Расчёт — `scripts/h70_leverage.py`.", "",
          "**Правило и критерий (банк, до расчёта):** ситуация выбирается только по 2016–2020 из трёх кандидатов (режим «боковик» "
          "карты H75; стоп 6 ATR ближе медианы выбора; оба) — наибольший PF при ≥ 10 сделках в год. Подтвердилась, если на "
          "проверке «плечо 2 только в ситуации» даёт итог к просадке выше, чем «плечо 2 на всё» и «без плеча».", "", verdict, "",
          "## Кандидаты", "", "| ситуация | выбор: сделок | PF | средняя % | худшая % | проверка: сделок | PF | средняя % | худшая % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in C.iterrows():
        md.append(f"| {'**' + r['ситуация'] + ' — выбрана**' if r['ситуация'] == pick else r['ситуация']} | {r['выбор: сделок']} | {r['выбор: PF']:.2f} | "
                  f"{r['выбор: средняя %']:+.2f} | {r['выбор: худшая %']:+.1f} | {r['проверка: сделок']} | {r['проверка: PF']:.2f} | "
                  f"{r['проверка: средняя %']:+.2f} | {r['проверка: худшая %']:+.1f} |")
    md += ["", "## Депозит", "", "| вариант | период | сделок | итог, % депозита | в год, % | просадка, % | итог / просадка | худшая сделка, % депозита |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['итог % депозита']:+.1f} | {r['в год %']:+.1f} | {r['просадка %']:.1f} | "
                  f"{r['итог / просадка']:.2f} | {r['худшая сделка % депозита']:+.1f} |")
    md += ["", f"Критерий выполнен: {'да' if ok2 else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok2, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
