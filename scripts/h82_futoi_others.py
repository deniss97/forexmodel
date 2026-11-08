"""H82: проверка находки H80 на других акциях — правило S42 только после дня, когда юрлица нарастили чистую позицию по
фьючерсу на эту акцию. Отчёт — docs/results/h82_futoi_others.md.

В H80 (SBER, GAZP, LKOH) окно 1 день было запасным, а не основным, — найдено после взгляда на данные. Здесь — 15 акций,
которых в H80 не было и у которых есть фьючерс с FUTOI. Сделки — правило S42 на этих акциях (импульс 6 ч ≥ 3 ATR,
гейт C, лонг; прогоны H66 `<бумага>_lo__impulse_6_3_long`), издержки биржи, без наложений, с 03.01.2020. FUTOI —
последний срез дня (`latest=1`), data/raw/moex/futoi_daily_<код>.csv.

Критерий (банк, до расчёта): на 2021–2026 по 15 акциям вместе PF и итог к просадке с фильтром выше, чем без него, и у
≥ 6 из 10 акций (было 9 из 15; у 5 бумаг нет истории FUTOI) PF с фильтром выше; в 2020 PF не хуже.

    python scripts/h82_futoi_others.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "h82_futoi_others.md"
FUT = ROOT / "data" / "raw" / "moex"
# у AFKS (ak), IRAO (ir), RTKM (rt), SNGSP (sg), TRNFP (tn) FUTOI с историей нет (ISS: данные с 2025–2026 или нет совсем)
CODE = {"aflt": "af", "alrs": "al", "gmkn": "gk", "magn": "mg", "moex": "me", "nlmk": "nm", "nvtk": "nk", "rosn": "rn",
        "tatn": "tt", "vtbr": "vb"}
MIN_BETTER = 6           # из 10 — та же доля, что 9 из 15 в исходном критерии
START = pd.Timestamp("2020-01-03")
SLOTS = 3
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x) -> float:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def account(t: pd.DataFrame, n_inst: int) -> dict:
    """Инструменты поровну: сделка = 1/3 слота × 3 / число бумаг — итог и просадка в % депозита портфеля из n бумаг."""
    x = t.sort_values("close_dt")
    eq = (x["profit_pct"] / n_inst).cumsum().to_numpy()
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    end = float(eq[-1]) if len(eq) else 0.0
    return {"сделок": len(x), "PF": pf(x["profit_pct"]), "средняя %": float(x["profit_pct"].mean()) if len(x) else np.nan,
            "итог %": end, "просадка %": dd, "итог / просадка": end / abs(dd) if dd else float("nan")}


def futoi_d1(code: str) -> pd.Series:
    f = pd.read_csv(FUT / f"futoi_daily_{code}.csv")
    p = f.pivot_table(index="tradedate", columns="clgroup", values=["pos", "pos_long"], aggfunc="last")
    d = pd.DataFrame({"yur": p[("pos", "YUR")], "oi": p[("pos_long", "YUR")] + p[("pos_long", "FIZ")]}).dropna()
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    return (d["yur"] - d["yur"].shift(1)) / d["oi"]


def main() -> int:
    parts, info = [], {}
    for inst, code in CODE.items():
        d1 = futoi_d1(code)
        run = f"{inst}_lo__impulse_6_3_long"
        t = load_trades({"id": run, "sources": {inst: [run]}, "costs": {"profile": "exchange"}, "dividends": True})
        t = no_overlaps(t[t["open_dt"] >= START]).reset_index(drop=True)
        days = d1.index.to_numpy()
        j = np.searchsorted(days, t["open_dt"].dt.normalize().to_numpy(), side="left") - 1
        t["d1"] = np.where(j >= 0, d1.to_numpy()[j.clip(0)], np.nan)
        info[inst] = {"дней FUTOI": len(d1), "с": d1.index.min().date(), "сделок": len(t), "без признака": int(t["d1"].isna().sum())}
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    n = len(CODE)
    rows, per_inst = [], []
    for per, pm in (("2020", T["open_dt"] < TEST_START), ("проверка 2021–2026", T["open_dt"] >= TEST_START)):
        x = T[pm]
        rows.append({"вариант": "правило S42", "период": per, **account(x, n)})
        rows.append({"вариант": "юрлица нарастили за день", "период": per, **account(x[x["d1"] > 0], n)})
        rows.append({"вариант": "юрлица сократили за день", "период": per, **account(x[x["d1"] < 0], n)})
        for inst in CODE:
            y = x[x["instrument"] == inst]
            per_inst.append({"бумага": inst, "период": per, "сделок": len(y), "PF": pf(y["profit_pct"]),
                             "с фильтром: сделок": int((y["d1"] > 0).sum()), "PF с фильтром": pf(y.loc[y["d1"] > 0, "profit_pct"]),
                             "PF обратной половины": pf(y.loc[y["d1"] < 0, "profit_pct"])})
    S, P = pd.DataFrame(rows), pd.DataFrame(per_inst)
    pd.set_option("display.width", 250)
    print(info, "\n", S.round(2).to_string(), "\n", P.round(2).to_string())
    s = S.set_index(["вариант", "период"])
    V, B, Pr = "юрлица нарастили за день", "правило S42", "проверка 2021–2026"
    pp = P[P["период"] == Pr]
    better = int((pp["PF с фильтром"] > pp["PF"]).sum())
    ok = (s.loc[(V, Pr), "PF"] > s.loc[(B, Pr), "PF"] and s.loc[(V, Pr), "итог / просадка"] > s.loc[(B, Pr), "итог / просадка"]
          and better >= MIN_BETTER and s.loc[(V, "2020"), "PF"] >= s.loc[(B, "2020"), "PF"])
    # перестановочная проверка: та же доля сделок, взятая случайно, — насколько редко PF не ниже, чем у фильтра
    rng = np.random.default_rng(82)
    x = T[T["open_dt"] >= TEST_START]
    k = int((x["d1"] > 0).sum())
    v = x["profit_pct"].to_numpy()
    sims = np.array([pf(v[rng.choice(len(v), k, replace=False)]) for _ in range(5000)])
    p_perm = float((sims >= s.loc[(V, Pr), "PF"]).mean())
    print("лучше у", better, "из", n, "; p перестановки", p_perm, "; критерий:", ok)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Фильтр «юрлица нарастили позицию за день» на 10 других акциях (H82)", "",
          "В H80 на SBER, GAZP, LKOH фильтр S42 по росту чистой позиции юрлиц во фьючерсе за последний день был лучше S42 в "
          "обоих периодах, но это окно было запасным — найдено после взгляда на данные. Проверка на акциях, которых в H80 "
          "не было: правило S42 (импульс, гейт C, лонг) на 10 бумагах с фьючерсами и историей FUTOI (у AFKS, IRAO, RTKM, SNGSP, TRNFP её нет), с 03.01.2020, издержки биржи. "
          "Расчёт — `scripts/h82_futoi_others.py`.", "",
          "**Критерий (банк, до расчёта):** на 2021–2026 по акциям вместе PF и итог к просадке с фильтром выше, у ≥ 6 из 10 "
          "акций PF с фильтром выше; в 2020 PF не хуже.", "", verdict, "",
          "## 10 акций вместе (поровну)", "", "| вариант | период | сделок | PF | средняя % | итог, % | просадка, % | итог / просадка |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} | {r['итог %']:+.1f} | "
                  f"{r['просадка %']:.1f} | {r['итог / просадка']:.2f} |")
    md += ["", f"PF с фильтром выше у {better} из {n} акций на проверке. Перестановочная проверка: случайные {k} сделок проверки из "
           f"{len(v)} дают PF не ниже, чем у фильтра, в {p_perm * 100:.1f}% случаев.", "",
           "## По акциям", "", "| бумага | период | сделок | PF | с фильтром: сделок | PF с фильтром | PF обратной половины |", "|---|---|---|---|---|---|---|"]
    for _, r in P.iterrows():
        md.append(f"| {r['бумага']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['с фильтром: сделок']} | {r['PF с фильтром']:.2f} | "
                  f"{r['PF обратной половины']:.2f} |")
    md += ["", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
