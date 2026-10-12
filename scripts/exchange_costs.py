"""Акции на Мосбирже вместо CFD (гипотеза H04): сколько может стоить шорт, чтобы импульс по акциям жил.

Сделки — база B01 (прогоны *_costs__impulse_6_3_h240_tr6: цены акций скорректированы на дивиденды, стоп
по гэпу). Издержки пересчитываются по готовым сделкам (путь сделки от них не зависит) по профилю
configs/costs/exchange.yaml: лонг своими деньгами без переноса, шорт платит за каждую календарную ночь,
металлы — как в базе (CFD). Сетка: плата за шорт 0–0.15% в день × комиссия 0.04 / 0.10% за круг.
Протокол журнала: проверка — сделки, открытые с 2021 года.

    python scripts/exchange_costs.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import metrics, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "reports" / "walk_forward"
OUT = ROOT / "docs" / "results" / "exchange"
STOCKS = ["lkoh", "gazp", "sber", "moex", "mtss"]
NAMES = {"lkoh": "LKOH", "gazp": "GAZP", "sber": "SBER", "moex": "MOEX", "mtss": "MTSS"}
FEES = [0.0, 0.02, 0.05, 0.08, 0.12, 0.15]
COMMISSIONS = [0.04, 0.10]
BG, INK, INK2, GRID, BLUE, ORANGE, GRAY = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc", "#2459b3", "#c8641e", "#9aa3ae"
FS = 14


def trades(inst: str) -> pd.DataFrame:
    return pd.read_csv(WF / f"{inst}_costs__impulse_6_3_h240_tr6_trades.csv", parse_dates=["open_dt", "close_dt"])


def priced(profile: str, inst: str, overrides: dict | None = None) -> pd.DataFrame:
    t = recost(trades(inst), profile_entry(ROOT / "configs" / "costs" / profile, inst, overrides))
    return standardize(t, inst, inst)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rows = []
    cfd = pd.concat([priced("base.yaml", i) for i in STOCKS + ["silver"]], ignore_index=True)
    for per in ("выбор", "проверка", "всё"):
        rows.append({"вариант": "CFD (база B01)", "комиссия": 0.04, "шорт_в_день": 0.068, "период": per,
                     **{f"{k}_{sc}": metrics(cfd, mem, per).get(k) for sc, mem in (("акции", STOCKS), ("п4", ["silver", "lkoh", "gazp", "sber"]))
                        for k in ("итог", "PF", "без_2_лучших")}})
    by_inst = []
    for com in COMMISSIONS:
        for fee in FEES:
            ov = {"commission_pct": com, "swap_short_pct": -fee}
            T = pd.concat([priced("exchange.yaml", i, ov) for i in STOCKS] + [priced("exchange.yaml", "silver")], ignore_index=True)
            for per in ("выбор", "проверка", "всё"):
                rows.append({"вариант": "биржа", "комиссия": com, "шорт_в_день": fee, "период": per,
                             **{f"{k}_{sc}": metrics(T, mem, per).get(k) for sc, mem in (("акции", STOCKS), ("п4", ["silver", "lkoh", "gazp", "sber"]))
                                for k in ("итог", "PF", "без_2_лучших")}})
                if com == 0.10 and fee == 0.05:
                    for i in STOCKS:
                        m = metrics(T, [i], per)
                        by_inst.append({"инструмент": NAMES[i], "период": per, "итог": m.get("итог"), "PF": m.get("PF"),
                                        "cfd": metrics(cfd, [i], per).get("итог")})
    # только лонг по акциям: шорт не открывается (дорог или недоступен), металлы как в базе
    for com in COMMISSIONS:
        T = pd.concat([priced("exchange.yaml", i, {"commission_pct": com}) for i in STOCKS] + [priced("exchange.yaml", "silver")], ignore_index=True)
        T = T[(T["side"] > 0) | (T["instrument"] == "silver")]
        for per in ("выбор", "проверка", "всё"):
            rows.append({"вариант": "биржа, только лонг", "комиссия": com, "шорт_в_день": np.nan, "период": per,
                         **{f"{k}_{sc}": metrics(T, mem, per).get(k) for sc, mem in (("акции", STOCKS), ("п4", ["silver", "lkoh", "gazp", "sber"]))
                            for k in ("итог", "PF", "без_2_лучших")}})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "short_fee_grid.csv", index=False)
    B = pd.DataFrame(by_inst)
    B.to_csv(OUT / "by_instrument.csv", index=False)
    # доля шортов и переносов на сделку — для безубыточной платы
    allt = pd.concat([trades(i).assign(inst=i) for i in STOCKS])
    test = allt[allt["open_dt"] >= "2021-01-01"]
    short = test[test["side"] != "buy"]
    nights = ((short["close_dt"] - short["open_dt"]).dt.total_seconds() / 86400).mean()
    lines = ["ЧУВСТВИТЕЛЬНОСТЬ К ПЛАТЕ ЗА ШОРТ (акции — 5 бумаг поровну; п4 — серебро через CFD + LKOH, GAZP, SBER на бирже)",
             R[R["период"] != "всё"].round(2).to_string(index=False),
             f"\nпроверка 2021–2026: шортов {len(short)} из {len(test)} сделок акций ({len(short) / len(test):.0%}), "
             f"шорт держится в среднем {nights:.1f} суток",
             "\nПО БУМАГАМ при комиссии 0.10% за круг и шорте 0.05% в день:\n" + B.round(1).to_string(index=False)]
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")

    # график: итог акций на проверке от платы за шорт, две комиссии, ориентир CFD
    f, ax = plt.subplots(figsize=(11, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=FS - 1, length=0)
    for com, color, ls in ((0.04, BLUE, "-"), (0.10, BLUE, "--")):
        x = R[(R["вариант"] == "биржа") & (R["комиссия"] == com) & (R["период"] == "проверка")]
        ax.plot(x["шорт_в_день"], x["итог_акции"], color=color, linestyle=ls, linewidth=2.5, marker="o", markersize=8,
                label=f"биржа, комиссия {com:.2f}% за круг")
    lo = R[(R["вариант"] == "биржа, только лонг") & (R["комиссия"] == 0.10) & (R["период"] == "проверка")]["итог_акции"].iloc[0]
    ax.axhline(lo, color=INK, linewidth=2.5, linestyle=":", label=f"биржа, только лонг (комиссия 0.10%): {lo:+.0f}%")
    c = R[(R["вариант"] == "CFD (база B01)") & (R["период"] == "проверка")]["итог_акции"].iloc[0]
    ax.axhline(c, color=ORANGE, linewidth=2.5, label=f"CFD Альфа-Форекс (база B01): {c:+.0f}%")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("плата за шорт, % цены за ночь", fontsize=FS, color=INK2)
    ax.set_ylabel("итог 5 акций, 2021–2026, %", fontsize=FS, color=INK2)
    ax.legend(frameon=False, fontsize=FS - 1, labelcolor=INK)
    f.tight_layout()
    f.savefig(OUT / "short_fee.png", dpi=140, facecolor=BG)
    plt.close(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
