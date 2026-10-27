"""Разбор вопросов созвона 26.10 к отчёту № 4: почему портфель S42 + S46 + S48 даёт меньше одного S42, правило счёта
15 тыс. ₽ на истории, худшие сделки S42 (стопы и плечо), сделки через крупные дивидендные отсечки и позиция по стопу
(предложение 27.10: акций столько, чтобы потеря на стопе была 10% капитала) против слота 5 тыс.

Сделки — те же, что в H67 (scripts/launch_portfolio.py): S42 на SBER, GAZP, LKOH, S46 и S48 на 19 акциях, открытые до
мая 2026. Правило счёта — функция allocate из scripts/live_account.py, тот же код, что в контуре: слот 5 тыс. ₽ вниз до
целых лотов, лот дороже слота — один лот, если хватает денег, до 3 позиций, одна позиция на бумагу, при равных входах —
S42, S46, S48. Капитал без реинвестирования: прибыль в слоты не добавляется. Стоимость лота — по сырой цене входа
(в единицах бумаги на сегодня), результат сделки — по ценам с поправкой на дивиденды, как в бэктесте.

    python scripts/launch_questions.py
"""
from __future__ import annotations

import heapq
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import launch_portfolio as lp  # noqa: E402
from experiments import BG, BLUE, GRAY, INK, INK2, ROOT, STOCK_FILES, load_catalog, load_trades, raw_prices  # noqa: E402
from forexmodel.evaluation import portfolio as pf  # noqa: E402
from forexmodel.evaluation.experiments import TEST_START  # noqa: E402
from live_account import allocate  # noqa: E402

OUT = ROOT / "docs" / "results" / "launch_questions"
ORANGE, AQUA = "#c8641e", "#199e70"          # проверено валидатором палитры на фоне BG вместе с BLUE
PERIODS = {"выбор": (pd.Timestamp("2016-01-01"), TEST_START), "проверка": (TEST_START, lp.END)}
THIRD = {"S42": 1 / 3, "S46": 1 / 3, "S48": 1 / 3}
DIV_EXPS = {"S50": "лонг по фильтру, SBER/GAZP/LKOH", "S51": "S42 на 16 новых", "S56": "лонг по фильтру, 16 новых"}


def years(per: str) -> float:
    a, b = PERIODS[per]
    return (b - a).days / 365.25


def strategies_table(parts: dict) -> tuple[pd.DataFrame, dict, float]:
    """Каждая стратегия на весь капитал, портфель по трети и он же при той же наибольшей загрузке капитала, что у S42."""
    k = lp.exposure(lp.portfolio(parts, {"S42": 1.0}))["макс"] / lp.exposure(lp.portfolio(parts, THIRD))["макс"]
    variants = {"S42 один": {"S42": 1.0}, "S46 один": {"S46": 1.0}, "S48 один": {"S48": 1.0}, "D: по трети": THIRD,
                f"D ×{k:.1f}": {s: v * k for s, v in THIRD.items()}}
    rows, curves = [], {}
    for name, shares in variants.items():
        x = lp.portfolio(parts, shares)
        r = {"вариант": name}
        for per in PERIODS:
            m = pf.metrics(x, per)
            r.update({f"{per}: {c}": m.get(c) for c in ("в_год", "итог", "PF", "без_2_лучших", "просадка", "итог_к_просадке", "t_мес")})
            r[f"{per}: в год %"] = m.get("итог", 0) / years(per)
        r.update({f"загрузка: {c}": v for c, v in lp.exposure(x).items()})
        rows.append(r)
        curves[name] = pf.monthly(x).cumsum()
    return pd.DataFrame(rows), curves, k


def account_run(parts: dict, names: list[str], per: str, cfg: dict, lots: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Правило счёта на истории: (взятые сделки с суммой и результатом в ₽, не взятые с причиной)."""
    a, b = PERIODS[per]
    t = pd.concat([parts[s][0].assign(strategy=s) for s in names], ignore_index=True)
    t = t[(t["open_dt"] >= a) & (t["open_dt"] < b)].copy()
    t["raw_open"] = raw_prices(t)["open_price"].to_numpy()
    size = t["size"].fillna(1.0) if "size" in t else 1.0
    t["unit"] = t["profit_pct"] / size                       # результат на единицу позиции после издержек, %
    t = t.sort_values("open_dt", kind="stable")
    held, taken, missed = [], [], []                         # held — куча (close_dt, n, позиция)
    for at, group in t.groupby("open_dt", sort=True):
        while held and held[0][0] <= at:
            heapq.heappop(held)
        signals = [{"стратегия": r.strategy, "бумага": r.instrument.upper(), "вход_в": at, "цена": r.raw_open,
                    "стоп": r.raw_open * 0.5, "i": i} for i, r in group.iterrows()]
        take, alt = allocate(signals, [p for _, _, p in held], cfg, lots)
        for s in take:
            r = group.loc[s["i"]]
            pos = {"бумага": s["бумага"], "сумма": s["сумма"]}
            heapq.heappush(held, (r["close_dt"], len(taken), pos))
            taken.append({"стратегия": r["strategy"], "бумага": s["бумага"], "open_dt": at, "close_dt": r["close_dt"],
                          "сумма": s["сумма"], "unit": r["unit"], "pnl": s["сумма"] * r["unit"] / 100})
        missed += [{"стратегия": s["стратегия"], "бумага": s["бумага"], "open_dt": at, "причина": s["причина"].split(":")[0],
                    "unit": group.loc[s["i"], "unit"]} for s in alt]
    return pd.DataFrame(taken), pd.DataFrame(missed)


def account_table(parts: dict) -> tuple[pd.DataFrame, dict]:
    cfg = yaml.safe_load((ROOT / "configs" / "account.yaml").read_text(encoding="utf-8"))
    lots = {k: v["lot"] for k, v in json.loads((ROOT / cfg["lots_file"]).read_text(encoding="utf-8")).items()}
    rows, runs = [], {}
    for name, names in {"S42 один": ["S42"], "S42 + S46": ["S42", "S46"], "S42 + S46 + S48": ["S42", "S46", "S48"]}.items():
        for per in PERIODS:
            took, alt = account_run(parts, names, per, cfg, lots)
            runs[(name, per)] = (took, alt)
            took = took.sort_values("close_dt")
            eq = took["pnl"].cumsum().to_numpy()
            under = eq - np.maximum.accumulate(np.maximum(eq, 0))
            cap = cfg["capital"]
            r = {"набор": name, "период": per, "сигналов": len(took) + len(alt), "взято": len(took),
                 "итог ₽": took["pnl"].sum(), "итог %": took["pnl"].sum() / cap * 100,
                 "в год %": took["pnl"].sum() / cap * 100 / years(per), "просадка %": under.min() / cap * 100,
                 "дно просадки": took["close_dt"].iloc[int(under.argmin())].date(),
                 "худшая сделка ₽": took["pnl"].min(), "средняя сумма ₽": took["сумма"].mean(),
                 "упущено ₽": (alt["unit"] / 100 * cfg["slot"]).sum() if len(alt) else 0.0}
            for s in names:
                mine = took[took["стратегия"] == s]
                r[f"{s}: сигналов"] = len(mine) + (alt["стратегия"] == s).sum() if len(alt) else len(mine)
                r[f"{s}: взято"] = len(mine)
                r[f"{s}: итог ₽"] = mine["pnl"].sum()
            for why, n in (alt["причина"].value_counts() if len(alt) else pd.Series(dtype=int)).items():
                r[f"не взято: {why}"] = n
            rows.append(r)
    return pd.DataFrame(rows), runs


def worst_trades(parts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S42: распределение результата сделки и расстояние до стопа 6 ATR в % цены — к вопросу про стопы 3–5%."""
    t = parts["S42"][0].copy()
    t["unit"] = t["profit_pct"] / (t["size"].fillna(1.0) if "size" in t else 1.0)
    t["стоп %"] = 6 * t["atr_at_entry"] / t["open_price"] * 100
    rows = []
    for per, (a, b) in PERIODS.items():
        x = t[(t["open_dt"] >= a) & (t["open_dt"] < b)]
        u = x["unit"]
        rows.append({"период": per, "сделок": len(x), "≤ −3%": (u <= -3).sum(), "≤ −5%": (u <= -5).sum(),
                     "≤ −10%": (u <= -10).sum(), "худшая %": u.min(), "лучшая %": u.max(),
                     "стоп 6 ATR, медиана %": x["стоп %"].median(), "стоп 6 ATR, 90% сделок не дальше %": x["стоп %"].quantile(0.9),
                     "стоп 6 ATR, макс %": x["стоп %"].max(),
                     **{f"стоп дальше {c}%, доля": (x["стоп %"] > c).mean() * 100 for c in (3, 4, 5)}})
    worst = t.nsmallest(8, "unit")[["instrument", "open_dt", "close_dt", "unit", "стоп %", "exit_reason"]]
    return pd.DataFrame(rows), worst


def risk_run(t: pd.DataFrame, risk_rub: float, max_lev: float, cfg: dict, lots: dict) -> dict:
    """Позиция по риску (предложение 27.10): акций столько, чтобы потеря на стопе была risk_rub; все позиции вместе —
    не больше max_lev капиталов (плечо брокера); меньше лота — сделка пропускается."""
    cap = cfg["capital"]
    held, rows, skipped = [], [], 0
    for r in t.itertuples():
        while held and held[0][0] <= r.open_dt:
            heapq.heappop(held)
        room = max_lev * cap - sum(a for _, _, a in held)
        want = min(risk_rub / r.d, room)
        cost = lots.get(r.instrument.upper(), 1) * r.raw_open
        amount = np.floor(max(want, 0) / cost) * cost
        if amount <= 0:
            skipped += 1
            continue
        heapq.heappush(held, (r.close_dt, len(rows), amount))
        rows.append({"close_dt": r.close_dt, "amount": amount, "pnl": amount * r.unit / 100, "capped": risk_rub / r.d > room,
                     "exposure": (sum(a for _, _, a in held)) / cap})
    x = pd.DataFrame(rows).sort_values("close_dt")
    eq = x["pnl"].cumsum().to_numpy()
    under = eq - np.maximum.accumulate(np.maximum(eq, 0))
    streak = run = 0
    for p in x["pnl"]:
        run = run + 1 if p < 0 else 0
        streak = max(streak, run)
    return {"взято": len(x), "пропущено": skipped, "итог ₽": x["pnl"].sum(), "итог %": x["pnl"].sum() / cap * 100,
            "без лучшей сделки ₽": x["pnl"].sum() - x["pnl"].max(), "лучшая сделка ₽": x["pnl"].max(),
            "просадка %": under.min() / cap * 100, "дно просадки": x["close_dt"].iloc[int(under.argmin())].date(),
            "худшая сделка ₽": x["pnl"].min(), "позиция, медиана ₽": x["amount"].median(), "позиция, макс ₽": x["amount"].max(),
            "упёрлись в лимит, %": x["capped"].mean() * 100, "все позиции, макс × капитала": x["exposure"].max(),
            "убытков подряд, макс": streak}


def risk_table(parts: dict) -> pd.DataFrame:
    """S42 на счёте 15 тыс. ₽: слот 5 тыс. (сейчас) против позиции по риску 10% / 5% / 2% капитала при плече 1, 2, 4."""
    cfg = yaml.safe_load((ROOT / "configs" / "account.yaml").read_text(encoding="utf-8"))
    lots = {k: v["lot"] for k, v in json.loads((ROOT / cfg["lots_file"]).read_text(encoding="utf-8")).items()}
    t = parts["S42"][0].copy()
    t["raw_open"] = raw_prices(t)["open_price"].to_numpy()
    t["unit"] = t["profit_pct"] / (t["size"].fillna(1.0) if "size" in t else 1.0)
    t["d"] = 6 * t["atr_at_entry"] / t["open_price"]         # стоп 6 ATR от входа, доля цены
    rows = []
    for per, (a, b) in PERIODS.items():
        x = t[(t["open_dt"] >= a) & (t["open_dt"] < b)].sort_values("open_dt")
        took, _ = account_run(parts, ["S42"], per, cfg, lots)
        eq = took.sort_values("close_dt")["pnl"].cumsum().to_numpy()
        rows.append({"правило": "слот 5 тыс. (сейчас)", "период": per, "взято": len(took), "итог ₽": took["pnl"].sum(),
                     "итог %": took["pnl"].sum() / cfg["capital"] * 100,
                     "без лучшей сделки ₽": took["pnl"].sum() - took["pnl"].max(), "лучшая сделка ₽": took["pnl"].max(),
                     "просадка %": (eq - np.maximum.accumulate(np.maximum(eq, 0))).min() / cfg["capital"] * 100,
                     "худшая сделка ₽": took["pnl"].min(), "позиция, медиана ₽": took["сумма"].median()})
        for risk in (0.10, 0.05, 0.02):
            for lev in (1, 2, 4):
                r = risk_run(x, risk * cfg["capital"], lev, cfg, lots)
                rows.append({"правило": f"риск {risk:.0%} капитала, плечо до {lev}", "период": per, **r})
    out = pd.DataFrame(rows)
    out["в год %"] = out["итог %"] / out["период"].map(years)
    return out


def dividend_crossings(parts: dict) -> pd.DataFrame:
    """Сделки через дивидендную отсечку: на графике журнала (сырые цены) и в итоге (цены с поправкой) — разный знак."""
    _, exps = load_catalog()
    sets = {"S42": parts["S42"][0], "S46": parts["S46"][0], "S48": parts["S48"][0]}
    for code in DIV_EXPS:
        sets[code] = load_trades(exps[code])
    divs = pd.read_csv(ROOT / "data" / "dividends" / "moex_dividends.csv", parse_dates=["ex_time"])
    rows = []
    for code, t in sets.items():
        t = t[t["instrument"].isin(STOCK_FILES)].copy()
        if t.empty:
            continue
        raw = raw_prices(t)
        t["на графике %"] = (raw["exit_price"] / raw["open_price"] - 1) * 100
        for _, r in t.iterrows():
            d = divs[(divs["ticker"] == r["instrument"].upper()) & (divs["ex_time"] > r["open_dt"]) & (divs["ex_time"] <= r["close_dt"])]
            if len(d):
                rows.append({"эксперимент": code, "бумага": r["instrument"].upper(), "вход": r["open_dt"], "выход": r["close_dt"],
                             "на графике %": r["на графике %"], "в итоге %": r["profit_pct"],
                             "дивиденд %": d["dividend_pct"].sum(), "отсечка": d["ex_time"].max().date()})
    return pd.DataFrame(rows)


def plot_equity(curves: dict, k: float) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=BG)
    lp._axes(ax)
    style = {"S42 один": (GRAY, "-", 2.0), "S46 один": (ORANGE, "-", 1.6), "S48 один": (AQUA, "-", 1.6),
             "D: по трети": (BLUE, "-", 2.4), f"D ×{k:.1f}": (BLUE, "--", 1.8)}
    labels = {"S42 один": "S42 один на весь капитал", "S46 один": "S46 один на весь капитал",
              "S48 один": "S48 один на весь капитал", "D: по трети": "D: капитал по трети (слайд отчёта № 4)",
              f"D ×{k:.1f}": f"D ×{k:.1f}: та же наибольшая загрузка капитала, что у S42"}
    for name, (color, ls, lw) in style.items():
        m = curves[name]
        x = m.index.to_timestamp()
        ax.plot(x, m.to_numpy(), color=color, linestyle=ls, linewidth=lw, label=labels[name])
        ax.annotate(f"{name.split(' ')[0] if name.startswith('S') else name.split(':')[0]} {m.iloc[-1]:+.0f}%",
                    (x[-1], m.iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax.axvline(TEST_START, color=INK2, linewidth=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=INK2, fontsize=8.5)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("накоплено, % капитала", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_xlim(right=ax.get_xlim()[1] + 200)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110, facecolor=BG)
    plt.close(fig)


def plot_account(runs: dict) -> None:
    """Правило счёта 15 тыс. ₽ на проверке: накопленный результат в рублях, S42 один против набора."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8), facecolor=BG)
    lp._axes(ax)
    for name, color, lw in (("S42 один", GRAY, 2.0), ("S42 + S46", ORANGE, 1.6), ("S42 + S46 + S48", BLUE, 2.4)):
        took = runs[(name, "проверка")][0].sort_values("close_dt")
        eq = took["pnl"].cumsum()
        under = (eq - eq.clip(lower=0).cummax()).min()
        label = f"{name}: {eq.iloc[-1]:+,.0f} ₽, просадка {under:,.0f} ₽".replace(",", " ")
        ax.plot(took["close_dt"], eq, color=color, linewidth=lw, label=label, drawstyle="steps-post")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("накоплено, ₽ (счёт 15 тыс.)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "account.png", dpi=110, facecolor=BG)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = lp.strategies()
    res, curves, k = strategies_table(parts)
    acc, runs = account_table(parts)
    worst, worst8 = worst_trades(parts)
    div = dividend_crossings(parts)
    risk = risk_table(parts)
    risk.to_csv(OUT / "risk_sizing.csv", index=False)
    res.to_csv(OUT / "strategies.csv", index=False)
    acc.to_csv(OUT / "account.csv", index=False)
    worst.to_csv(OUT / "s42_losses.csv", index=False)
    worst8.to_csv(OUT / "s42_worst.csv", index=False)
    div.to_csv(OUT / "dividend_trades.csv", index=False)
    plot_equity(curves, k)
    plot_account(runs)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
    for title, df in (("стратегии", res), ("счёт", acc), ("S42 убытки", worst), ("S42 худшие", worst8), ("позиция по риску", risk)):
        print(f"\n== {title}\n{df.to_string(index=False)}")
    print(f"\n== отсечки: сделок через отсечку {len(div)}; по экспериментам\n{div.groupby('эксперимент').size().to_string()}")
    flip = div[(div["на графике %"] < 0) & (div["в итоге %"] > 0)].sort_values("дивиденд %", ascending=False)
    print(f"\n== на графике минус, в итоге плюс: {len(flip)}\n{flip.head(15).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
