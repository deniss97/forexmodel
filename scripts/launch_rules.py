"""Вопросы Тимура 28.10 к разбору 27.10 (docs/results/launch_questions.md, раздел 7):
  1. какой набор стратегий и размер сделки берут ВСЕ сигналы на депозите 15 тыс. ₽ при плече до 4 — и что это даёт;
  2. при какой просадке останавливать стратегию — пороги «внимание» и «стоп» по распределению просадок на горизонте
     пробного этапа (3, 6, 12 месяцев): перестановки месяцев проверки и скользящие окна истории;
  3. как решать, когда возобновлять, — правила «стоп по просадке → бумага → возврат» на истории против торговли без остановок.
Плюс T51: сколько сделок S42 на истории открывалось в выходные.

Сделки и правило счёта — те же, что в launch_questions.py (allocate из контура). Взять всё: позиций одновременно — столько,
сколько их бывало открыто на истории, сделка = депозит × плечо / это число, одна бумага может быть в нескольких
стратегиях сразу (у брокера это одна позиция, стопы — на свои части). Итог и просадка — в % депозита 15 тыс. ₽, без
реинвестирования и без платы за плечо.

    python scripts/launch_rules.py
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import launch_portfolio as lp  # noqa: E402
import launch_questions as lq  # noqa: E402
from experiments import BG, BLUE, GRAY, INK, INK2, ROOT  # noqa: E402

OUT = ROOT / "docs" / "results" / "launch_rules"
DEPOSIT = 15_000
SETS = {"S42": ["S42"], "S42 + S46": ["S42", "S46"], "S42 + S46 + S48": ["S42", "S46", "S48"]}
LEVERAGE = (1, 2, 4)
ORANGE, AQUA = "#c8641e", "#199e70"


def lots() -> dict:
    return {k: v["lot"] for k, v in json.loads((ROOT / "docs/results/launch_portfolio/lots.json").read_text(encoding="utf-8")).items()}


def no_overlaps(t: pd.DataFrame) -> pd.DataFrame:
    """Убрать сделки, открытые до закрытия прошлой сделки той же стратегии по той же бумаге: артефакт годовых прогонов
    walk-forward (3 января новый прогон стартует, хотя декабрьская сделка ещё открыта). Живой контур так не откроет."""
    t = t.sort_values(["instrument", "open_dt"])
    keep, last = [], {}
    for i, r in zip(t.index, t.itertuples()):
        if r.instrument in last and r.open_dt < last[r.instrument]:
            continue
        keep.append(i)
        last[r.instrument] = r.close_dt
    return t.loc[keep].sort_values("open_dt")


def concurrency(parts: dict, names: list[str]) -> int:
    """Сколько сделок набора бывало открыто одновременно (вся история, без правила «одна позиция на бумагу»)."""
    t = pd.concat([parts[s][0] for s in names], ignore_index=True)
    ev = pd.concat([pd.DataFrame({"t": t["open_dt"], "d": 1}), pd.DataFrame({"t": t["close_dt"], "d": -1})])
    return int(ev.sort_values(["t", "d"])["d"].cumsum().max())        # закрытия раньше открытий в ту же минуту


def exposure(took: pd.DataFrame) -> float:
    ev = pd.concat([pd.DataFrame({"t": took["open_dt"], "d": took["сумма"]}), pd.DataFrame({"t": took["close_dt"], "d": -took["сумма"]})])
    return float(ev.sort_values(["t", "d"])["d"].cumsum().max())


def dd(pnl: np.ndarray) -> float:
    eq = np.cumsum(pnl)
    return float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0


def config(capital: float, slot: float, positions: int, per_stock: bool) -> dict:
    return {"capital": capital, "slot": slot, "max_positions": positions, "lot_over_slot": True, "one_position_per_stock": per_stock,
            "priority": ["S42", "S46", "S48"], "max_loss_pct": 10, "weekends": True}


def all_trades(parts: dict) -> tuple[pd.DataFrame, dict]:
    """Вопрос 1: набор × плечо — сделка = депозит × плечо / наибольшее число одновременных сделок."""
    rows, runs, lt = [], {}, lots()
    for name, names in SETS.items():
        n = concurrency(parts, names)
        for lev in LEVERAGE:
            slot = DEPOSIT * lev / n
            cfg = config(DEPOSIT * lev, slot, n, per_stock=False)
            for per in lq.PERIODS:
                took, alt = lq.account_run(parts, names, per, cfg, lt)
                took = took.sort_values("close_dt")
                runs[(name, lev, per)] = took
                rows.append({"набор": name, "плечо до": lev, "период": per, "одновременно сделок, макс": n, "сделка ₽": round(slot),
                             "сигналов": len(took) + len(alt), "взято": len(took), "взято %": len(took) / (len(took) + len(alt)) * 100,
                             "не взято: лот дороже": int(alt["причина"].str.startswith("не хватает").sum()) if len(alt) else 0,
                             "итог ₽": took["pnl"].sum(), "в год % депозита": took["pnl"].sum() / DEPOSIT * 100 / lq.years(per),
                             "просадка % депозита": dd(took["pnl"].to_numpy()) / DEPOSIT * 100,
                             "худшая сделка ₽": took["pnl"].min(), "в бумагах макс × депозита": exposure(took) / DEPOSIT})
    return pd.DataFrame(rows), runs


def monthly(took: pd.DataFrame, per: str) -> pd.Series:
    a, b = lq.PERIODS[per]
    m = took.groupby(took["close_dt"].dt.to_period("M"))["pnl"].sum()
    return m.reindex(pd.period_range(a, b - pd.Timedelta(days=1), freq="M"), fill_value=0.0)


def drawdown_table(runs: dict, draws: int = 20000, seed: int = 7) -> pd.DataFrame:
    """Вопрос 2: наибольшая просадка за 3, 6, 12 месяцев — перестановки месяцев проверки (5–99-й процентиль) и
    худшее скользящее окно истории той же длины. В % депозита."""
    rng = np.random.default_rng(seed)
    rows = []
    for (name, lev) in (("S42", 1), ("S42", 2), ("S42", 4), ("S42 + S46", 4), ("S42 + S46 + S48", 4)):
        m = monthly(runs[(name, lev, "проверка")], "проверка").to_numpy()
        for h in (3, 6, 12):
            paths = np.cumsum(m[rng.integers(0, len(m), size=(draws, h))], axis=1)
            mdd = (paths - np.maximum.accumulate(np.maximum(paths, 0), axis=1)).min(axis=1) / DEPOSIT * 100
            roll = [dd(m[i:i + h]) / DEPOSIT * 100 for i in range(len(m) - h + 1)]
            rows.append({"набор": name, "плечо до": lev, "месяцев": h, "медиана": np.percentile(mdd, 50), "хуже 10%": np.percentile(mdd, 10),
                         "хуже 5% (внимание)": np.percentile(mdd, 5), "хуже 1% (стоп)": np.percentile(mdd, 1),
                         "худшее окно истории": min(roll), "итог окна, медиана": np.median([m[i:i + h].sum() for i in range(len(m) - h + 1)]) / DEPOSIT * 100})
    return pd.DataFrame(rows)


def stop_resume(took: pd.DataFrame, stop_pct: float, rule: str) -> dict:
    """Вопрос 3: торгуем, пока просадка счёта от пика меньше stop_pct депозита; остановились — стратегия идёт на бумаге,
    возвращаемся по правилу: «половина» — бумага отыграла от своего дна половину порога; «выше остановки» — бумага на полпорога выше
    уровня, где остановились; «10 сделок» — прошло 10 бумажных сделок и их сумма в плюсе."""
    t = took.reset_index(drop=True)
    ev = sorted([(r.close_dt, 0, i) for i, r in t.iterrows()] + [(r.open_dt, 1, i) for i, r in t.iterrows()])
    stop = stop_pct / 100 * DEPOSIT
    active, real, peak, paper, low, at_stop, n_since, sum_since = True, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
    taken, real_path, stops, off_days, off_from = set(), [], 0, 0.0, None
    for when, kind, i in ev:
        if kind == 1:                                        # вход: берём, только если торгуем
            if active:
                taken.add(i)
            continue
        pnl = float(t.at[i, "pnl"])
        paper += pnl
        if i in taken:
            real += pnl
            real_path.append(pnl)
            peak = max(peak, real)
        if active and peak - real >= stop:
            active, stops, at_stop, low, n_since, sum_since, off_from = False, stops + 1, paper, paper, 0, 0.0, when
            continue
        if not active:
            low, n_since, sum_since = min(low, paper), n_since + 1, sum_since + pnl
            back = ((rule == "половина" and paper - low >= stop / 2) or (rule == "выше остановки" and paper > at_stop + stop / 2)
                    or (rule == "10 сделок" and n_since >= 10 and sum_since > 0))
            if back:
                active, peak = True, real                   # просадку считаем заново от возврата
                off_days += (when - off_from).days
    if not active:
        off_days += (t["close_dt"].max() - off_from).days
    span = (t["close_dt"].max() - t["open_dt"].min()).days
    return {"итог ₽": real, "просадка % депозита": dd(np.array(real_path)) / DEPOSIT * 100, "сделок взято": len(taken),
            "остановок": stops, "вне рынка, % времени": off_days / span * 100}


def resume_table(runs: dict) -> pd.DataFrame:
    rows = []
    for (name, lev) in (("S42", 1), ("S42", 2)):
        for per in lq.PERIODS:
            took = runs[(name, lev, per)]
            base = {"итог ₽": took["pnl"].sum(), "просадка % депозита": dd(took["pnl"].to_numpy()) / DEPOSIT * 100,
                    "сделок взято": len(took), "остановок": 0, "вне рынка, % времени": 0.0}
            rows.append({"набор": name, "плечо до": lev, "период": per, "правило": "без остановок", **base})
            for sp in (10, 15, 20):
                for rule in ("половина", "выше остановки", "10 сделок"):
                    rows.append({"набор": name, "плечо до": lev, "период": per, "правило": f"стоп −{sp * lev}%, возврат: {rule}",
                                 **stop_resume(took, sp * lev, rule)})
    return pd.DataFrame(rows)


def plot(runs: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=BG)
    lp._axes(ax)
    for (name, lev), color, ls, lw in ((("S42", 1), GRAY, "-", 2.0), (("S42", 2), ORANGE, "-", 1.6), (("S42", 4), BLUE, "-", 2.2),
                                        (("S42 + S46", 4), AQUA, "-", 2.0), (("S42 + S46 + S48", 4), AQUA, "--", 1.4)):
        took = runs[(name, lev, "проверка")]
        eq = took["pnl"].cumsum() / DEPOSIT * 100
        label = f"{name}, плечо до {lev}: {eq.iloc[-1]:+.0f}%, просадка {dd(took['pnl'].to_numpy()) / DEPOSIT * 100:.0f}%"
        ax.plot(took["close_dt"], eq, color=color, linestyle=ls, linewidth=lw, label=label, drawstyle="steps-post")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_ylabel("накоплено, % депозита 15 тыс. ₽", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "all_trades.png", dpi=110, facecolor=BG)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {k: (no_overlaps(t), insts) for k, (t, insts) in lp.strategies().items()}
    q1, runs = all_trades(parts)
    q2 = drawdown_table(runs)
    q3 = resume_table(runs)
    s42 = parts["S42"][0]
    wk = {"S42 сделок": len(s42), "в выходные": int((s42["open_dt"].dt.weekday >= 5).sum()),
          "с 2025 года": int((s42["open_dt"] >= "2025-01-01").sum()),
          "в выходные с 2025": int(((s42["open_dt"] >= "2025-01-01") & (s42["open_dt"].dt.weekday >= 5)).sum())}
    q1.to_csv(OUT / "all_trades.csv", index=False)
    q2.to_csv(OUT / "drawdown.csv", index=False)
    q3.to_csv(OUT / "stop_resume.csv", index=False)
    (OUT / "weekends.json").write_text(json.dumps(wk, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(runs)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.precision", 1)
    for title, df in (("1. взять всё", q1), ("2. просадки", q2), ("3. стоп и возврат", q3)):
        print(f"\n== {title}\n{df.to_string(index=False)}")
    print("\n== T51", wk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
