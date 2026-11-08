"""H09: лимит общей просадки и учёт одновременных сделок на связанных рынках — на наборе счёта (S42 на GAZP, SBER, LKOH
в лонг). Отчёт — docs/results/h09_limits.md.

Счёт как в жизни: слот 1/3 депозита, не больше 3 позиций, сигнал без свободного слота пропускается; % депозита =
результат сделки / 3. Правила (банк, до расчёта, без подбора):
  (а) не больше 2 позиций одновременно — три акции одного рынка движутся вместе;
  (б) лимит просадки: счёт по закрытым сделкам ушёл на 10% депозита ниже максимума → 20 календарных дней новых сделок не
      открываем, затем максимум считается заново;
  (в) оба правила.
Подтвердилась, если в обоих периодах просадка меньше, итог ниже не более чем на 10% от базы, итог к просадке выше.

    python scripts/h09_limits.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "h09_limits.md"
STOCKS = ["gazp", "sber", "lkoh"]
SLOTS = 3
DD_LIMIT, PAUSE_DAYS = 10.0, 20
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
PER = {"выбор 2016–2020": (pd.Timestamp("2016-01-01"), TEST_START), "проверка 2021–2026": (TEST_START, pd.Timestamp("2100-01-01"))}


def simulate(t: pd.DataFrame, max_pos: int = 3, dd_limit: float | None = None, pause_days: int = PAUSE_DAYS) -> dict:
    """Проход по сигналам по времени входа; закрытые сделки обновляют счёт к моменту следующего входа."""
    t = t.sort_values("open_dt").reset_index(drop=True)
    open_pos: list[tuple[pd.Timestamp, float]] = []      # (время выхода, % депозита)
    eq, peak, paused_until = 0.0, 0.0, pd.Timestamp.min
    closed: list[tuple[pd.Timestamp, float]] = []
    skipped_slot = skipped_pause = pauses = 0

    def settle(until):
        nonlocal eq, peak, paused_until, pauses, open_pos
        done = sorted([p for p in open_pos if p[0] <= until])
        open_pos = [p for p in open_pos if p[0] > until]
        for when, r in done:
            eq += r
            closed.append((when, r))
            if eq > peak:
                peak = eq
            if dd_limit is not None and peak - eq >= dd_limit and when >= paused_until:
                paused_until = when + pd.Timedelta(days=pause_days)
                pauses += 1
                peak = eq                                  # после паузы максимум считается заново

    for r in t.itertuples():
        settle(r.open_dt)
        if r.open_dt < paused_until:
            skipped_pause += 1
            continue
        if len(open_pos) >= min(max_pos, SLOTS):
            skipped_slot += 1
            continue
        open_pos.append((r.close_dt, r.profit_pct / SLOTS))
    settle(pd.Timestamp.max)
    c = pd.DataFrame(closed, columns=["when", "dep"]).sort_values("when")
    curve = c["dep"].cumsum().to_numpy()
    dd = float((curve - np.maximum.accumulate(np.maximum(curve, 0))).min()) if len(curve) else 0.0
    end = float(curve[-1]) if len(curve) else 0.0
    return {"сделок": len(c), "пропущено: нет слота": skipped_slot, "пропущено: пауза": skipped_pause, "пауз": pauses,
            "итог % депозита": end, "просадка %": dd, "итог / просадка": end / abs(dd) if dd else float("nan"),
            "худший месяц %": float(c.set_index("when")["dep"].resample("ME").sum().min()) if len(c) else 0.0}


def main() -> int:
    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    t = no_overlaps(t[t["instrument"].isin(STOCKS)]).reset_index(drop=True)
    plans = {"S42 как на счёте (3 позиции)": dict(), "(а) не больше 2 позиций": dict(max_pos=2),
             f"(б) пауза {PAUSE_DAYS} дн. при просадке {DD_LIMIT:g}%": dict(dd_limit=DD_LIMIT),
             "(в) оба правила": dict(max_pos=2, dd_limit=DD_LIMIT),
             "для сравнения: 1 позиция": dict(max_pos=1)}
    rows = []
    for name, kw in plans.items():
        for per, (a, b) in PER.items():
            x = t[(t["open_dt"] >= a) & (t["open_dt"] < b)]
            rows.append({"вариант": name, "период": per, **simulate(x, **kw)})
    R = pd.DataFrame(rows)
    # сколько позиций бывало одновременно у S42 без ограничений
    ev = pd.concat([pd.Series(1, index=t["open_dt"]), pd.Series(-1, index=t["close_dt"])]).sort_index()
    level = ev.cumsum().to_numpy()
    dur = np.diff(ev.index.to_numpy()).astype("timedelta64[s]").astype(float)
    share = {k: float(dur[level[:-1] == k].sum()) for k in (0, 1, 2, 3)}
    tot = sum(share.values())
    share = {k: v / tot * 100 for k, v in share.items()}
    pd.set_option("display.width", 250)
    print(R.round(2).to_string(), "\nвремя с k открытыми позициями, %:", {k: round(v, 1) for k, v in share.items()})
    s = R.set_index(["вариант", "период"])
    base = "S42 как на счёте (3 позиции)"
    verdicts = {}
    for name in list(plans)[1:4]:
        verdicts[name] = all(s.loc[(name, p), "просадка %"] > s.loc[(base, p), "просадка %"]
                             and s.loc[(name, p), "итог % депозита"] >= 0.9 * s.loc[(base, p), "итог % депозита"]
                             and s.loc[(name, p), "итог / просадка"] > s.loc[(base, p), "итог / просадка"] for p in PER)
    print(verdicts)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Лимит просадки и связанные рынки на счёте S42 (H09)", "",
          "Гипотеза (отчёт 05.10, техдолг): лимит общей просадки и учёт одновременных сделок на связанных рынках снизят "
          "просадку без потери итога. Набор счёта — S42 на GAZP, SBER, LKOH в лонг, издержки биржи; слот 1/3 депозита, не "
          "больше 3 позиций, сигнал без слота пропускается. Расчёт — `scripts/h09_limits.py`.", "",
          "**Критерий (банк, до расчёта):** правила (а) не больше 2 позиций, (б) пауза 20 дней после просадки 10% депозита, "
          "(в) оба. Подтвердилась, если в обоих периодах просадка меньше, итог ниже не более чем на 10%, итог к просадке выше.", "",
          verdict, "", "## Счёт", "",
          "| вариант | период | сделок | пропущено: нет слота | пропущено: пауза | пауз | итог, % депозита | просадка, % | итог / просадка | худший месяц, % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['пропущено: нет слота']} | {r['пропущено: пауза']} | {r['пауз']} | "
                  f"{r['итог % депозита']:+.1f} | {r['просадка %']:.1f} | {r['итог / просадка']:.2f} | {r['худший месяц %']:+.1f} |")
    md += ["", "Доля времени с k открытыми позициями S42 (без ограничений, 2016–2026): "
           + ", ".join(f"{k} — {v:.0f}%" for k, v in share.items()) + ".", "",
           "Критерий: " + "; ".join(f"{k} — {'да' if v else 'нет'}" for k, v in verdicts.items()) + "."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
