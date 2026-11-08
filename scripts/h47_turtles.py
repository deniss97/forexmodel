"""H47 (скептик): пробой 55 дней по гейту C («Черепахи», ТЗ 15.10) — случайность или нет, и что он даёт к S42 в
общем портфеле акций. Отчёт — docs/results/h47_turtles.md.

1. Случайные входы. Сделки пробоя — прогоны 15.10 `<рынок>_tz__pattern_donch_55` (8 рынков). Для каждого рынка,
   периода и стороны — столько же случайных входов на часах, где гейт C (reports/night_search/feat, строка 2)
   разрешает эту сторону; 1000 наборов. Выход у всех одинаковый и считается одним и тем же упрощённым движком на часовых
   свечах: вход по открытию следующего часа, стоп 6 ATR(14 ч), трейлинг 6 ATR от максимума после активации +3 ATR,
   выход по стопу (с гэпом — по открытию), не дольше 240 ч по часам. Издержки — средние издержки B01 по сделкам рынка.
   Тем же движком пересчитаны и сами входы пробоя, так что сравнение «вход против случайного» без поправок на
   минутную точность.
2. Портфель. Акции GAZP, SBER, LKOH в лонг, издержки биржи, без наложений: S42 один против S42 + пробой 55 дней;
   слот 1/3 депозита, не больше 3 позиций одновременно (как на счёте), лишние сигналы пропускаются.

    python scripts/h47_turtles.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START
from launch_rules import no_overlaps

from experiments import ROOT, load_catalog, load_trades

DOC = ROOT / "docs" / "results" / "h47_turtles.md"
OUT = ROOT / "docs" / "results" / "h47_turtles"
FEAT = ROOT / "reports" / "night_search" / "feat"
MARKETS = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
STOCKS = ["gazp", "sber", "lkoh"]
GROUPS = {"металлы": ["silver", "gold"], "портфель 4": ["silver", "lkoh", "gazp", "sber"], "все 8": MARKETS,
          "акции GAZP, SBER, LKOH": STOCKS}
RUN = "{}_tz__pattern_donch_55"
GATE_C = 2                       # строка гейта C в reports/night_search/feat/<рынок>.pkl (trend_grid.gate_items)
SL, ACT, HOURS = 6.0, 3.0, 240
REPS = 1000
END = pd.Timestamp("2100-01-01")     # проверка — до конца данных, как в журнале
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x: np.ndarray, axis=None):
    pos = np.where(x > 0, x, 0).sum(axis=axis)
    neg = -np.where(x < 0, x, 0).sum(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pos / neg


class Engine:
    """Упрощённые выходы S42/B01 на часовых свечах — для честного сравнения входов между собой."""

    def __init__(self, inst: str):
        d = pd.read_pickle(FEAT / f"{inst}.pkl")
        b = d["bars"]
        self.time = pd.to_datetime(b["time"]).to_numpy()
        self.o, self.h, self.l, self.c = (b[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
        self.atr = d["f"]["atr14"].astype(np.float64)
        self.gate = d["gates"][GATE_C]
        n = len(b)
        # последний бар окна: первый бар, начинающийся через ≥ 240 ч после входа (вход — бар s + 1)
        self.last = np.searchsorted(self.time, self.time + np.timedelta64(HOURS, "h"), side="left").clip(max=n - 1)
        self.L = int((self.last - np.arange(n)).max()) + 1

    def run(self, s: np.ndarray, side: np.ndarray, chunk: int = 8000) -> np.ndarray:
        """Результат сделки до издержек, % (s — бар сигнала, вход по открытию s + 1)."""
        out = np.empty(len(s))
        n = len(self.o)
        for a in range(0, len(s), chunk):
            ss, sd = s[a:a + chunk], side[a:a + chunk].astype(np.float64)
            e = ss + 1
            idx = e[:, None] + np.arange(self.L)[None, :]
            valid = idx <= self.last[e][:, None]
            idx = idx.clip(max=n - 1)
            E = self.o[e]
            A = self.atr[ss]
            # в «длинных» координатах: для шорта цены зеркалятся
            hi = np.where(sd[:, None] > 0, self.h[idx], -self.l[idx])
            lo = np.where(sd[:, None] > 0, self.l[idx], -self.h[idx])
            op = np.where(sd[:, None] > 0, self.o[idx], -self.o[idx])
            Ex = E * sd
            cmh = np.maximum.accumulate(np.where(valid, hi, -np.inf), axis=1)
            prev = np.concatenate([np.full((len(ss), 1), -np.inf), cmh[:, :-1]], axis=1)
            sl0 = (Ex - SL * A)[:, None]
            stop = np.where(prev >= (Ex + ACT * A)[:, None], np.maximum(sl0, prev - SL * A[:, None]), sl0)
            hit = (lo <= stop) & valid
            first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            r = np.arange(len(ss))
            k_end = valid.sum(axis=1) - 1
            exit_stop = np.minimum(op[r, first.clip(0)], stop[r, first.clip(0)])
            exit_time = np.where(sd > 0, self.c[idx[r, k_end]], -self.c[idx[r, k_end]])
            X = np.where(first >= 0, exit_stop, exit_time)
            out[a:a + chunk] = (X - Ex) / E * 100
        return out


def period_mask(time: np.ndarray, per: str) -> np.ndarray:
    t = pd.to_datetime(time)
    return (t < TEST_START) if per == "выбор" else ((t >= TEST_START) & (t < END))


def skeptic():
    rng = np.random.default_rng(47)
    rows, real, rand = [], {}, {}
    for inst in MARKETS:
        t = load_trades({"id": "H47", "sources": {inst: [RUN.format(inst)]}, "costs": {"profile": "base"}, "dividends": True})
        cost = float((t["gross_pct"] - t["profit_pct"]).mean())
        eng = Engine(inst)
        sig = pd.read_csv(ROOT / "reports" / "walk_forward" / f"{RUN.format(inst)}_trades.csv", parse_dates=["signal_dt"])["signal_dt"]
        s = np.searchsorted(eng.time, sig.to_numpy())
        ok = (s < len(eng.time) - 2) & (eng.time[s.clip(max=len(eng.time) - 1)] == sig.to_numpy())
        side = t["side"].to_numpy()[ok]
        s = s[ok]
        g_eng = eng.run(s, side)
        g_sim = t["gross_pct"].to_numpy()[ok]
        for per in ("выбор", "проверка"):
            pm = period_mask(eng.time[s], per)
            real[(inst, per)] = g_eng[pm] - cost
            sims = []
            for sd in (1, -1):
                n = int(((side == sd) & pm).sum())
                if not n:
                    continue
                pool = np.flatnonzero((eng.gate == sd) & period_mask(eng.time, per) & (np.arange(len(eng.time)) < len(eng.time) - 2)
                                      & np.isfinite(eng.atr) & (eng.atr > 0))
                pick = rng.choice(pool, size=(REPS, n), replace=True)
                sims.append(eng.run(pick.ravel(), np.full(pick.size, sd)).reshape(REPS, n) - cost)
            rand[(inst, per)] = np.concatenate(sims, axis=1) if sims else np.zeros((REPS, 0))
            rows.append({"рынок": inst, "период": per, "сделок": int(pm.sum()), "PF симулятора": float(pf(g_sim[pm] - cost)),
                         "PF движка": float(pf(real[(inst, per)])), "средняя движка %": float(real[(inst, per)].mean()) if pm.any() else np.nan,
                         "PF случайных, медиана": float(np.nanmedian(pf(rand[(inst, per)], axis=1))),
                         "доля случайных ниже": float(np.mean(pf(rand[(inst, per)], axis=1) < pf(real[(inst, per)])))})
        print(inst, "издержки", round(cost, 3), "последний вход", t["open_dt"].max(), flush=True)
    G, dist = [], {}
    for g, insts in GROUPS.items():
        for per in ("выбор", "проверка"):
            x = np.concatenate([real[(i, per)] for i in insts])
            R = np.concatenate([rand[(i, per)] for i in insts], axis=1)
            pr = pf(R, axis=1)
            dist[f"{g} · {per}"] = pr
            dist[f"{g} · {per} · пробой"] = np.full(REPS, pf(x))
            G.append({"группа": g, "период": per, "сделок": len(x), "PF пробоя": float(pf(x)), "средняя %": float(x.mean()),
                      "PF случайных: медиана": float(np.nanmedian(pr)), "95%": float(np.nanpercentile(pr, 95)),
                      "доля случайных ниже": float(np.mean(pr < pf(x)))})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dist).to_csv(OUT / "random_pf.csv", index=False)       # для графика еженедельного отчёта
    return pd.DataFrame(rows), pd.DataFrame(G)


def account(t: pd.DataFrame, per: str, slots: int = 3) -> dict:
    a, b = (pd.Timestamp("2016-01-01"), TEST_START) if per == "выбор" else (TEST_START, END)
    x = t[(t["open_dt"] >= a) & (t["open_dt"] < b)].sort_values("open_dt")
    open_until, took = [], []
    for r in x.itertuples():
        open_until = [c for c in open_until if c > r.open_dt]
        if len(open_until) >= slots:
            continue
        open_until.append(r.close_dt)
        took.append(r.Index)
    y = x.loc[took].sort_values("close_dt")
    eq = (y["profit_pct"] / slots).cumsum().to_numpy()
    dd = float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()) if len(eq) else 0.0
    by = y.groupby("стратегия").size().to_dict()
    return {"сделок": len(y), "из них пробой": int(by.get("пробой 55", 0)), "пропущено (нет слота)": len(x) - len(y),
            "итог % депозита": float(eq[-1]) if len(eq) else 0.0, "просадка %": dd,
            "итог / просадка": float(eq[-1]) / abs(dd) if dd else float("nan"),
            "в год %": (float(eq[-1]) if len(eq) else 0.0) / ((min(b, x["close_dt"].max()) - a).days / 365.25) if len(x) else 0.0}


def portfolio():
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    s42 = no_overlaps(s42[s42["instrument"].isin(STOCKS)]).assign(стратегия="S42")
    tu = load_trades({"id": "H47", "sources": {i: [RUN.format(i)] for i in STOCKS}, "costs": {"profile": "exchange"},
                      "dividends": True, "long_only": STOCKS})
    tu = no_overlaps(tu).assign(стратегия="пробой 55")
    both = pd.concat([s42, tu], ignore_index=True)
    rows = []
    for name, t in (("S42", s42), ("пробой 55 дней", tu), ("S42 + пробой 55 дней", both)):
        for per in ("выбор", "проверка"):
            rows.append({"вариант": name, "период": per, **account(t.reset_index(drop=True), per)})
    m = lambda t: t.set_index("close_dt")["profit_pct"].resample("ME").sum()  # noqa: E731
    mm = pd.concat([m(s42), m(tu)], axis=1, keys=["S42", "пробой"]).fillna(0)
    corr = {per: float(mm[(mm.index < TEST_START) == (per == "выбор")].corr().iloc[0, 1]) for per in ("выбор", "проверка")}
    # общие сделки: пробой, открытый, пока по той же бумаге открыта сделка S42
    same = 0
    for r in tu.itertuples():
        z = s42[(s42["instrument"] == r.instrument) & (s42["open_dt"] <= r.open_dt) & (s42["close_dt"] > r.open_dt)]
        same += len(z) > 0
    return pd.DataFrame(rows), corr, same / max(len(tu), 1), len(tu)


def main() -> int:
    R, G = skeptic()
    P, corr, same, ntu = portfolio()
    print(R.round(3).to_string(), "\n", G.round(3).to_string(), "\n", P.round(2).to_string(), "\n", corr, same)
    g = G.set_index(["группа", "период"])
    ok1 = all(g.loc[(k, p), "доля случайных ниже"] >= 0.95 for k in ("металлы", "все 8") for p in ("выбор", "проверка"))
    p = P.set_index(["вариант", "период"])["итог / просадка"]
    ok2 = all(p[("S42 + пробой 55 дней", per)] > p[("S42", per)] for per in ("выбор", "проверка"))

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Пробой 55 дней («Черепахи»): случайность или нет, и что даёт к S42 (H47)", "",
          "15.10 пробой 55-дневного уровня по гейту C был единственным новым входом в плюсе и на выборе, и на проверке "
          "(docs/results/trend_filter_tz.md, раздел 5), но с t < 1. Дополнение к гипотезе: проверить, насколько это "
          "случайно, и как он работает вместе с S42. Расчёт — `scripts/h47_turtles.py`.", "",
          "**Критерий (банк, до расчёта):** (1) PF пробоя выше 95% наборов случайных входов (то же число сделок в периоде "
          "и та же сторона, только там, где гейт C разрешает эту сторону, те же выходы и издержки) и на 2016–2020, и на "
          "2021–2026 — на металлах и на всех 8 рынках; (2) в портфеле акций GAZP, SBER, LKOH в лонг S42 + пробой даёт итог к "
          "просадке выше, чем S42 один, в обоих периодах.", "", verdict, "",
          "## 1. Пробой против случайных входов в том же тренде", "",
          "Выходы у всех одинаковые: упрощённый движок на часовых свечах (вход по открытию следующего часа, стоп и трейлинг "
          "6 ATR, активация +3 ATR, ≤ 240 ч), издержки B01 — средние по сделкам рынка. «PF симулятора» — исходные сделки "
          f"15.10 по минуткам, для сверки движка. {REPS} наборов случайных входов.", "",
          "| группа | период | сделок | PF пробоя | средняя % | PF случайных: медиана | 95% случайных | случайных ниже пробоя |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in G.iterrows():
        md.append(f"| {r['группа']} | {r['период']} | {r['сделок']} | {r['PF пробоя']:.2f} | {r['средняя %']:+.2f} | "
                  f"{r['PF случайных: медиана']:.2f} | {r['95%']:.2f} | {r['доля случайных ниже'] * 100:.0f}% |")
    md += ["", "### По рынкам", "", "| рынок | период | сделок | PF симулятора | PF движка | PF случайных, медиана | случайных ниже |",
           "|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        md.append(f"| {r['рынок']} | {r['период']} | {r['сделок']} | {r['PF симулятора']:.2f} | {r['PF движка']:.2f} | "
                  f"{r['PF случайных, медиана']:.2f} | {r['доля случайных ниже'] * 100:.0f}% |")
    md += ["", "## 2. Вместе с S42 на акциях (лонг, биржа)", "",
           "Слот 1/3 депозита, не больше 3 позиций одновременно, сигнал без свободного слота пропускается. Корреляция "
           f"месячных результатов S42 и пробоя: {corr['выбор']:+.2f} на выборе, {corr['проверка']:+.2f} на проверке; "
           f"{same * 100:.0f}% сделок пробоя ({ntu}) открываются, пока по той же бумаге уже открыта сделка S42.", "",
           "| вариант | период | сделок | из них пробой | пропущено | итог, % депозита | в год, % | просадка, % | итог / просадка |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in P.iterrows():
        md.append(f"| {r['вариант']} | {r['период']} | {r['сделок']} | {r['из них пробой']} | {r['пропущено (нет слота)']} | "
                  f"{r['итог % депозита']:+.1f} | {r['в год %']:+.1f} | {r['просадка %']:.1f} | {r['итог / просадка']:.2f} |")
    md += ["", f"Критерий (1) выполнен: {'да' if ok1 else 'нет'}; критерий (2): {'да' if ok2 else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok1, ok2, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
