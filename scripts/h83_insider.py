"""H83 «Поиск инсайдерских сделок и роботов» — проверка по критерию из банка (записан 09.11 до расчёта).

Вход — дневные свёртки ленты ALGOPACK по 21 акции (scripts/h83_tape_daily.py → data/derived/h83_tape/*.parquet).
Три следа, событие — z-оценка против прошлых 60 торговых дней бумаги ≥ +2.5 (покупки) или ≤ −2.5 (продажи):
  A «крупные заявки»  — чистый объём заявок не меньше 99-го перцентиля размера заявки за прошлые 60 дней / объём дня;
  B «роботы-нарезчики» — чистый объём серий ≥ 30 заявок одного размера и стороны / объём дня;
  C «тихое накопление» — чистый поток дня / объём при |ход дня| < 0.5 ATR(20) (z — по всем дням, событие — в тихий).
Исход — с открытия следующего дня до закрытия 5-го минус средний ход 21 бумаги за то же окно. Выбор 2023–2024,
проверка 01.2025–06.04.2026. Сделки по одной бумаге не накладываются (пока открыта — новые события пропускаются).

    python scripts/h83_insider.py            # → docs/results/h83_insider.md, reports/h83/*.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "derived" / "h83_tape"
OUT_MD = ROOT / "docs" / "results" / "h83_insider.md"
OUT_DIR = ROOT / "reports" / "h83"
WIN, MINP, Z, HOLD, COST = 60, 40, 2.5, 5, 0.0005
PERIODS = {"выбор 2023–2024": ("2023-01-01", "2024-12-31"), "проверка 2025–04.2026": ("2025-01-01", "2026-12-31")}
NAMES = {"A": "A крупные заявки", "B": "B роботы-нарезчики", "C": "C тихое накопление"}
RNG = np.random.default_rng(83)


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(p) for p in sorted(SRC.glob("*.parquet"))], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"].dt.dayofweek < 5]                      # сессии выходного дня — не торговые дни для сигнала
    return d.sort_values(["tk", "date"]).reset_index(drop=True)


def zscore(x: pd.Series) -> pd.Series:
    m = x.shift(1).rolling(WIN, min_periods=MINP).mean()
    s = x.shift(1).rolling(WIN, min_periods=MINP).std()
    return (x - m) / s


def large_net(g: pd.DataFrame) -> pd.Series:
    """Чистый объём крупных заявок: порог — корзина 99-го перцентиля числа заявок за прошлые 60 дней."""
    hn = np.stack(g["hist_n"].to_numpy())
    hb, hs = np.stack(g["hist_b"].to_numpy()), np.stack(g["hist_s"].to_numpy())
    cs = np.cumsum(np.vstack([np.zeros(hn.shape[1]), hn]), axis=0)
    out = np.full(len(g), np.nan)
    for i in range(MINP, len(g)):
        prev = cs[i] - cs[max(0, i - WIN)]
        tot = prev.sum()
        if tot <= 0:
            continue
        k = int(np.searchsorted(np.cumsum(prev) / tot, 0.99))
        out[i] = (hb[i, k:].sum() - hs[i, k:].sum()) / g["vol"].iloc[i]
    return pd.Series(out, index=g.index)


def features(d: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tk, g in d.groupby("tk"):
        g = g.copy()
        pc = g["c"].shift(1)
        tr = pd.concat([g["h"] - g["l"], (g["h"] - pc).abs(), (g["l"] - pc).abs()], axis=1).max(axis=1)
        g["atr"] = tr.rolling(20, min_periods=15).mean()
        g["flow"] = (g["buy"] - g["sell"]) / g["vol"]
        g["zA"] = zscore(large_net(g))
        g["zB"] = zscore((g["robot_b"] - g["robot_s"]) / g["vol"])
        quiet = (g["c"] - pc).abs() < 0.5 * g["atr"].shift(1)
        g["zC"] = zscore(g["flow"]).where(quiet)
        # исход: вход — открытие следующего дня ленты, выход — закрытие 5-го; окно не длиннее 9 календарных дней
        g["o1"] = g["o"].shift(-1)
        g["c5"] = g["c"].shift(-HOLD)
        g["d5"] = g["date"].shift(-HOLD)
        g["ret"] = g["c5"] / g["o1"] - 1
        g.loc[(g["d5"] - g["date"]).dt.days > 9, "ret"] = np.nan
        g["big"] = (g["c5"] - g["o1"]).abs() > 2 * g["atr"]
        g["i"] = np.arange(len(g))
        parts.append(g)
    f = pd.concat(parts, ignore_index=True)
    f["mkt"] = f.groupby("date")["ret"].transform("mean")
    f["ex"] = f["ret"] - f["mkt"]
    f["period"] = None
    for name, (a, b) in PERIODS.items():
        f.loc[(f["date"] >= a) & (f["date"] <= b), "period"] = name
    return f


def events(f: pd.DataFrame, v: str, side: int) -> pd.DataFrame:
    """События следа v стороны side без наложений по бумаге (сделка держится HOLD дней)."""
    e = f[(f[f"z{v}"] * side >= Z) & f["ret"].notna() & f["period"].notna()]
    keep, busy = [], {}
    for r in e.itertuples():
        if r.i <= busy.get(r.tk, -1):
            continue
        keep.append(r.Index)
        busy[r.tk] = r.i + HOLD
    return e.loc[keep]


def tstat_by_date(e: pd.DataFrame) -> tuple[float, float, int]:
    m = e.groupby("date")["ex"].mean()
    if len(m) < 3:
        return float("nan"), float("nan"), len(m)
    return float(m.mean()), float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))), len(m)


def pf(x: np.ndarray) -> float:
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def random_pf(f: pd.DataFrame, e: pd.DataFrame, n: int = 1000) -> np.ndarray:
    """Случайные входы: та же бумага, тот же месяц, то же число сделок."""
    pool = f[f["ret"].notna()].assign(m=lambda x: x["date"].dt.to_period("M"))
    groups = {k: g["ret"].to_numpy() for k, g in pool.groupby(["tk", "m"])}
    need = e.assign(m=e["date"].dt.to_period("M")).groupby(["tk", "m"]).size()
    out = np.empty(n)
    for j in range(n):
        x = np.concatenate([RNG.choice(groups[k], size=c, replace=True) for k, c in need.items()])
        out[j] = pf(x - 2 * COST)
    return out


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load()
    f = features(d)
    days = d.groupby(d["date"].dt.year)["date"].nunique().to_dict()
    rows, verdict, trades = [], {}, []
    for v in "ABC":
        ok_v = True
        for side, sname in ((1, "покупки"), (-1, "продажи")):
            for p in PERIODS:
                e = events(f[f["period"] == p], v, side)
                mean, t, nd = tstat_by_date(e)
                r = {"след": NAMES[v], "сторона": sname, "период": p, "событий": len(e), "дат": nd,
                     "изб. ход, %": round(100 * mean, 3) if nd else None, "t": round(t, 2) if nd >= 3 else None}
                if side == 1:
                    x = e["ret"].to_numpy() - 2 * COST
                    rp = random_pf(f[f["period"] == p], e) if len(e) else np.array([np.nan])
                    r["PF лонга"] = round(pf(x), 2) if len(x) else None
                    r["лучше случайных, %"] = round(100 * float(np.mean(rp < pf(x))), 1) if len(x) else None
                    per = e.groupby("tk")["ex"].agg(["size", "mean"])
                    per = per[per["size"] >= 5]
                    r["бумаг ≥5 соб., в плюсе"] = f"{int((per['mean'] > 0).sum())} из {len(per)}"
                    r["_share"] = float((per["mean"] > 0).mean()) if len(per) else 0.0
                    trades.append(e.assign(variant=v)[["variant", "period", "tk", "date", "o1", "c5", "ret", "ex", f"z{v}"]])
                r["_mean"], r["_t"] = mean, t
                rows.append(r)
        sel, chk = PERIODS
        g = {(r["сторона"], r["период"]): r for r in rows if r["след"] == NAMES[v]}
        b1, b2, s1, s2 = g[("покупки", sel)], g[("покупки", chk)], g[("продажи", sel)], g[("продажи", chk)]
        c1 = (b1["_mean"] > 0 and b2["_mean"] > 0 and (b1["_t"] or 0) >= 2.4 and (b2["_t"] or 0) >= 1.65)
        c2 = s1["_mean"] < 0 and s2["_mean"] < 0
        c3 = all((r["PF лонга"] or 0) >= 1.2 and (r["лучше случайных, %"] or 0) >= 95 for r in (b1, b2))
        c4 = b2["_share"] >= 0.6
        verdict[v] = {"(1) ход после покупок": bool(c1), "(2) обратный знак продаж": bool(c2),
                      "(3) PF ≥ 1.2 и лучше 95% случайных": bool(c3), "(4) ≥ 60% бумаг в плюсе": bool(c4)}
    # справочно: предвестник сильных движений
    big = []
    for v in "ABC":
        for p in PERIODS:
            fp = f[(f["period"] == p) & f["ret"].notna()]
            ev = fp[fp[f"z{v}"].abs() >= Z]
            big.append({"след": NAMES[v], "период": p, "доля ходов > 2 ATR после событий, %": round(100 * ev["big"].mean(), 1),
                        "в обычные дни, %": round(100 * fp["big"].mean(), 1),
                        "отношение": round(ev["big"].mean() / fp["big"].mean(), 2)})
    tab = pd.DataFrame(rows).drop(columns=[c for c in pd.DataFrame(rows).columns if c.startswith("_")])
    tab.to_csv(OUT_DIR / "h83_stats.csv", index=False)
    pd.concat(trades).to_csv(OUT_DIR / "h83_trades.csv", index=False)
    (OUT_DIR / "h83_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = [v for v, c in verdict.items() if all(c.values())]
    with open(OUT_DIR / "h83_summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"дней ленты по годам (будни): {days}\n\n")
        fh.write(md(tab) + "\n\n")
        fh.write(md(pd.DataFrame([{"след": NAMES[v], **c} for v, c in verdict.items()])) + "\n\n")
        fh.write(md(pd.DataFrame(big)) + "\n\n")
        fh.write(f"прошли все пункты: {', '.join(passed) or 'ни один след'}\n")
    print((OUT_DIR / "h83_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
