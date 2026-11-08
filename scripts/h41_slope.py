"""H41 (Тимур): наклон тренда за окно и sin / cos его угла как признак — шаг 1, связь с дальнейшим ходом (IC).
Шаг 2 (модель) — scripts/queues/h41_slope.sh, сводка — `python scripts/h41_slope.py --model`. Отчёт —
docs/results/h41_slope.md.

Признак (forexmodel/features/extension.py, features.slope_windows): подъём линии регрессии close за w часов в ATR(14) —
slope_atr_w; угол atan(подъём / √w) — slope_sin_w, slope_cos_w. Сравнение — со смещением за окно runup_w, которое у
модели уже есть (6, 12, 24 ч). Цель — ход за 10 ч в ATR (горизонт разметки). IC — ранговая корреляция Спирмена;
t ≈ IC·√(N/10) (цели соседних часов перекрываются). Частная IC — после вычета рангового влияния runup_6/12/24.
«Внутри гейта» — только часы, где гейт C задаёт сторону, признак и цель умножены на неё (как модель и торгует).

    python scripts/h41_slope.py            # IC
    python scripts/h41_slope.py --model    # IC + сводка прогонов модели
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from forexmodel.evaluation.experiments import TEST_START, metrics

from experiments import ROOT, load_trades

DOC = ROOT / "docs" / "results" / "h41_slope.md"
FEAT = ROOT / "reports" / "night_search" / "feat"
MARKETS = ["silver", "lkoh", "gazp", "sber"]
WINDOWS = [24, 72, 168]
RUNUP = [6, 12, 24]
H = 10
SEEDS = (42, 7, 1)
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
RUNS = {"lkoh": ("lkoh_h63_multiclass_s{}", "lkoh_h41_slope_s{}", {"profile": "exchange"}),
        "silver": ("silver_h41_base_s{}", "silver_h41_slope_s{}", {"profile": "base"})}


def linreg_rise(c: pd.Series, w: int) -> pd.Series:
    from forexmodel.features.indicators import linreg_slope

    return linreg_slope(c, w) * (w - 1)


def frame(inst: str) -> pd.DataFrame:
    d = pd.read_pickle(FEAT / f"{inst}.pkl")
    b = d["bars"].reset_index(drop=True)
    c = b["close"].astype(float)
    atr = pd.Series(d["f"]["atr14"].astype(float)).replace(0, np.nan)
    out = pd.DataFrame({"time": pd.to_datetime(b["time"]), "gate": d["gates"][2].astype(float)})
    out.loc[out["gate"] == 99, "gate"] = np.nan
    for w in RUNUP:
        out[f"runup_{w}"] = (c - c.shift(w)) / atr
    for w in WINDOWS:
        rise = linreg_rise(c, w) / atr
        ang = np.arctan(rise / np.sqrt(w))
        out[f"slope_atr_{w}"], out[f"slope_sin_{w}"], out[f"slope_cos_{w}"] = rise, np.sin(ang), np.cos(ang)
    out["fwd"] = (c.shift(-H) - c) / atr
    return out


def ic(x: np.ndarray, y: np.ndarray, ctrl: np.ndarray | None = None) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    if ctrl is not None:
        m &= np.isfinite(ctrl).all(axis=1)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    rx, ry = rankdata(x[m]), rankdata(y[m])
    if ctrl is not None:
        C = np.column_stack([np.ones(m.sum())] + [rankdata(z) for z in ctrl[m].T])
        rx = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
        ry = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def ic_table() -> pd.DataFrame:
    rows = []
    feats = [f"runup_{w}" for w in RUNUP] + [f"slope_{k}_{w}" for w in WINDOWS for k in ("atr", "sin", "cos")]
    for inst in MARKETS:
        f = frame(inst)
        for per in ("выбор", "проверка"):
            p = f[(f["time"] < TEST_START) == (per == "выбор")]
            g = p["gate"].to_numpy()
            ing = np.isfinite(g) & (g != 0)
            ctrl = p[[f"runup_{w}" for w in RUNUP]].to_numpy()
            for col in feats:
                x, y = p[col].to_numpy(), p["fwd"].to_numpy()
                # cos не знаковый: с модулем хода (сила движения), остальные — с ходом
                yy = np.abs(y) if "_cos_" in col else y
                r = {"рынок": inst, "период": per, "признак": col}
                r["IC"], n = ic(x, yy)
                r["t"] = r["IC"] * np.sqrt(n / H)
                if col.startswith("slope_"):
                    r["частная IC"], _ = ic(x, yy, ctrl if "_cos_" not in col else np.abs(ctrl))
                    ru = p["runup_24"].to_numpy()
                    r["корр. с runup_24"] = ic(x, np.abs(ru) if "_cos_" in col else ru)[0]
                if "_cos_" not in col:
                    r["IC внутри гейта"], ng = ic((x * g)[ing], (y * g)[ing])
                    if col.startswith("slope_"):
                        r["частная IC внутри гейта"], _ = ic((x * g)[ing], (y * g)[ing], (ctrl * g[:, None])[ing])
                rows.append(r)
    return pd.DataFrame(rows)


def model_table() -> pd.DataFrame:
    rows = []
    for inst, (base, new, costs) in RUNS.items():
        for tag, pat in (("без наклона", base), ("с наклоном", new)):
            for sd in SEEDS:
                run = pat.format(sd)
                if not (ROOT / "reports" / "walk_forward" / f"{run}_trades.csv").exists():
                    continue
                t = load_trades({"id": run, "sources": {inst: [run]}, "costs": costs, "dividends": True})
                for per in ("выбор", "проверка"):
                    m = metrics(t, [inst], per)
                    rows.append({"рынок": inst, "модель": tag, "seed": sd, "период": per, "сделок": m.get("сделок", 0),
                                 "итог %": m.get("итог", np.nan), "PF": m.get("PF", np.nan), "просадка": m.get("просадка", np.nan)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    a = ap.parse_args()
    T = ic_table()
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    M = model_table() if a.model else pd.DataFrame()
    if len(M):
        print(M.round(2).to_string())

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Наклон тренда за окно и sin / cos его угла как признак (H41)", "",
          "Идея Тимура: мерить не наклон свечи, а наклон тренда за период, через sin и cos угла; возможно, пригодится и "
          "в тренд-фильтре. Признак — подъём линии регрессии close за 24, 72 и 168 часов в ATR(14) и угол atan(подъём / √w) "
          "(у случайного блуждания подъём порядка √w ATR, поэтому 45° — «тренд сильнее шума»). Код — "
          "`features.slope_windows` в `forexmodel/features/extension.py`; расчёт — `scripts/h41_slope.py`, прогоны модели — "
          "`scripts/queues/h41_slope.sh`.", "",
          "**Важно до цифр:** sin угла — монотонная функция наклона, для деревьев CatBoost это тот же признак; cos — его "
          "модуль. Новое здесь только длинные окна (72 и 168 ч) и регрессия вместо разности двух точек: у модели уже есть "
          "смещение за 6, 12 и 24 ч (runup).", "",
          "**Критерий (банк, до расчёта):** шаг 1 — IC с ходом за 10 ч в обоих периодах, в т. ч. сверх runup (частная IC); "
          "шаг 2 — PF модели с наклоном выше той же модели без него в обоих периодах на LKOH и серебре (среднее по 3 seed).", "",
          verdict, "", "## Шаг 1. Связь с ходом за 10 часов", "",
          "IC × 100. cos сравнивается с модулем хода (плоский тренд → маленький ход?). Частная IC — после вычета runup_6/12/24.", "",
          "| рынок | период | признак | IC | t | частная IC | корр. с runup_24 | IC внутри гейта | частная внутри гейта |",
          "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "" if pd.isna(v) else f"{v * 100:+.1f}"  # noqa: E731
    for _, r in T.iterrows():
        md.append(f"| {r['рынок']} | {r['период']} | {r['признак']} | {f(r['IC'])} | {r['t']:+.1f} | {f(r.get('частная IC'))} | "
                  f"{f(r.get('корр. с runup_24'))} | {f(r.get('IC внутри гейта'))} | {f(r.get('частная IC внутри гейта'))} |")
    if len(M):
        A = M.groupby(["рынок", "модель", "период"], sort=False)[["сделок", "итог %", "PF", "просадка"]].mean()
        md += ["", "## Шаг 2. Модель с наклоном против модели без него (среднее по 3 seed)", "",
               "LKOH — настройки H63 (база L10: CatBoost без NN и меты, гейт C, разметка atr_asym, годовой walk-forward "
               "2018–2026, издержки биржи); серебро — configs/silver.yaml без NN и меты, сигнал CatBoost, walk-forward "
               "2019–2026, издержки B01, база пересчитана на тех же настройках.", "",
               "| рынок | модель | период | сделок | итог % | PF | просадка |", "|---|---|---|---|---|---|---|"]
        for (i, m, p), r in A.iterrows():
            md.append(f"| {i} | {m} | {p} | {r['сделок']:.0f} | {r['итог %']:+.1f} | {r['PF']:.2f} | {r['просадка']:.1f} |")
        md += ["", "### По seed", "", "| рынок | модель | seed | период | сделок | итог % | PF |", "|---|---|---|---|---|---|---|"]
        for _, r in M.iterrows():
            md.append(f"| {r['рынок']} | {r['модель']} | {r['seed']} | {r['период']} | {r['сделок']} | {r['итог %']:+.1f} | {r['PF']:.2f} |")
        g = lambda i, m, p: float(A.loc[(i, m, p), "PF"])  # noqa: E731
        try:
            ok = all(g(i, "с наклоном", p) > g(i, "без наклона", p) for i in RUNS for p in ("выбор", "проверка"))
            md += ["", f"Критерий шага 2 выполнен: {'да' if ok else 'нет'}."]
            print("критерий шага 2:", ok)
        except KeyError:
            pass
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
