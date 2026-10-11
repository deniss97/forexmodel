"""Улучшение импульсной стратегии на базе B01 (неделя 12.10): что отличает убыточные сделки и помогают ли
простые правила их отсечь.

Сделки — база B01 (данные страницы журнала: docs/experiments/site/data/exp/B01.json — цены в масштабе
сырых свечей, ATR на входе) и часовые свечи страницы (data/candles/*.json, scripts/build_journal_candles.py).
Издержки — профиль базы configs/costs/base.yaml. Протокол журнала: выбор правила — по сделкам, открытым
до 2021 года; проверка 2021–2026 — один раз для выбранного варианта.

  1. H32, разведка: худший ход против позиции за первые N часов у выигрышных и убыточных сделок.
  2. H32, ранний стоп штатным симулятором: прогоны *_es (scripts/queues/early_stop.sh) — стоп es ATR первые
     eh часов, пока трейлинг не включился; выбор es/eh по PF металлов и портфеля 4 на 2016–2020.
  3. H34: выход из «застывших» сделок — через K часов трейлинг не включился и цена ближе Y ATR от входа
     (быстрый расчёт поверх сделок B01 по закрытию часа K).
  4. H33: результат сделки в зависимости от хода цены за 72 / 120 ч до сигнала (только выбор).

    python scripts/strategy_week_1012.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forexmodel.evaluation.experiments import metrics, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost, rollover_count  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "experiments" / "site" / "data"
WF = ROOT / "reports" / "walk_forward"
OUT = ROOT / "docs" / "results" / "strategy_week_1012"
PROFILE = ROOT / "configs" / "costs" / "base.yaml"
INST = ["silver", "gold", "lkoh", "gazp", "sber", "moex", "mtss", "eth"]
SCOPES = {"портфель 4": ["silver", "lkoh", "gazp", "sber"], "металлы": ["silver", "gold"], "все": INST}
ES_GRID = [(a, h) for a in (2, 3, 4) for h in (12, 24, 48)]
BG, INK, INK2, GRID, BLUE, ORANGE, GRAY = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc", "#2459b3", "#c8641e", "#9aa3ae"
FS = 13


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=FS - 1, length=0, pad=6)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def load_b01():
    e = json.loads((SITE / "exp" / "B01.json").read_text(encoding="utf-8"))
    candles = {}
    for inst in e["trades"]:
        c = json.loads((SITE / "candles" / f"{inst}.json").read_text(encoding="utf-8"))
        t = c["t0"] + np.cumsum(np.array(c["dt"])) * 3600
        candles[inst] = (t, *(np.array(c[k], dtype=float) for k in ("h", "l", "c")))
    return e, candles


def pf(v: pd.Series) -> float:
    loss = -v[v < 0].sum()
    return float(v[v > 0].sum() / loss) if loss > 0 else float("inf")


def mae_table(e, candles) -> pd.DataFrame:
    rows = []
    for inst, trs in e["trades"].items():
        t, H, L, _ = candles[inst]
        for x in trs:
            o, cl, d, op, p, atr = x[0], x[1], x[2], x[3], x[5], x[8]
            if not atr:
                continue
            i0 = int(np.searchsorted(t, o, "right") - 1)
            i1 = int(np.searchsorted(t, cl, "right") - 1)
            r = {"inst": inst, "year": pd.Timestamp(o, unit="s").year, "p": p, "win": p > 0}
            for n in (6, 12, 24, 48):
                j = min(i1, i0 + n)
                r[f"mae{n}"] = ((op - L[i0:j + 1].min()) if d > 0 else (H[i0:j + 1].max() - op)) / atr
            rows.append(r)
    return pd.DataFrame(rows)


def early_stop_grid() -> pd.DataFrame:
    rows = []
    for v in ["impulse_6_3"] + [f"impulse_6_3_es{a}_eh{h}" for a, h in ES_GRID]:
        parts = []
        for inst in INST:
            p = WF / f"{inst}_es__{v}_trades.csv"
            if p.exists():
                t = recost(pd.read_csv(p, parse_dates=["open_dt", "close_dt"]), profile_entry(PROFILE, inst))
                parts.append(standardize(t, inst, v))
        if not parts:
            continue
        T = pd.concat(parts, ignore_index=True)
        for sc, mem in SCOPES.items():
            if not set(mem) <= set(T["instrument"]):
                continue
            for per in ("выбор", "проверка"):
                m = metrics(T, mem, per)
                rows.append({"вариант": v, "срез": sc, "период": per, **{k: m.get(k) for k in ("сделок", "итог", "PF", "без_2_лучших", "t", "лет_в_плюсе")}})
    return pd.DataFrame(rows)


def stagnation(e, candles, K: int, Y: float) -> pd.DataFrame:
    out = []
    for inst, trs in e["trades"].items():
        t, H, L, C = candles[inst]
        pr = profile_entry(PROFILE, inst)
        for x in trs:
            o, cl, d, op, p, atr = x[0], x[1], x[2], x[3], x[5], x[8]
            new = p
            if atr and cl - o > K * 3600:
                i0 = int(np.searchsorted(t, o, "right") - 1)
                ik = int(np.searchsorted(t, o + K * 3600, "right") - 1)
                best = (H[i0:ik + 1].max() - op) if d > 0 else (op - L[i0:ik + 1].min())
                move = d * (C[ik] - op)
                if best < 3 * atr and abs(move) < Y * atr:
                    n = rollover_count(pd.Timestamp(o, unit="s"), pd.Timestamp(t[ik] + 3600, unit="s"), pr["rollover_hour"], pr["triple_weekday"])
                    new = move / op * 100 - pr["spread_pct"] + (pr["swap_long_pct"] if d > 0 else pr["swap_short_pct"]) * n
            out.append((inst, pd.Timestamp(o, unit="s").year, p, new))
    return pd.DataFrame(out, columns=["inst", "year", "base", "new"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    e, candles = load_b01()
    lines = []

    # 1. разведка H32
    M = mae_table(e, candles)
    med = M.groupby("win")[["mae6", "mae12", "mae24", "mae48"]].median().round(2)
    M["bin"] = pd.cut(M["mae24"], [-1, 1, 2, 3, 4, 5, 6, 99], labels=["<1", "1–2", "2–3", "3–4", "4–5", "5–6", "≥6"])
    bins = M.groupby("bin", observed=True).agg(сделок=("p", "size"), средний=("p", "mean"), выигрышей=("win", "mean"))
    lines += ["1. ХУДШИЙ ХОД ПРОТИВ ПОЗИЦИИ (ATR на входе), медиана, все сделки B01:", med.to_string(),
              "\nрезультат сделки по ходу против за первые 24 ч:", bins.round(2).to_string()]
    f, ax = plt.subplots(figsize=(10, 4.6), facecolor=BG)
    style(ax)
    k = np.arange(len(bins))
    ax.bar(k, bins["средний"], 0.62, color=[BLUE if v > 0 else ORANGE for v in bins["средний"]])
    for kk, (v, n) in enumerate(zip(bins["средний"], bins["сделок"])):
        ax.text(kk, v + (0.08 if v >= 0 else -0.08), f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top", color=INK, fontsize=FS - 1)
    ax.set_xticks(k)
    ax.set_xticklabels([f"{b} ATR\n{n} сд." for b, n in zip(bins.index, bins["сделок"])], fontsize=FS - 1, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("средний результат сделки, %", color=INK2, fontsize=FS)
    ax.set_xlabel("насколько цена ушла против позиции за первые 24 часа", color=INK2, fontsize=FS)
    ax.grid(axis="x", visible=False)
    f.tight_layout()
    f.savefig(OUT / "mae24_bins.png", dpi=140, facecolor=BG)
    plt.close(f)

    # 2. ранний стоп
    G = early_stop_grid()
    G.to_csv(OUT / "early_stop_grid.csv", index=False)
    lab = lambda v: "база B01" if v == "impulse_6_3" else v.replace("impulse_6_3_", "").replace("es", "стоп ").replace("_eh", " ATR / ") + " ч"  # noqa: E731
    G["вариант"] = G["вариант"].map(lab)
    for sc in SCOPES:
        x = G[G["срез"] == sc]
        if len(x):
            lines.append(f"\n2. РАННИЙ СТОП · {sc}:\n" + x.pivot_table(index="вариант", columns="период", values=["итог", "PF", "без_2_лучших"], sort=False).round(2).to_string())
    f, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
    for ax, per in zip(axes, ("выбор", "проверка")):
        x = G[(G["срез"] == "металлы") & (G["период"] == per)]
        base = float(x[x["вариант"] == "база B01"]["PF"].iloc[0]) if len(x) else np.nan
        grid = np.full((3, 3), np.nan)
        for (a, h), v in zip(ES_GRID, [f"impulse_6_3_es{a}_eh{h}" for a, h in ES_GRID]):
            y = x[x["вариант"] == lab(v)]
            if len(y):
                grid[(2, 3, 4).index(a), (12, 24, 48).index(h)] = float(y["PF"].iloc[0])
        im = ax.imshow(grid - base, cmap="RdBu", vmin=-0.25, vmax=0.25, aspect="auto")
        for (r, c), v in np.ndenumerate(grid):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=FS, color=INK, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["12 ч", "24 ч", "48 ч"], fontsize=FS - 1, color=INK2)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["2 ATR", "3 ATR", "4 ATR"], fontsize=FS - 1, color=INK2)
        ax.set_title(f"{'2016–2020' if per == 'выбор' else '2021–2026'}: PF металлов (база {base:.2f})", fontsize=FS, color=INK, loc="left")
        ax.set_xlabel("сколько часов действует ранний стоп", fontsize=FS - 1, color=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("ранний стоп", fontsize=FS - 1, color=INK2)
    f.tight_layout()
    f.savefig(OUT / "early_stop_pf.png", dpi=140, facecolor=BG)
    plt.close(f)

    # 3. H34: выход из «застывших» сделок
    rows = []
    for K in (24, 48, 72, 96):
        for Y in (0.5, 1.0, 1.5):
            D = stagnation(e, candles, K, Y)
            for per, q in (("выбор", D["year"] < 2021), ("проверка", D["year"] >= 2021)):
                for sc in ("портфель 4", "металлы"):
                    s = D[q & D["inst"].isin(SCOPES[sc])]
                    w = 1 / len(SCOPES[sc])
                    rows.append({"K": K, "Y": Y, "период": per, "срез": sc, "база": s["base"].sum() * w, "правило": s["new"].sum() * w,
                                 "PF база": pf(s["base"]), "PF правило": pf(s["new"])})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "stagnation_exit.csv", index=False)
    lines.append("\n3. ВЫХОД ИЗ «ЗАСТЫВШИХ» СДЕЛОК (выбор):\n" + S[S["период"] == "выбор"].drop(columns="период").round(2).to_string(index=False))
    lines.append("   проверка для выбранного 72 ч / 1 ATR:\n" + S[(S["период"] == "проверка") & (S["K"] == 72) & (S["Y"] == 1.0)].round(2).to_string(index=False))

    # 4. H33: ход до сигнала
    rows = []
    for inst, trs in e["trades"].items():
        t, _, _, C = candles[inst]
        for x in trs:
            if not x[8]:
                continue
            i = int(np.searchsorted(t, x[0], "right") - 1) - 1
            rows.append({"year": pd.Timestamp(x[0], unit="s").year, "p": x[5],
                         **{f"m{n}": x[2] * (C[i] - C[max(0, i - n)]) / x[8] for n in (72, 120)}})
    P = pd.DataFrame(rows)
    P = P[P["year"] < 2021]
    for col in ("m72", "m120"):
        b = pd.cut(P[col], [-99, -6, -3, 0, 3, 6, 10, 99])
        lines.append(f"\n4. ХОД ЗА {col[1:]} Ч ДО СИГНАЛА В СТОРОНУ СДЕЛКИ (выбор 2016–2020):\n"
                     + P.groupby(b, observed=True)["p"].agg(["size", "mean"]).round(2).to_string())
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
