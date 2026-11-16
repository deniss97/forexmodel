"""H74, продолжение: разметка фигур кодом Тимура (docs/timur/2026-11-15/chart_patterns.py, прислан 15.11) против
моей разметки (scripts/h74_patterns.py) на тех же данных и по тому же протоколу.

Данные — дневные свечи 19 акций из часового кеша (как в h74_patterns.py), 2015–2026. Код автора — как есть, настройки
по умолчанию (Config(): ZigZag 1.5 / 2.5 / 4 / 6 ATR, min_score 0.55, tol 1.0).

Что считается:
  1. сколько фигур каждого типа находит код автора, по годам и в половинах «выбор» (до 2021) / «проверка» (с 2021);
  2. совпадение с моей разметкой: та же бумага, тот же тип, пробой в пределах ±5 торговых дней;
  3. заглядывание вперёд: пробой раньше, чем подтверждены все экстремумы фигуры (known_at) — в торговле такой пробой
     увидеть нельзя; честный вход — max(пробой, known_at);
  4. ход цены после честного входа в ATR часа против обычного часа (как в моём H74), со знаком направления фигуры;
  5. исходы «цель / стоп» функцией автора evaluate (без комиссий).
    python scripts/h74_timur.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "docs" / "timur" / "2026-11-15"))
import chart_patterns as cp  # noqa: E402
import h74_patterns as mine  # noqa: E402

OUT = ROOT / "docs" / "results" / "h74_timur"
OUT.mkdir(parents=True, exist_ok=True)
REP = ROOT / "reports" / "h74"
SPLIT = mine.SPLIT
# мои фигуры → имена автора (у меня только бычьи версии; bump-and-run у автора нет)
MAP = {"db": "double_bottom", "tb": "triple_bottom", "ihs": "inv_head_shoulders", "htf": "high_tight_flag",
       "tri": "asc_triangle"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#888888"


def md(df: pd.DataFrame, index: bool = False) -> str:
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                              (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    hours, rows = {}, []
    for t in mine.TICKERS:
        h = mine.hourly(t)
        hours[t] = h
        d = mine.to_daily(h)
        pats = cp.detect_patterns(d, cp.Config())
        stats = cp.evaluate(d, pats, cp.Config())  # заполняет outcome / ret у подтверждённых
        del stats
        idx = d.index
        for p in pats:
            bo = p.breakout_idx
            ka = p.known_at
            sig = None if bo is None else (max(bo, ka) if ka is not None else None)
            rows.append({"ticker": t, "pattern": p.name, "ru": p.name_ru, "dir": p.direction, "score": p.score,
                         "scale": p.scale, "start": idx[p.start_idx], "end": idx[p.end_idx],
                         "known_at": idx[ka] if ka is not None else pd.NaT, "status": p.status,
                         "breakout": idx[bo] if bo is not None else pd.NaT,
                         "peek": bo is not None and (ka is None or bo < ka),
                         "peek_bars": (ka - bo) if (bo is not None and ka is not None and bo < ka) else 0,
                         "signal": idx[sig] if sig is not None else pd.NaT,
                         "outcome": p.outcome, "ret": p.ret})
        print(t, len(pats), flush=True)
    a = pd.DataFrame(rows)
    a["year"] = a["start"].dt.year
    a["part"] = np.where(a["breakout"].fillna(a["end"]) < SPLIT, "выбор", "проверка")
    a.to_csv(OUT / "timur_patterns.csv", index=False)
    lines = ["## Разметка кодом автора", ""]

    # 1. сколько и каких
    cnt = a.pivot_table(index="ru", columns="part", values="ticker", aggfunc="size", fill_value=0)
    conf = a[a["status"] == "confirmed"].pivot_table(index="ru", columns="part", values="ticker", aggfunc="size",
                                                     fill_value=0).add_prefix("с пробоем, ")
    peek = a[a["status"] == "confirmed"].groupby("ru")["peek"].mean().mul(100).round(0).rename("пробой раньше known_at, %")
    t1 = cnt.join(conf).join(peek).fillna(0)
    t1.to_csv(OUT / "counts.csv")
    lines += [f"Всего фигур: {len(a)} на {a['ticker'].nunique()} бумагах; с пробоем — {(a['status'] == 'confirmed').sum()}.",
              "", md(t1, True), ""]

    # 2. совпадение с моей разметкой
    my = pd.read_csv(REP / "events.csv", parse_dates=["date"])
    my = my[my["pat"].isin(MAP)]
    cf = a[a["status"] == "confirmed"]
    mrows = []
    for code, name in MAP.items():
        m = my[my["pat"] == code]
        t_ = cf[cf["pattern"] == name]
        hit_m = 0
        for e in m.itertuples():
            tt = t_[t_["ticker"] == e.ticker]["breakout"]
            if len(tt) and (tt - e.date).abs().min() <= pd.Timedelta(days=7):
                hit_m += 1
        hit_t = 0
        for e in t_.itertuples():
            mm = m[m["ticker"] == e.ticker]["date"]
            if len(mm) and (mm - e.breakout).abs().min() <= pd.Timedelta(days=7):
                hit_t += 1
        mrows.append({"фигура": name, "моих": len(m), "у автора (с пробоем)": len(t_),
                      "моих нашёл автор, %": round(100 * hit_m / max(len(m), 1)),
                      "его нашёл я, %": round(100 * hit_t / max(len(t_), 1))})
    t2 = pd.DataFrame(mrows)
    t2.to_csv(OUT / "overlap.csv", index=False)
    lines += ["Совпадение с моей разметкой (та же бумага и тип, пробой ±7 календарных дней):", "", md(t2), ""]

    # 3–4. ход после честного входа
    ev = cf.dropna(subset=["signal"]).rename(columns={"pattern": "pat"}).copy()
    ev["date"] = ev["signal"]
    ev["part"] = np.where(ev["date"] < SPLIT, "выбор", "проверка")
    ev["year"] = ev["date"].dt.year
    fw = mine.forward(ev[["ticker", "pat", "date", "part", "year"]], hours)
    fw = fw.merge(ev[["ticker", "pat", "date", "dir"]], on=["ticker", "pat", "date"], how="left")
    for q in (6, 24, 72):
        fw[f"h{q}"] = fw[f"h{q}"] * np.where(fw["dir"] == 0, 1, fw["dir"])
    base = mine.baseline(hours).groupby(["part", "h"]).apply(lambda g: np.average(g["mean"], weights=g["n"]))
    fw["half"] = np.where(fw["date"] < SPLIT, "выбор 2015–2020",
                          np.where(fw["date"].dt.year <= 2023, "проверка 2021–2023", "проверка 2024–2026"))
    fw.to_csv(OUT / "forward.csv", index=False)
    rows = []
    for (pat, half), g in fw.groupby(["pat", "half"]):
        part = "выбор" if half.startswith("выбор") else "проверка"
        sgn = int(np.sign(g["dir"].iloc[0])) or 1
        r = {"фигура": cp.NAMES_RU.get(pat, pat), "период": half, "n": len(g)}
        for q in (24, 72):
            b = sgn * base[(part, q)]
            r[f"ход {q} ч, ATR"] = round(g[f"h{q}"].mean(), 2)
            r[f"обычный час {q} ч"] = round(b, 2)
            r[f"t {q} ч"] = round((g[f"h{q}"].mean() - b) / (g[f"h{q}"].std(ddof=1) / np.sqrt(len(g))), 1) \
                if len(g) > 2 else np.nan
        rows.append(r)
    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "forward_summary.csv", index=False)
    lines += ["Ход цены после честного входа (max(пробой, known_at), первый час следующего дня) в сторону фигуры, ATR "
              "часа, против обычного часа (для медвежьих фигур — со знаком минус):", "", md(t3), ""]

    # 5. цель / стоп автора
    ev2 = cf.dropna(subset=["outcome"])
    rows = []
    for (ru, part), g in ev2.groupby(["ru", "part"]):
        rows.append({"фигура": ru, "период": part, "сделок": len(g),
                     "цель, %": round(100 * (g["outcome"] == "target").mean()),
                     "стоп, %": round(100 * (g["outcome"] == "stop").mean()),
                     "средняя, %": round(100 * g["ret"].mean(), 2),
                     "с заглядыванием, %": round(100 * g["peek"].mean())})
    t4 = pd.DataFrame(rows)
    t4.to_csv(OUT / "evaluate.csv", index=False)
    lines += ["Исходы «цель или стоп» функцией автора (вход на закрытии пробоя, без комиссий):", "", md(t4), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    charts(a, t3)
    page(hours)


def charts(a: pd.DataFrame, t3: pd.DataFrame) -> None:
    # число фигур по годам у автора
    y = a[a["status"] == "confirmed"].groupby(["year", "ru"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    y.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_title("H74: фигуры с пробоем по годам — код Тимура, 19 акций, дневные свечи")
    ax.set_xlabel("")
    ax.set_ylabel("фигур")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "counts_years.png", dpi=120)
    plt.close(fig)
    # ход за 72 ч по фигурам и периодам
    t3 = t3[t3.groupby("фигура")["n"].transform("min") >= 5]  # фигуры, где случаев меньше 5, не рисуем
    halves = ["выбор 2015–2020", "проверка 2021–2023", "проверка 2024–2026"]
    figs = sorted(t3["фигура"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, h in zip(axes, halves):
        g = t3[t3["период"] == h].set_index("фигура").reindex(figs)
        x = np.arange(len(figs))
        ax.bar(x - 0.2, g["ход 72 ч, ATR"], 0.4, color=C1, label="после фигуры")
        ax.bar(x + 0.2, g["обычный час 72 ч"], 0.4, color=C3, label="обычный час")
        for i, n in enumerate(g["n"]):
            if np.isfinite(n):
                ax.text(i, 0, f"{int(n)}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(figs, rotation=60, ha="right", fontsize=8)
        ax.set_title(h, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("ход в сторону фигуры за 72 ч, ATR часа")
    axes[0].legend(fontsize=8)
    fig.suptitle("H74, код Тимура: ход цены после честного входа (пробой и все экстремумы подтверждены)")
    fig.tight_layout()
    fig.savefig(OUT / "forward72.png", dpi=120)
    plt.close(fig)


def page(hours: dict) -> None:
    """Интерактивная разметка автора на LKOH (последние 3 года) — для глаза."""
    d = mine.to_daily(hours["LKOH"])
    pats = cp.detect_patterns(d, cp.Config())
    cp.evaluate(d, pats, cp.Config())
    cp.plot_patterns(d, pats, str(OUT / "lkoh_timur.html"), last=750, no_gaps=True, plotlyjs=True,
                     title="LKOH, дневные свечи — разметка фигур кодом Тимура (наведите на фигуру — оценки условий)")


if __name__ == "__main__":
    main()
