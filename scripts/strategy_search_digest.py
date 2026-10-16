"""Промежуточные результаты перебора стратегий (H50) для разбора — задача T21.

reports/strategy_search/trials.sqlite (2.6 ГБ, 1.12 млн проб) хранит каждую пробу: параметры и по каждому
срезу и году [итог %, прибыль, убыток, сделок]. Скрипт:

  flatten — раскладывает базу в таблицу reports/strategy_search/trials_flat.parquet: блоки стратегии и по
            каждому срезу итог / PF / сделок на выборе (2016–2020) и проверке (2021–2026) и итог по годам;
  digest  — сводка по блокам (что зашло, что нет) docs/results/strategy_search/blocks.md и выборка лучших
            проб с цифрами проверки docs/results/strategy_search/top_trials.csv.

Статистика по блокам — только по случайным пробам («разведка»): мутации лучших смещены к хорошим.

    python scripts/strategy_search_digest.py flatten    # ~10 мин
    python scripts/strategy_search_digest.py digest
"""
from __future__ import annotations

import json
import sqlite3
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from strategy_search import DB, OUT, ROOT, SEL_END, YEARS

FLAT = OUT / "trials_flat.parquet"
DOCS = ROOT / "docs" / "results" / "strategy_search"
SLICES = {"металлы": "met", "портфель 4": "p4", "все": "all"}
MIN_TRADES = {"met": 140, "p4": 200, "all": 360}  # 30% сделок B01 на выборе, как в цели поиска


def family(entry: str) -> tuple[str, str]:
    """Семейство входа и таймфрейм: 'macd:12:26:9:sig @4h' → ('macd', '4h'); уровни — как есть, 'дни'."""
    if " @" in entry:
        code, tf = entry.rsplit(" @", 1)
        return code.split(":")[0], {"1h": "1 ч", "4h": "4 ч", "24h": "дни"}.get(tf, tf)
    return entry.split(" + ")[0].replace(" → пробой", ""), "дни"


def flatten() -> int:
    # 1.12 млн проб: числа — в заранее выделенные массивы float32, названия блоков — в коды (машине хватает 7 ГБ)
    con = sqlite3.connect(DB)
    total = con.execute("select count(*) from trials").fetchone()[0]
    names = ("entry", "family", "tf", "f1", "f2", "exit", "atr", "size", "how")
    codes = {k: np.zeros(total, np.int16) for k in names}
    vocab: dict[str, dict] = {k: {} for k in names}
    nums = [f"{c}_{part}_{m}" for c in SLICES.values() for part in ("sel", "test") for m in ("pf", "net", "n")]
    nums += [f"{c}_{y}" for c in SLICES.values() for y in YEARS]
    num = {k: np.full(total, np.nan, np.float32) for k in nums}
    for n, (params, res, how) in enumerate(con.execute("select params, res, how from trials")):
        p, r = json.loads(params), json.loads(res)
        fam, tf = family(p["entry"])
        for k, v in zip(names, (p["entry"], fam, tf, p["f1"], p["f2"], p["exit"], str(p["atr"]), p["size"], how)):
            codes[k][n] = vocab[k].setdefault(v, len(vocab[k]))
        for s, c in SLICES.items():
            agg = r.get(s) or {}
            for part, sel in (("sel", True), ("test", False)):
                ys = [y for y in agg if (int(y) < SEL_END) == sel]
                pos, neg = sum(agg[y][1] for y in ys), sum(agg[y][2] for y in ys)
                num[f"{c}_{part}_pf"][n] = pos / neg if neg > 0 else np.nan
                num[f"{c}_{part}_net"][n] = sum(agg[y][0] for y in ys)
                num[f"{c}_{part}_n"][n] = sum(agg[y][3] for y in ys)
            for y in YEARS:
                if str(y) in agg:
                    num[f"{c}_{y}"][n] = agg[str(y)][0]
        if (n + 1) % 100_000 == 0:
            print(f"{n + 1} проб", flush=True)
    con.close()
    df = pd.DataFrame(num)
    for k in names:
        cats = sorted(vocab[k], key=vocab[k].get)
        df.insert(len(df.columns) if k == "how" else list(names).index(k), k, pd.Categorical.from_codes(codes[k], cats))
    df["atr"] = df["atr"].astype(int)
    df.to_parquet(FLAT, index=False)
    print(f"{len(df)} проб → {FLAT.relative_to(ROOT)}")
    return 0


def block_table(df: pd.DataFrame, by: str, c: str, top: int | None = None) -> list[str]:
    ok = df[df[f"{c}_sel_n"] >= MIN_TRADES[c]]
    g = ok.groupby(by, observed=True)
    t = pd.DataFrame({
        "проб": g.size(),
        "PF выбор, медиана": g[f"{c}_sel_pf"].median(),
        "PF > 1 на выборе": g[f"{c}_sel_pf"].apply(lambda v: (v > 1).mean()),
        "PF проверка, медиана": g[f"{c}_test_pf"].median(),
        "PF > 1 на проверке": g[f"{c}_test_pf"].apply(lambda v: (v > 1).mean()),
        "связь выбор → проверка": g.apply(lambda x: x[f"{c}_sel_pf"].corr(x[f"{c}_test_pf"], method="spearman")),
    }).sort_values("PF проверка, медиана", ascending=False)
    if top:
        t = t.head(top)
    out = [f"| {by} | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    for k, r in t.iterrows():
        out.append(f"| {k} | {int(r['проб'])} | {r.iloc[1]:.2f} | {r.iloc[2]:.0%} | {r.iloc[3]:.2f} | {r.iloc[4]:.0%} | {r.iloc[5]:.2f} |")
    return out


def digest() -> int:
    df = pd.read_parquet(FLAT)
    rnd = df[df["how"] == "разведка"]
    md = ["# Перебор стратегий: что зашло, что нет", "",
          f"Сводка по блокам {len(df):,} проб перебора H50 (`scripts/strategy_search_digest.py`). Для честного сравнения "
          f"блоков взяты только случайные пробы — {len(rnd):,} («разведка»; мутации лучших смещены к хорошим). "
          "Пробы, у которых на выборе меньше 30% сделок базы B01, не учитываются. «Связь выбор → проверка» — "
          "ранговая корреляция PF внутри блока: около нуля — хороший PF на выборе ничего не говорит о проверке.".replace(",", " "), "",
          "Издержки B01 (CFD в обе стороны для всех рынков). Цифры — быстрый часовой движок, сверенный с минутным "
          "симулятором (число сделок ±0.5%).", ""]
    for s, c in SLICES.items():
        md += [f"## Срез «{s}»", "", "### Семейство входа", ""] + block_table(rnd, "family", c) + [""]
        md += ["### Таймфрейм входа", ""] + block_table(rnd, "tf", c) + [""]
        md += ["### Выход — 12 лучших на проверке", ""] + block_table(rnd, "exit", c, 12) + [""]
        md += ["### Первый фильтр", ""] + block_table(rnd, "f1", c) + [""]
        md += ["### Размер позиции", ""] + block_table(rnd, "size", c) + [""]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "blocks.md").write_text("\n".join(md), encoding="utf-8")

    # лучшие пробы по каждому срезу: как их видел поиск (выбор) и что было дальше (проверка)
    keep = ["entry", "f1", "f2", "exit", "atr", "size", "how"]
    tops = []
    for s, c in SLICES.items():
        ok = df[df[f"{c}_sel_n"] >= MIN_TRADES[c]].nlargest(300, f"{c}_sel_pf")
        x = ok[keep + [f"{c}_{p}_{m}" for p in ("sel", "test") for m in ("pf", "net", "n")]].copy()
        x.columns = keep + ["PF выбор", "итог выбор %", "сделок выбор", "PF проверка", "итог проверка %", "сделок проверка"]
        x.insert(0, "срез", s)
        tops.append(x)
    pd.concat(tops).round(3).to_csv(DOCS / "top_trials.csv", index=False)
    print(f"→ {(DOCS / 'blocks.md').relative_to(ROOT)}, {(DOCS / 'top_trials.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"flatten": flatten, "digest": digest}.get(cmd, lambda: print(__doc__) or 1)())
