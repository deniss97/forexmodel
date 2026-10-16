"""Честный поиск второй стратегии для портфеля с набором для запуска (гипотеза H54).

Тот же перебор блоков ТЗ, что H50 (scripts/strategy_search.py: 128 входов, до двух фильтров, 36 выходов, период
ATR, размер), но на наборе для запуска и с новой целью:
  * рынки — серебро и золото через CFD в обе стороны (издержки B01), GAZP, SBER, LKOH на бирже только в лонг
    (издержки configs/costs/exchange.yaml; шорты гасятся до расчёта сделок и не занимают время);
  * отбор — только по выбору 2016–2020: не меньше 50 сделок (10 в год на набор), корреляция месячных результатов
    с S42 ниже 0.3, цель — PF; устойчивость — медиана PF пробы и 6 соседей;
  * финалисты — лучшие по приросту t месяцев портфеля «S42 + стратегия» (капитал пополам) на выборе;
  * честная оценка самого поиска — walk-forward: на год Y берётся лучшая по годам до Y, торгует год Y;
  * проверка 2021–2026 — один раз, для финалистов; они же пересчитываются штатным минутным симулятором.

    python scripts/second_search.py search --hours 6     # 3 процесса
    python scripts/second_search.py report               # docs/results/second_strategy.md

База — reports/second_search/trials.sqlite; помесячный результат каждой пробы хранится (float32), чтобы
считать связь с импульсом и walk-forward поиска по всем пробам.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import strategy_search as ss

ROOT = ss.ROOT
OUT = ROOT / "reports" / "second_search"
DB = OUT / "trials.sqlite"
DOC = ROOT / "docs" / "results" / "second_strategy.md"
SET = ["silver", "gold", "gazp", "sber", "lkoh"]
STOCKS = {"gazp", "sber", "lkoh"}
MONTHS = pd.period_range("2016-01", "2026-06", freq="M")
SEL = MONTHS < pd.Period("2021-01", "M")
MIN_SEL_TRADES = 50          # 10 сделок в год на набор за 5 лет выбора (T19)
MAX_CORR = 0.3               # связь месячных результатов с S42 на выборе
ROBUST_MIN = 1.2
MARK = ("<!-- вывод -->", "<!-- /вывод -->")

_entries = ss.entries_for


def entries_launch(ins, prm):
    """Входы как в переборе H50, но по акциям только лонг (шорты гасятся до цепочки сделок)."""
    i, s = _entries(ins, prm)
    if ins.inst in STOCKS:
        keep = s > 0
        i, s = i[keep], s[keep]
    return i, s


ss.entries_for = entries_launch      # часовой движок и минутный симулятор берут входы отсюда


def load_data() -> dict:
    from forexmodel.simulation.costs import profile_entry

    data = {i: ss.Inst(i) for i in SET}
    for i in STOCKS:
        e = profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", i)
        data[i].spread, data[i].swap = e["spread_pct"], {1: e["swap_long_pct"], -1: e["swap_short_pct"]}
    return data


def s42_monthly() -> np.ndarray:
    from experiments import load_catalog, load_trades

    _, exps = load_catalog()
    t = load_trades(exps["S42"])
    m = (t["profit_pct"] / len(SET)).groupby(t["close_dt"].dt.to_period("M")).sum()
    return m.reindex(MONTHS, fill_value=0.0).to_numpy(np.float64)


def evaluate(data: dict, prm: dict) -> dict:
    """Помесячный итог набора (рынки поровну) и по периодам: прибыль, убыток, сделок; по годам — то же."""
    mon = np.zeros(len(MONTHS))
    years: dict[int, list] = {}
    for inst in SET:
        ins = data[inst]
        tr = ss.trades(ins, prm)
        if not len(tr["net"]):
            continue
        p = tr["net"] / len(SET)
        xm = pd.PeriodIndex(pd.to_datetime(ins.t[tr["x"]]), freq="M")
        idx = MONTHS.get_indexer(xm)
        ok = idx >= 0
        np.add.at(mon, idx[ok], p[ok])
        for y, v in zip(tr["year"], p):
            a = years.setdefault(int(y), [0.0, 0.0, 0.0, 0])
            a[0] += v; a[1] += max(v, 0); a[2] += max(-v, 0); a[3] += 1
    return {"m": mon, "years": years}


def part(years: dict, ys) -> tuple[float, float, int, float]:
    pos = sum(years[y][1] for y in ys if y in years)
    neg = sum(years[y][2] for y in ys if y in years)
    n = sum(years[y][3] for y in ys if y in years)
    return (pos / neg if neg > 0 else (np.inf if pos > 0 else 0.0)), sum(years[y][0] for y in ys if y in years), n, neg


def tstat(v: np.ndarray) -> float:
    sd = v.std(ddof=1)
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else np.nan


def judge(r: dict, ref: np.ndarray) -> dict:
    """Оценки пробы: цель на выборе (PF или 0), связь с S42, t портфеля на выборе и (для отчёта) на проверке."""
    pf_s, _, n_s, _ = part(r["years"], range(2016, 2021))
    pf_t, net_t, n_t, _ = part(r["years"], range(2021, 2027))
    m = r["m"]
    corr = float(np.corrcoef(m[SEL], ref[SEL])[0, 1]) if m[SEL].std() > 0 else np.nan
    ok = n_s >= MIN_SEL_TRADES and np.isfinite(corr) and corr < MAX_CORR
    return {"score": float(min(pf_s, 10.0)) if ok else 0.0, "sel_pf": float(min(pf_s, 99)), "sel_n": n_s, "corr": corr,
            "pt_sel": tstat(0.5 * ref[SEL] + 0.5 * m[SEL]), "test_pf": float(min(pf_t, 99)), "test_n": n_t, "test_net": net_t,
            "pt_test": tstat(0.5 * ref[~SEL] + 0.5 * m[~SEL])}


def db():
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS trials (key TEXT PRIMARY KEY, score REAL, robust REAL, sel_pf REAL, sel_n INT,
                   corr REAL, pt_sel REAL, test_pf REAL, test_n INT, test_net REAL, pt_test REAL,
                   params TEXT, years TEXT, m BLOB, how TEXT, t TEXT)""")
    for c in ("score", "robust"):
        con.execute(f"CREATE INDEX IF NOT EXISTS i_{c} ON trials ({c})")
    return con


def worker(args) -> int:
    wid, hours, seed = args
    logging.disable(logging.WARNING)
    rng = random.Random(seed * 1000 + wid + int(time.time()))
    data, sp, con, ref = load_data(), ss.space(), db(), s42_monthly()
    pending, robust_upd = [], []

    def flush():
        for _ in range(30):
            try:
                con.executemany("INSERT OR IGNORE INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", pending)
                con.executemany("UPDATE trials SET robust = ? WHERE key = ?", robust_upd)
                con.commit()
                break
            except sqlite3.OperationalError:
                time.sleep(1 + rng.random())
        pending.clear()
        robust_upd.clear()

    t_end, n, cut, best = time.time() + hours * 3600, 0, 0.0, []
    while time.time() < t_end:
        if n % 100 == 0:
            flush()
            tops = con.execute("SELECT score FROM trials ORDER BY score DESC LIMIT 100").fetchall()
            cut = tops[-1][0] if len(tops) >= 100 else 0.0
            best = [json.loads(p) for (p,) in con.execute("SELECT params FROM trials WHERE robust IS NOT NULL ORDER BY robust DESC LIMIT 30")]
        if best and rng.random() > max(0.3, 1 - n / 5000):
            prm, how = ss.mutate(rng, sp, rng.choice(best)), "мутация"
        else:
            prm, how = ss.sample(rng, sp), "разведка"
        k = ss.key(prm)
        n += 1
        if con.execute("SELECT 1 FROM trials WHERE key=?", (k,)).fetchone() or any(p[0] == k for p in pending):
            continue
        r = evaluate(data, prm)
        j = judge(r, ref)
        rb = None
        if j["score"] > cut and j["score"] > 0:
            vals = [j["score"]] + [judge(evaluate(data, q), ref)["score"] for q in ss.neighbours(rng, sp, prm, 6)]
            rb = float(np.median(vals))
        pending.append((k, j["score"], rb, j["sel_pf"], j["sel_n"], j["corr"], j["pt_sel"], j["test_pf"], j["test_n"],
                        j["test_net"], j["pt_test"], json.dumps(prm, ensure_ascii=False),
                        json.dumps({str(y): [round(v, 4) for v in a[:3]] + [a[3]] for y, a in r["years"].items()}),
                        r["m"].astype(np.float32).tobytes(), how, time.strftime("%m-%d %H:%M")))
    flush()
    con.close()
    return n


def search(hours: float, seed: int, procs: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    db().close()
    with ProcessPoolExecutor(procs) as ex:
        counts = list(ex.map(worker, [(w, hours, seed) for w in range(procs)]))
    con = db()
    print(f"проб за запуск: {sum(counts)}, всего: {con.execute('SELECT COUNT(*) FROM trials').fetchone()[0]}", flush=True)
    con.close()
    return 0


# ---------------------------------------------------------------- отчёт

def load_all(con) -> tuple[pd.DataFrame, np.ndarray, list]:
    rows = con.execute("SELECT params, years, m, score, robust, sel_pf, sel_n, corr, pt_sel, test_pf, test_n, test_net, pt_test, how "
                       "FROM trials").fetchall()
    df = pd.DataFrame(rows, columns=["params", "years", "m", "score", "robust", "sel_pf", "sel_n", "corr", "pt_sel", "test_pf",
                                     "test_n", "test_net", "pt_test", "how"])
    M = np.frombuffer(b"".join(df["m"]), dtype=np.float32).reshape(len(df), len(MONTHS))
    years = [json.loads(y) for y in df["years"]]
    return df.drop(columns="m"), M, years


def walk_forward(df: pd.DataFrame, M: np.ndarray, years: list, ref: np.ndarray) -> pd.DataFrame:
    """На год Y — лучшая по PF за годы до Y среди прошедших отбор (сделок ≥ 10 в год, связь с S42 < 0.3 по месяцам
    до Y); торгует год Y. Честная оценка того, что дал бы сам поиск."""
    out = []
    for Y in range(2021, 2027):
        ys = [str(y) for y in range(2016, Y)]
        pos = np.array([sum(y.get(k, [0, 0, 0, 0])[1] for k in ys) for y in years])
        neg = np.array([sum(y.get(k, [0, 0, 0, 0])[2] for k in ys) for y in years])
        n = np.array([sum(y.get(k, [0, 0, 0, 0])[3] for k in ys) for y in years])
        before = MONTHS < pd.Period(f"{Y}-01", "M")
        Mb, rb = M[:, before], ref[before]
        Mc, rc = Mb - Mb.mean(axis=1, keepdims=True), rb - rb.mean()
        corr = (Mc @ rc) / (np.sqrt((Mc ** 2).sum(axis=1)) * np.sqrt((rc ** 2).sum()) + 1e-12)
        pf = np.where(neg > 0, pos / np.maximum(neg, 1e-12), 0)
        ok = (n >= 10 * len(ys)) & (corr < MAX_CORR)
        if not ok.any():
            continue
        k = int(np.argmax(np.where(ok, pf, -1)))
        yr = MONTHS.year == Y
        out.append({"год": Y, "стратегия": ss.describe(json.loads(df["params"].iat[k])), "PF до года": float(pf[k]),
                    "итог года": float(M[k, yr].sum()), "S42 за год": float(ref[yr].sum()),
                    "портфель пополам": float(0.5 * M[k, yr].sum() + 0.5 * ref[yr].sum())})
    return pd.DataFrame(out)


def finalists_minute(picks: list[dict]) -> pd.DataFrame:
    """Штатный минутный симулятор для финалистов: те же входы (акции — только лонг), издержки набора."""
    from forexmodel.simulation.costs import profile_entry, recost

    data = load_data()
    rows = []
    for inst in SET:
        minute = ss.load_minute(inst)
        prof = "exchange" if inst in STOCKS else "base"
        for k, prm in enumerate(picks):
            mt = ss.minute_trades(inst, prm, data[inst], minute)
            if len(mt):
                mt = recost(mt, profile_entry(ROOT / "configs" / "costs" / f"{prof}.yaml", inst))
                rows.append(pd.DataFrame({"k": k, "instrument": inst, "open_dt": pd.to_datetime(mt["open_dt"]),
                                          "close_dt": pd.to_datetime(mt["close_dt"]), "profit_pct": mt["profit_pct"]}))
        print(f"[{inst}] минутный симулятор: финалисты посчитаны", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def report(top_n: int) -> int:
    from forexmodel.evaluation import portfolio as pf
    from experiments import load_catalog, load_trades

    con = db()
    df, M, years = load_all(con)
    con.close()
    ref = s42_monthly()
    passed = df[(df["score"] > 0) & (df["robust"] >= ROBUST_MIN)].sort_values("pt_sel", ascending=False)
    picks = [json.loads(p) for p in passed["params"].head(top_n)]
    wf = walk_forward(df, M, years, ref)
    mt = finalists_minute(picks) if picks else pd.DataFrame()
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    alone = pf.combine({"S42": s42}, SET)
    rnd = df[df["how"] == "разведка"]
    lines = []
    for k, prm in enumerate(picks):
        x = mt[mt["k"] == k].assign(run="m")
        r = passed.iloc[k]
        both = pf.combine({"S42": s42, "C": x}, SET)
        for per in ("выбор", "проверка"):
            s, p = pf.metrics(pf.combine({"C": x}, SET), per), pf.metrics(both, per)
            lines.append({"№": k + 1, "стратегия": ss.describe(prm), "период": per, "сделок в год": s.get("в_год", 0),
                          "итог": s.get("итог", 0), "PF": s.get("PF", np.nan), "без 2 лучших": s.get("без_2_лучших", np.nan),
                          "t мес": s.get("t_мес", np.nan), "с S42: t мес": p.get("t_мес", np.nan),
                          "с S42: итог/просадка": p.get("итог_к_просадке", np.nan), "S42: t мес": pf.metrics(alone, per)["t_мес"],
                          "S42: итог/просадка": pf.metrics(alone, per)["итог_к_просадке"], "корр. месяцев (выбор)": r["corr"],
                          "устойчивая оценка": r["robust"]})
    L = pd.DataFrame(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_csv(OUT / "finalists.csv", index=False)
    wf.to_csv(OUT / "walk_forward.csv", index=False)

    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    fmt = lambda v, d=2: "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"  # noqa: E731
    md = ["# Честный поиск второй стратегии (H54)", "",
          f"Перебор блоков ТЗ (как H50) на наборе для запуска: серебро и золото через CFD в обе стороны, GAZP, SBER, LKOH на "
          f"бирже только в лонг. Отбор — только по 2016–2020: не меньше {MIN_SEL_TRADES} сделок за выбор, корреляция месяцев с "
          f"S42 ниже {MAX_CORR}, цель — PF, устойчивая оценка (медиана PF пробы и 6 соседей) не ниже {ROBUST_MIN}; финалисты — по "
          "приросту t месяцев портфеля «S42 + стратегия» на выборе. Проверка 2021–2026 — один раз, финалисты — штатным минутным "
          "симулятором. Расчёт — `scripts/second_search.py`.", "",
          f"Проб: **{len(df):,}** (случайных {len(rnd):,}); прошли отбор на выборе (сделки, связь с S42, PF > 1): "
          f"{int(((df['score'] > 1)).sum()):,}; устойчивых (оценка ≥ {ROBUST_MIN}): {len(passed):,}.".replace(",", " "), "",
          verdict, "", "## Walk-forward самого поиска — честная оценка", "",
          "На каждый год берётся лучшая стратегия по годам до него (среди прошедших отбор) и торгует этот год.", "",
          "| год | выбранная стратегия | PF до года | итог года, % | S42 за год, % | портфель пополам, % |", "|---|---|---|---|---|---|"]
    md += [f"| {r['год']} | {r['стратегия']} | {r['PF до года']:.2f} | {r['итог года']:+.1f} | {r['S42 за год']:+.1f} | {r['портфель пополам']:+.1f} |"
           for _, r in wf.iterrows()]
    if len(wf):
        md += [f"| **2021–2026** | | | **{wf['итог года'].sum():+.1f}** | **{wf['S42 за год'].sum():+.1f}** | **{wf['портфель пополам'].sum():+.1f}** |", ""]
    md += ["## Финалисты — штатный минутный симулятор", "",
           "| № | стратегия | период | сделок в год | итог % | PF | без 2 лучших | t мес | с S42: t мес (S42 один) | с S42: итог/просадка (S42 один) | корр. с S42 | устойч. оценка |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['№']} | {r['стратегия']} | {r['период']} | {fmt(r['сделок в год'], 1)} | {r['итог']:+.1f} | {fmt(r['PF'])} | "
           f"{fmt(r['без 2 лучших'], 1)} | {fmt(r['t мес'])} | {fmt(r['с S42: t мес'])} ({fmt(r['S42: t мес'])}) | "
           f"{fmt(r['с S42: итог/просадка'])} ({fmt(r['S42: итог/просадка'])}) | {fmt(r['корр. месяцев (выбор)'])} | {fmt(r['устойчивая оценка'])} |"
           for _, r in L.iterrows()]
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--hours", type=float, default=6.0)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--procs", type=int, default=3)
    r = sub.add_parser("report")
    r.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    return search(a.hours, a.seed, a.procs) if a.cmd == "search" else report(a.top)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
