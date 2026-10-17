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

WF_YEARS = list(range(2021, 2027))


def walk_forward(con, ref: np.ndarray) -> pd.DataFrame:
    """На год Y — лучшая по PF за годы до Y среди прошедших отбор (сделок ≥ 10 в год, связь с S42 < 0.3 по месяцам
    до Y); торгует год Y. Честная оценка того, что дал бы сам поиск. Один проход по базе (1 млн проб не помещаются
    в память машины целиком): на каждый год хранится только текущая лучшая."""
    cut = {Y: int(MONTHS.get_loc(pd.Period(f"{Y}-01", "M"))) for Y in WF_YEARS}
    yr = {Y: MONTHS.year == Y for Y in WF_YEARS}
    best = {Y: (-1.0, None, 0.0) for Y in WF_YEARS}
    rc = {Y: ref[:cut[Y]] - ref[:cut[Y]].mean() for Y in WF_YEARS}
    for n_row, (params, years, blob) in enumerate(con.execute("SELECT params, years, m FROM trials"), 1):
        y = json.loads(years)
        m = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
        for Y in WF_YEARS:
            ks = [str(k) for k in range(2016, Y) if str(k) in y]
            pos = sum(y[k][1] for k in ks)
            neg = sum(y[k][2] for k in ks)
            n = sum(y[k][3] for k in ks)
            if n < 10 * (Y - 2016) or neg <= 0 or pos / neg <= best[Y][0]:
                continue
            mb = m[:cut[Y]] - m[:cut[Y]].mean()
            den = np.sqrt((mb ** 2).sum() * (rc[Y] ** 2).sum())
            if den > 0 and (mb @ rc[Y]) / den < MAX_CORR:
                best[Y] = (pos / neg, params, float(m[yr[Y]].sum()))
        if n_row % 200_000 == 0:
            print(f"walk-forward: {n_row} проб", flush=True)
    out = []
    for Y, (pfv, params, net) in best.items():
        if params is None:
            continue
        r = float(ref[yr[Y]].sum())
        out.append({"год": Y, "стратегия": ss.describe(json.loads(params)), "PF до года": pfv, "итог года": net,
                    "S42 за год": r, "портфель пополам": 0.5 * net + 0.5 * r})
    return pd.DataFrame(out)


def wf_png(wf: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments import BG, BLUE, GRAY, GRID, INK, INK2

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(BG)
    x, w = np.arange(len(wf)), 0.38
    ax.bar(x - w / 2, wf["S42 за год"], width=w, color=GRAY, label=f"S42 один: {wf['S42 за год'].sum():+.1f}% за 2021–2026")
    ax.bar(x + w / 2, wf["итог года"], width=w, color=BLUE, label=f"стратегия, выбранная поиском по прошлым годам: {wf['итог года'].sum():+.1f}%")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, wf["год"].astype(str))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("итог года, % (5 рынков поровну)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


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
    ref = s42_monthly()
    cols = ["params", "score", "robust", "sel_pf", "sel_n", "corr", "pt_sel", "test_pf", "test_n", "test_net", "pt_test"]
    passed = pd.DataFrame(con.execute(f"SELECT {', '.join(cols)} FROM trials WHERE score > 0 AND robust >= ? ORDER BY pt_sel DESC",
                                      (ROBUST_MIN,)).fetchall(), columns=cols)
    stats = dict(zip(["всего", "случайных", "прошли"], con.execute(
        "SELECT COUNT(*), SUM(how = 'разведка'), SUM(score > 1) FROM trials").fetchone()))
    # финалисты: без перестановок фильтров и не больше одного на вход — иначе верхушка — одна стратегия в разных обличьях
    picks, seen_entry = [], set()
    for p in passed["params"]:
        prm = json.loads(p)
        if prm["entry"] not in seen_entry:
            seen_entry.add(prm["entry"])
            picks.append(prm)
        if len(picks) >= top_n:
            break
    # правило отбора целиком: что все прошедшие показали на проверке (оценка правила, а не удачной верхушки)
    rule = {"прошедших": len(passed), "PF проверки, медиана": float(passed["test_pf"].median()) if len(passed) else np.nan,
            "PF проверки > 1.1": float((passed["test_pf"] > 1.1).mean()) if len(passed) else np.nan,
            "t портфеля на проверке, медиана": float(passed["pt_test"].median()) if len(passed) else np.nan,
            "t портфеля на проверке выше S42": float((passed["pt_test"] > tstat(ref[~SEL])).mean()) if len(passed) else np.nan,
            "связь выбор → проверка": float(passed[["sel_pf", "test_pf"]].corr(method="spearman").iat[0, 1]) if len(passed) > 2 else np.nan,
            "S42 t выбор": tstat(ref[SEL]), "S42 t проверка": tstat(ref[~SEL])}
    wf = walk_forward(con, ref)
    con.close()
    print(f"отбор: {stats}, устойчивых {len(passed)}; walk-forward готов", flush=True)
    mt = finalists_minute(picks) if picks else pd.DataFrame()
    _, exps = load_catalog()
    s42 = load_trades(exps["S42"])
    alone = pf.combine({"S42": s42}, SET)
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
    (DOC.parent / "second_strategy").mkdir(parents=True, exist_ok=True)
    wf_png(wf, DOC.parent / "second_strategy" / "walk_forward.png")

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
          f"Проб: **{stats['всего']:,}** (случайных {stats['случайных']:,}); прошли отбор на выборе (сделки, связь с S42, PF > 1): "
          f"{stats['прошли']:,}; устойчивых (оценка ≥ {ROBUST_MIN}): {len(passed):,}.".replace(",", " "), "",
          verdict, "", "## Правило отбора целиком — что прошедшие показали на проверке", "",
          f"Все {rule['прошедших']} устойчивых проб, отобранных только по 2016–2020. Это оценка самого правила отбора, а не "
          "удачной верхушки.", "",
          "| PF проверки, медиана | доля с PF проверки > 1.1 | t портфеля с S42 на проверке, медиана | доля, где портфель лучше S42 по t | связь PF выбор → проверка | S42 один: t выбор / проверка |",
          "|---|---|---|---|---|---|",
          f"| {rule['PF проверки, медиана']:.2f} | {rule['PF проверки > 1.1']:.0%} | {rule['t портфеля на проверке, медиана']:.2f} | "
          f"{rule['t портфеля на проверке выше S42']:.0%} | {rule['связь выбор → проверка']:.2f} | {rule['S42 t выбор']:.2f} / {rule['S42 t проверка']:.2f} |", "",
          "## Walk-forward самого поиска — честная оценка", "",
          "На каждый год берётся лучшая стратегия по годам до него (среди прошедших отбор) и торгует этот год.", "",
          "![Walk-forward поиска против S42](second_strategy/walk_forward.png)", "",
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
