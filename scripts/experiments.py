"""Журнал экспериментов: одинаковые метрики на одном периоде проверки для всех экспериментов.

Читает каталог docs/experiments/catalog.yaml и сделки прогонов (reports/walk_forward/<имя>_trades.csv),
пересчитывает издержки, где они заданы профилем брокера, и для каждого эксперимента пишет
  docs/experiments/<id>/summary.md   — карточка: что поменяли, метрики против базы, график;
  docs/experiments/<id>/equity.png   — накопленный итог эксперимента и базы;
  docs/experiments/<id>/trades.csv.gz — журнал сделок (вход, выход, цены, результат, причина выхода);
и общие docs/experiments/registry.csv, docs/experiments/README.md и данные страницы журнала
(docs/experiments/site/data/).

Протокол: «проверка» — сделки, открытые с 1 января 2021 года; «выбор» — до 2021 года. Портфель —
инструменты поровну, прогоны одного инструмента (seed) поровну. База сравнивается на тех же
инструментах, что и эксперимент.

    python scripts/experiments.py            # всё
    python scripts/experiments.py S03 S17    # только эти эксперименты (+ их базы)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from forexmodel.data.dividends import adjustment_factors, load_dividends  # noqa: E402
from forexmodel.evaluation.experiments import CORE, PERIODS, TEST_START, metrics, scopes, standardize  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "reports" / "walk_forward"
DOC = ROOT / "docs" / "experiments"
SITE = DOC / "site" / "data"
GH = "https://github.com/deniss97/forexmodel/blob/main/docs/"
NAMES = {"silver": "серебро", "gold": "золото", "lkoh": "LKOH", "gazp": "GAZP", "sber": "SBER", "moex": "MOEX",
         "mtss": "MTSS", "eth": "ETH", "все": "все инструменты", "портфель 4": "портфель 4",
         "металлы": "металлы (серебро + золото)"}
STOCK_FILES = {"lkoh": "LKOH_1m_15-26", "gazp": "GAZP_1m_15-26_upd", "sber": "SBER_1m_15-26",
               "moex": "MOEX_1m_15-26", "mtss": "MTSS_1m_15-26"}
REASON = {"stop_loss": "стоп", "trailing_stop": "трейлинг", "timeout": "время", "take_profit": "тейк", "trend_flip": "разворот",
          "early_stop": "ранний стоп", "signal": "сигнал"}
METRIC_COLS = ["сделок", "итог", "выигрышей", "ожидание", "ср_выигрыш", "ср_проигрыш", "PF", "без_2_лучших",
               "просадка", "t", "лет_в_плюсе", "с", "по"]
BG, INK, INK2, GRID, BLUE, GRAY = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc", "#2459b3", "#9aa3ae"


def load_trades(exp: dict) -> pd.DataFrame:
    costs = exp.get("costs")
    parts = []
    for inst, runs in exp["sources"].items():
        for run in runs:
            p = WF / f"{run}_trades.csv"
            if not p.exists():
                raise FileNotFoundError(f"{exp['id']}: нет сделок {p.name}")
            t = pd.read_csv(p, parse_dates=["open_dt", "close_dt"])
            if "size" not in t:
                t["size"] = 1.0
            if "gross_unit_pct" not in t:
                t["gross_unit_pct"] = t["gross_pct"] / t["size"]
            if costs:
                name = (costs.get("override") or {}).get(inst, inst)
                t = recost(t, profile_entry(ROOT / "configs" / "costs" / f"{costs['profile']}.yaml", name, costs.get("set")))
            if inst in (exp.get("long_only") or []):
                # только лонг (шорт недоступен или дорог): шорты убираются из готовых сделок
                t = t[t["side"].astype(str).isin(["buy", "1", "2"])]
            parts.append(standardize(t, inst, run))
    return pd.concat(parts, ignore_index=True)


_factor_cache: dict = {}


def raw_prices(t: pd.DataFrame) -> pd.DataFrame:
    """Цены сделок по скорректированным на дивиденды котировкам -> в масштаб сырых свечей (для графика)."""
    t = t.copy()
    for inst in t["instrument"].unique():
        if inst not in STOCK_FILES:
            continue
        if inst not in _factor_cache:
            m = pd.read_pickle(ROOT / "reports" / "_cache" / f"{STOCK_FILES[inst]}_1m.pkl").sort_values("time")
            divs = load_dividends(ROOT / "data" / "dividends" / "moex_dividends.csv", inst)
            _factor_cache[inst] = (m["time"].to_numpy(), adjustment_factors(m, divs))
        tt, f = _factor_cache[inst]
        sel = t["instrument"] == inst
        cols = [("open_price", "open_dt"), ("exit_price", "close_dt")]
        if "atr_at_entry" in t:
            cols.append(("atr_at_entry", "open_dt"))
        for col, tcol in cols:
            i = np.clip(np.searchsorted(tt, t.loc[sel, tcol].to_numpy(), "left"), 0, len(f) - 1)
            t.loc[sel, col] = t.loc[sel, col].to_numpy() / f[i]
    return t


def exit_params(cat: dict, exp: dict, inst: str) -> dict:
    """Параметры выхода для линий стопа на графике: стоп, трейлинг, активация (в ATR на входе), часы."""
    fam = (cat.get("exit_defaults") or {}).get(exp["family"], {})
    return {**fam.get("all", {}), **fam.get(inst, {}), **(exp.get("exit") or {})}


def fmt(v, k):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if k in ("сделок", "лет_в_плюсе", "с", "по"):
        return str(v)
    if k in ("PF", "t"):
        return f"{v:.2f}"
    if k in ("выигрышей",):
        return f"{v:.0f}%"
    if k in ("ожидание", "ср_выигрыш", "ср_проигрыш"):
        return f"{v:+.3f}%"
    return f"{v:+.1f}"


def equity_png(exp: dict, t: pd.DataFrame, base: pd.DataFrame | None, inst: list, path: Path) -> None:
    f, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(BG)
    for d, label, color, lw in ((base, f"база {exp.get('baseline')}", GRAY, 1.8), (t, exp["id"], BLUE, 2.4)):
        if d is None:
            continue
        x = d[d["instrument"].isin(inst)].sort_values("close_dt")
        if x.empty:
            continue
        runs = x.groupby("instrument")["run"].nunique()
        w = x["instrument"].map(lambda i: 1 / len(inst) / runs[i])
        ax.plot(x["close_dt"], (w * x["profit_pct"]).cumsum(), color=color, linewidth=lw, label=label)
    ax.axvspan(TEST_START, pd.Timestamp("2026-12-31"), color=BLUE, alpha=0.06, lw=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.text(TEST_START, ax.get_ylim()[1], "  проверка →", va="top", color=BLUE, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel("накопленный итог, %", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    f.tight_layout()
    f.savefig(path, dpi=110, facecolor=BG)
    plt.close(f)


BANK = ROOT / "docs" / "hypothesis_bank" / "journal.json"
HYPS = ROOT / "docs" / "hypothesis_bank" / "hypotheses.json"  # снимок банка гипотез (scripts/bank_snapshot.py)
METRICS = ("сделок", "итог", "PF", "без_2_лучших")
KINDS = {"base": "база", "costs": "издержки и исполнение", "clean": "чистая проверка", "variation": "принятая вариация"}


def write_bank_digest(reg: dict) -> None:
    """Краткий реестр для банка гипотез: эксперимент → гипотеза, вердикт и метрики проверки основного среза."""
    pick = lambda m: {k: m.get(k) for k in METRICS} if m else None  # noqa: E731
    rows = []
    for e in reg["experiments"]:
        r = (e["res"].get("проверка") or {}).get(e["main"]) or {}
        rows.append({"id": e["id"], "family": e["family"], "title": e["title"], "hypothesis": e.get("hypothesis"),
                     "baseline": e.get("baseline"), "verdict": e.get("verdict", ""), "outcome": e.get("outcome", ""),
                     "kind": e.get("kind", "clean"), "archived": e.get("archived", ""),
                     "scope": reg["names"].get(e["main"], e["main"]), "m": pick(r.get("m")), "b": pick(r.get("b"))})
    BANK.write_text(json.dumps({"families": reg["families"], "kinds": KINDS, "experiments": rows}, ensure_ascii=False,
                               separators=(",", ":")), encoding="utf-8")


def load_catalog() -> tuple[dict, dict]:
    cat = yaml.safe_load((DOC / "catalog.yaml").read_text(encoding="utf-8"))
    exps = {e["id"]: e for e in cat["experiments"]}
    desc_path = DOC / "descriptions.yaml"
    desc = yaml.safe_load(desc_path.read_text(encoding="utf-8")) if desc_path.exists() else {}
    for k, e in exps.items():
        e.update({f: (desc.get(k) or {}).get(f, "") for f in ("what", "why", "outcome")})
    return cat, exps


def meta(e: dict) -> dict:
    """Описательные поля эксперимента для страницы журнала (без метрик)."""
    return {"id": e["id"], "family": e["family"], "title": e["title"], "change": e.get("change", ""), "what": e.get("what", ""),
            "why": e.get("why", ""), "outcome": e.get("outcome", ""), "hypothesis": e.get("hypothesis"),
            "baseline": e.get("baseline"), "verdict": e.get("verdict", ""), "kind": e.get("kind", "clean"),
            "archived": e.get("archived", ""), "doc": GH + e["doc"], "card": GH + f"experiments/{e['id']}/summary.md"}


def hypotheses_for(codes) -> dict:
    """Название, описание и критерий гипотез из снимка банка — для карточки эксперимента."""
    if not HYPS.exists():
        return {}
    bank = json.loads(HYPS.read_text(encoding="utf-8"))["hypotheses"]
    return {c: {k: bank[c].get(k, "") for k in ("title", "text", "criterion", "status", "area")} for c in codes if c in bank}


def refresh_meta() -> int:
    """--meta: обновить в registry.json описания, вид, архив и гипотезы из каталога без пересчёта метрик."""
    reg = json.loads((SITE / "registry.json").read_text(encoding="utf-8"))
    _, exps = load_catalog()
    reg["experiments"] = [{**e, **meta(exps[e["id"]])} for e in reg["experiments"] if e["id"] in exps]
    reg["kinds"] = KINDS
    reg["hypotheses"] = hypotheses_for({e["hypothesis"] for e in reg["experiments"] if e.get("hypothesis")})
    (SITE / "registry.json").write_text(json.dumps(reg, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_bank_digest(reg)
    print(f"обновлено описаний: {len(reg['experiments'])}, гипотез: {len(reg['hypotheses'])}")
    return 0


def main(argv=None) -> int:
    if argv and list(argv) == ["--bank"]:  # только реестр для банка гипотез из готового registry.json
        write_bank_digest(json.loads((SITE / "registry.json").read_text(encoding="utf-8")))
        return 0
    if argv and list(argv) == ["--meta"]:
        return refresh_meta()
    cat, exps = load_catalog()
    want = list(argv) if argv else list(exps)
    need = set(want) | {exps[i]["baseline"] for i in want if exps[i].get("baseline")}
    trades = {}
    for i in [k for k in exps if k in need]:
        trades[i] = load_trades(exps[i])
        print(f"[{i}] {len(trades[i])} сделок, {trades[i]['instrument'].nunique()} инстр.", flush=True)
    rows, cards = [], []
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "exp").mkdir(exist_ok=True)
    for i in want:
        exp, t = exps[i], trades[i]
        b = trades.get(exp.get("baseline")) if exp.get("baseline") else None
        inst = list(exp["sources"])
        res = {}
        for scope, members in scopes(inst).items():
            for period in PERIODS:
                m = metrics(t, members, period)
                bm = metrics(b, members, period) if b is not None and set(members) <= set(b["instrument"]) else None
                res.setdefault(period, {})[scope] = {"m": m, "b": bm}
                rows.append({"id": i, "семейство": cat["families"][exp["family"]], "эксперимент": exp["title"],
                             "база": exp.get("baseline") or "", "период": period, "срез": scope,
                             **{k: m.get(k) for k in METRIC_COLS},
                             **({f"база_{k}": bm.get(k) for k in ("итог", "ожидание", "PF", "без_2_лучших")} if bm else {}),
                             "вердикт": exp.get("verdict", "")})
        # карточка в репозитории
        d = DOC / i
        d.mkdir(parents=True, exist_ok=True)
        std = t.copy()
        std.to_csv(d / "trades.csv.gz", index=False, compression="gzip")
        main_scope = exp.get("main") or ("портфель 4" if "портфель 4" in res["проверка"] else "все")
        equity_png(exp, t, b, scopes(inst)[main_scope], d / "equity.png")
        lines = [f"# {i}. {exp['title']}", "",
                 f"**Семейство:** {cat['families'][exp['family']]} · **база:** {exp.get('baseline') or '—'} · "
                 f"**гипотеза:** {exp.get('hypothesis') or '—'} · **вердикт:** {exp.get('verdict', '')}", ""]
        if exp.get("what"):
            lines += [f"**Что проверяли.** {exp['what']}", ""]
        elif exp.get("change"):
            lines += [f"**Что поменяли:** {exp['change']}", ""]
        if exp.get("why"):
            lines += [f"**Зачем.** {exp['why']}", ""]
        if exp.get("outcome"):
            lines += [f"**Итог простыми словами.** {exp['outcome']}", ""]
        if exp.get("costs"):
            lines += [f"**Издержки:** профиль `{exp['costs']['profile']}` (спред и свопы брокера)", ""]
        lines += [f"Подробности: [{exp['doc']}](../../{exp['doc']})", "",
                  "Проверка — сделки, открытые с 1 января 2021 года; выбор — до 2021. В скобках — база на тех же инструментах.", ""]
        hdr = ["срез", "период", "сделок", "итог %", "выигрышей", "ожидание на сделку", "PF", "без 2 лучших", "просадка", "t", "лет в плюсе"]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for scope in scopes(inst):
            for period in ("проверка", "выбор"):
                m, bm = res[period][scope]["m"], res[period][scope]["b"]
                if not m.get("сделок"):
                    continue
                cell = lambda k: fmt(m.get(k), k) + (f" ({fmt(bm.get(k), k)})" if bm and bm.get("сделок") else "")  # noqa: E731
                lines.append("| " + " | ".join([NAMES.get(scope, scope), period, cell("сделок"), cell("итог"), cell("выигрышей"),
                                                cell("ожидание"), cell("PF"), cell("без_2_лучших"), cell("просадка"),
                                                cell("t"), cell("лет_в_плюсе")]) + " |")
        lines += ["", f"![накопленный итог: {NAMES.get(main_scope, main_scope)}](equity.png)", "",
                  "Журнал сделок: `trades.csv.gz` (инструмент, прогон, сторона, вход, выход, цены, результат после издержек, причина выхода)."]
        (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        cards.append((exp, res, main_scope))
        # данные страницы: сделки в сыром масштабе цен (для графика со свечами)
        raw = raw_prices(t) if exp.get("dividends") else t
        payload, runs_by = {}, {}
        num = lambda g, c, k: [round(float(x), k) if np.isfinite(x) else None for x in g[c]] if c in g else [None] * len(g)  # noqa: E731
        for ins, g in raw.groupby("instrument"):
            g = g.sort_values("open_dt")
            runs = {r: k for k, r in enumerate(sorted(g["run"].unique()))}   # прогоны (seed) — внутри инструмента
            runs_by[ins] = len(runs)
            # [вход, выход, сторона, цена входа, цена выхода, результат %, причина, прогон, ATR на входе, своп %]
            payload[ins] = [[int(a.timestamp()), int(c.timestamp()), int(s), round(float(op), 6), round(float(ep), 6),
                             round(float(p), 4), REASON.get(r, r), runs[rn], atr, sw]
                            for a, c, s, op, ep, p, r, rn, atr, sw in zip(g["open_dt"], g["close_dt"], g["side"], g["open_price"],
                                                                          g["exit_price"], g["profit_pct"], g["exit_reason"], g["run"],
                                                                          num(g, "atr_at_entry", 6), num(g, "swap_pct", 4))]
        (SITE / "exp" / f"{i}.json").write_text(json.dumps({"id": i, "runs": max(runs_by.values()), "runs_by": runs_by, "trades": payload,
                                                            "exit": {ins: exit_params(cat, exp, ins) for ins in payload}},
                                                           ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    clean = lambda v: None if isinstance(v, float) and not np.isfinite(v) else v  # noqa: E731
    fresh = {e["id"]: {**meta(e),
                       "costs": (e.get("costs") or {}).get("profile"), "dividends": bool(e.get("dividends")),
                       "instruments": list(e["sources"]), "main": ms,
                       "res": {p: {s: {"m": {k: clean(v) for k, v in r["m"].items()},
                                       "b": {k: clean(v) for k, v in r["b"].items()} if r["b"] else None}
                                   for s, r in sc.items()} for p, sc in res.items()}}
             for e, res, ms in cards}
    reg = pd.DataFrame(rows)
    if argv:   # частичная сборка: новые и пересчитанные эксперименты вливаются в готовый реестр, остальное не трогаем
        old = json.loads((SITE / "registry.json").read_text(encoding="utf-8"))["experiments"]
        fresh = {**{e["id"]: e for e in old if e["id"] in exps}, **fresh}
        prev = pd.read_csv(DOC / "registry.csv", dtype=str, keep_default_na=False)   # старые строки — как были, до знака
        reg = pd.concat([prev[~prev["id"].isin(want)], reg], ignore_index=True)
        reg["_o"] = reg["id"].map({k: n for n, k in enumerate(exps)})
        reg = reg.sort_values("_o", kind="stable").drop(columns="_o")
    entries = [fresh[k] for k in exps if k in fresh]
    reg.to_csv(DOC / "registry.csv", index=False)
    # README: сводная таблица по проверке
    lines = ["# Журнал экспериментов", "",
             "Все эксперименты посчитаны одинаково: **проверка — сделки, открытые с 1 января 2021 года** (параметры выбирались "
             "раньше), метрики после издержек эксперимента. Портфель — инструменты поровну; база — на тех же инструментах. "
             "С 11.10 эксперименты сравниваются с базой **B01** — все неизбежные издержки: дивиденды, стоп по гэпу, комиссия 0.04%, "
             "проскальзывание 0.05% на сторону, свопы за перенос (configs/costs/base.yaml). "
             "Интерактивная версия с журналом сделок на графике — https://claude.ai/artifact/E4teCbCVvojCf96CJy9UQU. Каталог — [catalog.yaml](catalog.yaml), сборка — `scripts/experiments.py`.", ""]
    for fam, fam_name in cat["families"].items():
        lines += [f"## {fam_name}", "",
                  "| id | эксперимент | срез | сделок | итог % | Δ к базе | ожидание | PF | без 2 лучших | t | вердикт |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in entries:
            if e["family"] != fam:
                continue
            ms = e["main"]
            m, bm = e["res"]["проверка"][ms]["m"], e["res"]["проверка"][ms]["b"]
            delta = f"{m['итог'] - bm['итог']:+.1f}" if bm and bm.get("сделок") and m.get("сделок") else "—"
            lines.append(f"| [{e['id']}]({e['id']}/summary.md) | {e['title']} | {NAMES.get(ms, ms)} | {fmt(m.get('сделок'), 'сделок')} | "
                         f"{fmt(m.get('итог'), 'итог')} | {delta} | {fmt(m.get('ожидание'), 'ожидание')} | {fmt(m.get('PF'), 'PF')} | "
                         f"{fmt(m.get('без_2_лучших'), 'без_2_лучших')} | {fmt(m.get('t'), 't')} | {e.get('verdict', '')} |")
        lines.append("")
    lines += ["## Проверено без журнала сделок", "", "| проверка | итог | где |", "|---|---|---|"]
    lines += [f"| {x['title']} | {x['result']} | [{x['doc']}](../{x['doc']}) |" for x in cat.get("no_log", [])]
    (DOC / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # реестр для страницы
    out = {"protocol": {"test_start": "2021-01-01", "periods": list(PERIODS)}, "families": cat["families"], "names": NAMES, "kinds": KINDS,
           "hypotheses": hypotheses_for({e["hypothesis"] for e in entries if e.get("hypothesis")}),
           "experiments": entries,
           "no_log": [{**x, "doc": GH + x["doc"]} for x in cat.get("no_log", [])]}
    (SITE / "registry.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_bank_digest(out)
    print(f"готово: {len(cards)} пересчитано, в реестре {len(entries)} → {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
