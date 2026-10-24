"""Сверка сделок журнала с правилом и с графиком (предложение созвона 23.10, T42): то, что видно на странице журнала,
согласовано с тем, что требует правило.

По каждому эксперименту семейств «импульс», «перебор» и «тренд-фильтр» (модели — со своими данными, не здесь):
  * цена входа и цена выхода лежат внутри часовой свечи страницы того часа, что указан в сделке (свечи —
    docs/experiments/site/data/candles, цены сделок — в масштабе сырых свечей, как на графике: experiments.raw_prices);
  * сделка не дольше предела правила (выход эксперимента или семейства, часы);
  * направление входа — по тренд-фильтру на часе сигнала (час до входа), где правило этого требует: импульс — гейт C,
    эксперименты с зонами трёхслойного фильтра (H59, H60 и их перенос) — трёхслойный фильтр.
Сдвиг цен целиком (все входы выше или ниже свечей) значит, что сделки посчитаны на других ценах, чем свечи страницы —
например, со старым файлом дивидендов.

    python scripts/trade_audit.py            # все эксперименты → docs/results/trade_audit.md
    python scripts/trade_audit.py S42 S53    # только эти
"""
from __future__ import annotations

import gc
import json
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import experiments as xp
from experiments import ROOT, SITE, load_catalog, load_trades, raw_prices

DOC = ROOT / "docs" / "results" / "trade_audit.md"
CSV = ROOT / "docs" / "results" / "trade_audit.csv"
FAMILIES = ("strategy", "search", "trend")
GATED = {"strategy"}            # импульс: сторону входа подтверждает гейт C
OTHER_FILTER = {"S14": "без фильтра", "S26": "фильтр ТЗ 15.10", "S27": "фильтр ТЗ 15.10"}   # импульс не с гейтом C
TOL = 2e-4                      # относительный допуск: свечи страницы округлены до шага цены
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def candles(inst: str) -> dict | None:
    p = SITE / "candles" / f"{inst}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    n = len(d["dt"])
    t = d["t0"] + np.cumsum(np.asarray(d["dt"], dtype=np.int64)) * 3600

    def fill(ch):
        g = np.zeros(n, np.int8)
        for k, (i, v) in enumerate(ch):
            g[i:ch[k + 1][0] if k + 1 < len(ch) else n] = v
        return g
    return {"t": t, "h": np.asarray(d["h"], float), "l": np.asarray(d["l"], float),
            "gate": fill(d.get("gate") or []), "tl": fill(d["gate_tl"]) if d.get("gate_tl") else None}


def bar(c: dict, ts: pd.Series) -> np.ndarray:
    """Индекс часовой свечи страницы, в которую попадает момент (последняя свеча со временем ≤ момента)."""
    x = (ts.dt.floor("h") - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return np.clip(np.searchsorted(c["t"], x.to_numpy(), side="right") - 1, 0, len(c["t"]) - 1)


def check(exp: dict, cat: dict, inst: str, c: dict) -> dict:
    t = load_trades({**exp, "sources": {inst: exp["sources"][inst]}})
    if t.empty:
        return {}
    if exp.get("dividends"):                    # как на странице: без поправки на дивиденды цены уже сырые
        t = raw_prices(t)
    hours = (exp.get("exit") or (cat.get("exit_defaults") or {}).get(exp["family"], {}).get("all") or {}).get("hours")
    bi, bo = bar(c, t["open_dt"]), bar(c, t["close_dt"])
    lo_i, hi_i, lo_o, hi_o = c["l"][bi], c["h"][bi], c["l"][bo], c["h"][bo]
    p_in, p_out = t["open_price"].to_numpy(float), t["exit_price"].to_numpy(float)
    ok_in = (p_in >= lo_i * (1 - TOL)) & (p_in <= hi_i * (1 + TOL))
    ok_out = (p_out >= lo_o * (1 - TOL)) & (p_out <= hi_o * (1 + TOL))
    r = {"эксперимент": exp["id"], "бумага": inst, "сделок": len(t),
         "вход в свече, %": ok_in.mean() * 100, "выход в свече, %": ok_out.mean() * 100,
         "входы выше свечи, %": (p_in > hi_i * (1 + TOL)).mean() * 100, "входы ниже свечи, %": (p_in < lo_i * (1 - TOL)).mean() * 100}
    held = (t["close_dt"] - t["open_dt"]) / pd.Timedelta(hours=1)
    # выход раньше входа — движок перебора закрывает сделку на открытии часа входа, если сигнал выхода уже стоит
    r["выход раньше входа"] = int((held < 0).sum())
    if hours:
        r["не дольше предела, %"] = (held <= hours + 1).mean() * 100
    gate = c["tl"] if exp.get("zones") == "tl" else c["gate"] if exp["family"] in GATED and exp["id"] not in OTHER_FILTER else None
    if gate is not None:
        sig = np.clip(bi - 1, 0, len(gate) - 1)                     # час сигнала — свеча перед входом
        side = t["side"].to_numpy()
        r["по фильтру, %"] = (gate[sig] == side).mean() * 100
        r["фильтр"] = "трёхслойный" if exp.get("zones") == "tl" else "гейт C"
    bad = t[~(ok_in & ok_out)].head(3)
    r["примеры"] = "; ".join(f"{a:%Y-%m-%d %H:%M} → {b:%Y-%m-%d %H:%M} {x} {p:.4g}"
                             for a, b, x, p in zip(bad["open_dt"], bad["close_dt"], bad["exit_reason"], bad["exit_price"]))
    return r


def main(argv=None) -> int:
    cat, exps = load_catalog()
    want = [e for k, e in exps.items() if e["family"] in FAMILIES and (not argv or k in argv)]
    insts = sorted({i for e in want for i in e["sources"]})
    rows = []
    for inst in insts:                           # по одной бумаге: минутки для пересчёта цен в масштаб графика
        c = candles(inst)
        if c is None:
            print(f"[{inst}] нет свечей страницы — пропуск", flush=True)
            continue
        for e in want:
            if inst in e["sources"]:
                r = check(e, cat, inst, c)
                if r:
                    rows.append(r)
        xp._factor_cache.clear()
        gc.collect()
        print(f"[{inst}] проверено", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(CSV, index=False)
    write_doc(d, exps)
    return 0


def write_doc(d: pd.DataFrame, exps: dict) -> None:
    f = lambda v: "—" if pd.isna(v) else f"{v:.1f}"  # noqa: E731
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a, b = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a:b + len(MARK[1])] if a >= 0 and b > a else f"{MARK[0]}\n## Вывод\n\n(дописать)\n{MARK[1]}"
    by = d.groupby("эксперимент", sort=False)
    md = ["# Сверка сделок журнала с правилом и графиком", "",
          "Предложение созвона 23.10 (T42): то, что видно на графике журнала, машинно сверяется с правилом стратегии. "
          "По каждому эксперименту импульса, перебора и тренд-фильтра: цена входа и выхода — внутри часовой свечи страницы "
          "того часа; сделка не дольше предела правила; вход — по тренд-фильтру на часе сигнала, где правило этого требует "
          "(импульс — гейт C, эксперименты с зонами трёхслойного фильтра — трёхслойный). Расчёт — `scripts/trade_audit.py`, "
          "по бумагам — [trade_audit.csv](trade_audit.csv).", "", verdict, "",
          "| эксперимент | сделок | вход в свече, % | выход в свече, % | не дольше предела, % | по фильтру, % | выход раньше входа | где расходится |",
          "|---|---|---|---|---|---|---|---|"]
    for k, g in by:
        n = g["сделок"].sum()
        w = lambda col: (g[col] * g["сделок"]).sum() / n if col in g and g[col].notna().any() else np.nan  # noqa: E731
        worst = g.assign(bad=100 - g[["вход в свече, %", "выход в свече, %"]].min(axis=1)).sort_values("bad", ascending=False)
        where = ", ".join(f"{r['бумага']} ({r['вход в свече, %']:.0f}% / {r['выход в свече, %']:.0f}%)"
                          for r in worst.to_dict("records") if r["bad"] > 0.5)[:200]
        md.append(f"| {k} | {n} | {f(w('вход в свече, %'))} | {f(w('выход в свече, %'))} | {f(w('не дольше предела, %'))} | "
                  f"{f(w('по фильтру, %'))} | {int(g['выход раньше входа'].sum())} | {where or '—'} |")
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
