"""Сверка живой торговли с симуляцией — чтобы вовремя остановиться (условие Дениса к ТЗ инференса, 26.10).

Раз в сутки, по каждой стратегии (S42, S46, S48) и бумаге:
  1. Стратегия против бэктеста. Сделки стратегии на живых данных с начала режима (24.10): S42 —
     reports/live/s42_journal.csv, S46 и S48 — reports/live/iterator_journal.csv. Сравнение — со сделками той же стратегии на
     тех же бумагах на проверке 2021–2026 (журнал: S42; S46 + S53; S48 + S55), результат на единицу позиции после
     издержек. Средняя живая сделка против распределения средних из стольких же случайных сделок бэктеста (5000 повторов):
     ниже 5% повторов — «внимание», ниже 1% — «стоп»: стратегия попадает в reports/live/strategy_stop.json, и счёт
     (scripts/live_account.py) перестаёт брать её сигналы, пока её не уберут оттуда руками. Меньше 5 сделок — «мало сделок».
     Плюс серия убытков подряд против 99-го процентиля такой серии в случайных выборках бэктеста — «внимание».
  2. Наше исполнение против сигналов. Позиции счёта (reports/live/account_trades.csv, account_state.json) против
     исполненных сделок у брокера (только чтение): проскальзывание на входе и выходе, опоздание, сигналы, которые мы
     не исполнили. Проскальзывание в среднем хуже 0.15% на сторону (в бэктесте заложено 0.05%) от 5 сделок — «внимание».
  3. Журнал альтернативных сделок: чем закончились сигналы, которые не взяли на счёт (не хватило слота или денег).
Отчёт — reports/live/deviation.md; при смене статуса стратегии — уведомление.

    python scripts/live_deviation.py
"""
from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import notify
from experiments import NEW_STOCKS, ROOT, load_catalog, load_trades

OUT = ROOT / "reports" / "live"
DOC = OUT / "deviation.md"
STATUS = OUT / "deviation_status.json"
STOP = OUT / "strategy_stop.json"
START = pd.Timestamp("2026-10-24")
TEST = pd.Timestamp("2021-01-01")
COST = 0.2                     # издержки биржи за круг, % (0.1 комиссии + 2 × 0.05 проскальзывания) — как exchange.yaml
REF = {"S42": ["S42"], "S46": ["S46", "S53"], "S48": ["S48", "S55"]}
STOCKS = {"S42": ["sber", "gazp", "lkoh"], "S46": ["sber", "gazp", "lkoh", *NEW_STOCKS], "S48": ["sber", "gazp", "lkoh", *NEW_STOCKS]}
DRAWS = 5000
MIN_TRADES = 5


def reference() -> dict[str, pd.DataFrame]:
    """Сделки бэктеста на проверке: результат на единицу позиции после издержек, по стратегиям."""
    _, exps = load_catalog()
    out = {}
    for code, ids in REF.items():
        t = pd.concat([load_trades(exps[i]) for i in ids], ignore_index=True)
        t = t[t["instrument"].isin(STOCKS[code]) & (t["open_dt"] >= TEST)]
        size = t["size"].to_numpy(float) if "size" in t else np.ones(len(t))
        out[code] = t.assign(unit=t["profit_pct"].to_numpy(float) / np.where(size > 0, size, 1))
    return out


def live() -> dict[str, pd.DataFrame]:
    """Закрытые сделки стратегий на живых данных с начала режима, результат на единицу позиции после издержек."""
    parts = []
    p42 = OUT / "s42_journal.csv"
    if p42.exists():
        t = pd.read_csv(p42, parse_dates=["вход", "выход"])
        t = t.assign(unit=(t["цена выхода"] / t["цена входа"] - 1) * 100 - COST)
        parts.append(t[["стратегия", "бумага", "вход", "выход", "unit"]])
    pit = OUT / "iterator_journal.csv"
    if pit.exists():
        t = pd.read_csv(pit, parse_dates=["вход", "выход"])
        t = t[t["выход"].notna()]
        t = t.assign(unit=t["результат %"] / t["размер"].where(t["размер"] > 0, 1))
        parts.append(t[["стратегия", "бумага", "вход", "выход", "unit"]])
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["стратегия", "бумага", "вход", "выход", "unit"])
    t = t[t["вход"] >= START]
    return {code: t[t["стратегия"] == code] for code in REF}


def losing_streak(v: np.ndarray) -> int:
    best = cur = 0
    for x in v:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def judge(live_unit: np.ndarray, ref_unit: np.ndarray, rng: np.random.Generator) -> dict:
    """Насколько живые сделки похожи на бэктест: доля случайных выборок бэктеста со средней не выше живой."""
    n = len(live_unit)
    r = {"сделок": n, "средняя": float(live_unit.mean()) if n else np.nan, "средняя бэктеста": float(ref_unit.mean()),
         "сумма": float(live_unit.sum()) if n else 0.0}
    if n < MIN_TRADES:
        return {**r, "статус": "мало сделок", "p": np.nan}
    draws = ref_unit[rng.integers(0, len(ref_unit), (DRAWS, n))]
    p = float((draws.mean(axis=1) <= live_unit.mean()).mean())
    streak = losing_streak(live_unit)
    lim = float(np.percentile([losing_streak(d) for d in draws[:1000]], 99))
    status = "стоп" if p < 0.01 else "внимание" if p < 0.05 or streak > lim else "в норме"
    return {**r, "p": p, "убытков подряд": streak, "предел серии": lim, "статус": status}


def execution() -> dict:
    """Наши сделки против сигналов: по позициям счёта и исполненным сделкам у брокера (если есть токен)."""
    trades = pd.read_csv(OUT / "account_trades.csv", parse_dates=["вход_в", "назначено"]) if (OUT / "account_trades.csv").exists() else pd.DataFrame()
    state = json.loads((OUT / "account_state.json").read_text(encoding="utf-8")) if (OUT / "account_state.json").exists() else {"позиции": []}
    pos = pd.DataFrame(state.get("позиции", []))
    signals = pd.concat([trades, pos], ignore_index=True) if len(trades) or len(pos) else pd.DataFrame()
    if signals.empty:
        return {"сигналов": 0}
    try:
        import yaml
        from forexmodel.broker.tinvest import TInvest
        from live_account import CFG, account_id

        api = TInvest(sandbox=False)
        acc = account_id(api, yaml.safe_load(CFG.read_text(encoding="utf-8")).get("account"))
        uids = json.loads((OUT / "uids.json").read_text())
        back = {u: t for t, u in uids.items()}
        ops = api.operations(acc, pd.to_datetime(signals["назначено"]).min() - pd.Timedelta(hours=3), pd.Timestamp.now("UTC").tz_localize(None))
    except Exception as e:
        return {"сигналов": len(signals), "ошибка брокера": str(e)[:200]}
    from forexmodel.broker.tinvest import q2d

    fills = pd.DataFrame([{"бумага": back.get(o.get("instrumentUid")), "тип": o.get("operationType"), "штук": int(o.get("quantity", 0)),
                           "цена": float(q2d(o["price"])) if o.get("price") else np.nan,
                           "время": pd.Timestamp(o["date"]).tz_convert("Europe/Moscow").tz_localize(None)} for o in ops])
    fills = fills[fills["бумага"].notna() & fills["тип"].isin(["OPERATION_TYPE_BUY", "OPERATION_TYPE_SELL"])]
    rows = []
    for s in signals.to_dict("records"):
        t0 = pd.Timestamp(s["назначено"])
        b = fills[(fills["бумага"] == s["бумага"]) & (fills["тип"] == "OPERATION_TYPE_BUY") & (fills["время"] >= t0 - pd.Timedelta(minutes=30))
                  & (fills["время"] <= t0 + pd.Timedelta(hours=6))]
        if b.empty:
            rows.append({"стратегия": s["стратегия"], "бумага": s["бумага"], "исполнено": False})
            continue
        px = float((b["цена"] * b["штук"]).sum() / b["штук"].sum())
        rows.append({"стратегия": s["стратегия"], "бумага": s["бумага"], "исполнено": True,
                     "проскальзывание входа, %": (px / float(s["цена"]) - 1) * 100,
                     "опоздание, мин": (b["время"].iloc[0] - pd.Timestamp(s["вход_в"])).total_seconds() / 60})
    return {"сигналов": len(signals), "сделки": rows}


def alternatives() -> pd.DataFrame:
    """Чем закончились сигналы, не взятые на счёт: результат той же сделки стратегии."""
    if not (OUT / "alt_journal.csv").exists():
        return pd.DataFrame()
    a = pd.read_csv(OUT / "alt_journal.csv", parse_dates=["вход_в"])
    res = []
    for p, code in ((OUT / "iterator_journal.csv", None), (OUT / "s42_journal.csv", "S42")):
        if p.exists():
            t = pd.read_csv(p, parse_dates=["вход"])
            if code:
                t = t.assign(**{"результат %": (t["цена выхода"] / t["цена входа"] - 1) * 100 - COST})
            res.append(t[["стратегия", "бумага", "вход", "результат %"]])
    if not res:
        return a.assign(**{"результат %": np.nan})
    r = pd.concat(res, ignore_index=True).rename(columns={"вход": "вход_в"})
    return a.merge(r, on=["стратегия", "бумага", "вход_в"], how="left")


def main() -> int:
    rng = np.random.default_rng(26)
    ref, lv = reference(), live()
    old = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    stopped = set(json.loads(STOP.read_text())) if STOP.exists() else set()
    rows, msgs = [], []
    for code in REF:
        r = judge(lv[code]["unit"].to_numpy(float), ref[code]["unit"].to_numpy(float), rng)
        rows.append({"стратегия": code, **r})
        if r["статус"] != old.get(code) and r["статус"] in ("внимание", "стоп"):
            msgs.append(f"{'⛔' if r['статус'] == 'стоп' else '⚠️'} Сверка с симуляцией: {code} — {r['статус']}. {r['сделок']} сделок на живых "
                        f"данных, средняя {r['средняя']:+.2f}% против {r['средняя бэктеста']:+.2f}% в бэктесте (так плохо — в {r['p'] * 100:.1f}% "
                        f"случайных выборок бэктеста)" + (". Новые сигналы стратегии не берём, пока её не уберут из reports/live/strategy_stop.json."
                                                          if r["статус"] == "стоп" else "."))
        if r["статус"] == "стоп":
            stopped.add(code)
    per = []
    for code in REF:
        for inst, g in lv[code].groupby("бумага"):
            rr = ref[code][ref[code]["instrument"] == inst.lower()]["unit"]
            per.append({"стратегия": code, "бумага": inst, "сделок": len(g), "сумма": g["unit"].sum(), "средняя": g["unit"].mean(),
                        "средняя бэктеста": rr.mean() if len(rr) else np.nan})
    ex, alt = execution(), alternatives()
    if ex.get("сделки"):
        e = pd.DataFrame(ex["сделки"])
        done = e[e["исполнено"]]
        for code, g in done.groupby("стратегия"):
            if len(g) >= MIN_TRADES and g["проскальзывание входа, %"].mean() > 0.15 and old.get("исп " + code) != "внимание":
                msgs.append(f"⚠️ Исполнение {code} хуже симуляции: проскальзывание на входе {g['проскальзывание входа, %'].mean():+.2f}% в среднем "
                            f"({len(g)} сделок), в бэктесте заложено 0.05%.")
                old["исп " + code] = "внимание"
    STOP.write_text(json.dumps(sorted(stopped), ensure_ascii=False))
    STATUS.write_text(json.dumps({**old, **{r["стратегия"]: r["статус"] for r in rows}}, ensure_ascii=False))
    write_doc(pd.DataFrame(rows), pd.DataFrame(per), ex, alt, stopped)
    for m in msgs:
        notify.send(m)
        print(m)
    return 0


def write_doc(s: pd.DataFrame, per: pd.DataFrame, ex: dict, alt: pd.DataFrame, stopped: set) -> None:
    f = lambda v, d=2: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{d}f}"  # noqa: E731
    now = pd.Timestamp.now("Europe/Moscow").tz_localize(None)
    md = ["# Сверка живой торговли с симуляцией", "", f"На {now:%d.%m.%Y %H:%M}. Сделки стратегий на живых данных с {START:%d.%m.%Y} против "
          "бэктеста той же стратегии на тех же бумагах (проверка 2021–2026), результат на единицу позиции после издержек. "
          "p — доля случайных выборок бэктеста из стольких же сделок со средней не выше живой: ниже 5% — «внимание», ниже 1% — «стоп».",
          "", "| стратегия | сделок | средняя, % | в бэктесте, % | сумма, % | p | убытков подряд (предел) | статус |", "|---|---|---|---|---|---|---|---|"]
    for r in s.to_dict("records"):
        streak = f"{r.get('убытков подряд', '—')} ({r.get('предел серии', 0):.0f})" if "убытков подряд" in r and pd.notna(r.get("убытков подряд")) else "—"
        md.append(f"| {r['стратегия']} | {r['сделок']} | {f(r['средняя'])} | {f(r['средняя бэктеста'])} | {f(r['сумма'], 1)} | "
                  f"{'—' if pd.isna(r['p']) else f'{r['p']:.3f}'} | {streak} | {r['статус']} |")
    md += ["", f"Остановлены сверкой (счёт не берёт их сигналы): {', '.join(sorted(stopped)) or 'нет'} — reports/live/strategy_stop.json.", ""]
    if len(per):
        md += ["## По бумагам", "", "| стратегия | бумага | сделок | сумма, % | средняя, % | в бэктесте, % |", "|---|---|---|---|---|---|"]
        md += [f"| {r['стратегия']} | {r['бумага']} | {r['сделок']} | {f(r['сумма'], 1)} | {f(r['средняя'])} | {f(r['средняя бэктеста'])} |"
               for r in per.to_dict("records")]
        md.append("")
    md += ["## Наше исполнение", ""]
    if not ex.get("сигналов"):
        md.append("Сигналов на счёт ещё не было.")
    elif ex.get("ошибка брокера"):
        md.append(f"Сигналов на счёт: {ex['сигналов']}; сделки у брокера не прочитаны: {ex['ошибка брокера']}.")
    else:
        e = pd.DataFrame(ex["сделки"])
        done = e[e["исполнено"]]
        md.append(f"Сигналов на счёт: {ex['сигналов']}, исполнено: {len(done)}; проскальзывание на входе в среднем "
                  f"{f(done['проскальзывание входа, %'].mean()) if len(done) else '—'}%, опоздание {done['опоздание, мин'].median() if len(done) else '—'} мин.")
    md += ["", "## Альтернативные сделки (не взяты на счёт)", ""]
    if alt.empty:
        md.append("Пока нет.")
    else:
        md.append(f"Сигналов: {len(alt)}; закрыто: {alt['результат %'].notna().sum()}, сумма результата {f(alt['результат %'].sum(), 1)}% на позицию "
                  "(сколько упустили — или от чего ушли).")
        md += ["", "| причина | сигналов | закрыто | сумма, % |", "|---|---|---|---|"]
        for why, g in alt.groupby(alt["причина"].str.split(":").str[0]):
            md.append(f"| {why} | {len(g)} | {g['результат %'].notna().sum()} | {f(g['результат %'].sum(), 1)} |")
    md.append("")
    DOC.write_text("\n".join(md), encoding="utf-8")
    print(f"→ {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
