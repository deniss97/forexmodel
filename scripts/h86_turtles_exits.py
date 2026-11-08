"""H86 (скептик 08.11 к H47): пробой 55 дней с выходами настоящих «Черепах» против случайных входов в том же тренде. Отчёт —
docs/results/h86_turtles_exits.md.

Дневные свечи — из часовых баров (с поправкой на дивиденды; reports/night_search/feat и reports/new_stocks/feat), гейт C — на
последнем часе дня. Вход в лонг по открытию следующего дня, если закрытие выше максимума 55 предыдущих дней и гейт C смотрит
вверх. Выход: стоп 2 × ATR(20 дней) от входа (по стопу или по открытию, если гэп ниже), или закрытие ниже минимума 20
предыдущих дней — по открытию следующего дня; без срока; одна позиция на бумагу. Издержки — 0.2% за круг (биржа).
Случайные входы — столько же по бумаге и периоду, в дни с гейтом C вверх, с теми же выходами; 500 наборов.
Критерий (банк, до расчёта): на 18 акциях без GAZP, SBER, LKOH PF лучше 95% случайных наборов в обоих периодах и выше 1.1.

    python scripts/h86_turtles_exits.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.evaluation.experiments import TEST_START

from experiments import ROOT

DOC = ROOT / "docs" / "results" / "h86_turtles_exits.md"
FEAT = {**{i: ROOT / "reports" / "night_search" / "feat" for i in ("gazp", "sber", "lkoh")},
        **{i: ROOT / "reports" / "new_stocks" / "feat" for i in ("afks", "aflt", "alrs", "chmf", "gmkn", "irao", "magn", "moex", "mtss",
                                                                  "nlmk", "nvtk", "plzl", "rosn", "rtkm", "sngsp", "tatn", "trnfp", "vtbr")}}
MAIN = ("gazp", "sber", "lkoh")
COST = 0.2
REPS = 500
MARK = ("<!-- вывод -->", "<!-- /вывод -->")


def pf(x) -> float:
    x = np.asarray(x, float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float("nan")


def daily(inst: str) -> pd.DataFrame:
    d = pd.read_pickle(FEAT[inst] / f"{inst}.pkl")
    b = d["bars"][["time", "open", "high", "low", "close"]].copy()
    b["time"] = pd.to_datetime(b["time"])
    b["gate"] = np.asarray(d["gates"][2], dtype=float)
    g = b.groupby(b["time"].dt.normalize())
    D = pd.DataFrame({"o": g["open"].first(), "h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last(), "gate": g["gate"].last()})
    D = D[D.index >= "2015-06-01"]
    tr = np.maximum(D["h"] - D["l"], np.maximum((D["h"] - D["c"].shift()).abs(), (D["l"] - D["c"].shift()).abs()))
    D["atr"] = tr.rolling(20).mean()
    D["hi55"] = D["h"].shift(1).rolling(55).max()
    D["lo20"] = D["l"].shift(1).rolling(20).min()
    return D


def exits(D: pd.DataFrame, entries: np.ndarray) -> np.ndarray:
    """Результат сделок (%, до издержек) для входов по открытию дня e."""
    o, l, c, atr, lo20 = (D[k].to_numpy(float) for k in ("o", "l", "c", "atr", "lo20"))
    n = len(D)
    brk = c < lo20                                           # закрытие ниже минимума 20 дней → выход по открытию завтра
    nxt = np.full(n, n - 1)
    last = n - 1
    for k in range(n - 1, -1, -1):
        if brk[k]:
            last = k
        nxt[k] = last
    out = np.full(len(entries), np.nan)
    for j, e in enumerate(entries):
        if e >= n - 1 or not np.isfinite(atr[e - 1]):
            continue
        E = o[e]
        stop = E - 2 * atr[e - 1]
        nb = nxt[e]
        w = l[e:nb + 1] <= stop
        if w.any():
            k = e + int(w.argmax())
            X = min(o[k], stop) if k > e else stop
        else:
            X = o[min(nb + 1, n - 1)] if nb < n - 1 else c[-1]
        out[j] = (X / E - 1) * 100
    return out


def signals(D: pd.DataFrame) -> np.ndarray:
    """Дни входа (по открытию) — по одному сигналу, пока позиция открыта, новые не берутся."""
    c, hi55, gate = D["c"].to_numpy(float), D["hi55"].to_numpy(float), D["gate"].to_numpy(float)
    sig = np.flatnonzero((c > hi55) & (gate == 1))
    ent, busy_until = [], -1
    o, l, atr, lo20 = (D[k].to_numpy(float) for k in ("o", "l", "atr", "lo20"))
    for s in sig:
        e = s + 1
        if e <= busy_until or e >= len(D) - 1:
            continue
        ent.append(e)
        # конец позиции — первый выход по стопу или каналу
        stop = o[e] - 2 * atr[e - 1] if np.isfinite(atr[e - 1]) else -np.inf
        k = e
        while k < len(D) - 1 and not (l[k] <= stop) and not (D["c"].iat[k] < lo20[k]):
            k += 1
        busy_until = k + 1
    return np.array(ent, dtype=int)


def main() -> int:
    rng = np.random.default_rng(86)
    rows, real, rand = [], {}, {}
    for inst in FEAT:
        D = daily(inst)
        ent = signals(D)
        gross = exits(D, ent)
        net = gross - COST
        when = D.index.to_numpy()[ent]
        for per, pm in (("выбор", when < np.datetime64(TEST_START)), ("проверка", when >= np.datetime64(TEST_START))):
            x = net[pm & np.isfinite(net)]
            real[(inst, per)] = x
            pool = np.flatnonzero((D["gate"].to_numpy() == 1) & ((D.index < TEST_START) == (per == "выбор"))
                                  & np.isfinite(D["atr"].to_numpy()) & (np.arange(len(D)) < len(D) - 2)) + 1
            sims = np.full((REPS, len(x)), np.nan)
            if len(x) and len(pool):
                pick = rng.choice(pool, size=(REPS, len(x)), replace=True)
                sims = exits(D, pick.ravel()).reshape(REPS, len(x)) - COST
            rand[(inst, per)] = sims
            rows.append({"бумага": inst, "период": per, "сделок": len(x), "PF": pf(x), "средняя %": float(np.mean(x)) if len(x) else np.nan})
    G = []
    for name, insts in (("18 акций без GAZP, SBER, LKOH", [i for i in FEAT if i not in MAIN]), ("GAZP, SBER, LKOH", list(MAIN))):
        for per in ("выбор", "проверка"):
            x = np.concatenate([real[(i, per)] for i in insts])
            R = np.concatenate([rand[(i, per)] for i in insts], axis=1)
            pos = np.where(R > 0, R, 0).sum(axis=1)
            neg = -np.where(R < 0, R, 0).sum(axis=1)
            pr = pos / np.where(neg > 0, neg, np.nan)
            G.append({"набор": name, "период": per, "сделок": len(x), "PF": pf(x), "средняя %": float(x.mean()),
                      "PF случайных: медиана": float(np.nanmedian(pr)), "95%": float(np.nanpercentile(pr, 95)),
                      "случайных ниже": float(np.mean(pr < pf(x)))})
    G = pd.DataFrame(G)
    pd.set_option("display.width", 220)
    print(G.round(3).to_string())
    g = G[G["набор"].str.startswith("18")].set_index("период")
    ok = bool(all(g.loc[p, "случайных ниже"] >= 0.95 and g.loc[p, "PF"] > 1.1 for p in ("выбор", "проверка")))
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Пробой 55 дней с выходами настоящих «Черепах» (H86)", "",
          "Гипотеза из скептика к H47: там пробой проверялся с короткими выходами S42, а не по системе «Черепах». Здесь — дневные свечи "
          "21 акции, вход по пробою 55-дневного максимума при гейте C вверх, стоп 2 × ATR(20 дней), выход по минимуму 20 дней, без "
          "срока, одна позиция на бумагу, издержки 0.2% за круг. Расчёт — `scripts/h86_turtles_exits.py`.", "",
          "**Критерий (банк, до расчёта):** на 18 акциях без GAZP, SBER, LKOH PF лучше 95% случайных входов в тренде с теми же выходами "
          f"в обоих периодах и выше 1.1. Случайных наборов — {REPS}.", "", verdict, "",
          "| набор | период | сделок | PF | средняя, % | PF случайных: медиана | 95% случайных | случайных ниже |", "|---|---|---|---|---|---|---|---|"]
    for _, r in G.iterrows():
        md.append(f"| {r['набор']} | {r['период']} | {r['сделок']} | {r['PF']:.2f} | {r['средняя %']:+.2f} | {r['PF случайных: медиана']:.2f} | "
                  f"{r['95%']:.2f} | {r['случайных ниже'] * 100:.0f}% |")
    md += ["", f"Критерий выполнен: {'да' if ok else 'нет'}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("критерий:", ok, "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
