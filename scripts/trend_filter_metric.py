"""Качество тренд-фильтра отдельно от стратегии (гипотезы H37, H38 — к встрече в Казани).

Эталон «настоящего тренда» задним числом: знак хода цены за следующие H часов, если ход больше K ATR,
иначе «нет тренда» (0). Это заглядывание вперёд — только для оценки, не для торговли. Фильтр на часе t
знает лишь прошлое и говорит +1 / −1 / 0. Метрики, отдельно для выбора (до 2021) и проверки:
  * говорит      — доля часов, где фильтр не нейтрален;
  * верно        — доля часов с верным знаком среди часов, где и фильтр, и эталон не нули;
  * ошибок       — доля часов, где фильтр показывает против настоящего тренда (среди тех же часов);
  * ход, ATR     — средний ход за H часов в сторону фильтра (польза для следования тренду); у «всегда
                   лонг» это просто дрейф рынка — планка, которую фильтр должен превзойти.
Фильтры: гейт C стратегии (из свечей журнала), пересечение EMA 20/50, пробой канала Дончиана 48 ч,
наклон EMA 4-часовых свечей, два таймфрейма (4 ч задаёт направление, 1 ч подтверждает), «всегда лонг».
Данные — часовые свечи страницы журнала (docs/experiments/site/data/candles, scripts/build_journal_candles.py).

    python scripts/trend_filter_metric.py
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

ROOT = Path(__file__).resolve().parents[1]
CANDLES = ROOT / "docs" / "experiments" / "site" / "data" / "candles"
OUT = ROOT / "docs" / "results" / "trend_metric"
INST = ["silver", "gold", "sber", "gazp", "lkoh"]
H, K = 72, 3.0
BG, INK, INK2, GRID, BLUE, ORANGE, GRAY = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc", "#2459b3", "#c8641e", "#9aa3ae"


def load(inst: str) -> pd.DataFrame:
    c = json.loads((CANDLES / f"{inst}.json").read_text(encoding="utf-8"))
    t = pd.to_datetime(c["t0"] + np.cumsum(np.array(c["dt"])) * 3600, unit="s").astype("datetime64[ns]")
    d = pd.DataFrame({"time": t, "c": np.array(c["c"], float), "h": np.array(c["h"], float),
                      "l": np.array(c["l"], float), "atr": np.array(c["atr"], float)})
    g = np.zeros(len(d), dtype=int)
    for k, (i, v) in enumerate(c["gate"]):
        g[i:(c["gate"][k + 1][0] if k + 1 < len(c["gate"]) else len(d))] = v
    d["gate C"] = g
    return d


def filters(d: pd.DataFrame) -> dict[str, np.ndarray]:
    c = d["c"]
    ema20, ema50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    hi, lo = d["h"].rolling(48).max().shift(1), d["l"].rolling(48).min().shift(1)
    brk = pd.Series(np.where(c > hi, 1, np.where(c < lo, -1, np.nan)), index=d.index).ffill().fillna(0)
    four = d.set_index("time")["c"].resample("4h").last().dropna()
    e4 = four.ewm(span=10, adjust=False).mean()
    slope4 = np.sign(e4.diff())
    slope4.index = slope4.index.astype("datetime64[ns]")
    slope4.index = slope4.index + pd.Timedelta("4h")                 # известно после закрытия 4-часовой свечи
    s4 = pd.merge_asof(d[["time"]], slope4.rename("s").reset_index().rename(columns={"index": "time"}).sort_values("time"),
                       on="time", direction="backward")["s"].fillna(0).to_numpy()
    s1 = np.sign(c - ema20).to_numpy()
    return {
        "gate C": d["gate C"].to_numpy(),
        "EMA 20/50": np.sign(ema20 - ema50).to_numpy(),
        "канал 48 ч": brk.to_numpy(),
        "наклон EMA 4ч": s4,
        "4ч + 1ч": np.where(s4 == s1, s4, 0),
        "всегда лонг": np.ones(len(d)),
    }


def score(f: np.ndarray, truth: np.ndarray, fwd: np.ndarray) -> dict:
    ok = np.isfinite(fwd)
    f, truth, fwd = f[ok], truth[ok], fwd[ok]
    both = (f != 0) & (truth != 0)
    return {"говорит": float((f != 0).mean()), "верно": float((f[both] == truth[both]).mean()) if both.any() else np.nan,
            "ошибок": float((f[both] == -truth[both]).mean()) if both.any() else np.nan,
            "ход, ATR": float(np.mean(f[f != 0] * fwd[f != 0])) if (f != 0).any() else np.nan}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rows = []
    for inst in INST:
        d = load(inst)
        fwd = ((d["c"].shift(-H) - d["c"]) / d["atr"]).to_numpy()
        truth = np.where(fwd > K, 1, np.where(fwd < -K, -1, 0))
        F = filters(d)
        for per, q in (("выбор", d["time"] < "2021-01-01"), ("проверка", d["time"] >= "2021-01-01")):
            m = q.to_numpy()
            for name, f in F.items():
                rows.append({"инструмент": inst, "период": per, "фильтр": name, **score(f[m], truth[m], fwd[m])})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "by_instrument.csv", index=False)
    S = R.groupby(["период", "фильтр"], sort=False)[["говорит", "верно", "ошибок", "ход, ATR"]].mean().round(3)
    text = (f"Эталон: знак хода за {H} ч, если больше {K} ATR. Средние по {', '.join(INST)}.\n\n" + S.to_string()
            + "\n\nПо инструментам (проверка):\n"
            + R[R["период"] == "проверка"].pivot_table(index="фильтр", columns="инструмент", values="ход, ATR", sort=False).round(3).to_string())
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")

    # график: верно (точность направления) и ход в сторону фильтра, проверка
    x = S.loc["проверка"]
    f, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)
    for ax, col, title in ((axes[0], "верно", "доля верного направления"), (axes[1], "ход, ATR", f"ход за {H} ч по фильтру, ATR")):
        ax.set_facecolor(BG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="x", color=GRID)
        ax.tick_params(colors=INK2, labelsize=12, length=0)
        k = np.arange(len(x))
        ax.barh(k, x[col], color=[BLUE if n == "gate C" else (GRAY if n == "всегда лонг" else ORANGE) for n in x.index], height=0.6)
        for kk, v in zip(k, x[col]):
            ax.text(v, kk, f" {v:.2f}", va="center", fontsize=12, color=INK)
        if col == "верно":
            ax.axvline(0.5, color=INK2, linewidth=1.5, linestyle="--")
            ax.text(0.5, len(x) - 0.4, " монетка", fontsize=11, color=INK2, va="bottom")
        ax.set_yticks(k)
        ax.set_yticklabels(x.index, fontsize=12, color=INK)
        ax.invert_yaxis()
        ax.set_title(f"2021–2026: {title}", fontsize=13, color=INK, loc="left")
    f.tight_layout()
    f.savefig(OUT / "filters.png", dpi=140, facecolor=BG)
    plt.close(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
