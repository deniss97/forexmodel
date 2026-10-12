"""Форвард-тест лучшей стратегии на свежих данных: сколько бы она заработала после конца истории.

Стратегия S25: акции на Мосбирже только в лонг (импульс 6 ч ≥ 3 ATR по тренд-гейту C, стоп и трейлинг
6 ATR, выход ≤ 240 ч), издержки биржи (configs/costs/exchange.yaml: комиссия 0.1% за круг, проскальзывание
0.05% на сторону, лонг без переноса). Данные — история + свежие минуты с Мосбиржи
(data/raw/live, scripts/fetch_moex_iss.py), дивиденды с отсечками 2026 (fetch_dividends.py --live).
Тот же код, что у walk-forward и живого контура (scripts/live_signals.py): окно 2026-01-01 … сейчас.

Две проверки:
  1. до конца истории — сделки должны совпасть со сделками бэктеста (тот же код, другой файл данных);
  2. после конца истории — честный тест вне выборки: ни один день не участвовал ни в выборе, ни в проверке.

    python scripts/forward_test.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from forexmodel.pipelines.dataset import build_dataset  # noqa: E402
from forexmodel.simulation.costs import profile_entry, recost  # noqa: E402
from forexmodel.simulation.signals import build_signal_column  # noqa: E402
from forexmodel.simulation.simulator import simulate_trades  # noqa: E402
from live_signals import FLAT, SHORT, instrument_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "forward_test"
WF = ROOT / "reports" / "walk_forward"
STOCKS = {"sber": "SBER_1m_15-26.csv", "gazp": "GAZP_1m_15-26_upd.csv", "lkoh": "LKOH_1m_15-26.csv"}
START = pd.Timestamp("2026-01-01")
BG, INK, INK2, GRID, BLUE, ORANGE, GRAY = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc", "#2459b3", "#c8641e", "#9aa3ae"
COLORS = {"sber": "#2459b3", "gazp": "#c8641e", "lkoh": "#1d7a55"}


def run(inst: str, live: dict) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    spec = live["instruments"][inst]
    hist_end = pd.Timestamp(pd.read_csv(ROOT / "data" / "raw" / STOCKS[inst], usecols=["begin"]).iloc[-1, 0])
    last = pd.Timestamp(pd.read_csv(ROOT / spec["csv"], usecols=["begin"]).iloc[-1, 0])
    cfg = instrument_cfg(spec, live, START, last + pd.Timedelta(days=1))
    ds = build_dataset(cfg)
    sig = build_signal_column(ds.test.copy(), cfg)
    sig = sig[sig["time"] + pd.Timedelta(minutes=60) <= last + pd.Timedelta(minutes=1)]
    sig.loc[pd.to_numeric(sig["final_class"], errors="coerce") == SHORT, "final_class"] = FLAT     # только лонг
    t, _ = simulate_trades(sig, ds.minute_slice("test"), cfg)
    t = recost(t, profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))
    t["instrument"] = inst
    horizon = pd.Timedelta(minutes=cfg.simulation.horizon_minutes)
    t["открыта"] = (t["exit_reason"] == "timeout") & (t["close_dt"] >= last - pd.Timedelta(minutes=1)) & (t["close_dt"] - t["open_dt"] < horizon)
    t["период"] = np.where(t["open_dt"] > hist_end, "свежий", "история")
    return t, hist_end, last


def backtest(inst: str) -> pd.DataFrame:
    b = pd.read_csv(WF / f"{inst}_costs__impulse_6_3_h240_tr6_trades.csv", parse_dates=["open_dt", "close_dt"])
    b = b[(b["side"] == "buy") & (b["open_dt"] >= START)]
    return recost(b, profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    lines, parts, ends = [], [], {}
    for inst in STOCKS:
        t, hist_end, last = run(inst, live)
        ends[inst] = (hist_end, last)
        parts.append(t)
        # 1. сверка с бэктестом до конца истории (сделки, закрытые до конца истории)
        b = backtest(inst)
        b = b[b["close_dt"] <= hist_end]
        h = t[(t["период"] == "история") & (t["close_dt"] <= hist_end)]
        m = h.merge(b, on="open_dt", how="outer", suffixes=("", "_bt"), indicator=True)
        same = m[m["_merge"] == "both"]
        diff = (same["profit_pct"] - same["profit_pct_bt"]).abs()
        lines.append(f"[{inst}] история до {hist_end:%Y-%m-%d}: сделок контур {len(h)}, бэктест {len(b)}, совпали по входу {len(same)}, "
                     f"расхождение результата макс {diff.max() if len(diff) else float('nan'):.4f} п.п.; "
                     f"только в контуре {int((m['_merge'] == 'left_only').sum())}, только в бэктесте {int((m['_merge'] == 'right_only').sum())}")
    T = pd.concat(parts, ignore_index=True)
    T.to_csv(OUT / "trades.csv", index=False)
    F = T[T["период"] == "свежий"].sort_values("open_dt")
    lines.append("\nСВЕЖИЙ ПЕРИОД (после конца истории; открытые сделки — по последней цене):")
    cols = ["instrument", "open_dt", "close_dt", "open_price", "exit_price", "exit_reason", "profit_pct", "открыта"]
    lines.append(F[cols].round(3).to_string(index=False))
    # итоги: по бумагам и портфель (бумаги поровну)
    rows = []
    for inst, g in F.groupby("instrument"):
        p = g["profit_pct"]
        rows.append({"бумага": inst, "с": f"{ends[inst][0]:%Y-%m-%d}", "по": f"{ends[inst][1]:%Y-%m-%d}", "сделок": len(p),
                     "открыто": int(g["открыта"].sum()), "итог %": p.sum(), "средняя %": p.mean(), "выигрышей": (p > 0).mean()})
    S = pd.DataFrame(rows)
    lines.append("\nИТОГ ПО БУМАГАМ:\n" + S.round(3).to_string(index=False))
    port = S["итог %"].sum() / len(STOCKS)
    lines.append(f"\nПортфель из 3 бумаг поровну: {port:+.2f}% за свежий период; закрытых сделок {int((~F['открыта']).sum())}, открыто {int(F['открыта'].sum())}")
    # ожидание по бэктесту: S25 на проверке 2021–2026, средняя сделка по бумаге
    exp = []
    for inst in STOCKS:
        b = pd.read_csv(WF / f"{inst}_costs__impulse_6_3_h240_tr6_trades.csv", parse_dates=["open_dt", "close_dt"])
        b = recost(b[(b["side"] == "buy") & (b["open_dt"] >= "2021-01-01")], profile_entry(ROOT / "configs" / "costs" / "exchange.yaml", inst))
        days = (b["open_dt"].max() - b["open_dt"].min()).days
        exp.append({"бумага": inst, "средняя % (2021–26)": b["profit_pct"].mean(), "сделок в месяц": len(b) / days * 30.4,
                    "выигрышей": (b["profit_pct"] > 0).mean(), "ст.откл. сделки": b["profit_pct"].std()})
    E = pd.DataFrame(exp)
    lines.append("\nОЖИДАНИЕ ПО БЭКТЕСТУ (S25, проверка 2021–2026):\n" + E.round(3).to_string(index=False))
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")

    # график: накопленный итог свежих сделок по бумагам и портфель
    f, ax = plt.subplots(figsize=(12, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID)
    ax.tick_params(colors=INK2, labelsize=12, length=0)
    for inst, g in F.groupby("instrument"):
        g = g.sort_values("close_dt")
        y = g["profit_pct"].cumsum()
        ax.step(g["close_dt"], y, where="post", color=COLORS[inst], linewidth=2.2, label=f"{inst.upper()} {y.iloc[-1]:+.1f}%")
    g = F.sort_values("close_dt")
    ax.step(g["close_dt"], (g["profit_pct"] / len(STOCKS)).cumsum(), where="post", color=INK, linewidth=3, label=f"портфель (поровну) {port:+.1f}%")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("накопленный итог, %", fontsize=13, color=INK2)
    ax.legend(frameon=False, fontsize=12, labelcolor=INK, loc="upper left")
    f.tight_layout()
    f.savefig(OUT / "fresh_equity.png", dpi=140, facecolor=BG)
    plt.close(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
