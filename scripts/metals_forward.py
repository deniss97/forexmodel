"""Металлы после конца истории: приблизительный форвард-тест по часовым фьючерсам COMEX.

Свежего спота XAG/USD, XAU/USD у нас нет (история до 29.05.2026). Бесплатно есть часовые фьючерсы
COMEX (Yahoo: SI=F, GC=F) за два года — data/raw/live/{XAG,XAU}_fut_1h.csv; для сравнения тот же спот,
собранный в часы, — data/raw/live/{XAG,XAU}_spot_1h.csv. Правило — база B01 (импульс 6 ч ≥ 3 ATR по
тренд-гейту C, стоп и трейлинг 6 ATR, ≤ 240 ч), обе стороны, издержки базы (CFD). На часовых данных
вход — открытие следующего часа (open_delay 60), стоп проверяется по часовым high/low.

Ступени: (1) на общем отрезке 2024-12 … 2026-05 сравнить бэктест по минутам спота, правило по часам спота
и по часам фьючерса — насколько оценка по часам и по фьючерсу близка к настоящей; (2) свежий период
2026-05-29 … сейчас — только фьючерс.

    python scripts/metals_forward.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from forexmodel.pipelines.dataset import build_dataset
from forexmodel.simulation.costs import profile_entry, recost
from forexmodel.simulation.signals import build_signal_column
from forexmodel.simulation.simulator import simulate_trades
from live_signals import instrument_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "forward_test"
WF = ROOT / "reports" / "walk_forward"
START, SPLIT = pd.Timestamp("2024-12-01"), pd.Timestamp("2026-05-29")
METALS = {"silver": "XAG", "gold": "XAU"}


def hourly_trades(inst: str, csv: str, live: dict) -> pd.DataFrame:
    spec = {"config": "configs/silver.yaml", "csv": csv, "venue": "CFD"}
    last = pd.Timestamp(pd.read_csv(ROOT / csv, usecols=["time"]).iloc[-1, 0])
    cfg = instrument_cfg(spec, live, START, last + pd.Timedelta(days=1))
    cfg.simulation.open_delay_minutes = 60
    ds = build_dataset(cfg)
    sig = build_signal_column(ds.test.copy(), cfg)
    t, _ = simulate_trades(sig, ds.minute_slice("test"), cfg)
    t = recost(t, profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst))
    horizon = pd.Timedelta(minutes=cfg.simulation.horizon_minutes)
    t["открыта"] = (t["exit_reason"] == "timeout") & (t["close_dt"] >= last) & (t["close_dt"] - t["open_dt"] < horizon)
    return t


def stats(t: pd.DataFrame) -> dict:
    p = t["profit_pct"]
    loss = -p[p < 0].sum()
    return {"сделок": len(p), "итог %": round(p.sum(), 1), "средняя %": round(p.mean(), 3) if len(p) else np.nan,
            "PF": round(p[p > 0].sum() / loss, 2) if loss > 0 else np.nan, "выигрышей": round((p > 0).mean(), 2) if len(p) else np.nan}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    live = yaml.safe_load((ROOT / "configs" / "live.yaml").read_text(encoding="utf-8"))
    lines, fresh = [], []
    for inst, code in METALS.items():
        bt = pd.read_csv(WF / f"{inst}_costs__impulse_6_3_h240_tr6_trades.csv", parse_dates=["open_dt", "close_dt"])
        bt = recost(bt, profile_entry(ROOT / "configs" / "costs" / "base.yaml", inst))
        sp = hourly_trades(inst, f"data/raw/live/{code}_spot_1h.csv", live)
        fu = hourly_trades(inst, f"data/raw/live/{code}_fut_1h.csv", live)
        win = lambda t: t[(t["open_dt"] >= START) & (t["close_dt"] < SPLIT)]  # noqa: E731
        rows = [{"расчёт": "спот, минуты (бэктест)", **stats(win(bt))}, {"расчёт": "спот, часы", **stats(win(sp))},
                {"расчёт": "фьючерс, часы", **stats(win(fu))}]
        # совпадение сделок: вход в пределах 2 часов и в ту же сторону
        def match(a, b):
            if not len(a):
                return np.nan
            hit = [((b["side"] == r.side) & ((b["open_dt"] - r.open_dt).abs() <= pd.Timedelta(hours=2))).any() for r in a.itertuples()]
            return round(float(np.mean(hit)), 2)
        lines.append(f"\n== {inst}: общий отрезок {START:%Y-%m-%d} … {SPLIT:%Y-%m-%d}\n" + pd.DataFrame(rows).to_string(index=False)
                     + f"\nсделки бэктеста, найденные по часам спота: {match(win(bt), win(sp))}; по часам фьючерса: {match(win(bt), win(fu))}")
        f = fu[fu["open_dt"] >= SPLIT].assign(instrument=inst)
        fresh.append(f)
        lines.append(f"свежий период по фьючерсу ({SPLIT:%Y-%m-%d} … {fu['close_dt'].max():%Y-%m-%d}): {stats(f)}; открыто {int(f['открыта'].sum())}")
    F = pd.concat(fresh, ignore_index=True).sort_values("open_dt")
    F.to_csv(OUT / "metals_fresh_trades.csv", index=False)
    lines.append("\nСВЕЖИЕ СДЕЛКИ (фьючерс, часы):\n" + F[["instrument", "open_dt", "close_dt", "side", "open_price", "exit_price",
                                                          "exit_reason", "profit_pct", "открыта"]].round(3).to_string(index=False))
    lines.append(f"\nметаллы поровну, свежий период: {F.groupby('instrument')['profit_pct'].sum().sum() / 2:+.2f}%")
    text = "\n".join(lines)
    print(text)
    (OUT / "metals_summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
