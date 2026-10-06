"""Слой «досрочные закрытия» для карты сделок (docs/results/strategy_viz): где правило управления
позицией закрыло бы сделку и что было бы лучше — закрыть или держать.

Правило из scripts/position_management.py: в первый час сделки, где P(цель «ещё +3 ATR» раньше
стопа) по модели CatBoost (walk-forward, вне выборки, с 2017 года) ниже θ, сделка закрывается по
закрытию этого часа. Для каждой сделки карты и θ ∈ {0.2, 0.3} пишется: момент и цена досрочного
выхода, результат досрочного выхода после комиссии, часов в сделке. Сделки карты (штатный
симулятор) сопоставляются со строками «сделка × час» (быстрый движок) по времени входа (±90 мин).

Данные карты дополняются ключом `early`; прежние ключи не меняются.

    python scripts/build_early_exit_layer.py      # после position_management.py и build_strategy_viz.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results" / "strategy_viz" / "data"
ROWS = ROOT / "reports" / "patterns" / "position_rows.pkl"
THETAS = (0.2, 0.3)
COMMISSION = 0.04


def main() -> int:
    df = pd.read_pickle(ROWS)
    df = df[df["p_model"].notna()].copy()
    df["open_time"] = df["time"] + pd.Timedelta(minutes=61)
    summary = []
    for p in sorted(DATA.glob("*.json")):
        if p.stem in ("manifest", "sandbox"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        g_inst = df[df["inst"] == p.stem]
        if g_inst.empty:
            print(f"[{p.stem}] нет строк с вероятностями, пропуск")
            continue
        firsts = {}
        for th in THETAS:
            b = g_inst[g_inst["p_model"] < th].sort_values("hours_in").drop_duplicates("trade")
            firsts[th] = b.set_index("trade")
        opens = g_inst.drop_duplicates("trade").set_index("trade")["open_time"].sort_values()
        ot = opens.to_numpy()
        early = {str(th): {} for th in THETAS}
        for r in d["trades"]:
            t_open = np.datetime64(pd.Timestamp(r[0], unit="s"))
            k = int(np.searchsorted(ot, t_open))
            cand = [i for i in (k - 1, k) if 0 <= i < len(ot)]
            if not cand:
                continue
            i = min(cand, key=lambda i: abs(ot[i] - t_open))
            if abs(ot[i] - t_open) > np.timedelta64(90, "m"):
                continue
            trade = opens.index[i]
            for th in THETAS:
                f = firsts[th]
                if trade not in f.index:
                    continue
                x = f.loc[trade]
                ts = int((pd.Timestamp(r[0], unit="s") + pd.Timedelta(hours=float(x["hours_in"]))).timestamp())
                if ts >= r[1]:
                    continue                                   # сделка закрылась раньше, чем сработало бы правило
                res = float(x["now_gross"]) - COMMISSION
                price = r[3] * (1 + r[2] * float(x["now_gross"]) / 100)
                early[str(th)][str(r[0])] = [ts, round(price, d["decimals"]), round(res, 3), round(float(x["hours_in"]), 1)]
        d["early"] = {"source": "P модели CatBoost (вне выборки, с 2017)", "target_atr": 3, "commission": COMMISSION,
                      "cols": ["момент", "цена", "результат_%", "часов"], "by_theta": early}
        p.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        actual = {str(r[0]): r[5] for r in d["trades"]}
        for th in THETAS:
            e = early[str(th)]
            diff = [v[2] - actual[k] for k, v in e.items()]
            summary.append({"инструмент": p.stem, "θ": th, "сделок_с_вероятностью": int(g_inst["trade"].nunique()),
                            "закрыто_досрочно": len(e), "лучше_досрочно": int(sum(x > 0 for x in diff)),
                            "хуже_досрочно": int(sum(x < 0 for x in diff)), "разница_итога": round(sum(diff), 1)})
    s = pd.DataFrame(summary)
    print(s.to_string(index=False))
    print(s.groupby("θ")[["закрыто_досрочно", "лучше_досрочно", "хуже_досрочно", "разница_итога"]].sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
