"""H40: случайна ли разница между сделками у уровня и остальными (перестановочный тест).

Берёт признаки сделок из scripts/h40_trades.py (reports/h40_trades/trades_<тег>.csv), делит их на «большой уровень по ходу
сделки ближе порога» и «остальные» и считает, как часто такая же разница средних получается при случайном делении
(перестановки внутри года и бумаги — чтобы не ловить разницу рынка между годами).

    python scripts/h40_sig.py min_main 1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "h40_trades"
N = 5000


def pf(v: np.ndarray) -> float:
    pos, neg = v[v > 0].sum(), -v[v < 0].sum()
    return float(pos / neg) if neg > 0 else np.nan


def test(f: pd.DataFrame, thr: float, rng: np.random.Generator) -> dict:
    near = (f["d_block"] <= thr).to_numpy()
    v = f["profit_pct"].to_numpy(float)
    if near.sum() < 10:
        return {}
    obs = v[near].mean() - v[~near].mean()
    key = (f["year"].astype(str) + "/" + f["tk"]).to_numpy()
    idx = {k: np.flatnonzero(key == k) for k in np.unique(key)}
    cnt = 0
    for _ in range(N):
        perm = near.copy()
        for k, ii in idx.items():                       # перемешиваем метку группы внутри года и бумаги
            perm[ii] = rng.permutation(near[ii])
        d = v[perm].mean() - v[~perm].mean()
        cnt += abs(d) >= abs(obs)
    return {"порог_ATR": thr, "сделок_рядом": int(near.sum()), "сделок_остальных": int((~near).sum()),
            "PF_рядом": round(pf(v[near]), 3), "PF_остальные": round(pf(v[~near]), 3),
            "средняя_рядом": round(float(v[near].mean()), 3), "средняя_остальные": round(float(v[~near].mean()), 3),
            "разница": round(float(obs), 3), "p": round(float(cnt / N), 4)}


def main(argv: list[str]) -> int:
    tag = argv[0] if argv else "min_main"
    thrs = [float(x) for x in argv[1:]] or [0.5, 1.0]
    f = pd.read_csv(REP / f"trades_{tag}.csv")
    f["d_block"] = f["d_block"].fillna(np.inf)
    rng = np.random.default_rng(42)
    rows = []
    for thr in thrs:
        for lab, g in (("2022–2026", f), ("2022–2024", f[f["year"] <= 2024]), ("2025–2026", f[f["year"] >= 2025])):
            r = test(g, thr, rng)
            if r:
                rows.append({"период": lab, **r})
    t = pd.DataFrame(rows)
    t.to_csv(REP / f"sig_{tag}.csv", index=False)
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
