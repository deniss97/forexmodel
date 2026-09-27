"""Признаки мета-модели, которых нет у primary.

Мета-модель вырождалась в константу и после исправления OOF (docs/results/
model_improvements.md, §3): она видела ровно те же признаки, что primary, плюс её
вероятности, и сказать про исход сделки ей было нечего. Здесь три группы того,
чего у primary нет по построению.

1. Признаки ОТНОСИТЕЛЬНО НАПРАВЛЕНИЯ сделки (`mctx_s_*`): ход за последние бары,
   свеча, растяжение, тренд, наклон — умноженные на знак сигнала (+1 лонг,
   −1 шорт). Primary видит их без знака, и дереву меты, чтобы понять «хороший ли
   это ЛОНГ», пришлось бы каждый раз расщеплять по oof_pred. `mctx_entry_class` —
   класс входа из разбора сделок: 2 продолжение (≥ 0.3 ATR по ходу за 2 бара),
   1 плоско, −1 откат (0.3–1 ATR против), −2 нож (> 1 ATR против).

2. ИСХОД ПОСЛЕДНИХ СДЕЛОК (`mctx_recent_*`): winrate и средний результат
   гипотетических сделок по предыдущим сигналам, которые ЗАКРЫЛИСЬ до момента
   решения (закрытие бара сигнала). Всё и в своём направлении. Это то, что
   трейдер видит на графике и модель не видит никогда: «последние сигналы
   подряд выбивает».

3. Режимные `vr4_*` / `acf1_*` — считаются в сборке признаков (features/extension.py),
   здесь только подключаются.

Каузальность: сделка учитывается, только если её close_dt < time + bar,
где time — начало бара сигнала, а решение принимается на его закрытии.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SIGNED_SOURCES",
    "add_direction_context",
    "add_recent_outcomes",
    "context_columns",
]

LONG, SHORT = 2, 0

#: признак-источник -> имя знакового признака
SIGNED_SOURCES = {
    "log_return": "mctx_s_bar",
    "ret_lag_1_atr": "mctx_s_ret1",
    "ret_lag_2_atr": "mctx_s_ret2",
    "ret_lag_5_atr": "mctx_s_ret5",
    "runup_12": "mctx_s_runup12",
    "runup_24": "mctx_s_runup24",
    "z_close_24": "mctx_s_z24",
    "trend_4h": "mctx_s_trend",
    "slope_atr_4h": "mctx_s_slope",
    "trend_gate": "mctx_s_gate",
    "macd_hist_atr": "mctx_s_macd",
    "di_spread": "mctx_s_di",
}


def _direction(pred: pd.Series) -> np.ndarray:
    p = pd.to_numeric(pred, errors="coerce").to_numpy()
    return np.where(p == LONG, 1.0, np.where(p == SHORT, -1.0, np.nan))


def add_direction_context(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Знаковые признаки и класс входа. Для неполярных строк — NaN."""
    out = df.copy()
    sgn = _direction(out[pred_col])
    for src, name in SIGNED_SOURCES.items():
        if src in out.columns:
            out[name] = sgn * pd.to_numeric(out[src], errors="coerce").to_numpy()
    # растяжение в сторону сделки: для лонга — от минимума, для шорта — от максимума
    for w in (6, 24):
        lo, hi = f"ext_from_low_{w}", f"ext_from_high_{w}"
        if lo in out.columns and hi in out.columns:
            out[f"mctx_ext_{w}"] = np.where(sgn > 0, out[lo], np.where(sgn < 0, out[hi], np.nan))
            out[f"mctx_room_{w}"] = np.where(sgn > 0, out[hi], np.where(sgn < 0, out[lo], np.nan))
    if "ret_lag_2_atr" in out.columns:
        m = sgn * pd.to_numeric(out["ret_lag_2_atr"], errors="coerce").to_numpy()
        out["mctx_entry_class"] = np.where(
            np.isnan(m), np.nan, np.where(m >= 0.3, 2.0, np.where(m <= -1.0, -2.0, np.where(m <= -0.3, -1.0, 1.0)))
        )
    return out


def add_recent_outcomes(
    df: pd.DataFrame,
    outcomes: Optional[pd.DataFrame],
    pred_col: str,
    windows: Sequence[int] = (10, 30),
    bar: pd.Timedelta = pd.Timedelta("1h"),
    time_col: str = "time",
) -> pd.DataFrame:
    """Winrate и средний результат последних N закрытых сделок до момента решения.

    `outcomes`: сделки по сигналам с колонками close_dt, profit_pct, side
    ('buy'/'sell'). Сделки с одинаковым close_dt упорядочены стабильно; окно —
    последние N по времени закрытия.
    """
    out = df.copy()
    decision = pd.to_datetime(out[time_col]).to_numpy() + np.timedelta64(int(bar.total_seconds()), "s")
    sgn = _direction(out[pred_col])

    def fill(prefix: str, sub: pd.DataFrame, rows: np.ndarray) -> None:
        sub = sub.sort_values("close_dt", kind="stable")
        closes = pd.to_datetime(sub["close_dt"]).to_numpy()
        pnl = sub["profit_pct"].to_numpy(dtype=float)
        cs_pnl = np.concatenate([[0.0], np.cumsum(pnl)])
        cs_win = np.concatenate([[0.0], np.cumsum(pnl > 0)])
        idx = np.searchsorted(closes, decision[rows], side="left")  # закрытые СТРОГО до решения
        for n in windows:
            lo = np.maximum(idx - n, 0)
            cnt = (idx - lo).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.loc[out.index[rows], f"{prefix}_win_{n}"] = np.where(cnt > 0, (cs_win[idx] - cs_win[lo]) / cnt, np.nan)
                out.loc[out.index[rows], f"{prefix}_pnl_{n}"] = np.where(cnt > 0, (cs_pnl[idx] - cs_pnl[lo]) / cnt, np.nan)

    for n in windows:
        for p in ("mctx_recent", "mctx_recent_same"):
            out[f"{p}_win_{n}"] = np.nan
            out[f"{p}_pnl_{n}"] = np.nan
    if outcomes is None or outcomes.empty:
        return out

    polar = np.flatnonzero(~np.isnan(sgn))
    fill("mctx_recent", outcomes, polar)
    for s, side in ((1.0, "buy"), (-1.0, "sell")):
        rows = np.flatnonzero(sgn == s)
        sub = outcomes[outcomes["side"] == side]
        if len(rows) and len(sub):
            fill("mctx_recent_same", sub, rows)
    return out


def context_columns(windows: Sequence[int] = (10, 30), regime: Sequence[str] = ()) -> List[str]:
    cols = list(SIGNED_SOURCES.values())
    cols += [f"mctx_{k}_{w}" for w in (6, 24) for k in ("ext", "room")]
    cols += ["mctx_entry_class"]
    cols += [f"{p}_{m}_{n}" for n in windows for p in ("mctx_recent", "mctx_recent_same") for m in ("win", "pnl")]
    return cols + list(regime)
