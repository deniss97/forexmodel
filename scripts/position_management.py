"""Управление позицией по вероятности касания барьера (импульсная стратегия).

Вопрос: если в каждый час внутри сделки оценивать вероятность того, что цена дойдёт до цели
раньше стопа, и закрывать сделку при низкой вероятности, станет ли результат лучше?

Постановка для нашей стратегии (тейка нет, есть стоп с трейлингом и выход по времени):
  * SL — фактический текущий уровень трейлинг-стопа;
  * TP — цель «ещё +K ATR от текущей цены» в нашу сторону (K = `--target`, ATR — на входе,
    как у стопов);
  * событие — цена коснулась TP раньше, чем сделка закрылась (стоп, трейлинг, время).

Таблица «сделка × час»: для каждой сделки базовой стратегии (6 ч ≥ 3 ATR по гейту C, выход 240 ч
/ стоп и трейлинг 6 ATR / активация 3; быстрый движок) — точка проверки каждый час по часам
(внутри перерывов торгов дубли убираются). Признаки — только по прошлому: текущий результат,
расстояние до стопа, лучший ход за сделку, время в сделке, включён ли трейлинг, ATR сейчас к
ATR на входе, гейт сейчас, ход за 6 и 24 часа со знаком стороны, час суток, рынок.

Оценки P(TP раньше SL):
  * аналитическая — броуновское движение без дрейфа между двумя барьерами с конечным
    горизонтом (остаток до 240 ч), σ часа — по ATR часа на момент проверки;
  * модель — CatBoost на признаках, walk-forward по годам: модель года Y учится на сделках,
    закрывшихся до 1 января Y (все часы одной сделки — по одну сторону границы), общая на все
    инструменты.

Правила, которые проверяются на бэктесте (результат сделки после комиссии, сумма по 1/N):
  * «закрыть досрочно»: в первый час, где P < θ, сделка закрывается по цене закрытия этого
    часа (минус `--exit-slip`);
  * «фильтр на входе»: P в первый час < θ — сделка не открывается.
Порог θ выбирается по 2017–2020, проверка — 2021–2026. Освободившееся время другими сделками
не занимается (как в песочнице).

    python scripts/position_management.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from early_entry import simulate
from forexmodel.data.loader import load_minute_compact
from impulse_universe import hourly, signal
from pattern_lab import ALL_INSTRUMENTS, apply_exit, instrument_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "patterns"
SL, TRAIL, ACT, HORIZON_H = 6.0, 6.0, 3.0, 240
FEATS = ["unreal_atr", "dist_stop_atr", "mfe_atr", "hours_in", "trail_on", "vol_ratio_now", "atr_pct_now",
         "gate_agree", "move6", "move24", "hour", "market"]


def trade_rows(minute, h, trades, inst, market, k_target):
    """Строки «сделка × час» для сделок одного инструмента."""
    t = minute["time"].to_numpy()
    hi, lo, cl, op = (minute[c].to_numpy(dtype=float) for c in ("high", "low", "close", "open"))
    ht = h["t_close"].to_numpy()
    h_atr, h_gate, h_mv = h["atr"].to_numpy(), h["gate"].to_numpy(), h["move_atr"].to_numpy()
    h_close = h["close"].to_numpy()
    h_sig = (h["close"].diff().rolling(24 * 20, min_periods=24 * 5).std()).to_numpy()   # σ часа, прошлые 20 дней
    rows = []
    for k, tr in enumerate(trades.itertuples(index=False)):
        s, entry = int(tr.side), float(tr.entry)
        a = int(np.searchsorted(t, np.datetime64(tr.open_time), "left"))
        j = int(np.searchsorted(t, np.datetime64(tr.exit_time), "left"))
        hs = int(np.searchsorted(ht, np.datetime64(tr.time), "right")) - 1      # бар сигнала
        atr_e = float(h_atr[hs])
        seg_h, seg_l = hi[a:j + 1], lo[a:j + 1]
        if s > 0:
            best = np.maximum.accumulate(np.concatenate([[entry], seg_h[:-1]]))
            gain = best - entry
            stop = np.where(gain >= ACT * atr_e, np.maximum(entry - SL * atr_e, best - TRAIL * atr_e), entry - SL * atr_e)
        else:
            best = np.minimum.accumulate(np.concatenate([[entry], seg_l[:-1]]))
            gain = entry - best
            stop = np.where(gain >= ACT * atr_e, np.minimum(entry + SL * atr_e, best + TRAIL * atr_e), entry + SL * atr_e)
        final = float(tr.gross)
        # точки проверки: каждый час по часам, последняя минута не позже отметки
        marks = np.datetime64(tr.open_time) + np.arange(1, HORIZON_H) * np.timedelta64(1, "h")
        cs = np.unique(np.searchsorted(t, marks, "right") - 1)
        cs = cs[(cs > a) & (cs < j)]
        if not len(cs):
            continue
        # экстремумы по блокам между точками проверки: блок m — минуты cs[m]+1 .. cs[m+1] (последний — до j)
        bounds = np.r_[cs + 1, j + 1]
        ext = hi if s > 0 else lo
        red = np.maximum.reduceat if s > 0 else np.minimum.reduceat
        bx = red(ext, bounds[:-1])[: len(cs)] if bounds[-1] <= len(ext) else red(ext[: j + 1], bounds[:-1])[: len(cs)]
        for i, c in enumerate(cs):
            r = c - a
            px = cl[c]
            target = px + s * k_target * atr_e
            reach = (bx[i:] >= target) if s > 0 else (bx[i:] <= target)
            m = int(np.argmax(reach)) + i if reach.any() else -1
            if m == len(cs) - 1:
                # последний блок содержит выход: цель засчитывается, только если раньше минуты выхода
                seg = ext[cs[m] + 1: j]
                ok = (seg >= target).any() if s > 0 else (seg <= target).any()
                m = m if ok else -1
            hit = np.array([m]) if m >= 0 else np.array([])
            hb = int(np.searchsorted(ht, t[c], "right")) - 1                        # последний закрытый час
            atr_now = float(h_atr[hb]) if hb >= 0 else np.nan
            mv24 = (h_close[hb] - h_close[hb - 24]) / atr_now if hb >= 24 else np.nan
            rows.append((inst, market, k, np.datetime64(tr.time), np.datetime64(tr.exit_time),
                         (t[c] - np.datetime64(tr.open_time)) / np.timedelta64(1, "h"),
                         s * (px - entry) / atr_e, s * (px - stop[r]) / atr_e, gain[r] / atr_e,
                         float(gain[r] >= ACT * atr_e), atr_now / atr_e, atr_now / px * 100,
                         s * float(h_gate[hb]) if hb >= 0 else 0.0, s * float(h_mv[hb]) if hb >= 0 else np.nan,
                         s * mv24, pd.Timestamp(t[c]).hour,
                         int(hit.size > 0), final, s * (px - entry) / entry * 100,
                         s * (px - stop[r]) / px * 100, k_target * atr_e / px * 100,
                         float(h_sig[hb]) / px * 100 if hb >= 0 else np.nan))
    return rows


COLS = ["inst", "market", "trade", "time", "exit_time", "hours_in", "unreal_atr", "dist_stop_atr", "mfe_atr", "trail_on",
        "vol_ratio_now", "atr_pct_now", "gate_agree", "move6", "move24", "hour", "y_tp", "final_gross",
        "now_gross", "dist_stop_pct", "dist_tp_pct", "sigma_pct"]


def analytic_p(a, b, sigma, T, n_terms=60):
    """P(броуновское движение без дрейфа из точки a в (0, a+b) выйдет вверх раньше низа и до T).

    u(x, t) = x/L + Σ 2(−1)^n/(nπ) · sin(nπx/L) · exp(−(nπ)² σ² t / (2L²)),  L = a + b.
    """
    L = a + b
    x = a / L
    out = x.copy()
    for n in range(1, n_terms + 1):
        out += 2 * (-1) ** n / (n * np.pi) * np.sin(n * np.pi * x) * np.exp(-((n * np.pi) ** 2) * sigma ** 2 * T / (2 * L ** 2))
    return np.clip(out, 0, 1)


def walk_forward_p(df: pd.DataFrame, years, seed=42) -> pd.Series:
    from catboost import CatBoostClassifier

    p = pd.Series(np.nan, index=df.index)
    df_year = df["time"].dt.year
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        tr = df[df["exit_time"] < start]
        te = df[df_year == y]
        if len(tr) < 5000 or te.empty:
            continue
        # ранняя остановка — по последним 20% сделок обучения (по времени входа)
        trades_sorted = tr.drop_duplicates(["inst", "trade"]).sort_values("time")
        cut = trades_sorted.iloc[int(len(trades_sorted) * 0.8)]["time"]
        fit, val = tr[tr["exit_time"] < cut], tr[tr["time"] >= cut]
        m = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=10, random_seed=seed,
                               verbose=False, early_stopping_rounds=100, cat_features=["market"])
        m.fit(fit[FEATS], fit["y_tp"], eval_set=(val[FEATS], val["y_tp"]))
        p.loc[te.index] = m.predict_proba(te[FEATS])[:, 1]
        print(f"  модель {y}: обучение {len(fit)} строк, деревьев {m.tree_count_}", flush=True)
    return p


def apply_rule(df: pd.DataFrame, pcol: str, theta: float, comm: float, exit_slip: float, entry_only=False):
    """Результат сделок при правиле «закрыть в первый час, где P < θ» (или не входить)."""
    out = {}
    for key, g in df.groupby(["inst", "trade"], sort=False):
        g = g.sort_values("hours_in")
        first = g.iloc[0]
        if entry_only:
            out[key] = None if first[pcol] < theta else first["final_gross"] - comm
            continue
        below = g[g[pcol] < theta]
        if below.empty:
            out[key] = first["final_gross"] - comm
        else:
            out[key] = below.iloc[0]["now_gross"] - comm - exit_slip
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=float, default=3.0, help="цель TP: ещё +K ATR от текущей цены")
    ap.add_argument("--exit-slip", type=float, default=0.0, help="проскальзывание досрочного выхода, %")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)

    rows, comm = [], {}
    for inst in ALL_INSTRUMENTS:
        cfg = apply_exit(instrument_cfg(inst), "240:6:6:3")
        comm[inst] = cfg.simulation.commission_pct
        minute = load_minute_compact(ROOT / cfg.data.csv_path, cfg.data.time_col, ROOT / "reports" / "_cache")
        h = hourly(minute, cfg)
        trades = simulate(minute, signal(h, "atr", 3.0, "gate"), cfg, 0.0)
        market = {"silver": "fx", "gold": "fx", "eth": "crypto"}.get(inst, "stock")
        rows += trade_rows(minute, h, trades, inst, market, args.target)
        print(f"[{inst}] {len(trades)} сделок, строк всего {len(rows)}", flush=True)
        del minute
    df = pd.DataFrame(rows, columns=COLS)
    df["time"], df["exit_time"] = pd.to_datetime(df["time"]), pd.to_datetime(df["exit_time"])
    df = df.dropna(subset=["dist_stop_pct", "sigma_pct"]).reset_index(drop=True)

    # аналитика: расстояния в % цены, σ часа в % цены, горизонт — остаток часов
    df["p_analytic"] = analytic_p(df["dist_stop_pct"].clip(lower=1e-6).to_numpy(), df["dist_tp_pct"].to_numpy(),
                                  df["sigma_pct"].to_numpy(), (HORIZON_H - df["hours_in"]).clip(lower=1).to_numpy())
    df["p_model"] = walk_forward_p(df, range(2017, 2027))
    df.to_pickle(OUT / "position_rows.pkl")

    from sklearn.metrics import roc_auc_score

    oos = df[df["p_model"].notna()]
    print(f"\nстрок вне выборки: {len(oos)}, доля TP раньше SL: {oos['y_tp'].mean():.3f}")
    for col in ("p_analytic", "p_model"):
        print(f"AUC {col}: {roc_auc_score(oos['y_tp'], oos[col]):.3f} | средняя P {oos[col].mean():.3f}")
    # калибровка модели
    oos = oos.assign(bin=pd.cut(oos["p_model"], [0, .2, .3, .4, .5, .6, .7, 1]))
    print(oos.groupby("bin", observed=True).agg(строк=("y_tp", "size"), P=("p_model", "mean"), факт=("y_tp", "mean")).round(3))

    # правила на бэктесте
    res = []
    base = {k: g["final_gross"].iloc[0] - comm[k[0]] for k, g in oos.groupby(["inst", "trade"])}
    yrs = {k: g["time"].iloc[0].year for k, g in oos.groupby(["inst", "trade"])}
    n = oos["inst"].nunique()

    def summarize(name, outcome):
        for per, cond in (("2017–20", lambda y: y <= 2020), ("2021–26", lambda y: y >= 2021)):
            v = [x for k, x in outcome.items() if x is not None and cond(yrs[k])]
            b = [base[k] for k in outcome if cond(yrs[k])]
            res.append({"правило": name, "период": per, "сделок": len(v), "итог_портфеля": sum(v) / n,
                        "база": sum(b) / n, "на_сделку": np.mean(v) if v else np.nan})

    for pcol in ("p_analytic", "p_model"):
        for theta in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
            mean_comm = float(np.mean(list(comm.values())))
            summarize(f"закрыть, если {pcol} < {theta}", apply_rule(oos, pcol, theta, mean_comm, args.exit_slip))
        for theta in (0.3, 0.35, 0.4, 0.45):
            summarize(f"не входить, если {pcol} в 1-й час < {theta}", apply_rule(oos, pcol, theta, 0.04, 0, entry_only=True))
    r = pd.DataFrame(res)
    r.to_csv(OUT / "position_management.csv", index=False)
    print(r.pivot_table(index="правило", columns="период", values=["итог_портфеля", "база", "сделок"], sort=False)
          .round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
