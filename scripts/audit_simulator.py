"""Независимая проверка расчёта импульсной стратегии: свой симулятор с нуля и сверка.

Что проверяется:
  1. Данные: монотонность времени, дубли, high >= max(open, close) >= min >= low, выборочная сверка
     кэша минуток с исходным CSV.
  2. Расчёт: часовые бары, ATR(14), сигнал «ход за 6 ч ≥ 3 ATR», вход через минуту после закрытия
     часа по open, стоп и трейлинг 6 ATR (активация +3 ATR), выход не позже 240 ч, позиции без
     перекрытия, комиссия 0.04%. Всё написано заново явным поминутным циклом, без общего кода с
     scripts/early_entry.py и forexmodel/simulation. Общая часть одна — колонка гейта C из
     impulse_universe.hourly (так проверяется движок отдельно от гейта).
  3. Исполнение стопа при гэпе. Оба существующих движка исполняют стоп по уровню стопа, даже если
     минута открылась уже за ним (ночной гэп, дивидендная отсечка). Режим «гэп» исполняет по open
     минуты, если он хуже стопа. Разница — оценка оптимизма существующего расчёта.

    python scripts/audit_simulator.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "audit"
INSTRUMENTS = {"silver": ("XAGUSD_1m_full", "time"), "gold": ("XAUUSD_1m_full", "time"),
               "lkoh": ("LKOH_1m_15-26", "begin"), "gazp": ("GAZP_1m_15-26_upd", "begin"),
               "sber": ("SBER_1m_15-26", "begin"), "moex": ("MOEX_1m_15-26", "begin"),
               "mtss": ("MTSS_1m_15-26", "begin"), "eth": ("ETHUSDT_1m_clean", "time")}
COMMISSION, SL, TRAIL, ACT, HORIZON_MIN, WINDOW, THR = 0.04, 6.0, 6.0, 3.0, 240 * 60, 6, 3.0


def check_data(inst: str, m: pd.DataFrame) -> dict:
    t = m["time"].to_numpy()
    rep = {"баров": len(m), "время_монотонно": bool((np.diff(t) > np.timedelta64(0)).all()),
           "дублей": int(m["time"].duplicated().sum()),
           "ohlc_нарушений": int(((m["high"] < m[["open", "close"]].max(axis=1)) |
                                  (m["low"] > m[["open", "close"]].min(axis=1))).sum()),
           "нулевой_диапазон_%": round(100 * float((m["high"] == m["low"]).mean()), 2)}
    # выборочная сверка кэша с исходным CSV: 50 тыс. строк из середины файла
    fname, tcol = INSTRUMENTS[inst]
    raw = ROOT / "data" / "raw" / f"{fname}.csv"
    if raw.exists():
        n = sum(1 for _ in open(raw, "rb"))
        skip = n // 2
        head = pd.read_csv(raw, nrows=0).columns
        s = pd.read_csv(raw, skiprows=range(1, skip), nrows=50_000, names=head, header=0)
        s["time"] = pd.to_datetime(s[tcol], errors="coerce")
        s = s.dropna(subset=["time"]).drop_duplicates("time").set_index("time")[["open", "high", "low", "close"]]
        c = m.set_index("time").loc[s.index.intersection(m["time"])]
        s = s.loc[c.index]
        rep["сверка_с_csv_строк"] = len(c)
        rep["сверка_макс_расхождение"] = float(np.abs(c.to_numpy(dtype=float) - s.to_numpy(dtype=float)).max()) if len(c) else None
    return rep


def hourly_bars(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(m["time"].dt.floor("h"))
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "last_minute": g["time"].max()})
    h.index.name = "time"
    h = h.reset_index()
    pc = h["close"].shift(1)
    tr = np.maximum(h["high"] - h["low"], np.maximum((h["high"] - pc).abs(), (h["low"] - pc).abs()))
    h["atr"] = tr.rolling(14).mean()
    h["move"] = (h["close"] - h["close"].shift(WINDOW)) / h["atr"]
    return h


def simulate(m: pd.DataFrame, h: pd.DataFrame, gate: np.ndarray | None, gap_fill: bool) -> pd.DataFrame:
    t = m["time"].to_numpy()
    op, hi, lo, cl = (m[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    rows, busy = [], np.datetime64("1900-01-01")
    for i in range(len(h)):
        mv, atr = h["move"].iat[i], h["atr"].iat[i]
        if not (np.isfinite(mv) and np.isfinite(atr) and atr > 0) or abs(mv) < THR:
            continue
        s = 1 if mv > 0 else -1
        if gate is not None and gate[i] != s:
            continue
        entry_t = h["time"].iat[i].to_datetime64() + np.timedelta64(61, "m")   # закрытие часа + 1 минута
        if entry_t <= busy:
            continue
        a = int(np.searchsorted(t, entry_t, "left"))
        if a >= len(t):
            continue
        end_t = t[a] + np.timedelta64(HORIZON_MIN, "m")
        b = int(np.searchsorted(t, end_t, "right")) - 1
        if b <= a:
            continue
        fill = op[a]
        stop = fill - s * SL * atr
        best, active = fill, False
        exit_px, exit_i, reason = None, None, None
        for j in range(a, b + 1):
            worst = lo[j] if s > 0 else hi[j]
            if (worst - stop) * s <= 0:
                exit_px = stop
                if gap_fill and (op[j] - stop) * s < 0:      # минута открылась уже за стопом
                    exit_px = op[j]
                exit_i, reason = j, "trail" if active else "stop"
                break
            ext = hi[j] if s > 0 else lo[j]
            if (ext - best) * s > 0:
                best = ext
                if (best - fill) * s >= ACT * atr:
                    active = True
                    cand = best - s * TRAIL * atr
                    stop = max(stop, cand) if s > 0 else min(stop, cand)
        if exit_i is None:
            exit_i, exit_px, reason = b, cl[b], "time"
        gross = s * (exit_px - fill) / fill * 100
        busy = t[exit_i]
        rows.append((h["time"].iat[i], s, fill, exit_px, gross, gross - COMMISSION, reason,
                     (t[exit_i] - t[a]) / np.timedelta64(1, "h"), op[exit_i] if exit_i is not None else np.nan))
    return pd.DataFrame(rows, columns=["signal", "side", "fill", "exit", "gross", "net", "reason", "hours", "exit_open"])


def main() -> int:
    logging.disable(logging.WARNING)
    pd.set_option("display.width", 250)
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from impulse_universe import hourly as ref_hourly
    from pattern_lab import instrument_cfg

    checks, rows, trades = [], [], []
    for inst, (fname, _) in INSTRUMENTS.items():
        m = pd.read_pickle(ROOT / "reports" / "_cache" / f"{fname}_1m.pkl")
        m = m.sort_values("time").reset_index(drop=True)
        m[["open", "high", "low", "close"]] = m[["open", "high", "low", "close"]].astype(float)
        c = check_data(inst, m)
        h = hourly_bars(m)
        # гейт C — из существующего кода, выровнен по времени часа
        ref = ref_hourly(m, instrument_cfg(inst))
        gate = pd.Series(ref["gate"].to_numpy(), index=ref["time"]).reindex(h["time"]).fillna(0).to_numpy()
        # сверка часовых баров и ATR с существующим кодом
        j = ref.set_index("time").reindex(h.set_index("time").index)
        c["часовых_баров_совпало"] = bool(np.allclose(j["close"].to_numpy(dtype=float), h["close"].to_numpy(), equal_nan=True))
        c["atr_макс_расхождение"] = float(np.nanmax(np.abs(j["atr"].to_numpy(dtype=float) - h["atr"].to_numpy())))
        checks.append({"инструмент": inst, **c})
        for gname, g in (("без гейта", None), ("гейт C", gate)):
            for fill in (False, True):
                tr = simulate(m, h, g, fill)
                tr["год"] = pd.to_datetime(tr["signal"]).dt.year
                trades.append(tr.assign(инструмент=inst, гейт=gname, исполнение="гэп" if fill else "уровень"))
                rows.append({"инструмент": inst, "гейт": gname, "исполнение": "по open при гэпе" if fill else "по уровню стопа",
                             "сделок": len(tr), "итог": tr["net"].sum(), "стопов": int((tr["reason"] == "stop").sum())})
        print(f"[{inst}] готово", flush=True)
        del m
    pd.DataFrame(checks).to_csv(OUT / "data_checks.csv", index=False)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "audit_results.csv", index=False)
    T = pd.concat(trades, ignore_index=True)
    T.to_csv(OUT / "audit_trades.csv", index=False)
    print("\nПРОВЕРКА ДАННЫХ")
    print(pd.DataFrame(checks).to_string(index=False))
    print("\nНЕЗАВИСИМЫЙ СИМУЛЯТОР: итог % после комиссии, 2015–2026")
    p = r.pivot_table(index="инструмент", columns=["гейт", "исполнение"], values="итог").round(1)
    p.loc["портфель 8 (1/8)"] = p.mean()
    print(p.to_string())
    n = r.pivot_table(index="инструмент", columns=["гейт", "исполнение"], values="сделок")
    print("\nсделок:\n" + n.to_string())
    # сверка с существующим движком (исходы кандидатов посчитаны scripts/impulse_gate_tuner.py events)
    from impulse_gate_tuner import KX, _init, evaluate

    _init()
    q = KX.index((6, 3.0))
    ref_rows = []
    for gname, params in (("без гейта", None), ("гейт C", {})):
        _, _, S, C = evaluate((gname, params))
        for k, inst in enumerate(INSTRUMENTS):
            ref_rows.append({"инструмент": inst, "гейт": gname, "итог_существующий": S[0, q, k].sum(), "сделок_существующий": int(C[0, q, k].sum())})
    ref = pd.DataFrame(ref_rows)
    cmp = r[r["исполнение"] == "по уровню стопа"].merge(ref, on=["инструмент", "гейт"])
    cmp["разница_итог"] = cmp["итог"] - cmp["итог_существующий"]
    cmp["разница_сделок"] = cmp["сделок"] - cmp["сделок_существующий"]
    print("\nСВЕРКА С СУЩЕСТВУЮЩИМ ДВИЖКОМ (исполнение по уровню стопа)")
    print(cmp[["инструмент", "гейт", "сделок", "сделок_существующий", "итог", "итог_существующий", "разница_итог"]].round(2).to_string(index=False))
    cmp.to_csv(OUT / "audit_vs_existing.csv", index=False)
    # эффект гэпов по годам, портфель
    g = T[T["гейт"] == "гейт C"].groupby(["исполнение", "год"])["net"].sum().unstack(0) / 8
    g["разница"] = g["гэп"] - g["уровень"]
    print("\nГЕЙТ C, портфель по годам: исполнение по уровню стопа против по open при гэпе")
    print(g.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
