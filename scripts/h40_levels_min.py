"""H40: профиль объёма по уровням цены из минутных свечей (запас на всю историю 2015–2026, 19 акций).

Лента обезличенных сделок есть только с 2022 года и читается с Диска часами (scripts/h40_levels.py), а сделки
стратегий начинаются в 2016-м. Минутки лежат на сервере, и объём минуты попадает в уровень её средней цены
(value / volume — средневзвешенная цена минуты): уровень — шаг 0.1% цены, тот же, что у ленты
(номер = floor(log(цена) / log(1.001))). Внутри минуты акция обычно не выходит за один-два уровня, поэтому профиль
получается почти такой же, как по ленте, — совпадение проверяется в scripts/h40_trades.py на днях, где есть оба
источника.

Выход — data/derived/h40_levels_min/<бумага>.parquet: hour (МСК), b (уровень), vol (штук), n (минут).
Готовые бумаги пропускаются.

    python scripts/h40_levels_min.py                 # все 19 акций
    python scripts/h40_levels_min.py LKOH SBER
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "derived" / "h40_levels_min"
STEP = np.log(1.001)
TICKERS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
           "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS"]
SRC = {"SBER": RAW / "SBER_1m_15-26.csv", "GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv"}


def build(tk: str) -> pd.DataFrame:
    src = SRC.get(tk, RAW / "moex" / f"{tk}_1m.csv")
    parts = []
    for ch in pd.read_csv(src, usecols=["close", "value", "volume", "begin"], chunksize=400_000):
        ch = ch[ch["volume"] > 0]
        px = np.where(ch["value"] > 0, ch["value"] / ch["volume"].clip(lower=1), ch["close"])
        px = np.where(np.isfinite(px) & (px > 0), px, ch["close"])
        g = pd.DataFrame({"hour": pd.to_datetime(ch["begin"]).dt.floor("h").to_numpy(),
                          "b": np.floor(np.log(px) / STEP).astype("int32"),
                          "vol": ch["volume"].to_numpy()})
        parts.append(g.groupby(["hour", "b"], sort=False).agg(vol=("vol", "sum"), n=("vol", "size")))
    d = pd.concat(parts)
    d = d.groupby(level=[0, 1]).sum().reset_index()
    d["vol"] = d["vol"].astype("int64")
    d["n"] = d["n"].astype("int32")
    return d.sort_values(["hour", "b"])


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for tk in (argv or TICKERS):
        f = OUT / f"{tk}.parquet"
        if f.exists():
            print(f"{tk}: уже есть", flush=True)
            continue
        d = build(tk)
        d.to_parquet(f, index=False)
        print(f"{tk}: часов {d['hour'].nunique()}, строк {len(d)}, "
              f"{d['hour'].min():%Y-%m-%d} — {d['hour'].max():%Y-%m-%d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
