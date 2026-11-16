"""H40: профиль объёма по уровням цены для LKOH из секундных агрегатов ленты (data/raw/orderflow_lkoh_1s.parquet).

Тот же файл, что в h40_profile.py и на странице H84: обезличенные сделки TQBR, 01.2024 — 06.04.2026, с разбивкой
объёма секунды на покупки и продажи по инициатору. Уровень — шаг 0.1% цены (как у остальных источников H40), цена
секунды — её средневзвешенная (vwap), иначе цена последней сделки. Аукционы открытия и закрытия выброшены, как в
scripts/h40_levels.py.

Нужен, чтобы ответить на вопрос автора «покупки и продажи вместе или раздельно» и сверить уровни ленты и минуток на
длинном куске, не дожидаясь выгрузки всей ленты с Диска.

Выход — data/derived/h40_levels_lkoh.parquet: hour (МСК), b, buy, sell, n.

    python scripts/h40_levels_lkoh.py [путь к копии parquet]
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "orderflow_lkoh_1s.parquet"
OUT = ROOT / "data" / "derived" / "h40_levels_lkoh.parquet"
STEP = np.log(1.001)


def main(argv: list[str] | None = None) -> int:
    # читать файл прямо с Диска (data/raw — ссылка на yadisk) можно часами: если он скопирован на сервер,
    # путь к копии передаётся первым аргументом
    src = Path(argv[0]) if argv else SRC
    pf = pq.ParquetFile(src)
    parts = []
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["time", "price", "vwap", "volume", "buy_volume", "sell_volume"]).to_pandas()
        t = pd.to_datetime(d["time"])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(None)
        d["time"] = t + pd.Timedelta(hours=3)                                  # UTC → МСК
        d = d[d["volume"] > 0]
        hm = d["time"].dt.hour * 60 + d["time"].dt.minute
        d = d[~(((hm >= 9 * 60 + 50) & (hm < 10 * 60)) | ((hm >= 18 * 60 + 40) & (hm < 18 * 60 + 50)))]
        px = d["vwap"].where(d["vwap"] > 0, d["price"]).astype(float)
        g = pd.DataFrame({"hour": d["time"].dt.floor("h").to_numpy(),
                          "b": np.floor(np.log(px) / STEP).astype("int32"),
                          "buy": d["buy_volume"].to_numpy(), "sell": d["sell_volume"].to_numpy(),
                          "n": 1})
        parts.append(g.groupby(["hour", "b"], sort=False).sum())
        del d, g, px
        gc.collect()
    out = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index().sort_values(["hour", "b"])
    out.to_parquet(OUT, index=False)
    print(f"часов {out['hour'].nunique()}, строк {len(out)}, {out['hour'].min():%Y-%m-%d} — {out['hour'].max():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
