"""H40, шаг 1: профиль объёма по уровням цены из ленты сделок → свёртка «бумага × час × уровень × сторона» по дням.

Лента (две части, колонки одинаковые — TRADE_TS (UTC), TICKER_CC, DIRECTION (сторона инициатора), PRICE, QUANTITY,
TRADE_SOURCE, INSTRUMENT_UID):
  * ALGOPACK на Диске — yadisk:обезличенные сделки/<год>/<дата>.csv.gz (все инструменты, 50–190 МБ за день),
    03.01.2022 — 06.04.2026;
  * лента Т-Инвестиций по бумагам — yadisk:forexmodel/лента Т-Инвестиций/<бумага>/<год>/<дата>.csv.gz, с 07.04.2026
    (её качает scripts/tinvest_tape.py).
Файлы читаются потоком (rclone cat | gzip -dc | grep | awk), на сервер ничего не копируется. Уровень цены — шаг 0.1%
(номер уровня = floor(log(цена) / log(1.001))), как в scripts/h40_profile.py. Аукционы открытия (09:50–10:00 МСК) и
закрытия (18:40–18:50 МСК) не берём: там тысячи сделок с одной меткой времени и одной ценой — ложный «уровень».

Выход по дню — data/derived/h40_levels/<дата>.parquet: tk, hour (МСК), b (уровень), buy, sell (объём в лотах),
nbuy, nsell (число сделок). Готовые дни пропускаются, так что прерванный прогон продолжается с места.

    python scripts/h40_levels.py                      # всё недостающее, 4 потока
    python scripts/h40_levels.py --start 2026-04-07 --jobs 2
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "h40_levels"
RCLONE = Path.home() / "bin" / "rclone"
ALGO = "yadisk:обезличенные сделки"
MOUNT = Path.home() / "yadisk"
SECIDS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
          "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS"]
ALGO_END = "2026-04-06"
AWK = r'''
BEGIN { FS = ","; L = log(1.001) }
$6 == "EXCHANGE" && $5 > 0 && $4 > 0 && ($3 == "BUY" || $3 == "SELL") {
    t = substr($1, 12, 8)
    if (t >= "06:50:00" && t < "07:00:00") next
    if (t >= "15:40:00" && t < "15:50:00") next
    tk = $2; sub(/_.*/, "", tk)
    k = tk "\t" substr($1, 1, 13) "\t" int(log($4) / L) "\t" substr($3, 1, 1)
    v[k] += $5; c[k]++
}
END { for (k in v) print k "\t" v[k] "\t" c[k] }
'''
COLS = ["tk", "hour", "b", "buy", "sell", "nbuy", "nsell"]


def pipeline(cmd: str, day: str) -> pd.DataFrame:
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE)
    df = pd.read_csv(p.stdout, sep="\t", header=None, names=["tk", "h", "b", "side", "vol", "n"],
                     dtype={"tk": "category", "h": "string", "b": "int32", "side": "category",
                            "vol": "int64", "n": "int32"})
    rc = p.wait()
    if rc not in (0, 1):                       # grep без совпадений — 1, это не ошибка
        raise RuntimeError(f"{day}: конвейер вернул {rc}")
    return df


def read_algopack(day: str) -> pd.DataFrame:
    pat = r"^[^,]*,(" + "|".join(SECIDS) + r")_TQBR,"
    return pipeline(f"{RCLONE} cat '{ALGO}/{day[:4]}/{day}.csv.gz' | gzip -dc | grep -E '{pat}' | awk '{AWK}'", day)


def read_tinvest(day: str) -> pd.DataFrame:
    files = [MOUNT / "forexmodel" / "лента Т-Инвестиций" / tk / day[:4] / f"{day}.csv.gz" for tk in SECIDS]
    files = [f for f in files if f.exists()]
    if not files:
        return pd.DataFrame(columns=["tk", "h", "b", "side", "vol", "n"])
    args = " ".join(f"'{f}'" for f in files)
    return pipeline(f"gzip -dc {args} | awk '{AWK}'", day)


def shape(df: pd.DataFrame) -> pd.DataFrame:
    """Строка на «бумага × час (МСК) × уровень»: объём и число сделок покупок и продаж."""
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df["hour"] = pd.to_datetime(df["h"], format="%Y-%m-%dT%H") + pd.Timedelta(hours=3)
    w = df.pivot_table(index=["tk", "hour", "b"], columns="side", values=["vol", "n"],
                       aggfunc="sum", observed=True, fill_value=0)
    w.columns = [f"{a}{b}" for a, b in w.columns]
    w = w.rename(columns={"volB": "buy", "volS": "sell", "nB": "nbuy", "nS": "nsell"}).reset_index()
    for c in ("buy", "sell", "nbuy", "nsell"):
        if c not in w:
            w[c] = 0
    w["tk"] = w["tk"].astype(str)
    return w[COLS]


def days(start: str, end: str) -> list[str]:
    out = []
    for y in range(2022, 2027):
        d = MOUNT / "обезличенные сделки" / str(y)
        if d.exists():
            out += [f.name[:-7] for f in d.iterdir() if f.name.endswith(".csv.gz")]
    tin = MOUNT / "forexmodel" / "лента Т-Инвестиций" / "LKOH"
    for y in sorted(tin.iterdir()) if tin.exists() else []:
        out += [f.name[:-7] for f in y.iterdir() if f.name.endswith(".csv.gz")]
    return sorted({d for d in out if start <= d <= end})


def one(day: str) -> str:
    df = read_algopack(day) if day <= ALGO_END else read_tinvest(day)
    w = shape(df)
    tmp = OUT / f".{day}.tmp.parquet"
    w.to_parquet(tmp, index=False)
    tmp.rename(OUT / f"{day}.parquet")
    return f"бумаг {w['tk'].nunique()}, строк {len(w)}, объём {int(w['buy'].sum() + w['sell'].sum())}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days(a.start, a.end) if not (OUT / f"{d}.parquet").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"дней к расчёту {len(todo)}", flush=True)
    t0, errors = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futures = {ex.submit(one, d): d for d in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            day = futures[fut]
            try:
                msg = fut.result()
            except Exception as e:                                  # noqa: BLE001 — день не должен валить прогон
                errors += 1
                msg = f"ОШИБКА {e}"
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {day}: {msg} ({el / i:.1f} с/день, осталось ~{el / i * (len(todo) - i) / 60:.0f} мин)",
                  flush=True)
    print(f"готово, ошибок {errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
