"""H83: лента ALGOPACK с Яндекс Диска → дневные «следы» крупных заявок и роботов по 21 акции (потоком, без копии ленты).

Лента лежит на Диске: yadisk:обезличенные сделки/<год>/<дата>.csv.gz (все инструменты, 50–90 МБ за день; колонки
TRADE_TS (UTC), TICKER_CC, DIRECTION — сторона инициатора, PRICE, QUANTITY, TRADE_SOURCE, INSTRUMENT_UID). Файл
читается через `rclone cat | gzip -dc | grep` — на сервер ничего не копируется; на выходе по дню — маленький parquet
data/derived/h83_tape/<дата>.parquet (строка на бумагу). Готовые дни пропускаются, так что прерванный прогон
продолжается с места. Свёртки потом уходят на Диск (yadisk:forexmodel/h83_tape/) — `--yadisk`.

Что считается за день по бумаге (только биржевые сделки основной и вечерней сессии; аукционы открытия до 10:00 и
закрытия 18:40–18:50 МСК не входят — там одна метка времени у тысяч сделок):
  * o, h, l, c — первая / максимум / минимум / последняя цена; vol, n — объём и число сделок; buy, sell — объём по
    инициатору;
  * «заявка» — сделки одной метки времени и одной стороны (рыночная заявка, прошедшая несколько уровней);
    hist_b / hist_s — объём заявок покупки / продажи по корзинам размера (четверть октавы: floor(4·log2(размер)));
    hist_n — число заявок в корзине (для порога «крупная» по прошлым 60 дням); med_order — медиана размера заявки;
  * robot_b / robot_s — объём серий из ≥ 30 заявок одного размера и стороны за день при размере больше медианы.

    python scripts/h83_tape_daily.py 2023 2024 2025 2026 --jobs 3
    python scripts/h83_tape_daily.py --yadisk          # выложить свёртки на Диск
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "h83_tape"
RCLONE = Path.home() / "bin" / "rclone"
SRC = "yadisk:обезличенные сделки"
SECIDS = ["SBER", "GAZP", "LKOH", "TATN", "NVTK", "GMKN", "ROSN", "VTBR", "PLZL", "SNGSP", "IRAO", "CHMF", "NLMK",
          "MAGN", "RTKM", "TRNFP", "AFLT", "ALRS", "AFKS", "MOEX", "MTSS"]
NB = 100                       # корзин размера: 4 на октаву → до 2^25 лотов
ROBOT_MIN = 30


def remote_files(years: list[str]) -> list[str]:
    out = []
    for y in years:
        r = subprocess.run([str(RCLONE), "lsf", f"{SRC}/{y}/"], capture_output=True, text=True, check=True)
        out += [f"{y}/{f}" for f in sorted(r.stdout.split()) if f.endswith(".csv.gz")]
    return out


def read_day(rel: str) -> pd.DataFrame:
    pat = r"^[^,]*,(" + "|".join(SECIDS) + r")_TQBR,"
    cmd = f"{RCLONE} cat '{SRC}/{rel}' | gzip -dc | grep -E '{pat}'"
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE)
    df = pd.read_csv(p.stdout, header=None, usecols=[0, 1, 2, 3, 4, 5],
                     names=["ts", "tk", "dir", "price", "qty", "src", "uid"],
                     dtype={"tk": "category", "dir": "category", "src": "category", "price": "float64", "qty": "int64"})
    rc = p.wait()
    if rc not in (0, 1):       # grep без совпадений — 1, это не ошибка
        raise RuntimeError(f"{rel}: конвейер вернул {rc}")
    return df


def summarize(day: pd.DataFrame) -> pd.DataFrame:
    d = day[(day["src"] == "EXCHANGE") & day["dir"].isin(["BUY", "SELL"]) & (day["qty"] > 0)].copy()
    t = d["ts"].str.slice(11, 19)                       # чч:мм:сс UTC
    d = d[(t >= "07:00:00") & ~((t >= "15:40:00") & (t < "15:50:00"))]
    rows = []
    for tk, g in d.groupby("tk", observed=True):
        g = g.sort_values("ts", kind="stable")
        buy = g["dir"] == "BUY"
        o = g.groupby(["ts", "dir"], observed=True)["qty"].sum().reset_index()
        ob = (o["dir"] == "BUY").to_numpy()
        q = o["qty"].to_numpy()
        b = np.clip(np.floor(4 * np.log2(q)).astype(int), 0, NB - 1)
        med = float(np.median(q))
        ser = o.groupby(["qty", "dir"], observed=True).size()
        big = ser[(ser >= ROBOT_MIN) & (ser.index.get_level_values(0) > med)]
        rb = sum(int(k[0]) * int(n) for k, n in big.items() if k[1] == "BUY")
        rs = sum(int(k[0]) * int(n) for k, n in big.items() if k[1] == "SELL")
        rows.append({"tk": tk.split("_")[0], "o": g["price"].iloc[0], "h": g["price"].max(), "l": g["price"].min(),
                     "c": g["price"].iloc[-1], "vol": int(g["qty"].sum()), "n": len(g),
                     "buy": int(g.loc[buy, "qty"].sum()), "sell": int(g.loc[~buy, "qty"].sum()),
                     "n_orders": len(o), "med_order": med,
                     "hist_b": np.bincount(b[ob], weights=q[ob], minlength=NB).astype(np.int64),
                     "hist_s": np.bincount(b[~ob], weights=q[~ob], minlength=NB).astype(np.int64),
                     "hist_n": np.bincount(b, minlength=NB).astype(np.int64),
                     "robot_b": rb, "robot_s": rs})
    return pd.DataFrame(rows)


def one(rel: str) -> str:
    date = Path(rel).name.split(".")[0]
    out = OUT / f"{date}.parquet"
    if out.exists():
        return f"{date}: уже есть"
    day = read_day(rel)
    s = summarize(day)
    if s.empty:
        return f"{date}: акций нет (выходной?)"
    s.insert(0, "date", date)
    tmp = out.with_suffix(".tmp")
    s.to_parquet(tmp, index=False)
    tmp.rename(out)
    return f"{date}: {len(s)} бумаг, сделок {int(s['n'].sum())}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("years", nargs="*", default=["2023", "2024", "2025", "2026"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--yadisk", action="store_true", help="выложить готовые свёртки на Диск и выйти")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.yadisk:
        subprocess.run([str(RCLONE), "copy", str(OUT), "yadisk:forexmodel/h83_tape", "--include", "*.parquet"], check=True)
        print("[Диск] свёртки выложены в yadisk:forexmodel/h83_tape", flush=True)
        return 0
    files = remote_files(a.years)
    print(f"файлов: {len(files)}", flush=True)
    bad = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(one, f): f for f in files}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            try:
                print(f"[{k}/{len(files)}] {fu.result()}", flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"[{k}/{len(files)}] ошибка {futs[fu]}: {e}", file=sys.stderr, flush=True)
    print(f"готово, ошибок {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
