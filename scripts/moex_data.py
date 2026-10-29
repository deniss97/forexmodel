"""Свечи Мосбиржи через ISS (бесплатно, без ключа): акции, фьючерсы вместе с истёкшими контрактами, склейка фьючерсов.

Переделка ноутбука Тимура notebooks/moexalgo_test3.ipynb (T34) в один файл без библиотеки moexalgo: нужны только
стандартная библиотека и pandas, поэтому файл можно положить рядом с любым ноутбуком. CSV — в том же формате, что
data/raw/*_1m_15-26.csv (так выгружена наша история акций): open, close, high, low, value, volume, begin, end;
время московское. Что исправлено против ноутбука:
  * фьючерсы — все контракты серии, включая истёкшие (список из ISS с show_expired), а не только торгуемые сейчас;
  * интервал свечей передаётся в запрос (в ноутбуке фьючерсы качались без него);
  * докачка по месяцам дописывает только свечи позже последней в файле — без дублей и без чтения файла целиком;
  * склейка фьючерсов в непрерывный ряд с переходом на следующий контракт до экспирации.

Из командной строки (по умолчанию — в data/raw/moex/):
    python scripts/moex_data.py shares SBER GAZP --from 2015-01-01          # минутки, с докачкой
    python scripts/moex_data.py shares SBER --interval 60 --from 2020-01-01   # часы
    python scripts/moex_data.py futures SILV --from 2020-01-01 --interval 60  # контракты серебра по очереди
    python scripts/moex_data.py futures SILV --from 2020-01-01 --interval 60 --continuous
    python scripts/moex_data.py futoi sv gd --from 2020-01-03                 # позиции физлиц и юрлиц, 5 мин
    python scripts/moex_data.py list shares                                   # что торгуется (или futures)

Из ноутбука — notebooks/moex_download.ipynb.

Минутки за 10 лет по одной бумаге — около 4–5 тысяч запросов по 500 свечей, порядка получаса. Прервать можно в
любой момент: повторный запуск продолжит с последней свечи файла. Super Candles AlgoPack (tradestats, orderstats)
отдаются только подписчикам — здесь их нет.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

ISS ="https://iss.moex.com/iss"
COLS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]
# интервалы ISS: минуты 1 / 10 / 60, 24 — день, 7 — неделя, 31 — месяц
NAMES = {1: "1m", 10: "10m", 60: "1h", 24: "1d", 7: "1w", 31: "1mo"}
PATHS = {"shares": "engines/stock/markets/shares/boards/{board}/securities/{secid}",
         "futures": "engines/futures/markets/forts/securities/{secid}",
         "index": "engines/stock/markets/index/securities/{secid}"}
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "moex"
# на сервере свободно ~20 ГБ (24.10): выгрузка истории (гигабайты) останавливается, если на диске осталось меньше этого;
# докачка — с того же места. Почасовая дописка живого контура — сотни килобайт: у неё свой порог (28.10: диск съели
# соседние проекты, контур с порогом 10 ГБ встал на полдня)
MIN_FREE_GB = float(os.environ.get("FOREXMODEL_MIN_FREE_GB", "10"))
MIN_FREE_APPEND_GB = float(os.environ.get("FOREXMODEL_MIN_FREE_APPEND_GB", "1"))


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path if path.exists() else path.parent).free / 1024 ** 3


def check_disk(path: Path, min_gb: float | None = None) -> None:
    need = MIN_FREE_GB if min_gb is None else min_gb
    free = free_gb(path)
    if free < need:
        raise OSError(f"на диске свободно {free:.1f} ГБ — меньше {need:g} ГБ: запись данных остановлена")


def get_json(path: str, retries: int = 4, **params) -> dict:
    """Запрос к ISS с повтором при обрыве связи (как в ноутбуке: пауза растёт с каждой попыткой)."""
    params.setdefault("iss.meta", "off")
    url = f"{ISS}/{path}.json?{urlencode(params)}"
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                if "json" not in r.headers.get("Content-Type", ""):
                    raise PermissionError(f"ISS ответил не данными (нужна подписка?): {url}")
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k == retries - 1:
                raise
            print(f"   связь оборвалась ({e}), повтор через {5 * (k + 1)} с", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("недостижимо")


def candles(secid: str, start, end=None, interval: int = 1, kind: str = "shares", board: str = "TQBR",
            pause: float = 0.2) -> pd.DataFrame:
    """Свечи за [start, end] включительно (даты); ISS отдаёт по 500 строк — листаем смещением."""
    path = PATHS[kind].format(secid=secid, board=board) + "/candles"
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    rows, offset = [], 0
    while True:
        data = get_json(path, **{"from": str(start)[:10], "till": str(end)[:10], "interval": interval, "start": offset})["candles"]["data"]
        if not data:
            break
        rows += data
        offset += len(data)
        time.sleep(pause)                     # бережно к бесплатному API
    return pd.DataFrame(rows, columns=COLS)


def last_begin(path: Path) -> pd.Timestamp | None:
    """Время последней свечи в CSV — по хвосту файла, не читая его целиком (минутки за 10 лет — сотни МБ)."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as f:
        header = f.readline().decode().strip().split(",")
        f.seek(max(0, path.stat().st_size - 4096))
        tail = [x for x in f.read().decode(errors="ignore").splitlines() if x.strip()]
    if not tail or tail[-1].split(",") == header:
        return None
    return pd.Timestamp(tail[-1].split(",")[header.index("begin")])


def download(secid: str, start, end=None, interval: int = 1, kind: str = "shares", board: str = "TQBR",
             out: Path | str | None = None, quiet: bool = False) -> Path:
    """Докачивает свечи в CSV помесячно. Повторный запуск продолжает с последней свечи файла."""
    out = Path(out) if out else OUT / f"{secid}_{NAMES.get(interval, interval)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size:
        with open(out, encoding="utf-8") as f:
            cols = f.readline().strip().split(",")
        if cols != COLS:   # например, у data/raw/GAZP_1m_15-26_upd.csv лишняя колонка-индекс — дописывать нельзя
            raise ValueError(f"{out}: колонки {cols}, а нужны {COLS} — укажите другой файл")
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    have = last_begin(out)
    t0 = max(pd.Timestamp(start), have.normalize()) if have is not None else pd.Timestamp(start)
    added = 0
    for m in pd.date_range(t0.to_period("M").start_time, end, freq="MS"):
        a, b = max(m, t0), min(m + pd.offsets.MonthEnd(0), end)
        df = candles(secid, a, b, interval, kind, board)
        if have is not None:
            df = df[pd.to_datetime(df["begin"]) > have]
        if len(df):
            check_disk(out.parent)
            df.to_csv(out, mode="a", header=not out.exists() or out.stat().st_size == 0, index=False)
            have, added = pd.Timestamp(df["begin"].iloc[-1]), added + len(df)
        if not quiet:
            print(f"[{secid}] {a:%Y-%m-%d} — {b:%Y-%m-%d}: +{len(df)}", flush=True)
    print(f"[{secid}] добавлено {added} свечей, данные до {have} → {out}", flush=True)
    return out


# ---------------- фьючерсы ----------------

def futures_series(asset_code: str) -> pd.DataFrame:
    """Все контракты серии (SILV, Si, BR, GOLD, MIX, RTS…), включая истёкшие; календарные спреды убраны.
    У старых контрактов код с годом (SVZ5_2015) — качать именно по нему."""
    j = get_json("statistics/engines/futures/markets/forts/series", asset_code=asset_code, show_expired=1)["series"]
    s = pd.DataFrame(j["data"], columns=j["columns"])
    s = s[s["secid"].str.fullmatch(r"\w{2}[FGHJKMNQUVXZ]\d(_\d{4})?") & s["expiration_date"].notna()]
    for c in ("start_date", "expiration_date"):
        s[c] = pd.to_datetime(s[c])
    return s.sort_values("expiration_date").reset_index(drop=True)[["secid", "name", "start_date", "expiration_date"]]


def front_windows(series: pd.DataFrame, start, end=None, margin_days: int = 15) -> pd.DataFrame:
    """Когда контракт ближний: от экспирации предыдущего (минус запас на переход) до своей экспирации."""
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    s = series.copy()
    s["since"] = (s["expiration_date"].shift(1) - pd.Timedelta(days=margin_days)).fillna(s["start_date"])
    s["since"] = s[["since", "start_date"]].max(axis=1).clip(lower=pd.Timestamp(start))
    s["till"] = s["expiration_date"].clip(upper=end)
    return s[s["since"] <= s["till"]].reset_index(drop=True)


def download_futures(asset_code: str, start, end=None, interval: int = 1, folder: Path | str | None = None,
                     margin_days: int = 15) -> pd.DataFrame:
    """Каждый контракт — в свой CSV, только пока он ближний (с запасом на переход). Возвращает окна контрактов."""
    folder = Path(folder) if folder else OUT / asset_code
    w = front_windows(futures_series(asset_code), start, end, margin_days)
    for r in w.itertuples():
        download(r.secid, r.since, r.till, interval, "futures", out=folder / f"{r.secid}_{NAMES.get(interval, interval)}.csv", quiet=True)
    return w


def continuous(folder: Path | str, asset_code: str, interval: int = 1, roll_days: int = 5, adjust: bool = True,
               series: pd.DataFrame | None = None) -> pd.DataFrame:
    """Непрерывный ряд: на следующий контракт переходим за roll_days календарных дней до экспирации ближнего.
    adjust — пропорциональная поправка прошлого на разрыв цены при переходе (доходности сохраняются, уровни — нет);
    колонка contract — какой контракт в ряду. Объём и оборот не поправляются."""
    folder = Path(folder)
    s = series if series is not None else futures_series(asset_code)
    parts, prev_switch = [], None
    files = {r.secid: folder / f"{r.secid}_{NAMES.get(interval, interval)}.csv" for r in s.itertuples()}
    s = s[[files[x].exists() for x in s["secid"]]].reset_index(drop=True)
    for r in s.itertuples():
        df = pd.read_csv(files[r.secid], parse_dates=["begin", "end"])
        switch = r.expiration_date - pd.Timedelta(days=roll_days)
        part = df[(df["begin"] < switch) & (df["begin"] >= (prev_switch if prev_switch is not None else df["begin"].min()))].copy()
        if parts and adjust and len(part):
            # разрыв: последняя свеча прошлого куска против того же времени (или ближайшего раньше) у нового контракта
            last = parts[-1].iloc[-1]
            ref = df[df["begin"] <= last["begin"]]
            if len(ref):
                k = ref["close"].iloc[-1] / last["close"]
                for p in parts:
                    p[["open", "close", "high", "low"]] *= k
        part["contract"] = r.secid
        parts.append(part)
        prev_switch = switch
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS + ["contract"])


def continuous_csv(folder: Path | str, asset_code: str, out: Path | str, interval: int = 1, roll_days: int = 5,
                   series: pd.DataFrame | None = None, rename: dict | None = None, columns: list | None = None) -> int:
    """То же, что continuous (переход за roll_days до экспирации, пропорциональная поправка прошлого на разрыв), но
    потоком в CSV: первый проход — множители на переходах, второй — запись кусков, в памяти один контракт (у золота
    ряд ~3 млн минут — целиком не помещался в потолок памяти 24.10). rename / columns — имена и порядок колонок файла."""
    folder, out = Path(folder), Path(out)
    s = series if series is not None else futures_series(asset_code)
    files = {r.secid: folder / f"{r.secid}_{NAMES.get(interval, interval)}.csv" for r in s.itertuples()}
    s = s[[files[x].exists() for x in s["secid"]]].reset_index(drop=True)

    def part_of(r, prev_switch, cols):
        df = pd.read_csv(files[r.secid], usecols=cols, parse_dates=["begin"])
        switch = r.expiration_date - pd.Timedelta(days=roll_days)
        lo = prev_switch if prev_switch is not None else df["begin"].min()
        return df, df[(df["begin"] < switch) & (df["begin"] >= lo)], switch

    kept, k, prev_switch, last = [], [], None, None          # контракты с непустым куском и множитель перехода на них
    for r in s.itertuples():
        df, part, switch = part_of(r, prev_switch, ["begin", "close"])
        if len(part):
            if last is not None:
                ref = df[df["begin"] <= last[0]]
                k.append(ref["close"].iloc[-1] / last[1] if len(ref) else 1.0)
            else:
                k.append(1.0)
            kept.append((r, prev_switch))
            last = (part["begin"].iloc[-1], part["close"].iloc[-1])
        prev_switch = switch
    # множитель куска j — произведение множителей всех следующих переходов
    factor = np.cumprod(np.r_[k[1:], 1.0][::-1])[::-1] if k else np.array([])
    check_disk(out.parent)
    rows = 0
    with open(out, "w", encoding="utf-8") as fh:
        for n, ((r, prev), f) in enumerate(zip(kept, factor)):
            _, part, _ = part_of(r, prev, COLS)
            part = part.copy()
            part[["open", "close", "high", "low"]] *= f
            part["contract"] = r.secid
            if rename:
                part = part.rename(columns=rename)
            if columns:
                part = part[columns]
            part.to_csv(fh, header=n == 0, index=False)
            rows += len(part)
    return rows


# ---------------- открытые позиции по фьючерсам (FUTOI) ----------------

FUTOI_COLS = ["tradedate", "tradetime", "ticker", "clgroup", "pos", "pos_long", "pos_short", "pos_long_num", "pos_short_num"]


def futoi(ticker: str, start, end=None, out: Path | str | None = None, pause: float = 0.2) -> Path:
    """Открытые позиции физлиц (clgroup FIZ) и юрлиц (YUR) по фьючерсу каждые 5 минут, с 2020 года: pos — чистая
    позиция в контрактах, pos_long / pos_short — длинные и короткие, *_num — число лиц. Тикер — префикс серии:
    sv (серебро), gd (золото), br, si… Бесплатно — с задержкой 14 дней. Российский аналог отчёта CFTC о позициях.
    ISS отдаёт до 1000 строк, свежие первыми, и не листается — поэтому по одному дню (≈ 400 строк). С докачкой."""
    out = Path(out) if out else OUT / f"futoi_{ticker.lower()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    end = min(pd.Timestamp(end or pd.Timestamp.today()), pd.Timestamp.today().normalize() - pd.Timedelta(days=15))
    have = None
    if out.exists() and out.stat().st_size:
        with open(out, "rb") as f:
            f.seek(max(0, out.stat().st_size - 2048))
            have = pd.Timestamp(f.read().decode(errors="ignore").splitlines()[-1].split(",")[0])
    days = pd.date_range(have + pd.Timedelta(days=1) if have is not None else pd.Timestamp(start), end, freq="D")
    added = 0
    for d in days:
        j = get_json(f"analyticalproducts/futoi/securities/{ticker.lower()}", **{"from": f"{d:%Y-%m-%d}", "till": f"{d:%Y-%m-%d}"})["futoi"]
        if j["columns"] == ["ERROR_MESSAGE"]:
            raise PermissionError(j["data"][0][0])
        df = pd.DataFrame(j["data"], columns=j["columns"])
        if len(df):
            df = df[FUTOI_COLS].sort_values(["tradetime", "clgroup"])
            check_disk(out.parent)
            df.to_csv(out, mode="a", header=not out.exists() or out.stat().st_size == 0, index=False)
            added += len(df)
        time.sleep(pause)
    print(f"[futoi {ticker}] добавлено {added} строк за {len(days)} дн., данные до {days[-1]:%Y-%m-%d} → {out}" if len(days)
          else f"[futoi {ticker}] новых дней нет → {out}", flush=True)
    return out


# ---------------- справочники и проверка ----------------

def tickers(kind: str = "shares", board: str = "TQBR") -> pd.DataFrame:
    """Что торгуется сейчас: акции режима TQBR (код, название, лот, шаг цены) или фьючерсы (с кодом серии)."""
    if kind == "shares":
        j = get_json(f"engines/stock/markets/shares/boards/{board}/securities", **{"iss.only": "securities",
                     "securities.columns": "SECID,SHORTNAME,LOTSIZE,MINSTEP,PREVPRICE,ISSUESIZE"})["securities"]
    else:
        j = get_json("engines/futures/markets/forts/securities", **{"iss.only": "securities",
                     "securities.columns": "SECID,SHORTNAME,ASSETCODE,LASTTRADEDATE,MINSTEP,PREVSETTLEPRICE"})["securities"]
    return pd.DataFrame(j["data"], columns=[c.lower() for c in j["columns"]])


def check(path_or_df) -> dict:
    """Короткая проверка файла: строк, первая и последняя свеча, дубли по времени, торговых дней, дни без свечей
    среди будних (праздники биржи тоже попадут — смотреть глазами)."""
    df = pd.read_csv(path_or_df, usecols=["begin"]) if isinstance(path_or_df, (str, Path)) else path_or_df
    t = pd.to_datetime(df["begin"])
    days = t.dt.normalize().unique()
    bdays = pd.bdate_range(t.min().normalize(), t.max().normalize())
    return {"строк": len(df), "с": str(t.min()), "по": str(t.max()), "дублей": int(t.duplicated().sum()),
            "дней": len(days), "будних без свечей": int(len(bdays.difference(pd.DatetimeIndex(days))))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Свечи Мосбиржи через ISS")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("shares", "futures"):
        q = sub.add_parser(name)
        q.add_argument("codes", nargs="+", help="коды бумаг (SBER) или серий фьючерсов (SILV, Si, BR)")
        q.add_argument("--from", dest="start", required=True)
        q.add_argument("--till", dest="end")
        q.add_argument("--interval", type=int, default=1, choices=sorted(NAMES))
        q.add_argument("--out", type=Path, default=OUT)
        q.add_argument("--yadisk", action="store_true", help="после выгрузки перенести файлы на Яндекс Диск (~/bin/yd archive --link), на сервере — ссылка")
        if name == "futures":
            q.add_argument("--continuous", action="store_true", help="ещё и склеенный ряд <серия>_<интервал>_cont.csv")
            q.add_argument("--roll-days", type=int, default=5)
    q = sub.add_parser("futoi")
    q.add_argument("codes", nargs="+", help="префиксы фьючерсов: sv, gd, br, si")
    q.add_argument("--from", dest="start", default="2020-01-03")
    q.add_argument("--till", dest="end")
    q.add_argument("--out", type=Path, default=OUT)
    q.add_argument("--yadisk", action="store_true", help="после выгрузки перенести файлы на Яндекс Диск, на сервере — ссылка")
    q = sub.add_parser("list")
    q.add_argument("kind", choices=["shares", "futures"])
    a = p.parse_args(argv)
    if a.cmd == "futoi":
        done = [futoi(code, a.start, a.end, a.out / f"futoi_{code.lower()}.csv") for code in a.codes]
        return to_yadisk(done) if a.yadisk else 0
    if a.cmd == "list":
        with pd.option_context("display.max_rows", 500, "display.width", 160):
            print(tickers(a.kind).to_string(index=False))
        return 0
    done = []
    for code in a.codes:
        if a.cmd == "shares":
            f = a.out / f"{code}_{NAMES[a.interval]}.csv"
            print(check(download(code, a.start, a.end, a.interval, out=f)))
            done.append(f)
        else:
            done.append(a.out / code)
            w = download_futures(code, a.start, a.end, a.interval, a.out / code)
            print(f"[{code}] контрактов: {len(w)} ({w['secid'].iloc[0]} … {w['secid'].iloc[-1]})" if len(w) else f"[{code}] контрактов нет")
            if a.continuous and len(w):
                c = continuous(a.out / code, code, a.interval, a.roll_days, series=w)
                f = a.out / f"{code}_{NAMES[a.interval]}_cont.csv"
                c.to_csv(f, index=False)
                print(f"[{code}] склеено {len(c)} свечей → {f}", check(c))
                done.append(f)
    return to_yadisk(done) if a.yadisk else 0


def to_yadisk(paths: list[Path]) -> int:
    """Готовые выгрузки — на Яндекс Диск (yadisk:server-archive/<путь от домашней папки>), на сервере — ссылка: программы
    находят файл по прежнему пути, а место на сервере не занято (29.10: диск сервера заполняли большие выгрузки)."""
    yd = Path.home() / "bin" / "yd"
    if not yd.exists():
        print("~/bin/yd не найден — файлы остались на сервере")
        return 1
    return subprocess.run([str(yd), "archive", "--link", *map(str, paths)]).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
