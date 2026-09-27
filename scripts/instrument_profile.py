"""Профиль инструмента: насколько он склонен к трендам, откатам и резким разворотам.

Ответ на вопрос «можно ли математически описать сам инструмент»: у одних бумаг
тренд, начавшись, держится сутками, у других направление меняется каждый час.
Здесь на часовых барах считается набор безразмерных величин, по которым
инструменты (и годы одного инструмента) можно сравнивать между собой:

  * `vr_4h`, `vr_24h`  — variance ratio Ло–Маккинлея: дисперсия доходности за q
    часов, делённая на q дисперсий часовой доходности. > 1 — движения
    накапливаются (тренд), < 1 — гасятся (возврат к среднему), 1 — случайное
    блуждание;
  * `acf_1h`, `acf_4h` — автокорреляция часовой (и 4-часовой) доходности с лагом 1:
    знак говорит то же самое, что VR, но на самом коротком горизонте;
  * `hurst`           — показатель Хёрста (DFA по логарифму цены): > 0.5 тренд,
    < 0.5 возврат к среднему;
  * `p_cont`          — вероятность продолжения: если за прошлые 4 часа цена прошла
    больше 1 ATR, с какой вероятностью следующие 4 часа идут туда же;
  * `er_24_mean`, `er_24_hi` — коэффициент эффективности Кауфмана за 24 часа
    (|чистый ход| / сумма |часовых ходов|): средний и доля окон с ER > 0.5.
    Высокий ER — путь прямой, низкий — пила;
  * `trend_share`, `trend_len_h`, `reversals_month` — доля часов внутри «видимого»
    тренда (ход > 3 ATR за ±12 часов, задним числом), медианная длина такого
    эпизода и сколько раз в месяц направление видимого тренда меняется;
  * `v_turns_month`   — резкие развороты: ход > 1.5 ATR за 4 часа и сразу > 1.5 ATR
    обратно за следующие 4;
  * `kurtosis`, `q99_atr` — тяжесть хвостов часовой доходности и размер
    1%-самого крупного часового бара в ATR;
  * `hour_vol_ratio`  — сезонность: волатильность самого активного часа суток к
    самому тихому (у модели треть важности — час суток, это её почва).

Разрывы длиннее 12 часов (выходные) из расчётов исключены; ночной разрыв
LKOH и ежедневный часовой перерыв у металлов считаются обычным шагом ряда,
иначе для них не набирается ни одного непрерывного суточного окна.
`p_cont_24` — то же, что `p_cont`, но на сутках: ход > 2 ATR за 24 часа и
следующие 24. `trend_len_mean`, `long_trend_share` — средняя длина эпизода
видимого тренда и доля часов внутри эпизодов длиннее 12 часов.

    python scripts/instrument_profile.py                       # silver, gold, lkoh
    python scripts/instrument_profile.py --by-year silver lkoh

Минутные CSV читаются кусками в float32 и кэшируются в reports/_cache/
(`load_minute_compact`), часовые бары — там же, <name>_1h.pkl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from forexmodel.data.loader import load_minute_compact, resample_ohlcv
from forexmodel.features import indicators as ind

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "reports" / "_cache"

#: имя -> (минутный CSV, колонка времени)
INSTRUMENTS = {
    "silver": ("data/raw/XAGUSD_1m_full.csv", "time"),
    "gold": ("data/raw/XAUUSD_1m_full.csv", "time"),
    "lkoh": ("data/raw/LKOH_1m_15-26.csv", "begin"),
}


def hourly_bars(name: str) -> pd.DataFrame:
    cache = CACHE / f"{name}_1h.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    csv, time_col = INSTRUMENTS[name]
    h = resample_ohlcv(load_minute_compact(ROOT / csv, time_col, CACHE), "1h")
    cache.parent.mkdir(parents=True, exist_ok=True)
    h.to_pickle(cache)
    return h


def _hurst_dfa(x: np.ndarray, scales=(16, 32, 64, 128, 256)) -> float:
    """Detrended fluctuation analysis по кумулятивному ряду доходностей."""
    y = np.cumsum(x - x.mean())
    fl = []
    for s in scales:
        n = len(y) // s
        if n < 4:
            continue
        seg = y[: n * s].reshape(n, s)
        t = np.arange(s)
        coef = np.polyfit(t, seg.T, 1)
        fit = coef[0][:, None] * t + coef[1][:, None]
        fl.append(np.sqrt(np.mean((seg - fit) ** 2)))
    if len(fl) < 3:
        return np.nan
    used = [s for s in scales if len(x) // s >= 4][: len(fl)]
    return float(np.polyfit(np.log(used), np.log(fl), 1)[0])


def profile(h: pd.DataFrame) -> dict:
    h = h.sort_values("time").reset_index(drop=True)
    atr = ind.atr(h, 14)
    step = h["time"].diff()
    contiguous = step <= pd.Timedelta("12h")                  # рвём ряд только на выходных
    r = np.log(h["close"]).diff().where(contiguous)
    r1 = r.dropna().to_numpy()

    # 4-часовые доходности по непрерывным окнам
    r4 = np.log(h["close"]).diff(4).where(contiguous.rolling(4).sum() == 4)
    r24 = np.log(h["close"]).diff(24).where(contiguous.rolling(24).sum() == 24)
    vr4 = float(r4.var() / (4 * r.var()))
    vr24 = float(r24.var() / (24 * r.var()))
    acf1 = float(pd.Series(r1[1:]).corr(pd.Series(r1[:-1])))
    r4c = r4.dropna().iloc[::4].to_numpy()
    acf4 = float(pd.Series(r4c[1:]).corr(pd.Series(r4c[:-1])))

    # продолжение после хода > 1 ATR за 4 часа
    past = (h["close"] - h["close"].shift(4)) / atr
    fut = (h["close"].shift(-4) - h["close"]) / atr
    ok = (past.abs() > 1) & contiguous.rolling(4).sum().eq(4) & contiguous.shift(-4).rolling(4).sum().eq(4)
    p_cont = float((np.sign(past[ok]) == np.sign(fut[ok])).mean())

    # эффективность пути за 24 часа
    net = (h["close"] - h["close"].shift(24)).abs()
    path = h["close"].diff().abs().rolling(24).sum()
    er = (net / path.replace(0, np.nan)).where(contiguous.rolling(24).sum() == 24).dropna()

    # видимый тренд задним числом
    move = (h["close"].shift(-12) - h["close"].shift(12)) / atr
    truth = pd.Series(np.where(move > 3, 1, np.where(move < -3, -1, 0)), index=h.index)
    runs = truth.groupby((truth != truth.shift()).cumsum()).agg(["first", "size"])
    lab_runs = runs[runs["first"] != 0]
    months = max((h["time"].iloc[-1] - h["time"].iloc[0]).days / 30.44, 1)
    sign_changes = int((np.sign(lab_runs["first"]).diff().fillna(0) != 0).sum())

    # резкие развороты
    v = (past.abs() > 1.5) & (fut.abs() > 1.5) & (np.sign(past) == -np.sign(fut)) & ok
    past24 = (h["close"] - h["close"].shift(24)) / atr
    fut24 = (h["close"].shift(-24) - h["close"]) / atr
    ok24 = (past24.abs() > 2) & contiguous.rolling(24).sum().eq(24) & contiguous.shift(-24).rolling(24).sum().eq(24)
    p_cont24 = float((np.sign(past24[ok24]) == np.sign(fut24[ok24])).mean())
    long_runs = lab_runs[lab_runs["size"] >= 12]
    # сезонность волатильности по часу суток (p90/p10 часов, а не max/min — устойчиво к пустым часам)
    by_hour = r.abs().groupby(h["time"].dt.hour).mean().dropna()

    return {
        "часов": len(h),
        "ATR%": round(float((atr / h["close"]).median() * 100), 3),
        "vr_4h": round(vr4, 3),
        "vr_24h": round(vr24, 3),
        "acf_1h": round(acf1, 3),
        "acf_4h": round(acf4, 3),
        "hurst": round(_hurst_dfa(r1), 3),
        "p_cont": round(p_cont, 3),
        "p_cont_24": round(p_cont24, 3),
        "er_24_mean": round(float(er.mean()), 3),
        "er_24_hi": round(float((er > 0.5).mean()), 3),
        "trend_share": round(float((truth != 0).mean()), 3),
        "trend_len_mean": round(float(lab_runs["size"].mean()), 1) if len(lab_runs) else np.nan,
        "long_trend_share": round(float(long_runs["size"].sum() / len(h)), 3),
        "reversals_month": round(sign_changes / months, 2),
        "v_turns_month": round(float(v.sum()) / months, 2),
        "kurtosis": round(float(pd.Series(r1).kurt()), 1),
        "q99_atr": round(float((h["close"].diff().abs() / atr).where(contiguous).quantile(0.99)), 2),
        "hour_vol_ratio": round(float(by_hour.quantile(0.9) / by_hour.quantile(0.1)), 2) if len(by_hour) > 1 else np.nan,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help=f"инструменты из {sorted(INSTRUMENTS)}; по умолчанию все")
    ap.add_argument("--by-year", action="store_true", help="дополнительно разбивка по годам")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 250)

    names = args.names or list(INSTRUMENTS)
    rows, yearly = [], []
    for name in names:
        h = hourly_bars(name)
        rows.append({"инструмент": name, **profile(h)})
        if args.by_year:
            for year, hy in h.groupby(h["time"].dt.year):
                if len(hy) < 24 * 60:
                    continue
                yearly.append({"инструмент": name, "год": year, **profile(hy.reset_index(drop=True))})

    table = pd.DataFrame(rows).set_index("инструмент").T
    print("\nПРОФИЛЬ ИНСТРУМЕНТОВ (часовые бары, вся история)")
    print(table.to_string())
    out = ROOT / "reports" / "instrument_profile.csv"
    table.to_csv(out)
    if yearly:
        ydf = pd.DataFrame(yearly)
        keep = ["vr_4h", "vr_24h", "acf_1h", "hurst", "p_cont", "p_cont_24", "er_24_mean", "trend_share", "long_trend_share", "reversals_month", "v_turns_month", "hour_vol_ratio"]
        print("\nПО ГОДАМ")
        print(ydf.set_index(["инструмент", "год"])[keep].to_string())
        ydf.to_csv(ROOT / "reports" / "instrument_profile_by_year.csv", index=False)
    print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
