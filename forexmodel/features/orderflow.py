"""Признаки потока сделок (order flow) из секундных агрегатов.

Источник — секундные агрегаты ленты сделок (MOEX ALGOPACK): на каждую секунду
объём, число сделок, объём по стороне агрессора (buy/sell), средний размер
сделки. Отдельных сделок в них нет, поэтому из ноутбука S_Volume_Analysis
переносятся три блока, работающие на агрегатах:

  1. Volume profile — горизонтальные объёмы по уровням цены за скользящее окно:
     POC (уровень максимального объёма) и Value Area (70% объёма вокруг POC).
     В модель уходят расстояние до POC в ATR и положение цены внутри VA.
  2. Дельта агрессора — buy − sell за бар и за скользящие окна, нормированная
     на объём. Это то, чего нет в котировках: кто был инициатором сделок.
  3. Крупные игроки (`large_player_footprint`) — нетто-дельта только КРУПНЫХ
     сделок (`net_large_delta`) и её накопление за окно. Гипотеза ноутбука:
     рост накопленной дельты крупных при плоской цене — тихая аккумуляция,
     падение при растущей — дистрибуция. Для этого считается и явная
     дивергенция «дельта крупных минус ход цены».

«Крупная» секунда определяется квантилем объёма, посчитанным по ПРЕДЫДУЩИМ
дням — иначе порог знал бы будущее распределение. Iceberg-детектор из
ноутбука сюда не перенесён: ему нужны отдельные сделки, а не секундные суммы.

Все признаки безразмерные (доли, ATR, z-score): абсолютный объём в лотах
кодировал бы эпоху так же, как абсолютная цена.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..logging_utils import get_logger
from ..progress import Progress

log = get_logger(__name__)

__all__ = ["load_orderflow", "aggregate_orderflow", "add_orderflow_features", "orderflow_columns"]

_RAW_COLUMNS = [
    "time", "price", "volume", "n_trades", "signed_vol", "buy_volume", "sell_volume",
    "notional_sum", "avg_trade_sz",
]
_TICKER_COLUMN = "TICKER_CC"   # есть в parquet ALGOPACK; у фьючерсов — разные контракты


def load_orderflow(path: Path | str, tz_shift_hours: int = 3, tz: str | None = None) -> pd.DataFrame:
    """Читает parquet секундных агрегатов и переводит время во время котировок.

    `tz` — IANA-зона котировок: конвертация с учётом летнего времени (нью-йоркский
    форекс-фид живёт по UTC−5 зимой и UTC−4 летом, и фиксированный сдвиг полгода
    промахивался бы на час). Без `tz` — фиксированный сдвиг, для МСК его достаточно.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл потока сделок не найден: {path} (data.orderflow_path)")

    import pyarrow.parquet as pq

    have_ticker = _TICKER_COLUMN in pq.ParquetFile(path).schema_arrow.names
    of = pd.read_parquet(path, columns=_RAW_COLUMNS + ([_TICKER_COLUMN] if have_ticker else []))
    if have_ticker:
        of[_TICKER_COLUMN] = of[_TICKER_COLUMN].astype("category")
    t = pd.to_datetime(of["time"])
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize("UTC")
    if tz:
        of["time"] = t.dt.tz_convert(tz).dt.tz_localize(None)
        shift_note = f"зона {tz}"
    else:
        of["time"] = t.dt.tz_convert(None) + pd.Timedelta(hours=tz_shift_hours)
        shift_note = f"сдвиг +{tz_shift_hours}ч"
    of = of.sort_values("time").reset_index(drop=True)

    # ~10 млн строк: в int64/float64 это 700 МБ, а машина с 8 ГБ параллельно
    # держит ещё котировки. Объёмы и число сделок влезают в int32, цена и
    # средний размер сделки — в float32 (шаг цены 0.5 представим точно);
    # notional остаётся float64 — он до 1e9 и суммируется дальше
    for col in ("volume", "n_trades", "signed_vol"):
        of[col] = of[col].astype("int32")
    for col in ("buy_volume", "sell_volume", "price", "avg_trade_sz"):
        of[col] = of[col].astype("float32")
    log.info("Поток сделок: %d секунд, %s .. %s (%s)", len(of), of["time"].iloc[0], of["time"].iloc[-1], shift_note)
    return of


def _causal_daily_threshold(of: pd.DataFrame, col: str, quantile: float, lookback_days: int) -> pd.Series:
    """Порог «крупности» на день D = средний квантиль за предыдущие lookback_days дней.

    Квантиль по всему файлу знал бы будущее; квантиль по текущему дню — тоже
    (он известен только к вечеру). Первый день остаётся без порога, и крупных
    сделок в нём нет — это честная цена каузальности.
    """
    day = of["time"].dt.normalize()
    daily_q = of.groupby(day)[col].quantile(quantile)
    threshold = daily_q.shift(1).rolling(lookback_days, min_periods=1).mean()
    return day.map(threshold)


def aggregate_orderflow(of: pd.DataFrame, cfg: FeatureConfig, timeframe: str = "1h") -> pd.DataFrame:
    """Секунды -> бары рабочего ТФ. Возвращает сырые суммы (с префиксом of_) и
    уровни volume profile; безразмерные признаки из них делает
    `add_orderflow_features`.

    Кадр меняется НА МЕСТЕ (добавляются служебные колонки): копия секундных
    строк — это ещё полгигабайта, а вызывающий код удаляет `of` сразу после."""
    of["bar"] = of["time"].dt.floor(timeframe)

    thr_sec = _causal_daily_threshold(of, "volume", cfg.orderflow_large_quantile, cfg.orderflow_large_lookback_days)
    thr_trade = _causal_daily_threshold(of, "avg_trade_sz", cfg.orderflow_large_quantile, cfg.orderflow_large_lookback_days)
    large_sec = of["volume"] >= thr_sec          # много объёма за секунду
    large_trade = of["avg_trade_sz"] >= thr_trade  # крупные сделки в среднем
    log.info(
        "Крупные секунды: %.1f%% (порог ~%.0f лотов), крупные по размеру сделки: %.1f%% (порог ~%.1f)",
        100 * large_sec.mean(), float(thr_sec.median()), 100 * large_trade.mean(), float(thr_trade.median()),
    )

    of["_large_vol"] = np.where(large_sec, of["volume"], 0).astype("int32")
    of["_large_delta"] = np.where(large_sec, of["signed_vol"], 0).astype("int32")
    of["_big_vol"] = np.where(large_trade, of["volume"], 0).astype("int32")
    of["_big_delta"] = np.where(large_trade, of["signed_vol"], 0).astype("int32")
    del thr_sec, thr_trade, large_sec, large_trade

    hourly = of.groupby("bar").agg(
        of_volume=("volume", "sum"),
        of_buy=("buy_volume", "sum"),
        of_sell=("sell_volume", "sum"),
        of_delta=("signed_vol", "sum"),
        of_trades=("n_trades", "sum"),
        of_large_vol=("_large_vol", "sum"),
        of_large_delta=("_large_delta", "sum"),
        of_big_vol=("_big_vol", "sum"),
        of_big_delta=("_big_delta", "sum"),
    )

    # Ценовые величины — только по ФРОНТ-контракту бара (максимум объёма в баре).
    # У фьючерсов одновременно торгуются несколько сроков, и базис между ними —
    # проценты (у серебра на MOEX медиана 5.6%): смешав их, получаем размах цены
    # за бар 6% вместо 0.4%, прыгающий между контрактами close и профиль объёма,
    # размазанный по уровням, отстоящим на проценты. Объёмы же суммируются по всем:
    # фронт держит ~97% оборота, и дельта агрессора от этого не страдает.
    front = _front_contract_rows(of)
    price_src = of[front] if front is not None else of
    hourly = hourly.join(
        price_src.groupby("bar").agg(
            of_notional=("notional_sum", "sum"),
            _front_volume=("volume", "sum"),
            of_close=("price", "last"),
            of_high=("price", "max"),
            of_low=("price", "min"),
        )
    )
    hourly["of_vwap"] = hourly["of_notional"] / hourly["_front_volume"].replace(0, np.nan)
    hourly = hourly.drop(columns=["_front_volume"])

    profile = _rolling_volume_profile(price_src, hourly.index, cfg.orderflow_profile_bin, cfg.orderflow_profile_bars)
    hourly = hourly.join(profile)

    hourly.index.name = "time"
    log.info("Поток сделок агрегирован: %d баров %s", len(hourly), timeframe)
    return hourly.reset_index()


def _front_contract_rows(of: pd.DataFrame):
    """Маска строк фронт-контракта каждого бара (контракт с наибольшим объёмом в баре).

    None, если колонки тикера нет или тикер один — тогда фильтровать нечего.
    """
    if _TICKER_COLUMN not in of.columns or of[_TICKER_COLUMN].nunique() <= 1:
        return None
    per_contract = of.groupby(["bar", _TICKER_COLUMN], observed=True)["volume"].sum()
    front = per_contract.groupby(level=0).idxmax().map(lambda ix: ix[1])   # bar -> тикер
    mask = of[_TICKER_COLUMN].astype(str).to_numpy() == of["bar"].map(front).astype(str).to_numpy()
    log.info(
        "Контрактов в ленте: %d; ценовые величины — по фронт-контракту бара (%.1f%% строк, %.1f%% объёма)",
        of[_TICKER_COLUMN].nunique(), 100 * mask.mean(), 100 * of.loc[mask, "volume"].sum() / of["volume"].sum(),
    )
    return mask


def _rolling_volume_profile(of: pd.DataFrame, bars: pd.Index, bin_size: float, window: int) -> pd.DataFrame:
    """POC и Value Area (70%) по скользящему окну из `window` баров.

    Реализация: матрица «бар × ценовой бин» с объёмами, скользящая сумма по
    барам через cumsum, затем на каждом баре — argmax (POC) и классическое
    расширение от POC до 70% объёма (VA). Окно включает текущий бар: его
    сделки к закрытию бара уже известны, как и его close.
    """
    price_bin = (of["price"] / bin_size).round() * bin_size
    table = of.groupby(["bar", price_bin])["volume"].sum().unstack(fill_value=0.0)
    table = table.reindex(bars, fill_value=0.0)
    levels = table.columns.to_numpy(dtype=float)
    vol = table.to_numpy(dtype=float)

    cum = np.cumsum(vol, axis=0)
    rolling = cum.copy()
    rolling[window:] -= cum[:-window]

    n = len(bars)
    poc = np.full(n, np.nan)
    va_low = np.full(n, np.nan)
    va_high = np.full(n, np.nan)
    bar = Progress(n, label="Volume profile", unit="бар", logger=log, min_interval=15.0, min_total=2000)

    for i in range(n):
        bar.update()
        row = rolling[i]
        total = row.sum()
        if total <= 0:
            continue
        p = int(row.argmax())
        poc[i] = levels[p]
        lo = hi = p
        acc = row[p]
        target = 0.7 * total
        while acc < target and (lo > 0 or hi < len(row) - 1):
            down = row[lo - 1] if lo > 0 else -1.0
            up = row[hi + 1] if hi < len(row) - 1 else -1.0
            if down >= up:
                lo -= 1
                acc += row[lo]
            else:
                hi += 1
                acc += row[hi]
        va_low[i], va_high[i] = levels[lo], levels[hi]

    return pd.DataFrame({"of_poc": poc, "of_va_low": va_low, "of_va_high": va_high}, index=bars)


def add_orderflow_features(
    df: pd.DataFrame,
    of_bars: pd.DataFrame,
    cfg: FeatureConfig,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Считает безразмерные признаки потока и приклеивает их к барам рабочего ТФ.

    Все скользящие окна считаются на СОБСТВЕННЫХ барах потока сделок, а не на
    барах котировок. Причина: если котировки живут 24/5 (спот XAGUSD), а лента —
    только в часы биржи (фьючерсы MOEX, ~15 часов в сутки), то в любое 24-барное
    окно по барам котировок попадает ночной разрыв без ленты, и `rolling` даёт
    NaN на каждом баре — так у серебра целиком выпали все 24-барные признаки и
    z-score. На собственных барах потока «24 бара» — это 24 торговых часа подряд,
    как и задумано. Для LKOH обе сетки совпадают, и результат тот же.

    Там, где потока сделок нет (ночь, выходные, до начала данных), признаки
    остаются NaN — CatBoost работает с ними штатно, нейросеть заполняет нулями.
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    f = of_bars.copy()
    f["time"] = pd.to_datetime(f["time"])
    f = f.sort_values("time").reset_index(drop=True)

    # котировки подклеиваем К ПОТОКУ (а не наоборот): close/ATR нужны только для
    # дивергенции и, при одном инструменте, для ценовых признаков
    quotes = out[["time", "close", atr_col]].rename(columns={atr_col: "_atr"})
    f = f.merge(quotes, on="time", how="left")

    vol = f["of_volume"].replace(0, np.nan)
    if cfg.orderflow_same_instrument:
        ref_price = f["close"]
        scale = f["_atr"].replace(0, np.nan)
    else:
        # фьючерс против спота: в разность цен попал бы базис, а контракты разных
        # сроков ещё и торгуются одновременно — опора и масштаб только свои
        ref_price = f["of_close"]
        scale = (f["of_high"] - f["of_low"]).rolling(14).mean().replace(0, np.nan)

    # --- дельта агрессора и крупные игроки за бар ---
    f["of_delta_ratio"] = f["of_delta"] / vol
    f["of_large_delta_ratio"] = f["of_large_delta"] / vol      # net_large_delta / объём бара
    f["of_large_share"] = f["of_large_vol"] / vol
    f["of_big_delta_ratio"] = f["of_big_delta"] / vol
    f["of_big_share"] = f["of_big_vol"] / vol

    # --- скользящие окна: накопленная дельта (footprint) и её дивергенция с ценой ---
    for w in cfg.orderflow_windows:
        vol_w = f["of_volume"].rolling(w).sum().replace(0, np.nan)
        f[f"of_delta_ratio_{w}"] = f["of_delta"].rolling(w).sum() / vol_w
        f[f"of_large_delta_ratio_{w}"] = f["of_large_delta"].rolling(w).sum() / vol_w
        f[f"of_big_delta_ratio_{w}"] = f["of_big_delta"].rolling(w).sum() / vol_w

        # z-score накопленной дельты крупных: насколько текущее окно необычно
        ld = f["of_large_delta"].rolling(w).sum()
        f[f"of_large_delta_z_{w}"] = (ld - ld.rolling(w * 4).mean()) / ld.rolling(w * 4).std().replace(0, np.nan)

        # дивергенция: крупные покупают, а цена не растёт -> аккумуляция (>0);
        # крупные продают, а цена не падает -> дистрибуция (<0)
        price_move = (ref_price - ref_price.shift(w)) / (scale * np.sqrt(w))
        f[f"of_ld_price_div_{w}"] = f[f"of_large_delta_ratio_{w}"] - np.tanh(price_move)

    # --- активность ---
    f["of_trades_rel_24"] = f["of_trades"] / f["of_trades"].rolling(24).mean().replace(0, np.nan)
    avg_sz = f["of_volume"] / f["of_trades"].replace(0, np.nan)
    f["of_avg_sz_rel_24"] = avg_sz / avg_sz.rolling(24).mean().replace(0, np.nan)

    # --- цена относительно потока ---
    f["of_vwap_dev_atr"] = (ref_price - f["of_vwap"]) / scale
    f["of_poc_dist_atr"] = (ref_price - f["of_poc"]) / scale
    va_width = f["of_va_high"] - f["of_va_low"]
    f["of_va_pos"] = (ref_price - f["of_va_low"]) / va_width.replace(0, np.nan)   # 0..1 внутри VA
    f["of_va_width_atr"] = va_width / scale

    # в модель идут только безразмерные признаки; сырые суммы в лотах и
    # абсолютные уровни цены остаются здесь
    cols = orderflow_columns(cfg)
    out = out.merge(f[["time"] + cols], on="time", how="left")

    covered = out["of_delta_ratio"].notna().mean()
    all_nan = [c for c in cols if out[c].isna().all()]
    log.info("Признаков потока сделок: %d; покрытие баров: %.1f%%", len(cols), 100 * covered)
    if all_nan:
        log.warning("Признаки потока целиком NaN (проверьте окна и выравнивание времени): %s", all_nan)
    return out


def orderflow_columns(cfg: FeatureConfig) -> List[str]:
    """Имена признаков потока сделок (для мета-модели и отчётов)."""
    cols = ["of_delta_ratio", "of_large_delta_ratio", "of_large_share", "of_big_delta_ratio", "of_big_share"]
    for w in cfg.orderflow_windows:
        cols += [
            f"of_delta_ratio_{w}", f"of_large_delta_ratio_{w}", f"of_big_delta_ratio_{w}",
            f"of_large_delta_z_{w}", f"of_ld_price_div_{w}",
        ]
    cols += ["of_trades_rel_24", "of_avg_sz_rel_24", "of_vwap_dev_atr", "of_poc_dist_atr", "of_va_pos", "of_va_width_atr"]
    return cols
