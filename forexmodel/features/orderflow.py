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


def load_orderflow(path: Path | str, tz_shift_hours: int = 3) -> pd.DataFrame:
    """Читает parquet секундных агрегатов и переводит время в локальное (МСК)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл потока сделок не найден: {path} (data.orderflow_path)")

    of = pd.read_parquet(path, columns=_RAW_COLUMNS)
    t = pd.to_datetime(of["time"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    of["time"] = t + pd.Timedelta(hours=tz_shift_hours)
    of = of.sort_values("time").reset_index(drop=True)

    # ~10 млн строк: в int64/float64 это 700 МБ, а машина с 8 ГБ параллельно
    # держит ещё котировки. Объёмы и число сделок влезают в int32, цена и
    # средний размер сделки — в float32 (шаг цены 0.5 представим точно);
    # notional остаётся float64 — он до 1e9 и суммируется дальше
    for col in ("volume", "n_trades", "signed_vol"):
        of[col] = of[col].astype("int32")
    for col in ("buy_volume", "sell_volume", "price", "avg_trade_sz"):
        of[col] = of[col].astype("float32")
    log.info(
        "Поток сделок: %d секунд, %s .. %s (сдвиг +%dч к локальному времени)",
        len(of), of["time"].iloc[0], of["time"].iloc[-1], tz_shift_hours,
    )
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
        of_notional=("notional_sum", "sum"),
        of_large_vol=("_large_vol", "sum"),
        of_large_delta=("_large_delta", "sum"),
        of_big_vol=("_big_vol", "sum"),
        of_big_delta=("_big_delta", "sum"),
    )
    hourly["of_vwap"] = hourly["of_notional"] / hourly["of_volume"].replace(0, np.nan)

    profile = _rolling_volume_profile(of, hourly.index, cfg.orderflow_profile_bin, cfg.orderflow_profile_bars)
    hourly = hourly.join(profile)

    hourly.index.name = "time"
    log.info("Поток сделок агрегирован: %d баров %s", len(hourly), timeframe)
    return hourly.reset_index()


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
    """Приклеивает агрегаты к барам рабочего ТФ и считает безразмерные признаки.

    Там, где потока сделок нет (до начала данных), признаки остаются NaN —
    CatBoost работает с ними штатно, а нейросеть заполняет нулями.
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    of_bars = of_bars.copy()
    of_bars["time"] = pd.to_datetime(of_bars["time"])
    out = out.merge(of_bars, on="time", how="left")

    atr = out[atr_col].replace(0, np.nan)
    vol = out["of_volume"].replace(0, np.nan)
    close = out["close"]

    # --- дельта агрессора и крупные игроки за бар ---
    out["of_delta_ratio"] = out["of_delta"] / vol
    out["of_large_delta_ratio"] = out["of_large_delta"] / vol      # net_large_delta / объём бара
    out["of_large_share"] = out["of_large_vol"] / vol
    out["of_big_delta_ratio"] = out["of_big_delta"] / vol
    out["of_big_share"] = out["of_big_vol"] / vol

    # --- скользящие окна: накопленная дельта (footprint) и её дивергенция с ценой ---
    for w in cfg.orderflow_windows:
        vol_w = out["of_volume"].rolling(w).sum().replace(0, np.nan)
        out[f"of_delta_ratio_{w}"] = out["of_delta"].rolling(w).sum() / vol_w
        out[f"of_large_delta_ratio_{w}"] = out["of_large_delta"].rolling(w).sum() / vol_w
        out[f"of_big_delta_ratio_{w}"] = out["of_big_delta"].rolling(w).sum() / vol_w

        # z-score накопленной дельты крупных: насколько текущее окно необычно
        ld = out["of_large_delta"].rolling(w).sum()
        out[f"of_large_delta_z_{w}"] = (ld - ld.rolling(w * 4).mean()) / ld.rolling(w * 4).std().replace(0, np.nan)

        # дивергенция: крупные покупают, а цена не растёт -> аккумуляция (>0);
        # крупные продают, а цена не падает -> дистрибуция (<0)
        price_move = (close - close.shift(w)) / (atr * np.sqrt(w))
        out[f"of_ld_price_div_{w}"] = out[f"of_large_delta_ratio_{w}"] - np.tanh(price_move)

    # --- активность ---
    out["of_trades_rel_24"] = out["of_trades"] / out["of_trades"].rolling(24).mean().replace(0, np.nan)
    avg_sz = out["of_volume"] / out["of_trades"].replace(0, np.nan)
    out["of_avg_sz_rel_24"] = avg_sz / avg_sz.rolling(24).mean().replace(0, np.nan)

    # --- цена относительно потока ---
    out["of_vwap_dev_atr"] = (close - out["of_vwap"]) / atr
    out["of_poc_dist_atr"] = (close - out["of_poc"]) / atr
    va_width = (out["of_va_high"] - out["of_va_low"])
    out["of_va_pos"] = (close - out["of_va_low"]) / va_width.replace(0, np.nan)   # 0..1 внутри VA
    out["of_va_width_atr"] = va_width / atr

    # сырые суммы в лотах и абсолютные уровни цены в модель не идут
    raw = [
        "of_volume", "of_buy", "of_sell", "of_delta", "of_trades", "of_notional",
        "of_large_vol", "of_large_delta", "of_big_vol", "of_big_delta",
        "of_vwap", "of_poc", "of_va_low", "of_va_high",
    ]
    out = out.drop(columns=[c for c in raw if c in out.columns])

    covered = out["of_delta_ratio"].notna().mean()
    log.info("Признаков потока сделок: %d; покрытие баров: %.1f%%", len(orderflow_columns(cfg)), 100 * covered)
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
