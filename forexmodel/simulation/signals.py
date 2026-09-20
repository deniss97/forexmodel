"""Формирование торгового сигнала из предсказаний моделей.

Что исправлено относительно ноутбука:

  * ансамбль собирался построчным циклом `for idx in sig.index` с `.at[]` —
    на длинной выборке это минуты работы; здесь то же правило векторно;
  * мета-модель фактически не работала: `apply_meta_filter` писала
    `final_signal`, а `run_simulator` вызывался с `signal_source="ensemble"`
    и читал `y_pred_cb`/`y_pred_nn`, то есть фильтр не влиял ни на одну сделку.
    Теперь `signal_source="meta"` — явный режим, а не побочный эффект;
  * порог входа может считаться не по «уверенности >= 0.5», а по ожидаемой
    ценности: p*TP - (1-p)*SL - комиссия > порога.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "attach_model_predictions",
    "build_signal_column",
    "expected_value",
    "wire_meta_signal",
]

LONG, SHORT = 2, 0


def attach_model_predictions(
    df: pd.DataFrame,
    cb_preds: Optional[pd.DataFrame] = None,
    nn_preds: Optional[pd.DataFrame] = None,
    time_col: str = "time",
) -> pd.DataFrame:
    """Приклеивает предсказания CatBoost/NN к барам рабочего ТФ."""
    out = df.reset_index(drop=True).copy()

    for preds, suffix in ((cb_preds, "cb"), (nn_preds, "nn")):
        if preds is None:
            continue
        cols = {c: f"{c}_{suffix}" for c in preds.columns if c != time_col}
        out = out.merge(preds.rename(columns=cols), on=time_col, how="left")

    if cb_preds is not None:
        coverage = out["y_pred_cb"].notna().mean()
        log.info("Покрытие предсказаниями CatBoost: %.1f%%", 100 * coverage)
    if nn_preds is not None:
        log.info("Покрытие предсказаниями NN: %.1f%%", 100 * out["y_pred_nn"].notna().mean())
    return out


def _numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Колонка как float-Series; если её нет — Series из NaN нужной длины."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def expected_value(df: pd.DataFrame, direction: int, cfg: Config) -> pd.Series:
    """Ожидаемая ценность сделки в % от цены входа.

    p * TP - (1 - p) * SL - комиссия, где p — вероятность целевого класса по
    primary-модели. Порог «confidence >= 0.5» этой величины не учитывает:
    при TP=SL сделка с p=0.5 — это минус комиссия, то есть заведомый убыток.
    """
    sim = cfg.simulation
    proba_col = f"proba_{direction}_cb"
    if proba_col not in df.columns:
        raise KeyError(f"Нет колонки {proba_col} — фильтр по ожидаемой ценности требует вероятностей по классам")

    p = df[proba_col].astype(float)

    if sim.exit_mode == "fixed_pct":
        tp_pct = pd.Series(sim.take_profit_pct, index=df.index, dtype=float)
        sl_pct = pd.Series(sim.stop_loss_pct, index=df.index, dtype=float)
    else:
        atr_share = df[sim.atr_col] / df["close"] * 100.0
        tp_pct = sim.tp_atr * atr_share
        sl_pct = sim.sl_atr * atr_share

    return p * tp_pct - (1 - p) * sl_pct - sim.commission_pct


def build_signal_column(df: pd.DataFrame, cfg: Config, signal_col: str = "final_class") -> pd.DataFrame:
    """Единая точка, где решается «торгуем ли мы на этом баре и в какую сторону»."""
    sim = cfg.simulation
    out = df.copy()

    cb = _numeric_column(out, "y_pred_cb")
    nn = _numeric_column(out, "y_pred_nn")
    conf_cb = _numeric_column(out, "confidence_cb")
    conf_nn = _numeric_column(out, "confidence_nn")

    if sim.min_conf_cb is not None:
        cb = cb.where(conf_cb >= sim.min_conf_cb)
    if sim.min_conf_nn is not None:
        nn = nn.where(conf_nn >= sim.min_conf_nn)

    source = sim.signal_source
    if source == "cb":
        signal = cb
    elif source == "nn":
        signal = nn
    elif source == "meta":
        if "final_signal" not in out.columns:
            raise KeyError("signal_source='meta', но колонки final_signal нет — сначала примените мета-модель")
        signal = pd.to_numeric(out["final_signal"], errors="coerce")
    elif source == "ensemble":
        signal = _ensemble(cb, nn, conf_cb, conf_nn, sim.ensemble_rule)
    elif source == "rule":
        signal = _rule_signal(out, sim.rule_feature, sim.rule_threshold, sim.rule_invert)
    else:
        raise ValueError(f"Неизвестный signal_source: {source}")

    signal = signal.where(signal.isin([LONG, SHORT]))
    n_raw = int(signal.notna().sum())

    if sim.use_expected_value_filter and source != "nn":
        ev = pd.Series(np.nan, index=out.index)
        for direction in (LONG, SHORT):
            mask = signal == direction
            if mask.any():
                ev.loc[mask] = expected_value(out.loc[mask], direction, cfg)
        out["expected_value"] = ev
        signal = signal.where(ev > sim.min_expected_value_pct)
        log.info("Фильтр ожидаемой ценности: %d -> %d сигналов", n_raw, int(signal.notna().sum()))

    if sim.max_extension_atr is not None:
        signal = _extension_filter(out, signal, sim.max_extension_atr, sim.extension_col)

    out[signal_col] = signal
    log.info("Итого сигналов: %d (%s / %s)", int(signal.notna().sum()), source, sim.ensemble_rule)
    return out


def _rule_signal(df: pd.DataFrame, feature: str, threshold: float, invert: bool) -> pd.Series:
    """Сигнал по одному признаку без модели: проверка гипотезы в чистом виде.

    Если гипотеза «крупные покупают -> цена растёт» верна, то сама
    `of_large_delta_z_6` выше порога должна давать прибыльные лонги — без
    CatBoost, мета-модели и прочего. Если не даёт, никакая модель поверх
    этот признак не спасёт; если даёт — модель обязана его использовать.
    """
    if feature not in df.columns:
        raise KeyError(f"signal_source='rule': признака {feature!r} нет среди колонок")
    x = pd.to_numeric(df[feature], errors="coerce")
    long_mask, short_mask = x >= threshold, x <= -threshold
    if invert:
        long_mask, short_mask = short_mask, long_mask

    signal = pd.Series(np.nan, index=df.index, dtype=float)
    signal[long_mask] = LONG
    signal[short_mask] = SHORT
    log.info(
        "Правило %s %s %.3g: лонгов %d, шортов %d из %d баров",
        feature, "<=/>=" if not invert else "инвертировано", threshold,
        int(long_mask.sum()), int(short_mask.sum()), len(df),
    )
    return signal


def _ensemble(cb: pd.Series, nn: pd.Series, conf_cb: pd.Series, conf_nn: pd.Series, rule: str) -> pd.Series:
    polar_cb = cb.where(cb.isin([LONG, SHORT]))
    polar_nn = nn.where(nn.isin([LONG, SHORT]))

    if rule == "agreement":
        return polar_cb.where(polar_cb == polar_nn)
    if rule == "cb_priority":
        return polar_cb.fillna(polar_nn)
    if rule == "nn_priority":
        return polar_nn.fillna(polar_cb)
    if rule == "confidence":
        conf_cb = conf_cb.fillna(0.0)
        conf_nn = conf_nn.fillna(0.0)
        best = polar_nn.where(conf_nn >= conf_cb, polar_cb)
        return best.fillna(polar_cb).fillna(polar_nn)
    raise ValueError(f"Неизвестное правило ансамбля: {rule}")


def _extension_filter(df: pd.DataFrame, signal: pd.Series, max_ext: float, ext_col: str) -> pd.Series:
    """Грубый отсев поздних входов до переобучения модели.

    Для лонга смотрим, сколько ATR пройдено от минимума окна, для шорта — от
    максимума. Пара колонок определяется автоматически по имени ext_col.
    """
    low_col = ext_col
    high_col = ext_col.replace("ext_from_low", "ext_from_high")
    if low_col not in df.columns or high_col not in df.columns:
        log.warning("Колонок %s/%s нет — фильтр по растяжению пропущен", low_col, high_col)
        return signal

    too_late = ((signal == LONG) & (df[low_col] > max_ext)) | ((signal == SHORT) & (df[high_col] > max_ext))
    log.info("Фильтр растяжения (< %.2f ATR): отброшено %d сигналов", max_ext, int(too_late.sum()))
    return signal.where(~too_late)


def wire_meta_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Совместимость со старым кодом: перенести final_signal в y_pred_cb.

    Нужно только если вы хотите прогнать мета-сигнал через ветку signal_source='cb'.
    В штатном режиме используйте signal_source='meta'.
    """
    out = df.copy()
    out["y_pred_cb_raw"] = out["y_pred_cb"]
    out["y_pred_cb"] = out["final_signal"]
    return out
