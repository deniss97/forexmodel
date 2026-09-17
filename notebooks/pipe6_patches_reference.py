"""
pipe6_patches.py — дополнения к ML_ForexModelGPT_pipe6_clear.ipynb

Содержит:
  1. late_entry_diagnostic      — диагностика «входов на излёте»: PnL сделок в разрезе
                                   того, насколько цена уже ушла от локального экстремума
                                   на момент входа (в ATR).
  2. add_extension_features     — признаки «где мы внутри движения» (растяжение, возраст
                                   пробоя, efficiency ratio, серия свечей, относительный объём).
  3. generate_labels_atr_asym   — triple-barrier разметка с АСИММЕТРИЧНЫМИ барьерами в ATR:
                                   класс 2/0 только если движение впереди ещё «длинное».
                                   Именно это заставляет модель различать раннюю и позднюю фазу.
  4. add_trend_filter_early     — тренд-фильтр на 4H без ADX-подтверждения задним числом:
                                   наклон регрессии + растущий ADX + запрет на перегретый ADX.
                                   Мёрдж каузальный (через сдвиг на длину бина).
  5. wire_meta_signal           — фикс: final_signal мета-модели реально попадает в симулятор.
  6. simulate_trailing_exit     — обёртка: перевод TP/SL в ATR и трейлинг вместо фиксированного TP.

Все функции принимают df с колонками time/open/high/low/close (+ value/volume, если есть).
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Диагностика поздних входов
# ----------------------------------------------------------------------
def late_entry_diagnostic(trades_df, df_hour, atr_col='atr_14', lookback=24, bins=(0, 0.5, 1, 1.5, 2, 3, 99)):
    """
    Для каждой сделки считает extension_atr = сколько ATR цена прошла от
    противоположного экстремума за последние `lookback` часовых баров ДО входа:
      long : (open_price - min(low, lookback)) / ATR
      short: (max(high, lookback) - open_price) / ATR
    и группирует PnL по этому растяжению. Если winrate/avg PnL монотонно падают
    с ростом extension — модель действительно входит на излёте.
    """
    h = df_hour.sort_values('time').reset_index(drop=True).copy()
    h['roll_low'] = h['low'].rolling(lookback).min().shift(1)   # без текущего бара
    h['roll_high'] = h['high'].rolling(lookback).max().shift(1)
    h['atr_prev'] = h[atr_col].shift(1)
    lookup = h[['time', 'roll_low', 'roll_high', 'atr_prev']].rename(columns={'time': 'signal_dt'})

    t = trades_df.copy()
    t['signal_dt'] = pd.to_datetime(t['signal_dt'])
    t = t.merge(lookup, on='signal_dt', how='left')

    long_mask = t['side'].eq('buy')
    t['extension_atr'] = np.where(
        long_mask,
        (t['open_price'] - t['roll_low']) / t['atr_prev'],
        (t['roll_high'] - t['open_price']) / t['atr_prev'],
    )
    t['ext_bin'] = pd.cut(t['extension_atr'], bins=bins)

    summary = (
        t.groupby('ext_bin', observed=True)
         .agg(n=('profit_pct', 'size'),
              winrate=('profit_pct', lambda s: (s > 0).mean() * 100),
              avg_pnl=('profit_pct', 'mean'),
              tp=('exit_reason', lambda s: (s == 'take_profit').mean() * 100),
              sl=('exit_reason', lambda s: (s == 'stop_loss').mean() * 100))
         .round(2)
    )
    print(summary)
    return t, summary


# ----------------------------------------------------------------------
# 2. Признаки положения внутри движения
# ----------------------------------------------------------------------
def add_extension_features(df, atr_col='atr_14', windows=(6, 12, 24), vol_col=None):
    """
    Всё нормировано на ATR или безразмерно → не зависит от уровня цены
    (в отличие от close_lag_*, sma_*, ema_*, bb_up/bb_low в текущем feature_cols).
    """
    df = df.copy()
    atr = df[atr_col].replace(0, np.nan)

    for w in windows:
        # пройдено ATR от экстремума окна (растяжение)
        df[f'ext_from_low_{w}'] = (df['close'] - df['low'].rolling(w).min()) / atr
        df[f'ext_from_high_{w}'] = (df['high'].rolling(w).max() - df['close']) / atr
        # чистое смещение за окно в ATR
        df[f'runup_{w}'] = (df['close'] - df['close'].shift(w)) / atr
        # Kaufman efficiency ratio: 1 = прямая линия, ~0 = пила
        path = df['close'].diff().abs().rolling(w).sum()
        df[f'er_{w}'] = ((df['close'] - df['close'].shift(w)).abs() / path.replace(0, np.nan))
        # z-score цены относительно окна
        m, s = df['close'].rolling(w).mean(), df['close'].rolling(w).std()
        df[f'z_close_{w}'] = (df['close'] - m) / s.replace(0, np.nan)

    # серия одноцветных свечей (знак и длина)
    sign = np.sign(df['close'] - df['open'])
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    df['candle_streak'] = streak * sign

    # возраст пробоя: сколько баров назад был обновлён 24-барный экстремум
    hi24 = df['high'].rolling(24).max()
    lo24 = df['low'].rolling(24).min()
    new_high = (df['high'] >= hi24).astype(int)
    new_low = (df['low'] <= lo24).astype(int)
    df['bars_since_new_high'] = new_high.groupby(new_high.cumsum()).cumcount()
    df['bars_since_new_low'] = new_low.groupby(new_low.cumsum()).cumcount()

    # сжатие волатильности перед пробоем (низкая bb_width относительно истории)
    if 'bb_width' in df.columns:
        df['bb_width_pct'] = df['bb_width'].rolling(100).rank(pct=True)

    # относительный объём — главный разделитель ложных/истинных пробоев
    if vol_col and vol_col in df.columns:
        df['rel_vol_20'] = df[vol_col] / df[vol_col].rolling(20).mean().replace(0, np.nan)
        df['vol_x_body'] = df['rel_vol_20'] * df.get('body_to_range', 0)

    return df


# ----------------------------------------------------------------------
# 3. Асимметричная ATR-разметка (triple barrier)
# ----------------------------------------------------------------------
def generate_labels_atr_asym(df_main, df_fine, atr_col='atr_14', horizon=10,
                             tp_atr=1.5, sl_atr=0.75, min_minutes_after=1):
    """
    Класс 2: цена дошла до +tp_atr*ATR раньше, чем до -sl_atr*ATR.
    Класс 0: симметрично вниз.
    Класс 1: ни один из «прибыльных» барьеров не достигнут первым.

    Смысл: при tp_atr > sl_atr на излёте движения барьер-цель почти никогда
    не достигается раньше стопа → такие бары получают класс 1/противоположный,
    и модель ВЫНУЖДЕНА научиться отличать раннюю фазу от поздней. При
    симметричных барьерах (как сейчас, ±0.5%) излёт и старт движения для
    модели неотличимы — это coin flip в обоих случаях.
    """
    main = df_main.copy()
    fine = df_fine.copy()
    main['time'] = pd.to_datetime(main['time'])
    fine['time'] = pd.to_datetime(fine['time'])
    main = main.sort_values('time').reset_index(drop=True)
    fine = fine.sort_values('time').set_index('time')

    highs = fine['high'].values
    lows = fine['low'].values
    times = fine.index.values

    labels = np.full(len(main), np.nan)
    for i in range(len(main) - horizon - 1):
        atr = main.loc[i, atr_col]
        if not np.isfinite(atr) or atr <= 0:
            continue
        base = float(main.loc[i + 1, 'open'])
        up_l, up_s = base + tp_atr * atr, base - sl_atr * atr      # для лонга
        dn_l, dn_s = base - tp_atr * atr, base + sl_atr * atr      # для шорта

        t0 = np.datetime64(main.loc[i + 1, 'time'] + pd.Timedelta(minutes=min_minutes_after))
        t1 = np.datetime64(main.loc[i + 1 + horizon, 'time'])
        a, b = np.searchsorted(times, t0), np.searchsorted(times, t1, side='right')
        if b <= a:
            continue
        h, l = highs[a:b], lows[a:b]

        def first_idx(cond):
            idx = np.flatnonzero(cond)
            return idx[0] if len(idx) else np.inf

        long_win = first_idx(h >= up_l) < first_idx(l <= up_s)
        short_win = first_idx(l <= dn_l) < first_idx(h >= dn_s)

        if long_win and not short_win:
            labels[i] = 2
        elif short_win and not long_win:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels


# ----------------------------------------------------------------------
# 4. Ранний тренд-фильтр 4H (каузальный)
# ----------------------------------------------------------------------
def _merge_htf_causal(df, df_htf, timeframe, cols):
    shifted = df_htf.copy()
    shifted['time'] = shifted['time'] + pd.Timedelta(timeframe)   # доступно с момента закрытия бина
    return pd.merge_asof(df.sort_values('time'), shifted[['time'] + cols].sort_values('time'),
                         on='time', direction='backward')


def add_trend_filter_early(df, resample_fn, timeframe='4h', slope_period=6, adx_period=9,
                           adx_min=15, adx_max=40, require_adx_rising=True):
    """
    Тренд считается активным, если:
      - наклон регрессии цены за slope_period бинов (в ATR) имеет знак направления,
      - ADX выше adx_min И (опц.) растёт относительно предыдущего бина,
      - ADX НИЖЕ adx_max — перегретый тренд (ADX>40) = поздняя фаза, вход запрещён.
    Отличие от ADX>25 с ffill: сигнал появляется при разгоне тренда, а не после того,
    как он уже отработал.
    """
    import pandas_ta as ta
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    htf = resample_fn(df, timeframe=timeframe)

    tr = pd.concat([htf['high'] - htf['low'],
                    (htf['high'] - htf['close'].shift(1)).abs(),
                    (htf['low'] - htf['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean().replace(0, np.nan)

    x = np.arange(slope_period)
    slope = htf['close'].rolling(slope_period).apply(
        lambda y: np.polyfit(x, y, 1)[0] if not np.isnan(y).any() else np.nan, raw=True)
    htf['slope_atr_4h'] = slope / atr

    adx = ta.adx(high=htf['high'], low=htf['low'], close=htf['close'], length=adx_period)
    htf['adx_4h'] = adx[f'ADX_{adx_period}']
    htf['adx_slope_4h'] = htf['adx_4h'].diff()
    di_dir = np.sign(adx[f'DMP_{adx_period}'] - adx[f'DMN_{adx_period}'])

    ok = (htf['adx_4h'] > adx_min) & (htf['adx_4h'] < adx_max)
    if require_adx_rising:
        ok &= htf['adx_slope_4h'] > 0
    direction = np.sign(htf['slope_atr_4h'])
    agree = direction == di_dir
    htf['trend_4h'] = np.where(ok & agree, direction, 0).astype(float)
    htf['trend_age_4h'] = htf['trend_4h'].groupby((htf['trend_4h'] != htf['trend_4h'].shift()).cumsum()).cumcount()

    return _merge_htf_causal(df, htf, timeframe,
                             ['trend_4h', 'adx_4h', 'adx_slope_4h', 'slope_atr_4h', 'trend_age_4h'])


# ----------------------------------------------------------------------
# 5. Подключение мета-модели к симулятору
# ----------------------------------------------------------------------
def wire_meta_signal(df_sim_filtered):
    """
    В ноутбуке apply_meta_filter создаёт final_signal, но run_simulator вызывается
    с signal_source='ensemble' и читает y_pred_cb / y_pred_nn — фильтр не работает.
    Перезаписываем y_pred_cb отфильтрованным сигналом и зовём симулятор с signal_source='cb'.
    """
    df = df_sim_filtered.copy()
    df['y_pred_cb_raw'] = df['y_pred_cb']
    df['y_pred_cb'] = df['final_signal']           # NaN там, где мета-модель отказала
    return df
    # далее: run_simulator(df, ..., signal_source='cb', min_conf_cb=None)


# ----------------------------------------------------------------------
# 6. Выход по трейлингу (чтобы «не пропускать» большие движения)
# ----------------------------------------------------------------------
def trailing_exit(px_slice, side, open_price, atr, sl_atr=1.0, trail_atr=1.5, activate_atr=1.0):
    """
    px_slice: минутные бары от входа до конца горизонта (open/high/low/close).
    Стоп = open ∓ sl_atr*ATR; после хода в нашу сторону на activate_atr*ATR стоп
    подтягивается на trail_atr*ATR от лучшей цены. Фиксированного TP нет —
    большое движение забирается целиком, мелкий ложный пробой режется стопом.
    Возвращает (exit_idx, exit_price, reason).
    """
    sgn = 1 if side == 'buy' else -1
    stop = open_price - sgn * sl_atr * atr
    best = open_price
    for j, (h, l, c) in enumerate(zip(px_slice['high'].values, px_slice['low'].values, px_slice['close'].values)):
        worst = l if sgn == 1 else h
        if (worst - stop) * sgn <= 0:
            return j, stop, 'stop_loss' if best == open_price else 'trailing_stop'
        ext = h if sgn == 1 else l
        if (ext - best) * sgn > 0:
            best = ext
            if (best - open_price) * sgn >= activate_atr * atr:
                stop = max(stop, best - sgn * trail_atr * atr) if sgn == 1 else min(stop, best - sgn * trail_atr * atr)
    return len(px_slice) - 1, float(px_slice['close'].iloc[-1]), 'timeout'
