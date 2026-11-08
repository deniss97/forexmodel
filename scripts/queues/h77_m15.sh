#!/bin/bash
# H77 (созвон 05.11, Тимур: «посмотреть на пятнадцатиминутках»): импульс S42 на 15-минутных свечах, GAZP, SBER, LKOH в лонг.
# Сетка из банка: ход за N свечей (12 = 3 ч, 24 = 6 ч) ≥ X ATR (2, 3, 4) × стоп и трейлинг 6 или 12 ATR(15 мин), активация —
# половина. Остальное как у S42: дивиденды, стоп по гэпу, выход 240 ч, тренд-гейт C (удержание 8 свечей = 2 ч), издержки биржи.
# Вход — через минуту после закрытия 15-минутной свечи (open_delay 16 мин).
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=8 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
TF="--set data.base_timeframe=15min --set data.signal_tf_minutes=15 --set simulation.open_delay_minutes=16"
for sl in 6 12; do
  act=$((sl / 2))
  EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=$sl --set simulation.trail_atr=$sl --set simulation.activate_atr=$act --set simulation.stop_fill=gap"
  for i in LKOH_1m_15-26:lkoh GAZP_1m_15-26_upd:gazp SBER_1m_15-26:sber; do
    IFS=: read f n <<< "$i"
    [ -f reports/walk_forward/${n}_m15_sl${sl}__impulse_24_4_long_trades.csv ] && continue
    .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2016 2026 $EX $GATE $TF \
      --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*} --name ${n}_m15_sl${sl} \
      --variants impulse:12:2:long impulse:12:3:long impulse:12:4:long impulse:24:2:long impulse:24:3:long impulse:24:4:long \
      > $LOG/${n}_m15_sl${sl}.log 2>&1
  done
done
date > $LOG/queue_h77.done
