#!/bin/bash
# H71 (созвон 04.11, T46): S42 на акциях (GAZP, SBER, LKOH в лонг) с потолком убытка сделки 3, 4, 5% —
# стоп = ближний из «6 ATR» и «X% от входа» (simulation.max_loss_pct). Всё остальное — как у S42 (long_only.sh):
# дивиденды, стоп по гэпу, трейлинг 6 ATR с активацией +3, выход 240 ч, тренд-гейт C, издержки биржи.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
for cap in 3 4 5; do
  for i in LKOH_1m_15-26:lkoh GAZP_1m_15-26_upd:gazp SBER_1m_15-26:sber; do
    IFS=: read f n <<< "$i"
    [ -f reports/walk_forward/${n}_lo_cap${cap}__impulse_6_3_long_trades.csv ] && continue
    .venv/bin/python scripts/walk_forward.py -c configs/default.yaml --years 2016 2026 $EX $GATE --set simulation.max_loss_pct=$cap \
      --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*} --name ${n}_lo_cap${cap} \
      --variants impulse:6:3:long > $LOG/${n}_lo_cap${cap}.log 2>&1
  done
done
date > $LOG/queue_h71.done
