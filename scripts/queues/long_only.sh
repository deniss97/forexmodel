#!/bin/bash
# S25 штатно (12.10): акции только в лонг — шортовые сигналы гасятся до симуляции (в первой оценке S25 шорты
# убирались из готовых сделок и успевали занять время). Настройки как у базы B01: дивиденды, стоп по гэпу,
# выход 240 ч / 6 ATR; издержки биржи — по сделкам (configs/costs/exchange.yaml).
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
for i in LKOH_1m_15-26:lkoh GAZP_1m_15-26_upd:gazp SBER_1m_15-26:sber MOEX_1m_15-26:moex MTSS_1m_15-26:mtss; do
  IFS=: read f n <<< "$i"
  [ -f reports/walk_forward/${n}_lo__impulse_6_3_long.csv ] && continue
  .venv/bin/python scripts/walk_forward.py -c configs/default.yaml --years 2016 2026 $EX $GATE --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*} --name ${n}_lo --variants impulse:6:3:long > $LOG/${n}_lo.log 2>&1
done
date > $LOG/queue_long_only.done
