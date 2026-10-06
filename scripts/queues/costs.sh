#!/bin/bash
# импульс с новыми вводными (неделя 05.10–12.10): дивиденды акций, стоп по гэпу, сетка выхода
# (удержание 48–240 ч × стоп/трейлинг 3/4/6 ATR); издержки брокера — потом по сделкам (scripts/broker_costs.py)
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
V=""
for h in 48 72 120 168 240; do for tr in 3 4 6; do V="$V impulse:6:3:h$h:tr$tr"; done; done
for i in XAGUSD_1m_full:silver:silver.yaml:2016 LKOH_1m_15-26:lkoh:default.yaml:2016 GAZP_1m_15-26_upd:gazp:default.yaml:2016 SBER_1m_15-26:sber:default.yaml:2016 XAUUSD_1m_full:gold:silver.yaml:2016 MOEX_1m_15-26:moex:default.yaml:2016 MTSS_1m_15-26:mtss:default.yaml:2016 ETHUSDT_1m_clean:eth:silver.yaml:2021; do
  IFS=: read f n c y <<< "$i"
  [ -f reports/walk_forward/${n}_costs__impulse_6_3_h240_tr6.csv ] && continue
  if [ "$c" = default.yaml ]; then G="$GATE --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*}"
  else G="--set data.csv_path=data/raw/$f.csv"; fi
  .venv/bin/python scripts/walk_forward.py -c configs/$c --years $y 2026 $EX $G --name ${n}_costs --variants $V > $LOG/${n}_costs.log 2>&1
done
date > $LOG/queue_costs.done
