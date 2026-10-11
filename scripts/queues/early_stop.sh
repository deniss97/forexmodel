#!/bin/bash
# гипотеза H32 (неделя 12.10): ранний стоп — ближний стоп es ATR первые eh часов после входа, пока трейлинг
# не включился. Те же настройки, что у базы B01 (дивиденды акций, стоп по гэпу, выход 240 ч / 6 ATR), все
# 8 рынков; издержки базы — потом по сделкам (scripts/experiments.py). Выбор es/eh — по 2016–2020.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
V="impulse:6:3"
for es in 2 3 4; do for eh in 12 24 48; do V="$V impulse:6:3:es$es:eh$eh"; done; done
for i in XAGUSD_1m_full:silver:silver.yaml:2016 LKOH_1m_15-26:lkoh:default.yaml:2016 GAZP_1m_15-26_upd:gazp:default.yaml:2016 SBER_1m_15-26:sber:default.yaml:2016 XAUUSD_1m_full:gold:silver.yaml:2016 MOEX_1m_15-26:moex:default.yaml:2016 MTSS_1m_15-26:mtss:default.yaml:2016 ETHUSDT_1m_clean:eth:silver.yaml:2021; do
  IFS=: read f n c y <<< "$i"
  [ -f reports/walk_forward/${n}_es__impulse_6_3_es4_eh48.csv ] && continue
  if [ "$c" = default.yaml ]; then G="$GATE --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*}"
  else G="--set data.csv_path=data/raw/$f.csv"; fi
  .venv/bin/python scripts/walk_forward.py -c configs/$c --years $y 2026 $EX $G --name ${n}_es --variants $V > $LOG/${n}_es.log 2>&1
done
date > $LOG/queue_early_stop.done
