#!/bin/bash
# ТЗ 15.10, гипотеза H49: выходы из списка параметров — фиксированные стоп 1.5 / 2 / 3 ATR × тейк 2 / 3 / 4 ATR,
# Chandelier (трейлинг 3 ATR с первой минуты), выход по времени 10 / 20 / 50 ч (стоп и трейлинг базы).
# Вход — импульс базы B01 по гейту C, 8 рынков; издержки — по сделкам.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
until [ -f $LOG/queue_tz_features.done ]; do sleep 30; done
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
V="impulse:6:3:tr3:act0 impulse:6:3:h10 impulse:6:3:h20 impulse:6:3:h50"
for sl in 1.5 2 3; do for tp in 2 3 4; do V="$V impulse:6:3:sl$sl:tp$tp"; done; done
for i in XAGUSD_1m_full:silver:silver.yaml:2016 XAUUSD_1m_full:gold:silver.yaml:2016 LKOH_1m_15-26:lkoh:default.yaml:2016 GAZP_1m_15-26_upd:gazp:default.yaml:2016 SBER_1m_15-26:sber:default.yaml:2016 MOEX_1m_15-26:moex:default.yaml:2016 MTSS_1m_15-26:mtss:default.yaml:2016 ETHUSDT_1m_clean:eth:silver.yaml:2021; do
  IFS=: read f n c y <<< "$i"
  [ -f reports/walk_forward/${n}_tzx__impulse_6_3_sl3_tp4.csv ] && continue
  if [ "$c" = default.yaml ]; then G="$GATE --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*}"
  else G="--set data.csv_path=data/raw/$f.csv"; fi
  systemd-run --user --scope -q -p MemoryMax=5G .venv/bin/python -W ignore scripts/walk_forward.py -c configs/$c --years $y 2026 $EX $G --name ${n}_tzx --variants $V > $LOG/${n}_tzx.log 2>&1
done
date > $LOG/queue_tz_exits.done
