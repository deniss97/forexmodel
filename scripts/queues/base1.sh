#!/bin/bash
# перепрогон экспериментов импульса на новой базе B01 (неделя 12.10): у акций цены скорректированы на
# дивиденды, стоп по гэпу, выход 240 ч / 6 ATR; все 8 инструментов. Издержки (комиссия, проскальзывание,
# свопы — configs/costs/base.yaml) — потом по сделкам (scripts/experiments.py). Варианты, которые
# меняют путь сделки и которых нет в прогонах *_costs: без тренд-фильтра, только при высокой
# волатильности, импульс 8 ч / 3.5 ATR; отдельно — выход при развороте тренд-гейта.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
FLIP="--set simulation.close_on_trend_flip=true --set simulation.trend_flip_mode=opposite"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
V="impulse:6:3 impulse:6:3:notrend impulse:6:3:vol impulse:8:3.5"
for i in XAGUSD_1m_full:silver:silver.yaml:2016 LKOH_1m_15-26:lkoh:default.yaml:2016 GAZP_1m_15-26_upd:gazp:default.yaml:2016 SBER_1m_15-26:sber:default.yaml:2016 XAUUSD_1m_full:gold:silver.yaml:2016 MOEX_1m_15-26:moex:default.yaml:2016 MTSS_1m_15-26:mtss:default.yaml:2016 ETHUSDT_1m_clean:eth:silver.yaml:2021; do
  IFS=: read f n c y <<< "$i"
  if [ "$c" = default.yaml ]; then G="$GATE --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*}"
  else G="--set data.csv_path=data/raw/$f.csv"; fi
  [ -f reports/walk_forward/${n}_b1__impulse_8_3.5.csv ] || \
    .venv/bin/python scripts/walk_forward.py -c configs/$c --years $y 2026 $EX $G --name ${n}_b1 --variants $V > $LOG/${n}_b1.log 2>&1
  [ -f reports/walk_forward/${n}_b1flip__impulse_6_3.csv ] || \
    .venv/bin/python scripts/walk_forward.py -c configs/$c --years $y 2026 $EX $FLIP $G --name ${n}_b1flip --variants impulse:6:3 > $LOG/${n}_b1flip.log 2>&1
done
date > $LOG/queue_base1.done
