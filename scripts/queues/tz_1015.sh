#!/bin/bash
# ТЗ 15.10 (гипотезы H44–H47): тренд-фильтры и паттерны входа штатным walk-forward, настройки базы B01
# (дивиденды акций, стоп по гэпу, выход 240 ч / 6 ATR); издержки базы — по сделкам (scripts/experiments.py).
#   impulse:6:3                 — контроль: должен совпасть с B01
#   impulse:6:3:gatetz          — трёхслойный фильтр как в ТЗ (H44)
#   impulse:6:3:gatex4          — лучший трёхслойный по выбору 2016–2020 (пары EMA ×4) (H44)
#   impulse:6:3:gatedon55       — лучший классический по выбору (Дончиан 55/20 на 4 ч) (H45)
#   impulse:6:3:notrend:long    — без фильтра направления, только лонг (эталон «всегда лонг» перебора)
#   pattern:*                   — сжатие диапазона и уровни (H46, H47), с гейтом C и без
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
VARS='trend_gate.variants={tz: {mode: score}, x4: {mode: score, score_spans: [[32, 128], [64, 256], [128, 512]], score_vol_window: 192}, don55: {mode: classic, rule: "donchian:55:20", timeframe: 4h, hold_bars: 0}}'
V="impulse:6:3 impulse:6:3:gatetz impulse:6:3:gatex4 impulse:6:3:gatedon55 impulse:6:3:notrend:long"
for p in nr7 nr4 ib nr7ib donch:1 donch:20 donch:55 soup:20; do V="$V pattern:$p pattern:$p:notrend"; done
for i in XAGUSD_1m_full:silver:silver.yaml:2016 XAUUSD_1m_full:gold:silver.yaml:2016 LKOH_1m_15-26:lkoh:default.yaml:2016 GAZP_1m_15-26_upd:gazp:default.yaml:2016 SBER_1m_15-26:sber:default.yaml:2016 MOEX_1m_15-26:moex:default.yaml:2016 MTSS_1m_15-26:mtss:default.yaml:2016 ETHUSDT_1m_clean:eth:silver.yaml:2021; do
  IFS=: read f n c y <<< "$i"
  [ -f reports/walk_forward/${n}_tz__pattern_soup_20_notrend.csv ] && continue
  if [ "$c" = default.yaml ]; then G="$GATE --set data.csv_path=data/raw/$f.csv $DIV --set data.dividend_ticker=${f%%_*}"
  else G="--set data.csv_path=data/raw/$f.csv"; fi
  systemd-run --user --scope -q -p MemoryMax=5G .venv/bin/python -W ignore scripts/walk_forward.py -c configs/$c --years $y 2026 $EX $G --set "$VARS" --name ${n}_tz --variants $V > $LOG/${n}_tz.log 2>&1
done
date > $LOG/queue_tz_1015.done
