#!/bin/bash
# H68 (24.10): S42 по металлам на фьючерсах Мосбиржи — штатно, как металлы в queues/costs.sh: конфиг серебра (гейт C
# в нём), выход 240 ч / стоп и трейлинг 6 ATR (активация +3), стоп по гэпу, обе стороны; минутки — склейка фьючерса
# в формате XAG/XAU (scripts/metals_futures.py prepare); издержки биржи пересчитывает журнал (exchange.yaml: silvfut, goldfut).
#   systemd-run --user --unit=metals-fut-s42 -p MemoryMax=1200M scripts/queues/metals_futures_s42.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
for i in SILV:silvfut GOLD:goldfut; do
  IFS=: read f n <<< "$i"
  [ -f reports/walk_forward/${n}_costs__impulse_6_3_h240_tr6_trades.csv ] && continue
  .venv/bin/python scripts/walk_forward.py -c configs/silver.yaml --years 2016 2026 $EX --set data.csv_path=data/raw/moex/${f}_1m_fut.csv --name ${n}_costs --variants impulse:6:3:h240:tr6 > $LOG/${n}_costs.log 2>&1
done
date > $LOG/queue_metals_futures_s42.done
