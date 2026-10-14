#!/bin/bash
# ТЗ 15.10, гипотеза H48: компоненты трёхслойного тренд-фильтра и нормированные уровни как признаки CatBoost
# серебра. Те же настройки и seed, что у базы M13 (silver_gatevar_s*) и M17 (silver_emazone_s*).
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
until [ -f $LOG/queue_tz_1015.done ]; do sleep 30; done
for sd in 42 7 1; do
  [ -f reports/walk_forward/silver_tsfeat_s${sd}_trades.csv ] && continue
  systemd-run --user --scope -q -p MemoryMax=5G .venv/bin/python -W ignore scripts/walk_forward.py -c configs/silver.yaml --years 2019 2026 \
    --set catboost.random_seed=$sd --set nn.enabled=false --set meta.enabled=false --set simulation.signal_source=cb \
    --set features.use_trend_score_features=true --name silver_tsfeat_s$sd > $LOG/silver_tsfeat_s$sd.log 2>&1
done
date > $LOG/queue_tz_features.done
