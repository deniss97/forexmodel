#!/bin/bash
# H41 (Тимур): наклон тренда за 24 / 72 / 168 ч и sin / cos его угла как признаки модели (features.slope_windows).
# LKOH — настройки H63 (база L10 = lkoh_h63_multiclass_s*): CatBoost без NN и меты, сигнал по модели, тренд-гейт C,
# разметка atr_asym, годовой walk-forward 2018–2026. Серебро — configs/silver.yaml без NN и меты, сигнал CatBoost,
# walk-forward 2019–2026; база пересчитывается на тех же настройках. Seed 42, 7, 1.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
SLOPE="--set features.slope_windows=[24,72,168]"
LK="--set data.csv_path=data/raw/LKOH_1m_15-26.csv --set data.dividends_path=data/dividends/moex_dividends.csv --set data.dividend_ticker=LKOH --set nn.enabled=false --set meta.enabled=false --set simulation.signal_source=cb --set trend_gate.enabled=true --set simulation.trend_col=trend_gate"
AG="--set nn.enabled=false --set meta.enabled=false --set simulation.signal_source=cb"
for sd in 42 7 1; do
  n=lkoh_h41_slope_s${sd}
  [ -f reports/walk_forward/${n}_trades.csv ] || .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 $LK $SLOPE \
      --set catboost.random_seed=$sd --name $n > $LOG/$n.log 2>&1
  for v in base slope; do
    n=silver_h41_${v}_s${sd}
    X=""; [ $v = slope ] && X="$SLOPE"
    [ -f reports/walk_forward/${n}_trades.csv ] || .venv/bin/python -W ignore scripts/walk_forward.py -c configs/silver.yaml --years 2019 2026 $AG $X \
        --set catboost.random_seed=$sd --name $n > $LOG/$n.log 2>&1
  done
done
date > $LOG/queue_h41.done
