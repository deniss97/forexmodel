#!/bin/bash
# H63 (Тимур): модель LKOH на 3 класса против пары бинарных моделей (вверх да/нет, вниз да/нет) при одинаковых настройках:
# CatBoost без NN и меты, сигнал по модели (cb), тренд-гейт C, разметка atr_asym (+1.5 / −0.75 ATR за 10 ч), годовой
# walk-forward 2018–2026, seed 42, 7, 1. Издержки биржи — в журнале экспериментов.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
COMMON="--set data.csv_path=data/raw/LKOH_1m_15-26.csv --set data.dividends_path=data/dividends/moex_dividends.csv --set data.dividend_ticker=LKOH --set nn.enabled=false --set meta.enabled=false --set simulation.signal_source=cb --set trend_gate.enabled=true --set simulation.trend_col=trend_gate"
for sd in 42 7 1; do
  for obj in multiclass binary_pair; do
    n=lkoh_h63_${obj}_s${sd}
    [ -f reports/walk_forward/${n}_trades.csv ] && continue
    .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 $COMMON \
      --set catboost.random_seed=$sd --set catboost.objective=$obj --name $n > $LOG/$n.log 2>&1
  done
done
date > $LOG/queue_h63.done
