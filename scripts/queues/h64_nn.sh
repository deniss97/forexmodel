#!/bin/bash
# H64 (Тимур): нейросеть вместо CatBoost. A/B на LKOH при настройках H63 (база L10 — lkoh_h63_multiclass_s*):
# сигнал CNN-BiLSTM из конвейера (forexmodel/models/nn_cnn_bilstm.py, seq 20, 40 эпох, ранняя остановка), без меты,
# тренд-гейт C, разметка atr_asym (+1.5 / −0.75 ATR за 10 ч), годовой walk-forward 2018–2026, seed 42, 7, 1 (и у CatBoost,
# и у сети). Издержки биржи — в журнале экспериментов.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
COMMON="--set data.csv_path=data/raw/LKOH_1m_15-26.csv --set data.dividends_path=data/dividends/moex_dividends.csv --set data.dividend_ticker=LKOH --set nn.enabled=true --set nn.device=cpu --set meta.enabled=false --set simulation.signal_source=nn --set trend_gate.enabled=true --set simulation.trend_col=trend_gate"
for sd in 42 7 1; do
  n=lkoh_h64_nn_s${sd}
  [ -f reports/walk_forward/${n}_trades.csv ] && continue
  OMP_NUM_THREADS=4 .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 $COMMON \
    --set catboost.random_seed=$sd --set nn.random_seed=$sd --name $n > $LOG/$n.log 2>&1
done
date > $LOG/queue_h64.done
