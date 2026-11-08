#!/bin/bash
# Вторая очередь скептика к модели (после scripts/queues/skeptic_models.sh — ждёт её отметку): на базе с фиксированным числом
# деревьев (N из skeptic_pick.log) — (1) фильтр ожидаемой ценности с реальными издержками 0.2% и порогами 0 / 0.1 / 0.2%;
# (2) нейросеть как второе мнение к CatBoost (ensemble agreement) и CatBoost той же обученной пары для сравнения.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs
until [ -f $LOG/queue_skeptic_models.done ]; do sleep 60; done
N=$(grep -o "выбрано N=[0-9]*" $LOG/skeptic_pick.log | tail -1 | cut -d= -f2); N=${N:-100}
BASE="--set data.csv_path=data/raw/LKOH_1m_15-26.csv --set data.dividends_path=data/dividends/moex_dividends.csv --set data.dividend_ticker=LKOH --set meta.enabled=false --set trend_gate.enabled=true --set simulation.trend_col=trend_gate --set catboost.val_size=0 --set catboost.iterations=$N"
for sd in 42 7 1; do
  n=lkoh_sk_ev_n${N}_s$sd
  [ -f reports/walk_forward/${n}__ev_0.2_0.2_trades.csv ] || .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 $BASE \
    --set nn.enabled=false --set simulation.signal_source=cb --set catboost.random_seed=$sd --name $n \
    --variants ev:0.2:0 ev:0.2:0.1 ev:0.2:0.2 > $LOG/$n.log 2>&1
done
for sd in 42 7 1; do
  n=lkoh_sk_ens_n${N}_s$sd
  [ -f reports/walk_forward/${n}__ens_agreement_trades.csv ] || OMP_NUM_THREADS=4 .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 $BASE \
    --set nn.enabled=true --set nn.device=cpu --set catboost.random_seed=$sd --set nn.random_seed=$sd --name $n \
    --variants cb ens:agreement > $LOG/$n.log 2>&1
done
date > $LOG/queue_skeptic_models2.done
