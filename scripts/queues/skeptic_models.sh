#!/bin/bash
# Скептик к H63, H64, H41 (docs/results/skeptic_2026-11-08.md): модель LKOH с фиксированным числом деревьев (без ранней
# остановки, которая в 2021 и 2024 оставляла 1–10 деревьев и 0 сделок), N из {100, 300} — по 2018–2020; на нём — пара моделей,
# «ход × куда», наклон тренда; нейросеть — с порогами уверенности на той же обученной сети. Настройки — как в H63.
cd /home/dev/forexmodel
LOG=reports/walk_forward/logs; mkdir -p $LOG
COMMON="--set data.csv_path=data/raw/LKOH_1m_15-26.csv --set data.dividends_path=data/dividends/moex_dividends.csv --set data.dividend_ticker=LKOH --set nn.enabled=false --set meta.enabled=false --set simulation.signal_source=cb --set trend_gate.enabled=true --set simulation.trend_col=trend_gate"
run() { n=$1; shift; [ -f reports/walk_forward/${n}_trades.csv ] && return; .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 "$@" --name $n > $LOG/$n.log 2>&1; }
for N in 100 300; do for sd in 42 7 1; do
  run lkoh_sk_mc_n${N}_s$sd $COMMON --set catboost.val_size=0 --set catboost.iterations=$N --set catboost.random_seed=$sd
done; done
N=$(.venv/bin/python -W ignore scripts/skeptic_models.py --pick-n 2>> $LOG/skeptic_pick.log | tail -1)
echo "выбрано N=$N" >> $LOG/skeptic_pick.log
for sd in 42 7 1; do
  F="--set catboost.val_size=0 --set catboost.iterations=$N --set catboost.random_seed=$sd"
  run lkoh_sk_bp_n${N}_s$sd $COMMON $F --set catboost.objective=binary_pair
  run lkoh_sk_md_n${N}_s$sd $COMMON $F --set catboost.objective=move_dir
  run lkoh_sk_sl_n${N}_s$sd $COMMON $F "--set" "features.slope_windows=[24,72,168]"
done
# нейросеть: те же прогоны, что H64, но с порогами уверенности (имя файла — lkoh_h64_nn_s<seed>__nnconf_<порог>)
NN="--set data.csv_path=data/raw/LKOH_1m_15-26.csv --set data.dividends_path=data/dividends/moex_dividends.csv --set data.dividend_ticker=LKOH --set nn.enabled=true --set nn.device=cpu --set meta.enabled=false --set simulation.signal_source=nn --set trend_gate.enabled=true --set simulation.trend_col=trend_gate"
for sd in 42 7 1; do
  [ -f reports/walk_forward/lkoh_h64_nn_s${sd}__nnconf_0.6_trades.csv ] && continue
  OMP_NUM_THREADS=4 .venv/bin/python -W ignore scripts/walk_forward.py -c configs/default.yaml --years 2018 2026 $NN \
    --set catboost.random_seed=$sd --set nn.random_seed=$sd --name lkoh_h64_nn_s$sd \
    --variants nnconf:0.4 nnconf:0.45 nnconf:0.5 nnconf:0.55 nnconf:0.6 > $LOG/lkoh_h64_nnconf_s$sd.log 2>&1
done
date > $LOG/queue_skeptic_models.done
