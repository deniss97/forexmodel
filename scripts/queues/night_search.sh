#!/bin/bash
# Ночной перебор параметров импульса по списку ТЗ 15.10 (H50), scripts/night_search.py.
#   1) подготовка: сделки всех сигналов для 63 пар «период ATR × выход» (~2–2.5 ч, продолжает с места обрыва);
#   2) поиск по металлам с издержками B01 (CFD) — HOURS_METALS часов;
#   3) поиск по портфелю 4 с издержками биржи (акции на Мосбирже, S24/S25) — HOURS_P4 часов;
#   4) отчёты reports/night_search/report_<срез>_<издержки>.md.
# Запуск:  setsid nohup scripts/queues/night_search.sh > /dev/null 2>&1 &
# Повторный запуск продолжает: готовые сделки не пересчитываются, пробы копятся в trials.sqlite.
cd /home/dev/forexmodel
LOG=reports/night_search; mkdir -p $LOG
HOURS_METALS=${HOURS_METALS:-8}
HOURS_P4=${HOURS_P4:-4}
RUN="systemd-run --user --scope -q -p MemoryMax=5G .venv/bin/python -W ignore scripts/night_search.py"
[ -f $LOG/prepare.done ] || $RUN prepare --workers 4 >> $LOG/prepare.log 2>&1
$RUN search --hours $HOURS_METALS --workers 3 --scope металлы --costs base >> $LOG/search.log 2>&1
$RUN search --hours $HOURS_P4 --workers 3 --scope "портфель 4" --costs exchange >> $LOG/search.log 2>&1
$RUN report --scope металлы --costs base > /dev/null 2>&1
$RUN report --scope "портфель 4" --costs exchange > /dev/null 2>&1
date > $LOG/night.done
