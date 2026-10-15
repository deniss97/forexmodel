#!/bin/bash
# Перебор самостоятельных стратегий из блоков ТЗ 15.10 (H50), scripts/strategy_search.py.
#   1) поиск HOURS часов (по умолчанию 12): 3 процесса — цели «металлы», «портфель 4», «все рынки»;
#   2) отчёт reports/strategy_search/report.md (лучшие по выбору, устойчивые, walk-forward поиска);
#   3) финалисты (по TOP лучших устойчивых на срез) — штатный минутный симулятор с издержками B01 →
#      reports/walk_forward/*_ss_s<id>_trades.csv и записи журнала S36… (база B01);
#   4) журнал экспериментов: метрики финалистов против B01 на одном тесте (сделки с 2021 года).
# Запуск:  setsid nohup scripts/queues/strategy_search.sh > /dev/null 2>&1 &
# Данные — reports/night_search/feat (готовятся scripts/night_search.py prepare, один раз).
cd /home/dev/forexmodel
LOG=reports/strategy_search; mkdir -p $LOG
HOURS=${HOURS:-12}
TOP=${TOP:-2}
PY="systemd-run --user --scope -q -p MemoryMax=5G .venv/bin/python -W ignore"
[ -f reports/night_search/prepare.done ] || $PY scripts/night_search.py prepare --workers 4 >> $LOG/prepare.log 2>&1
$PY scripts/strategy_search.py search --hours $HOURS >> $LOG/search.log 2>&1
$PY scripts/strategy_search.py report > /dev/null 2>> $LOG/search.log
[ -f $LOG/finalize.done ] || { $PY scripts/strategy_search.py finalize --top $TOP >> $LOG/finalize.log 2>&1 && date > $LOG/finalize.done; }
$PY scripts/experiments.py >> $LOG/finalize.log 2>&1
date > $LOG/queue.done
