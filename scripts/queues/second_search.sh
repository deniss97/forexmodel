#!/bin/bash
# Честный поиск второй стратегии (H54): 6 ч перебора в 3 процессах, затем отчёт — walk-forward поиска и финалисты
# штатным минутным симулятором (docs/results/second_strategy.md).
#   setsid nohup scripts/queues/second_search.sh > /dev/null 2>&1 &
cd /home/dev/forexmodel || exit 1
LOG=reports/second_search; mkdir -p $LOG
HOURS=${HOURS:-6}
.venv/bin/python scripts/second_search.py search --hours "$HOURS" > $LOG/search.log 2>&1
.venv/bin/python scripts/second_search.py report --top "${TOP:-5}" > $LOG/report.log 2>&1
date > $LOG/queue.done
