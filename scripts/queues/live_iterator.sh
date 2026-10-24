#!/bin/bash
# Бумажный режим S46 и S48 на 19 акциях (H67, 24.10): раз в сутки после вечерней сессии — свежие минутки 16 новых
# бумаг из Т-Инвестиций, пересчёт стратегий движком перебора, журнал reports/live/iterator_journal.csv и отчёт
# reports/live/iterator_<дата>_<время>.md. Заявок нет. Минутки SBER, GAZP, LKOH докачивает live_hourly.sh.
#
# cron — в 00:40 по Москве (часы сервера +05), когда вечерняя сессия закрыта и live_hourly.sh в 00:05 уже отработал:
#   40 2 * * *  /home/dev/forexmodel/scripts/queues/live_iterator.sh
# Признаки пересобираются по одной бумаге (~0.5 ГБ памяти, 10–15 минут на все 19, когда минутки обновились).
cd /home/dev/forexmodel || exit 1
LOG=reports/live/logs; mkdir -p $LOG
exec 9> reports/live/.iterator.lock
flock -n 9 || exit 0                                  # предыдущий запуск ещё идёт
{
  date
  .venv/bin/python -W ignore scripts/live_iterator.py
} >> $LOG/iterator_$(date +%Y-%m-%d).log 2>&1
