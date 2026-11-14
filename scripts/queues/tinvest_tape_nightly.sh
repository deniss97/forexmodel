#!/bin/bash
# Обезличенные сделки Т-Инвестиций по 21 акции (открытый архив history-trades, скрипт Тимура от 12.11) → Яндекс Диск
# (yadisk:forexmodel/лента Т-Инвестиций/). Докачивает всё недостающее с 07.04.2026 по вчера. Подробно — scripts/tinvest_tape.py.
# cron (часы сервера +05), раз в сутки, мимо почасового контура (подключает Денис):
#   15 6 * * *  /home/dev/forexmodel/scripts/queues/tinvest_tape_nightly.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/live; mkdir -p $LOG
exec 9> $LOG/.tinvest_tape.lock
flock -n 9 || exit 0
{ echo "=== $(date '+%F %T')"; .venv/bin/python -W ignore scripts/tinvest_tape.py --alert "$@"; } >> $LOG/tinvest_tape-$(date +%Y-%m).log 2>&1
