#!/bin/bash
# Ночной архив минуток на Яндекс Диск (29.10): то, что контур дописал за сутки, — в yadisk:forexmodel/минутки/<бумага>/<год>.csv.gz.
# Горячие файлы остаются на сервере (контур читает их каждый час). Сбой — уведомление в группу. scripts/yadisk_archive.py.
# cron — раз в сутки после вечерней сессии (часы сервера +05: 02:30 = 00:30 по Москве):
#   30 2 * * *  /home/dev/forexmodel/scripts/queues/yadisk_nightly.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/live/logs; mkdir -p $LOG
exec 9> reports/live/.yadisk.lock
flock -n 9 || exit 0                                  # прошлая выгрузка ещё идёт (первая — долгая)
mountpoint -q "$HOME/yadisk" || systemctl --user start yadisk 2>/dev/null
{ date; .venv/bin/python -W ignore scripts/yadisk_archive.py; } >> $LOG/yadisk-$(date +%Y-%m).log 2>&1
