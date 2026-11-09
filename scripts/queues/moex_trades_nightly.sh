#!/bin/bash
# Лента сделок Мосбиржи (ISS, с направлением) по бумагам контура: ISS отдаёт только сделки текущего дня, поэтому каждые
# 15 минут дозабираем новые по номеру сделки, а законченные дни уходят на Яндекс Диск (yadisk:forexmodel/лента ISS/).
# Свёртки дней (agg/*.json) остаются на сервере — их читает страница H84. Подробно — scripts/moex_trades.py.
# cron (часы сервера +05), мимо почасового контура в :00:
#   5,20,35,50 * * * *  /home/dev/forexmodel/scripts/queues/moex_trades_nightly.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/h84; mkdir -p $LOG
exec 9> $LOG/.trades.lock
flock -n 9 || exit 0
mountpoint -q "$HOME/yadisk" || systemctl --user start yadisk 2>/dev/null
{ echo "=== $(date '+%F %T')"; .venv/bin/python -W ignore scripts/moex_trades.py; } >> $LOG/trades-$(date +%Y-%m).log 2>&1
