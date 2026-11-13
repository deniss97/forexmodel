#!/bin/bash
# Снимки стакана 21 акции раз в 15 с через API Т-Инвестиций (бесплатно), законченные дни — на Яндекс Диск
# (yadisk:forexmodel/стакан/<год>/<дата>/<час>.jsonl.gz). Подробно — scripts/orderbook_rec.py. Решение Дениса 14.11 (копить;
# на сервере только текущий час). cron (часы сервера +05):  * * * * *  /home/dev/forexmodel/scripts/queues/orderbook_rec.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/live; mkdir -p $LOG
exec 9> $LOG/.orderbook.lock
flock -n 9 || exit 0
.venv/bin/python -W ignore scripts/orderbook_rec.py >> $LOG/orderbook-$(date +%Y-%m).log 2>&1
