#!/bin/bash
# Сделки Мосбиржи по LKOH за прошедший торговый день (H84, страница «Объём по уровням LKOH» для Тагира): ISS отдаёт только
# сделки текущего дня, поэтому забираем их дважды за ночь — после вечерней сессии и до утренней (повтор не задваивает).
# cron (часы сервера +05): 02:20 и 08:20 = 00:20 и 06:20 по Москве
#   20 2,8 * * *  /home/dev/forexmodel/scripts/queues/moex_trades_nightly.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/h84; mkdir -p $LOG
exec 9> $LOG/.trades.lock
flock -n 9 || exit 0
{ echo "=== $(date '+%F %T')"; .venv/bin/python -W ignore scripts/moex_trades.py LKOH; } >> $LOG/trades-$(date +%Y-%m).log 2>&1
