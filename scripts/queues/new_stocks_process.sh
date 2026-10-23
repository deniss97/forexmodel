#!/bin/bash
# H66 (23.10): новые бумаги — по мере выгрузки (new_stocks_download.sh), по одной (память): проверка минуток →
# дивиденды → часовые свечи и признаки → сделки S45–S48 и H60 → S42 (walk-forward). Обработанные — в processed;
# бумагу, где шаг упал, повторяет на следующем круге. Выходит, когда выгрузка кончилась и всё докачанное обработано.
#   systemd-run --user --unit=new-stocks-proc -p MemoryMax=1000M scripts/queues/new_stocks_process.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/new_stocks; DONE=$LOG/processed; touch $DONE
PY=".venv/bin/python -W ignore"
loaded() { grep -oP '^\[\K[A-Z]+(?=\] добавлено)' $LOG/download.log | sort -u; }
while true; do
  for t in $(loaded); do
    grep -qx "$t" $DONE && continue
    { echo "== $t $(date '+%F %T')"
      $PY scripts/new_stocks.py prepare $t && \
      $PY scripts/fetch_dividends.py --tickers $t | grep -v "^ *$" | tail -n 30 && \
      $PY scripts/new_stocks.py bars $t && \
      $PY scripts/new_stocks.py trades $t && \
      scripts/queues/new_stocks_s42.sh $t && \
      tail -n 2 reports/walk_forward/logs/${t,,}_lo.log && echo "$t" >> $DONE
    } >> $LOG/process.log 2>&1
  done
  [ -f $LOG/download.done ] && [ "$(sort -u $DONE | wc -l)" -ge "$(loaded | wc -l)" ] && break
  sleep 120
done
date > $LOG/process.done
