#!/bin/bash
# Живой контур, первый этап запуска (T16): акции GAZP, SBER, LKOH на Мосбирже только в лонг.
# Раз в час, в первые минуты после закрытия часа: докачать свежие минуты с ISS и посчитать решения
# (reports/live/<время>.md и строка в reports/live/journal.csv). Раз в день — здоровье стратегии
# против порогов тревоги H43 (reports/live/health.md).
#
# cron — каждый час в :05, все дни (утренняя сессия с 07:00 и торги выходного дня тоже; вне торгов новых минут нет —
# решения не меняются). В :02 минута входа HH:01 — последняя в данных, и контур её не видит; будни 10–23 ч теряли
# треть входов (бумажный прогон 22.10, docs/results/paper_replay.md):
#   5 * * * *  /home/dev/forexmodel/scripts/queues/live_hourly.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/live/logs; mkdir -p $LOG
exec 9> reports/live/.hourly.lock
flock -n 9 || exit 0                                  # предыдущий запуск ещё идёт
{
  date
  # минутки: Т-Инвестиции без задержки, если есть токен (созвон 18.10); иначе бесплатный ISS Мосбиржи
  if [ -n "$TINVEST_TOKEN" ] || grep -qsv '^[[:space:]]*\(#\|$\)' configs/tinvest_token "$HOME/.config/forexmodel/tinvest_token"; then
    .venv/bin/python scripts/fetch_tinvest.py SBER GAZP LKOH || .venv/bin/python scripts/fetch_moex_iss.py SBER GAZP LKOH
  else
    .venv/bin/python scripts/fetch_moex_iss.py SBER GAZP LKOH
  fi
  .venv/bin/python scripts/live_signals.py --only sber gazp lkoh
  # здоровье — раз в день после вечерней сессии (часы сервера +05, торги — по Москве)
  if [ "$(TZ=Europe/Moscow date +%H)" = "23" ]; then .venv/bin/python scripts/live_health.py; fi
} >> $LOG/$(date +%Y-%m-%d).log 2>&1
