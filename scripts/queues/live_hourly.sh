#!/bin/bash
# Живой контур (ТЗ инференса 26.10): счёт 15 тыс. ₽ в Т-Инвестициях, S42 на GAZP, SBER, LKOH и S46, S48 на 19 акциях,
# только лонг; сделки на первом этапе — руками по уведомлениям (scripts/notify.py, Telegram).
# Раз в час, в первые минуты после закрытия часа:
#   1. свежие минутки (Т-Инвестиции без задержки; без токена — ISS Мосбиржи) — SBER, GAZP, LKOH здесь, 16 новых бумаг —
#      в live_iterator.py;
#   2. решения S42 (scripts/live_signals.py → reports/live/<время>.md, journal.csv, s42_state.json);
#   3. сигналы S46 и S48 (scripts/live_iterator.py → iterator_state.json, iterator_journal.csv);
#   4. счёт: что берём, что закрываем, уведомления, журнал альтернативных сделок (scripts/live_account.py).
# Раз в день, в 23 ч по Москве: здоровье S42 против порогов H43 (live_health.py) и сверка живой торговли с симуляцией
# (live_deviation.py: «стоп» — стратегия перестаёт брать сигналы).
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
  .venv/bin/python -W ignore scripts/live_iterator.py || .venv/bin/python scripts/notify.py "⚠️ Контур: сигналы S46/S48 не посчитаны — см. reports/live/logs"
  .venv/bin/python -W ignore scripts/live_account.py || .venv/bin/python scripts/notify.py "⚠️ Контур: счёт не обновлён — см. reports/live/logs"
  # раз в день после вечерней сессии (часы сервера +05, торги — по Москве)
  if [ "$(TZ=Europe/Moscow date +%H)" = "23" ]; then
    .venv/bin/python scripts/live_health.py
    .venv/bin/python -W ignore scripts/live_deviation.py
  fi
} >> $LOG/$(date +%Y-%m-%d).log 2>&1
