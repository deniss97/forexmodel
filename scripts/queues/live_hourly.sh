#!/bin/bash
# Живой контур (ТЗ инференса 26.10): счёт 15 тыс. ₽ в Т-Инвестициях, S42 на GAZP, SBER, LKOH и S46, S48 на 19 акциях,
# только лонг; сделки на первом этапе — руками по уведомлениям (scripts/notify.py, Telegram).
# Раз в час, в первые минуты после закрытия часа:
#   1. свежие минутки (Т-Инвестиции без задержки; без токена — ISS Мосбиржи) — SBER, GAZP, LKOH здесь, 16 новых бумаг —
#      в live_iterator.py;
#   2. решения S42 (scripts/live_signals.py → reports/live/<время>.md, journal.csv, s42_state.json) и сразу счёт: что
#      берём, что закрываем, уведомления, журнал альтернативных сделок (scripts/live_account.py);
#   3. сигналы S46 и S48 (scripts/live_iterator.py → iterator_state.json, iterator_journal.csv) и счёт ещё раз.
# Раз в день, в 23 ч по Москве: здоровье S42 против порогов H43 (live_health.py) и сверка живой торговли с симуляцией
# (live_deviation.py: «стоп» — стратегия перестаёт брать сигналы).
#
# cron — каждый час в :01, все дни (утренняя сессия с 07:00 и торги выходного дня тоже; вне торгов новых минут нет —
# решения не меняются). S42 — первым шагом и сразу счёт: в HH:01 час уже закрыт (Т-Инвестиции отдают только
# завершённые минуты), вход стратегии в HH:01 ещё впереди — «ОТКРЫТЬ» приходит в :02–:03, а не в :07 (план 03.11).
# Если минуты часа ещё не все, вход возьмёт следующий запуск по окну 30 минут (ENTRY_GRACE). Потом S46/S48 на
# 19 бумагах (~1 мин) и счёт ещё раз — для их сигналов (повторы не шлются: сигналы и тревоги помнятся в состоянии):
#   1 * * * *  /home/dev/forexmodel/scripts/queues/live_hourly.sh
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
  .venv/bin/python -W ignore scripts/live_account.py || .venv/bin/python scripts/notify.py "⚠️ Контур: счёт не обновлён — $(tail -n 1 $LOG/$(date +%Y-%m-%d).log | cut -c1-300)"
  # при сбое — причина из последней строки лога (её только что дописал упавший шаг)
  .venv/bin/python -W ignore scripts/live_iterator.py || .venv/bin/python scripts/notify.py "⚠️ Контур: сигналы S46/S48 не посчитаны — $(tail -n 1 $LOG/$(date +%Y-%m-%d).log | cut -c1-300)"
  .venv/bin/python -W ignore scripts/live_account.py || .venv/bin/python scripts/notify.py "⚠️ Контур: счёт не обновлён — $(tail -n 1 $LOG/$(date +%Y-%m-%d).log | cut -c1-300)"
  # раз в день после вечерней сессии (часы сервера +05, торги — по Москве)
  if [ "$(TZ=Europe/Moscow date +%H)" = "23" ]; then
    .venv/bin/python scripts/live_health.py
    .venv/bin/python -W ignore scripts/live_deviation.py
    .venv/bin/python -W ignore scripts/live_charts.py --send      # картинка «состояние контура» в группу (29.10)
  fi
} >> $LOG/$(date +%Y-%m-%d).log 2>&1
