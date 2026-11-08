#!/bin/bash
# Еженедельное обновление страницы «Объём по уровням LKOH» (H84, для Тагира): история + сделки, собранные за ночи
# (scripts/queues/moex_trades_nightly.sh), → docs/results/h84_volume/index.html и lkoh_volume_levels.html, коммит только
# этих файлов и push. Артефакт на claude.ai отсюда не обновляется — его перепубликует Claude в сессии.
# cron (часы сервера +05): понедельник 08:50 = 06:50 по Москве, после закрытия недели (сессия выходного дня — до 19:00 вс)
#   50 8 * * 1  /home/dev/forexmodel/scripts/queues/h84_weekly.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/h84; mkdir -p $LOG
exec 9> $LOG/.weekly.lock
flock -n 9 || exit 0
{
  echo "=== $(date '+%F %T')"
  .venv/bin/python -W ignore scripts/moex_trades.py LKOH
  .venv/bin/python -W ignore scripts/h84_volume_weeks.py || { echo "сборка не удалась"; exit 1; }
  F="docs/results/h84_volume/index.html docs/results/h84_volume/lkoh_volume_levels.html"
  if git diff --quiet -- $F && [ -z "$(git ls-files --others --exclude-standard -- $F)" ]; then echo "страница не изменилась"; exit 0; fi
  git add -- $F
  # коммит только файлов страницы: что бы ещё ни лежало в индексе, в этот коммит не попадёт
  git commit -q --only -m "Объём по уровням LKOH (H84): обновление $(date '+%d.%m.%Y')" -- $F && \
    { git push -q || { git pull -q --rebase && git push -q; } || echo "push не удался — повтор в следующий понедельник"; }
  git log --oneline -1
} >> $LOG/weekly-$(date +%Y).log 2>&1
