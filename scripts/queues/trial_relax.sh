#!/bin/bash
# Проба P42, 06.11 (Денис): если с 14:00 МСК не было ни одного «ПРОБА — ОТКРЫТЬ», в 15:59 МСК снизить порог входа
# до 0.5 ATR (было 1.0) и сообщить в группу. Запуск — разовым таймером systemd, см. ответ в сессии 06.11.
#   SINCE — с какого момента искать «ОТКРЫТЬ» в журнале уведомлений (время сервера, +05)
cd /home/dev/forexmodel || exit 1
crontab -l | grep -v "trial_relax.sh" | crontab -          # разовая задача — убрать себя из cron
SINCE="${1:-2026-11-06 16:00}"
LOG=reports/live/notifications.log
opened=$(awk -v s="$SINCE" '/^--- /{t=$2" "$3} /ПРОБА — ОТКРЫТЬ/ && t>=s {n++} END{print n+0}' "$LOG")
if [ "$opened" -gt 0 ]; then
  echo "$(date) проба уже открывала сделку ($opened) — порог не меняю"
  exit 0
fi
sed -i 's/^trial_impulse_atr: 1.0 /trial_impulse_atr: 0.5 /' configs/account.yaml
grep "^trial_impulse_atr" configs/account.yaml
.venv/bin/python scripts/notify.py "🧪 Проба P42: до 16:00 МСК сделки не было — порог входа снижен до 0.5 ATR (было 1.0), без тренд-гейта, срок сделки 2 ч, 1 лот. Сигналы — со следующего часового запуска (16:00). После 18:00 не пытаемся; выключить — /strategy P42 off."
