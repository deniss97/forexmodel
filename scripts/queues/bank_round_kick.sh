#!/bin/bash
# Будильник облачной половины обхода банка гипотез: расписание облачных routine у нас не срабатывает (13.11 два запуска
# по времени не случились, ручной run — работает), поэтому сервер сам запускает routine trig_01XMJTBi6sqCnrZAoDFbCu66
# через claude -p и инструмент RemoteTrigger (он доступен и без сессии VS Code). Серверная половина — bank_round.sh в :40.
# cron (часы сервера +05; 09:10 и 19:10 МСК):  10 11,21 * * *  /home/dev/forexmodel/scripts/queues/bank_round_kick.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/live; mkdir -p $LOG
{ echo "=== $(date '+%F %T') запуск облачного обхода"
  timeout 300 "$HOME/.local/bin/claude" -p 'Через ToolSearch загрузи select:RemoteTrigger и вызови action run, trigger_id trig_01XMJTBi6sqCnrZAoDFbCu66 (без body). Ответь одной строкой: запущен (session_id) или ошибка. Больше ничего не делай.' \
    --no-session-persistence --allowedTools "ToolSearch,RemoteTrigger" --output-format text
  echo "--- код $?"; } >> $LOG/bank_round-$(date +%Y-%m).log 2>&1
