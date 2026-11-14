#!/bin/bash
# Обход банка гипотез — серверная половина (просьба Дениса 12.11; переделано 13.11).
# Банк гипотез (ArtifactData) доступен только облачной routine claude.ai «forexmodel: обход банка гипотез»: в cron-запуске
# claude -p на сервере этого инструмента нет (он приходит только из открытой сессии VS Code). Поэтому:
#   облако (09:10 и 19:10 МСК) — банк: записывает итоги из bank_sync/results, пишет планы с вопросами, кладёт
#     подтверждённые планы в bank_sync/queue/Hxx.json, сообщения для группы — в bank_sync/outbox/*.md; ветка bank-sync;
#   сервер (этот скрипт, 09:40 и 19:40 МСК) — забирает ветку, шлёт сообщения из outbox, по одному заданию из queue
#     считает на своих данных (claude -p, задание scripts/queues/bank_round_prompt.md), шлёт «начинаю» / итог в группу и
#     кладёт итог в bank_sync/results/Hxx.json → push; в банк его запишет следующий облачный обход.
# Рабочая копия ветки — отдельный worktree ~/forexmodel-bank-sync, основная папка проекта не трогается.
# cron (часы сервера +05):  40 11,21 * * *  /home/dev/forexmodel/scripts/queues/bank_round.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/live; mkdir -p $LOG
exec 9> $LOG/.bank_round.lock
flock -n 9 || exit 0
exec >> $LOG/bank_round-$(date +%Y-%m).log 2>&1
echo "=== $(date '+%F %T')"
WT=$HOME/forexmodel-bank-sync
PY=/home/dev/forexmodel/.venv/bin/python

git fetch -q origin bank-sync 2>/dev/null || { echo "ветки bank-sync на GitHub ещё нет — облачный обход её создаст"; exit 0; }
if [ ! -d "$WT" ]; then git worktree add -q -B bank-sync "$WT" origin/bank-sync || exit 1; fi
git -C "$WT" checkout -q bank-sync && git -C "$WT" reset -q --hard origin/bank-sync || exit 1
mkdir -p "$WT/bank_sync/results" "$WT/bank_sync/queue" "$WT/bank_sync/outbox"

# 1. сообщения из облака → группа (каждое один раз)
SENT=$LOG/bank_outbox_sent.txt; touch $SENT
for f in "$WT"/bank_sync/outbox/*.md; do
  [ -e "$f" ] || continue
  id=$(basename "$f")
  grep -qxF "$id" $SENT && continue
  $PY scripts/hyp_post.py note --file "$f" && echo "$id" >> $SENT && echo "отправлено в группу: $id"
done

# 2. одно задание из очереди, по которому ещё нет итога
TODO=""
for q in "$WT"/bank_sync/queue/*.json; do
  [ -e "$q" ] || continue
  c=$(basename "$q" .json)
  [ -e "$WT/bank_sync/results/$c.json" ] && continue
  TODO=$c; break
done
if [ -z "$TODO" ]; then echo "заданий нет"; exit 0; fi
echo "считаю $TODO"
timeout 3h "$HOME/.local/bin/claude" -p "$(sed "s|{CODE}|$TODO|g; s|{WT}|$WT|g" scripts/queues/bank_round_prompt.md)" \
  --no-session-persistence --allowedTools "Read,Grep,Glob,Write,Edit,Bash" --output-format text
echo "--- код $?"

# 3. итог → GitHub (только папка bank_sync в ветке bank-sync)
if [ -n "$(git -C "$WT" status --porcelain bank_sync)" ]; then
  git -C "$WT" add bank_sync && git -C "$WT" commit -q -m "bank-sync: итог $TODO с сервера $(date '+%F %H:%M')" \
    && git -C "$WT" push -q origin bank-sync && echo "итог $TODO отправлен в ветку bank-sync"
fi
