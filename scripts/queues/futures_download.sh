#!/bin/bash
# H68 (24.10): минутки фьючерсов Мосбиржи на серебро (SILV) и золото (GOLD) с 2015 года — каждый контракт, пока он
# ближний (с запасом 15 дней на переход), в data/raw/moex/<серия>/<контракт>_1m.csv. Склейка в непрерывный ряд —
# отдельно, scripts/metals_futures.py prepare (вместе с выгрузкой склейка не помещалась в потолок памяти службы).
# Контракты — по восемь одновременно (ISS отвечает ~1.3 с на запрос), с докачкой; меньше 10 ГБ на диске — стоп.
#   systemd-run --user --unit=futures-dl -p MemoryMax=600M scripts/queues/futures_download.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/futures; mkdir -p $LOG
.venv/bin/python - >> $LOG/download.log 2>&1 <<'EOF'
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
import moex_data as md

jobs = []
for code in ("SILV", "GOLD"):
    s = md.futures_series(code)
    for r in md.front_windows(s, "2015-01-01").itertuples():
        jobs.append((code, r.secid, r.since, r.till))

def one(j):
    code, secid, a, b = j
    try:
        md.download(secid, a, b, 1, "futures", out=md.OUT / code / f"{secid}_1m.csv", quiet=True)
    except Exception as e:  # обрыв — повторный запуск докачает
        print(f"[{secid}] ошибка: {e}", flush=True)

with ThreadPoolExecutor(8) as ex:
    list(ex.map(one, jobs))
print("готово", flush=True)
EOF
date > $LOG/download.done
