#!/bin/bash
# T41 (созвон 23.10): минутки новых акций Мосбиржи с 2015 года — ISS, по восемь бумаг одновременно (ISS отвечает ~1.3 с на запрос — упор в ожидание, не в память), с докачкой
# (повторный запуск продолжает с последней свечи файла). Отбор — scripts/new_stocks.py: в индексе Мосбиржи, история
# на TQBR с начала 2015 года, медианный оборот 2016–2020 ≥ 100 млн ₽ в день, одна бумага на эмитента, без SBER, GAZP,
# LKOH (на них выбраны стратегии) и MOEX, MTSS (данные уже есть). Сверка файлов — после, по одной бумаге: целиком
# файл читается только там (~250 МБ на бумагу).
#   systemd-run --user --unit=new-stocks-dl -p MemoryMax=400M scripts/queues/new_stocks_download.sh
cd /home/dev/forexmodel || exit 1
LOG=reports/new_stocks; mkdir -p $LOG
.venv/bin/python - >> $LOG/download.log 2>&1 <<'EOF'
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
import moex_data as md

TICKERS = "TATN NVTK GMKN ROSN VTBR PLZL SNGSP IRAO CHMF NLMK MAGN RTKM TRNFP AFLT ALRS AFKS".split()

def one(t):
    try:
        md.download(t, "2015-01-01", out=md.OUT / f"{t}_1m.csv", quiet=True)
    except Exception as e:  # обрыв — повторный запуск докачает
        print(f"[{t}] ошибка: {e}", flush=True)

with ThreadPoolExecutor(8) as ex:
    list(ex.map(one, TICKERS))
print("готово", flush=True)
EOF
date > $LOG/download.done
