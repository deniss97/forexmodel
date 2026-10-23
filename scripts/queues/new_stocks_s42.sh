#!/bin/bash
# H66 (23.10): S42 на новых акциях — штатно, как long_only.sh (S25/S42): только лонг, гейт C, дивиденды, стоп по гэпу,
# выход 240 ч / 6 ATR; издержки биржи пересчитывает журнал (configs/costs/exchange.yaml). По одной бумаге — память.
# Минутки — scripts/new_stocks.py prepare, дивиденды — scripts/fetch_dividends.py --tickers.
#   systemd-run --user --unit=new-stocks-s42 -p MemoryMax=1500M scripts/queues/new_stocks_s42.sh [БУМАГИ]
cd /home/dev/forexmodel || exit 1
LOG=reports/walk_forward/logs; mkdir -p $LOG
EX="--set simulation.horizon_minutes=14400 --set simulation.sl_atr=6 --set simulation.trail_atr=6 --set simulation.activate_atr=3 --set simulation.stop_fill=gap"
GATE="--set trend_gate.enabled=true --set trend_gate.require_adx_rising=false --set trend_gate.update_every_bar=true --set trend_gate.hold_bars=2 --set simulation.trend_col=trend_gate"
DIV="--set data.dividends_path=data/dividends/moex_dividends.csv"
for f in ${@:-TATN NVTK GMKN ROSN VTBR PLZL SNGSP IRAO CHMF NLMK MAGN RTKM TRNFP AFLT ALRS AFKS}; do
  n=${f,,}
  [ -f reports/walk_forward/${n}_lo__impulse_6_3_long_trades.csv ] && continue
  .venv/bin/python scripts/walk_forward.py -c configs/default.yaml --years 2016 2026 $EX $GATE --set data.csv_path=data/raw/moex/${f}_1m.csv $DIV --set data.dividend_ticker=$f --name ${n}_lo --variants impulse:6:3:long > $LOG/${n}_lo.log 2>&1
done
date > $LOG/queue_new_stocks_s42.done
