"""Сквозная проверка Т-Инвестиций в песочнице (задача T30): всё, что понадобится живому контуру, на одной бумаге.

  1. счёт песочницы (открыть и пополнить, если его нет);
  2. бумага по тикеру: uid, лот, шаг цены, доступна ли торговля через API;
  3. минутки за последний торговый день и последняя цена;
  4. рыночная покупка 1 лота → цена исполнения и комиссия;
  5. стоп-лосс на −6 ATR и трейлинг-стоп (активация +3 ATR, отступ 6 ATR) — как выход импульса; ATR — по часам;
  6. активные стоп-заявки → отмена обеих → продажа 1 лота;
  7. позиции до и после.
Заявки в песочнице на биржу не уходят. Токен песочницы — TINVEST_TOKEN или ~/.config/forexmodel/tinvest_token.

    python scripts/tinvest_sandbox_check.py            # SBER
    python scripts/tinvest_sandbox_check.py GAZP
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
import pandas as pd

from forexmodel.broker.tinvest import TInvest, q2d


def main(argv=None) -> int:
    ticker = (argv or ["SBER"])[0].upper()
    api = TInvest(sandbox=True)
    accs = api.accounts()
    acc = accs[0]["id"] if accs else api.open_sandbox_account()
    print(f"1. счёт песочницы {acc[:6]}…, рублей {api.money(acc):,.0f}".replace(",", " "))
    s = api.share(ticker)
    print(f"2. {s.ticker}: лот {s.lot}, шаг цены {s.step}, торговля через API: {'да' if s.api_trade else 'НЕТ'}")
    end = pd.Timestamp.utcnow().tz_localize(None)
    c = api.candles_1m(s.uid, end - pd.Timedelta(days=4), end)
    px = api.last_price(s.uid)
    print(f"3. минуток за 4 дня: {len(c)}, последняя {c['time'].iloc[-1] if len(c) else '—'} UTC; последняя цена {px}")
    h = c.set_index("time").resample("1h").agg({"high": "max", "low": "min"}).dropna()
    atr = float((h["high"] - h["low"]).tail(14).mean()) if len(h) else float(px) * 0.01
    print(f"   ATR(14) часов ≈ {atr:.4f} ({atr / float(px) * 100:.2f}% цены)")
    before = api.positions(acc).get(s.uid, 0)
    r = api.market(acc, s.uid, 1, buy=True)
    entry = q2d(r.get("executedOrderPrice")) or px
    print(f"4. покупка 1 лота: статус {r.get('executionReportStatus')}, цена {entry}, комиссия {q2d(r.get('executedCommission'))}")
    sl = api.stop_loss(acc, s, 1, float(entry) - 6 * atr)
    tr = api.trailing(acc, s, 1, float(entry) + 3 * atr, 6 * atr)
    print(f"5. стоп-лосс ≈ {float(entry) - 6 * atr:.2f} (id {sl[:8]}…), трейлинг: активация ≈ {float(entry) + 3 * atr:.2f}, отступ ≈ {6 * atr:.2f} (id {tr[:8]}…)")
    active = api.stop_orders(acc)
    print(f"6. активных стоп-заявок: {len(active)} — " + ", ".join(f"{o.get('orderType')}/{o.get('takeProfitType', '')}" for o in active))
    for o in active:
        if o["stopOrderId"] in (sl, tr):
            api.cancel_stop(acc, o["stopOrderId"])
    r2 = api.market(acc, s.uid, 1, buy=False)
    print(f"   отменены; продажа 1 лота: статус {r2.get('executionReportStatus')}, цена {q2d(r2.get('executedOrderPrice'))}")
    print(f"7. позиция {s.ticker}: до {before}, после {api.positions(acc).get(s.uid, 0)} бумаг; стоп-заявок осталось {len(api.stop_orders(acc))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
