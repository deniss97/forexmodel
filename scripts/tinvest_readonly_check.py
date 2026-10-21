"""Проверка Т-Инвестиций без заявок (T29–T31): что видит токен, ничего не меняя ни на боевом счёте, ни в песочнице.

  1. счета: тип, статус, уровень доступа токена (только чтение / полный доступ); номера счетов маскируются;
  2. профиль: тариф, квалификация, лимиты запросов;
  3. деньги и бумаги на счетах (только чтение);
  4. по SBER, GAZP, LKOH: лот, шаг цены, доступна ли торговля через API, последняя цена;
  5. минутки брокера за последние дни против бесплатных минуток ISS Мосбиржи — совпадают ли цены и время;
  6. песочница: есть ли её счета (новый не открывается).
Заявок, стоп-заявок и открытия счетов здесь нет. Токен — configs/tinvest_token (или TINVEST_TOKEN), не печатается.

    python scripts/tinvest_readonly_check.py
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
import pandas as pd

from forexmodel.broker.tinvest import MSK, TInvest, TInvestError, q2d
from moex_data import candles as iss_candles

TICKERS = ["SBER", "GAZP", "LKOH"]
mask = lambda s: f"{str(s)[:4]}…" if s else "—"  # noqa: E731


def short(x: str) -> str:
    """ACCOUNT_ACCESS_LEVEL_FULL_ACCESS → full access."""
    return str(x).split("_", 3)[-1].lower().replace("_", " ") if x else "—"


def main(argv=None) -> int:
    api = TInvest(sandbox=False)
    print("1. Счета (боевой контур, только чтение)")
    accs = api.accounts()
    for a in accs:
        print(f"   {mask(a.get('id'))} {a.get('name', '')!r}: тип {short(a.get('type'))}, статус {short(a.get('status'))}, "
              f"доступ токена — {short(a.get('accessLevel'))}, открыт {str(a.get('openedDate', ''))[:10]}")
    if not accs:
        print("   счетов не видно — токен выпущен без доступа к счетам или только для песочницы")

    print("2. Профиль")
    try:
        info = api.call("UsersService", "GetInfo")
        print(f"   тариф: {info.get('tariff', '—')}, премиум: {'да' if info.get('premStatus') else 'нет'}, "
              f"квалифицированный: {'да' if info.get('qualStatus') else 'нет'}, доступно квал. инструментов: {len(info.get('qualifiedForWorkWith', []))}")
        lim = api.call("UsersService", "GetUserTariff")
        unary = {s: x.get("limitPerMinute") for x in lim.get("unaryLimits", []) for s in x.get("methods", [])}
        print(f"   лимиты: свечи {unary.get('tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles', '—')} в минуту, "
              f"заявки {unary.get('tinkoff.public.invest.api.contract.v1.OrdersService/PostOrder', '—')} в минуту, "
              f"потоков {sum(int(x.get('limit', 0)) for x in lim.get('streamLimits', []))}")
    except TInvestError as e:
        print(f"   не получилось: {e}")

    print("3. Деньги и бумаги на счетах")
    for a in accs:
        try:
            r = api.call("OperationsService", "GetPositions", {"accountId": a["id"]})
            rub = sum((q2d(x) for x in r.get("money", []) if x.get("currency") == "rub"), 0)
            print(f"   {mask(a['id'])}: рублей {float(rub):,.0f}, бумаг (позиций) {len(r.get('securities', []))}".replace(",", " "))
        except TInvestError as e:
            print(f"   {mask(a['id'])}: не получилось — {e}")

    print("4–5. Бумаги первого этапа и минутки брокера против ISS")
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("min")
    for t in TICKERS:
        s = api.share(t)
        px = api.last_price(s.uid)
        c = api.candles_1m(s.uid, end - pd.Timedelta(days=3), end)
        line = (f"   {t}: лот {s.lot}, шаг {s.step}, торговля через API {'да' if s.api_trade else 'НЕТ'}, последняя цена {px}; "
                f"минуток брокера за 3 дня {len(c)}")
        if len(c):
            c["begin"] = c["time"] + MSK                       # время Мосбиржи
            iss = iss_candles(t, f"{c['begin'].min():%Y-%m-%d}", f"{c['begin'].max():%Y-%m-%d}", 1, pause=0.1)
            iss["begin"] = pd.to_datetime(iss["begin"])
            m = c.merge(iss[["begin", "close"]], on="begin", suffixes=("", "_iss"))
            same = (m["close"] - m["close_iss"]).abs() < float(s.step) / 2
            line += (f", последняя {c['begin'].max():%d.%m %H:%M} МСК; общих с ISS минут {len(m)}, цена закрытия совпадает "
                     f"в {same.mean():.1%}")
        print(line)

    print("6. Песочница (только список счетов)")
    try:
        sb = TInvest(sandbox=True).accounts()
        print(f"   счетов песочницы: {len(sb)}" + ("" if sb else " — откроется при первом запуске проверки песочницы"))
    except TInvestError as e:
        print(f"   не получилось: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
