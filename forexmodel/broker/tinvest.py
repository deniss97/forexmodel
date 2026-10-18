"""Клиент Т-Инвестиций (Invest API) через REST — для запуска импульсной стратегии (задача T30, созвон 18.10).

REST повторяет методы gRPC (https://<хост>/rest/tinkoff.public.invest.api.contract.v1.<Сервис>/<Метод>, JSON с полями
в camelCase), поэтому внешних зависимостей нет: официальный Python SDK требует пакет `tinkoff` с gRPC-заглушками,
которого нет на pypi.org. Описание методов — protobuf в github.com/RussianInvestments/investAPI.

Сервер API подписан российским удостоверяющим центром (Russian Trusted Root CA, Минцифры), обычные системы ему не
доверяют. Корневой сертификат — configs/certs/russian_trusted_root_ca.pem (с Госуслуг, отпечаток совпадает с корнем,
который присылает сервер); подключается только к этому клиенту, системное доверие не меняется.

Токен — из переменной TINVEST_TOKEN или файла ~/.config/forexmodel/tinvest_token (права 600), никогда не печатается и
не попадает в репозиторий. Песочница — отдельный хост и отдельные методы (SandboxService): заявки не уходят на биржу.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CA_FILE = ROOT / "configs" / "certs" / "russian_trusted_root_ca.pem"
CA_SHA256 = "D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31"
TOKEN_FILE = Path.home() / ".config" / "forexmodel" / "tinvest_token"
HOSTS = {False: "https://invest-public-api.tinkoff.ru/rest/", True: "https://sandbox-invest-public-api.tinkoff.ru/rest/"}
PKG = "tinkoff.public.invest.api.contract.v1."
MSK = pd.Timedelta(hours=3)          # минутки Мосбиржи в проекте — по московскому времени, API отдаёт UTC


class TInvestError(RuntimeError):
    def __init__(self, status: int, payload: dict):
        self.status, self.payload = status, payload
        super().__init__(f"HTTP {status}: {payload.get('message', payload)} (код {payload.get('description', payload.get('code'))})")


def load_token() -> str:
    tok = os.environ.get("TINVEST_TOKEN", "").strip()
    if not tok and TOKEN_FILE.exists():
        if TOKEN_FILE.stat().st_mode & 0o077:
            raise PermissionError(f"{TOKEN_FILE}: доступ должен быть только у владельца (chmod 600)")
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not tok:
        raise RuntimeError("нет токена: задайте TINVEST_TOKEN или положите его в ~/.config/forexmodel/tinvest_token (chmod 600)")
    return tok


def ssl_context() -> ssl.SSLContext:
    pem = CA_FILE.read_bytes()
    der = ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))
    if hashlib.sha256(der).hexdigest().upper() != CA_SHA256:
        raise RuntimeError(f"{CA_FILE}: отпечаток корневого сертификата не совпадает — файл подменён")
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cadata=pem.decode("ascii"))
    return ctx


def q2d(q: dict | None) -> Decimal:
    """Quotation / MoneyValue {units, nano} → Decimal."""
    if not q:
        return Decimal(0)
    return Decimal(int(q.get("units", 0) or 0)) + Decimal(int(q.get("nano", 0) or 0)) / Decimal(10 ** 9)


def d2q(x) -> dict:
    d = Decimal(str(x))
    units = int(d.to_integral_value(rounding=ROUND_DOWN))
    return {"units": str(units), "nano": int((d - units) * 10 ** 9)}


def ts(t: pd.Timestamp) -> str:
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Share:
    ticker: str
    uid: str
    figi: str
    lot: int
    step: Decimal          # шаг цены
    api_trade: bool

    def round_price(self, price, down: bool = True) -> Decimal:
        """Цена, кратная шагу: для стопа на продажу — вниз (сработает не раньше уровня), иначе по ближайшему."""
        n = (Decimal(str(price)) / self.step).to_integral_value(rounding=ROUND_DOWN if down else ROUND_HALF_UP)
        return n * self.step


class TInvest:
    """Тонкий клиент: инструменты, свечи, счёт и позиции, рыночные заявки, стоп-лосс и трейлинг-стоп у брокера."""

    def __init__(self, sandbox: bool = True, token: str | None = None, timeout: float = 30.0):
        self.sandbox, self.timeout = sandbox, timeout
        self._token = token or load_token()
        self._ctx = ssl_context()
        self.base = HOSTS[sandbox]

    def call(self, service: str, method: str, body: dict | None = None, retries: int = 3) -> dict:
        req = urllib.request.Request(self.base + PKG + service + "/" + method, json.dumps(body or {}).encode("utf-8"),
                                     {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json",
                                      "x-app-name": "forexmodel"})
        for k in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
                    return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:
                payload = json.loads(e.read() or b"{}")
                if e.code == 429 and k < retries - 1:              # лимит запросов — подождать и повторить
                    time.sleep(2 * (k + 1))
                    continue
                raise TInvestError(e.code, payload) from None
            except urllib.error.URLError:
                if k == retries - 1:
                    raise
                time.sleep(1 + k)
        return {}

    # ---------------- инструменты и данные
    def share(self, ticker: str, class_code: str = "TQBR") -> Share:
        x = self.call("InstrumentsService", "ShareBy", {"idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": class_code, "id": ticker})["instrument"]
        return Share(x["ticker"], x["uid"], x["figi"], int(x["lot"]), q2d(x["minPriceIncrement"]), bool(x.get("apiTradeAvailableFlag")))

    def candles_1m(self, uid: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Минутные свечи [start, end) по UTC — кусками по суткам (ограничение API для минуток), только завершённые."""
        rows, a = [], pd.Timestamp(start)
        while a < end:
            b = min(a + pd.Timedelta(days=1), pd.Timestamp(end))
            r = self.call("MarketDataService", "GetCandles", {"instrumentId": uid, "from": ts(a), "to": ts(b),
                                                              "interval": "CANDLE_INTERVAL_1_MIN", "candleSourceType": "CANDLE_SOURCE_EXCHANGE"})
            rows += [c for c in r.get("candles", []) if c.get("isComplete")]
            a = b
            time.sleep(0.1)
        if not rows:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume_lots"])
        return pd.DataFrame({"time": pd.to_datetime([c["time"] for c in rows], utc=True).tz_convert(None),
                             "open": [float(q2d(c["open"])) for c in rows], "high": [float(q2d(c["high"])) for c in rows],
                             "low": [float(q2d(c["low"])) for c in rows], "close": [float(q2d(c["close"])) for c in rows],
                             "volume_lots": [int(c.get("volume", 0)) for c in rows]})

    def last_price(self, uid: str) -> Decimal:
        r = self.call("MarketDataService", "GetLastPrices", {"instrumentId": [uid]})
        return q2d(r["lastPrices"][0]["price"])

    # ---------------- счёт
    def accounts(self) -> list[dict]:
        svc, m = ("SandboxService", "GetSandboxAccounts") if self.sandbox else ("UsersService", "GetAccounts")
        return self.call(svc, m).get("accounts", [])

    def open_sandbox_account(self, rub: float = 1_000_000) -> str:
        assert self.sandbox, "только в песочнице"
        acc = self.call("SandboxService", "OpenSandboxAccount", {"name": "forexmodel"})["accountId"]
        self.call("SandboxService", "SandboxPayIn", {"accountId": acc, "amount": {"currency": "rub", **d2q(rub)}})
        return acc

    def positions(self, account: str) -> dict[str, int]:
        """uid → количество бумаг (штук), включая заблокированные заявками."""
        svc, m = ("SandboxService", "GetSandboxPositions") if self.sandbox else ("OperationsService", "GetPositions")
        r = self.call(svc, m, {"accountId": account})
        return {s["instrumentUid"]: int(s.get("balance", 0)) + int(s.get("blocked", 0)) for s in r.get("securities", [])}

    def money(self, account: str) -> Decimal:
        svc, m = ("SandboxService", "GetSandboxPositions") if self.sandbox else ("OperationsService", "GetPositions")
        r = self.call(svc, m, {"accountId": account})
        return sum((q2d(x) for x in r.get("money", []) if x.get("currency") == "rub"), Decimal(0))

    # ---------------- заявки
    def market(self, account: str, uid: str, lots: int, buy: bool) -> dict:
        """Рыночная заявка; возвращает ответ брокера (исполненная цена — executedOrderPrice, комиссия — executedCommission)."""
        svc, m = ("SandboxService", "PostSandboxOrder") if self.sandbox else ("OrdersService", "PostOrder")
        return self.call(svc, m, {"instrumentId": uid, "quantity": str(int(lots)), "accountId": account,
                                  "direction": "ORDER_DIRECTION_BUY" if buy else "ORDER_DIRECTION_SELL",
                                  "orderType": "ORDER_TYPE_MARKET", "orderId": str(uuid.uuid4())})

    def _stop(self, account: str, body: dict) -> str:
        svc, m = ("SandboxService", "PostSandboxStopOrder") if self.sandbox else ("StopOrdersService", "PostStopOrder")
        body = {"accountId": account, "expirationType": "STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL",
                "direction": "STOP_ORDER_DIRECTION_SELL", "orderId": str(uuid.uuid4()), **body}
        return self.call(svc, m, body)["stopOrderId"]

    def stop_loss(self, account: str, share: Share, lots: int, stop_price) -> str:
        """Защитный стоп лонга: при цене ≤ stop_price — продажа по рынку (как стоп в симуляторе)."""
        return self._stop(account, {"instrumentId": share.uid, "quantity": str(int(lots)), "stopOrderType": "STOP_ORDER_TYPE_STOP_LOSS",
                                    "stopPrice": d2q(share.round_price(stop_price, down=True)), "exchangeOrderType": "EXCHANGE_ORDER_TYPE_MARKET"})

    def trailing(self, account: str, share: Share, lots: int, activation_price, indent) -> str:
        """Трейлинг-стоп лонга (take-profit с подтипом trailing): после цены активации следит за максимумом и продаёт
        по рынку при откате на indent (в единицах цены) — как трейлинг импульса: активация +3 ATR, отступ 6 ATR."""
        return self._stop(account, {"instrumentId": share.uid, "quantity": str(int(lots)), "stopOrderType": "STOP_ORDER_TYPE_TAKE_PROFIT",
                                    "takeProfitType": "TAKE_PROFIT_TYPE_TRAILING", "exchangeOrderType": "EXCHANGE_ORDER_TYPE_MARKET",
                                    "stopPrice": d2q(share.round_price(activation_price, down=False)),
                                    "trailingData": {"indent": d2q(share.round_price(indent, down=False)), "indentType": "TRAILING_VALUE_ABSOLUTE",
                                                     "spread": d2q("0.1"), "spreadType": "TRAILING_VALUE_RELATIVE"}})

    def stop_orders(self, account: str) -> list[dict]:
        svc, m = ("SandboxService", "GetSandboxStopOrders") if self.sandbox else ("StopOrdersService", "GetStopOrders")
        return self.call(svc, m, {"accountId": account, "status": "STOP_ORDER_STATUS_ACTIVE"}).get("stopOrders", [])

    def cancel_stop(self, account: str, stop_order_id: str) -> None:
        svc, m = ("SandboxService", "CancelSandboxStopOrder") if self.sandbox else ("StopOrdersService", "CancelStopOrder")
        self.call(svc, m, {"accountId": account, "stopOrderId": stop_order_id})


def to_moex_minutes(c: pd.DataFrame, lot: int) -> pd.DataFrame:
    """Свечи API → формат минуток Мосбиржи в проекте (open, close, high, low, value, volume, begin, end; время московское)."""
    begin = c["time"] + MSK
    vol = c["volume_lots"] * lot
    return pd.DataFrame({"open": c["open"], "close": c["close"], "high": c["high"], "low": c["low"], "value": vol * c["close"],
                         "volume": vol, "begin": begin.dt.strftime("%Y-%m-%d %H:%M:%S"),
                         "end": (begin + pd.Timedelta(seconds=59)).dt.strftime("%Y-%m-%d %H:%M:%S")})
