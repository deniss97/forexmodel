"""H81 (Тимур): можно ли быстро и дёшево читать идеи из ресурсов для квантов (список — «Ресурсы для алготрейдеров.docx»,
прикреплён к H81 в банке). Шаг 1 — доступ: для каждого ресурса с нашего сервера пробуем ленту (RSS/Atom), API или
страницу и считаем, сколько материалов вышло за 7 и 30 дней. Шаг 2 — выгрузка за 30 дней из открытых лент в
reports/h81/items.json (заголовок, ссылка, дата, аннотация) для отбора идей. Отчёт — docs/results/h81_sources.md
(вывод и карточки идей дописываются вручную между метками).

    python scripts/h81_sources.py
"""
from __future__ import annotations

import json
import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import _bootstrap  # noqa: F401

from experiments import ROOT

DOC = ROOT / "docs" / "results" / "h81_sources.md"
OUT = ROOT / "reports" / "h81"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
NOW = datetime.now(timezone.utc)
MARK = ("<!-- вывод -->", "<!-- /вывод -->")

# (группа, ресурс, что пробуем, адрес)
SOURCES = [
    ("агрегаторы", "Quantocracy", "rss", "https://quantocracy.com/feed/"),
    ("агрегаторы", "Quantpedia (блог)", "rss", "https://quantpedia.com/feed/"),
    ("агрегаторы", "SSRN", "html", "https://www.ssrn.com/index.cfm/en/fin/"),
    ("агрегаторы", "arXiv q-fin.TR (торговля и микроструктура)", "arxiv", "q-fin.TR"),
    ("агрегаторы", "arXiv q-fin.ST (статистика финансов)", "arxiv", "q-fin.ST"),
    ("агрегаторы", "arXiv q-fin.PM (портфели)", "arxiv", "q-fin.PM"),
    ("фонды", "Man Group / Man AHL", "html", "https://www.man.com/insights"),
    ("фонды", "AQR Insights", "html", "https://www.aqr.com/Insights/Research"),
    ("фонды", "Alpha Architect", "rss", "https://alphaarchitect.com/feed/"),
    ("фонды", "Concretum Group", "rss", "https://concretumgroup.com/feed/"),
    ("блоги", "Rob Carver (qoppac)", "rss", "https://qoppac.blogspot.com/feeds/posts/default"),
    ("блоги", "Hudson & Thames", "rss", "https://hudsonthames.org/feed/"),
    ("блоги", "Robot Wealth", "rss", "https://robotwealth.com/feed/"),
    ("блоги", "QuantStart", "html", "https://www.quantstart.com/articles/"),
    ("блоги", "Jonathan Kinlay", "rss", "https://jonathankinlay.com/feed/"),
    ("блоги", "PredictNow.ai (Ernest Chan)", "html", "https://predictnow.ai/blog/"),
    ("сообщества", "Quant StackExchange", "api", "https://api.stackexchange.com/2.3/questions?site=quant&sort=creation&order=desc&pagesize=100"),
    ("сообщества", "Reddit r/algotrading", "api", "https://www.reddit.com/r/algotrading/new.json?limit=100"),
    ("сообщества", "Reddit r/quant", "api", "https://www.reddit.com/r/quant/new.json?limit=100"),
    ("сообщества", "QuantConnect (форум)", "html", "https://www.quantconnect.com/forum/discussions/1/"),
    ("сообщества", "Wilmott forum", "html", "https://forum.wilmott.com/"),
    ("сообщества", "Kaggle", "html", "https://www.kaggle.com/competitions"),
    ("подкасты", "Flirting with Models", "rss", "https://feeds.libsyn.com/127221/rss"),
    ("подкасты", "Top Traders Unplugged", "rss", "https://feeds.megaphone.fm/toptradersunplugged"),
    ("русскоязычные", "Smart-Lab (лента)", "rss", "https://smart-lab.ru/rss/"),
    ("русскоязычные", "Хабр: алготрейдинг", "rss", "https://habr.com/ru/rss/hubs/algo_trading/articles/?fl=ru"),
    ("русскоязычные", "ODS.ai", "html", "https://ods.ai/"),
    ("русскоязычные", "GitHub moexalgo / WISEPLAT", "api", "https://api.github.com/users/WISEPLAT/repos?sort=updated&per_page=100"),
]


def fetch(url: str, timeout: int = 25) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
            return r.status, r.read(5_000_000)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001
        return 0, str(e).encode()[:200]


def when(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def parse_feed(body: bytes) -> list[dict]:
    root = ET.fromstring(body)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    items = []
    for it in root.iter("item"):
        items.append({"title": strip(it.findtext("title")), "link": (it.findtext("link") or "").strip(),
                      "date": when(it.findtext("pubDate")), "summary": strip(it.findtext("description"))[:1500]})
    for it in root.findall("a:entry", ns):
        link = it.find("a:link[@rel='alternate']", ns) or it.find("a:link", ns)
        items.append({"title": strip(it.findtext("a:title", namespaces=ns)), "link": link.get("href") if link is not None else "",
                      "date": when(it.findtext("a:published", namespaces=ns) or it.findtext("a:updated", namespaces=ns)),
                      "summary": strip(it.findtext("a:summary", namespaces=ns) or it.findtext("a:content", namespaces=ns))[:1500]})
    return items


def arxiv(cat: str) -> tuple[int, list[dict]]:
    url = (f"http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending"
           f"&start=0&max_results=200")
    st, body = fetch(url, 60)
    return st, parse_feed(body) if st == 200 and body else []


def probe(kind: str, url: str) -> dict:
    if kind == "arxiv":
        st, items = arxiv(url)
        return {"код": st, "доступ": "API (Atom)" if items else "нет", "items": items}
    st, body = fetch(url)
    r = {"код": st, "доступ": "нет", "items": []}
    if st != 200 or not body:
        r["доступ"] = f"закрыт ({st or 'нет ответа'})"
        return r
    if kind == "rss":
        try:
            r["items"] = parse_feed(body)
            r["доступ"] = "RSS" if r["items"] else "RSS пустой"
        except ET.ParseError:
            r["доступ"] = "не RSS (страница)"
    elif kind == "api":
        try:
            j = json.loads(body)
            if "items" in j:            # StackExchange
                r["items"] = [{"title": strip(x.get("title")), "link": x.get("link"), "date": datetime.fromtimestamp(x["creation_date"], timezone.utc),
                               "summary": ", ".join(x.get("tags", []))} for x in j["items"]]
            elif "data" in j:           # Reddit
                r["items"] = [{"title": c["data"].get("title", ""), "link": "https://www.reddit.com" + c["data"].get("permalink", ""),
                               "date": datetime.fromtimestamp(c["data"]["created_utc"], timezone.utc), "summary": c["data"].get("selftext", "")[:1500]}
                              for c in j["data"].get("children", [])]
            elif isinstance(j, list):   # GitHub
                r["items"] = [{"title": x.get("name"), "link": x.get("html_url"), "date": when(x.get("pushed_at")), "summary": x.get("description") or ""} for x in j]
            r["доступ"] = "API (JSON)"
        except (ValueError, KeyError):
            r["доступ"] = "ответ не JSON"
    else:
        txt = body[:300000].decode("utf-8", "ignore")
        blocked = re.search(r"cf-chl|captcha|Just a moment|enable JavaScript|Access denied", txt, re.I)
        r["доступ"] = "страница за защитой" if blocked else "страница (HTML)"
    return r


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pool = [], []
    for grp, name, kind, url in SOURCES:
        r = probe(kind, url)
        d7 = sum(1 for x in r["items"] if x["date"] and x["date"] >= NOW - timedelta(days=7))
        d30 = sum(1 for x in r["items"] if x["date"] and x["date"] >= NOW - timedelta(days=30))
        last = max((x["date"] for x in r["items"] if x["date"]), default=None)
        rows.append({"группа": grp, "ресурс": name, "адрес": url if kind != "arxiv" else f"export.arxiv.org API, {url}",
                     "доступ": r["доступ"], "код": r["код"], "в ленте": len(r["items"]), "за 7 дней": d7, "за 30 дней": d30,
                     "последний": last.date().isoformat() if last else ""})
        for x in r["items"]:
            if x["date"] and x["date"] >= NOW - timedelta(days=30):
                pool.append({"источник": name, **x, "date": x["date"].isoformat()})
        print(f"{name:45s} {r['доступ']:22s} {len(r['items']):4d} {d7:4d} {d30:4d}", flush=True)
    (OUT / "items.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    a_, b_ = old.find(MARK[0]), old.find(MARK[1])
    verdict = old[a_:b_ + len(MARK[1])] if a_ >= 0 and b_ > a_ else f"{MARK[0]}\n{MARK[1]}"
    md = ["# Идеи из ресурсов для квантов: что читается автоматически (H81)", "",
          "Список ресурсов — файл Тимура «Ресурсы для алготрейдеров» (прикреплён к H81 в банке). Для каждого ресурса с нашего "
          f"сервера пробуем ленту (RSS/Atom), API или страницу; материалы считаются по датам публикации на {NOW:%d.%m.%Y}. "
          "Расчёт — `scripts/h81_sources.py`; материалы за 30 дней — `reports/h81/items.json`.", "",
          "**Критерий (банк, до расчёта):** автоматически читается большинство источников идей, и хотя бы 3 из 5 отобранных "
          "идей можно проверить на наших данных за 1–2 дня.", "", verdict, "",
          "## Доступ с сервера", "", "| группа | ресурс | доступ | в ленте | за 7 дней | за 30 дней | последний |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['группа']} | {r['ресурс']} | {r['доступ']} | {r['в ленте']} | {r['за 7 дней']} | {r['за 30 дней']} | {r['последний']} |")
    md += ["", f"Всего материалов за 30 дней из открытых лент: {len(pool)}."]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("материалов за 30 дней:", len(pool), "→", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
