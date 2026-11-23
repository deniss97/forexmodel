"""H94, просьба Тимура (исследовательский чат, 21.11, сообщения 409–418): все случаи «ход на 3 часовых ATR за 60 минут»
на одной бумаге на сплошном графике — два HTML: часовые свечи и 5-минутные, с августа 2025 по последнюю минуту.

Случаи — ровно как в H94 (scripts/h94_events.py): ход за 60 минут = закрытие минуты − последнее закрытие не позже чем
60 минут назад; ATR — средний истинный размах 14 прошлых закрытых часов; эпизод — подряд идущие минуты с ходом
≥ 1.5 ATR в одну сторону; в файл попадают эпизоды, у которых пик ≥ 3 ATR. Минутки — горячий файл контура
(data/raw/live/<бумага>_1m_live.csv), поэтому случаи — до последнего дня, а не до июня 2026, как в архиве H94.

На графике: полоса — 60 минут, за которые цена прошла пик хода; стрелка с номером — на свече пика. Справа — список
случаев: щелчок (или ← / →) — график переходит к случаю с запасом истории до него.

    python scripts/h94_html.py [--tk LKOH] [--k 3] [--since 2025-08-01]
    → docs/results/h94/<бумага>_<k>ATR_1h.html, <бумага>_<k>ATR_5m.html (автономные: библиотека графиков внутри)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import h94_events as ev  # noqa: E402

OUT = ROOT / "docs" / "results" / "h94"
LWC_URL = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
LWC = ROOT / "data" / "cache" / "js" / "lightweight-charts-4.2.0.js"
W = 60


def lwc_js() -> str:
    if not LWC.exists():
        LWC.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LWC_URL, timeout=60) as r:
            LWC.write_bytes(r.read())
    return LWC.read_text(encoding="utf-8")


def load(tk: str, since: pd.Timestamp) -> pd.DataFrame:
    src = ROOT / "data" / "raw" / "live" / f"{tk}_1m_live.csv"
    d = pd.read_csv(src, usecols=["begin", "open", "high", "low", "close", "volume"])
    d["begin"] = pd.to_datetime(d["begin"])
    d = d.drop_duplicates("begin").set_index("begin").sort_index()
    d.index.name = "t"
    return d[d.index >= since - pd.Timedelta(days=30)]          # запас на 14 часов ATR


def bars(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    b = d.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    return b


def epoch(t: pd.Timestamp) -> int:
    """Время МСК как «UTC» — чтобы шкала графика показывала московское время."""
    return int(pd.Timestamp(t).tz_localize(None).timestamp()) if pd.Timestamp(t).tzinfo else int(
        (pd.Timestamp(t) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))


def page(tk: str, tf: str, b: pd.DataFrame, cases: list[dict], k: float, since: str, last: str, rule: str,
         ctx_before: int, ctx_after: int) -> str:
    candles = [[epoch(t), round(r.open, 4), round(r.high, 4), round(r.low, 4), round(r.close, 4)]
               for t, r in b.iterrows()]
    meta = {"tk": tk, "tf": tf, "k": k, "w": W, "ctxBefore": ctx_before, "ctxAfter": ctx_after,
            "step": int(pd.Timedelta(rule).total_seconds())}
    up = sum(c["dir"] == 1 for c in cases)
    night = sum(c["cross"] for c in cases)
    title = f"{tk}: ходы на {k:g} часовых ATR за {W} минут — {tf}"
    data = json.dumps({"candles": candles, "cases": cases, "meta": meta}, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fff; --ink:#1d2433; --muted:#667085; --line:#e4e7ec; --up:#16a34a; --down:#dc2626; --sel:#eef4ff; --band:rgba(42,107,216,.13); }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11151c; --ink:#e6e9ef; --muted:#98a2b3; --line:#2a313c; --sel:#1d2a40; --band:rgba(96,148,255,.20); }} }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; height:100%; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
header {{ padding:10px 16px 8px; border-bottom:1px solid var(--line); }}
h1 {{ font-size:17px; margin:0 0 3px; }}
.sub {{ color:var(--muted); font-size:12.5px; }}
.sub b {{ color:var(--ink); }}
main {{ display:flex; height:calc(100% - 74px); }}
#chart {{ flex:1; min-width:0; position:relative; }}
aside {{ width:330px; border-left:1px solid var(--line); display:flex; flex-direction:column; }}
.nav {{ display:flex; gap:6px; padding:8px; border-bottom:1px solid var(--line); align-items:center; }}
.nav button {{ border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:6px; padding:4px 10px; cursor:pointer; font-size:13px; }}
.nav .pos {{ margin-left:auto; color:var(--muted); font-size:12.5px; }}
.list {{ overflow:auto; flex:1; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th {{ position:sticky; top:0; background:var(--bg); text-align:left; color:var(--muted); font-weight:500; padding:5px 6px; border-bottom:1px solid var(--line); }}
td {{ padding:4px 6px; border-bottom:1px solid var(--line); white-space:nowrap; }}
tr.c {{ cursor:pointer; }}
tr.c:hover {{ background:var(--sel); }}
tr.on {{ background:var(--sel); font-weight:600; }}
.u {{ color:var(--up); }} .d {{ color:var(--down); }}
.n {{ color:var(--muted); font-size:11px; }}
.legend {{ position:absolute; left:10px; top:8px; z-index:3; font-size:12px; color:var(--muted); background:color-mix(in srgb, var(--bg) 85%, transparent); padding:3px 6px; border-radius:4px; pointer-events:none; }}
@media (max-width: 760px) {{ main {{ flex-direction:column; height:auto; }} #chart {{ height:60vh; }} aside {{ width:auto; border-left:0; border-top:1px solid var(--line); height:40vh; }} }}
</style></head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">С {since} по {last} · случаев <b>{len(cases)}</b> (рост <b class="u">{up}</b>, падение <b class="d">{len(cases) - up}</b>; через ночной разрыв {night}) ·
  ход за {W} мин ≥ {k:g} × ATR, ATR — средний размах 14 прошлых часов. Полоса — те {W} минут, стрелка — пик хода. Время — МСК.</div>
</header>
<main>
  <div id="chart"><div class="legend" id="legend"></div></div>
  <aside>
    <div class="nav"><button id="prev">← пред.</button><button id="next">след. →</button><button id="all">весь график</button><span class="pos" id="pos"></span></div>
    <div class="list"><table><thead><tr><th>№</th><th>пик, МСК</th><th></th><th>ATR</th><th>ход</th></tr></thead><tbody id="rows"></tbody></table></div>
  </aside>
</main>
<script>{lwc_js().replace("</script", "<\\/script")}</script>
<script>
const D = {data};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = document.getElementById('chart');
const chart = LightweightCharts.createChart(el, {{
  autoSize: true,
  layout: {{ background: {{ color: css('--bg') }}, textColor: css('--muted'), fontSize: 12 }},
  grid: {{ vertLines: {{ color: css('--line') }}, horzLines: {{ color: css('--line') }} }},
  rightPriceScale: {{ borderColor: css('--line') }},
  timeScale: {{ borderColor: css('--line'), timeVisible: true, secondsVisible: false }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  localization: {{ locale: 'ru-RU' }},
}});
// полосы случаев — гистограмма на своей шкале во всю высоту
const band = chart.addHistogramSeries({{ priceScaleId: 'band', priceLineVisible: false, lastValueVisible: false }});
chart.priceScale('band').applyOptions({{ scaleMargins: {{ top: 0, bottom: 0 }}, visible: false }});
const cs = chart.addCandlestickSeries({{ upColor: css('--up'), downColor: css('--down'), borderVisible: false,
  wickUpColor: css('--up'), wickDownColor: css('--down'), priceLineVisible: false }});
const times = D.candles.map(c => c[0]);
cs.setData(D.candles.map(c => ({{ time: c[0], open: c[1], high: c[2], low: c[3], close: c[4] }})));
const step = D.meta.step;
const floorT = t => t - (t % step);
const idxOf = t => {{ let lo = 0, hi = times.length - 1; while (lo < hi) {{ const m = (lo + hi + 1) >> 1; if (times[m] <= t) lo = m; else hi = m - 1; }} return lo; }};
const inBand = new Map();
D.cases.forEach((c, i) => {{
  const a = idxOf(floorT(c.base)), b = idxOf(floorT(c.peak_t));
  for (let j = a; j <= b; j++) inBand.set(times[j], c.dir);
}});
band.setData(times.map(t => ({{ time: t, value: inBand.has(t) ? 1 : 0, color: inBand.has(t) ? css('--band') : 'rgba(0,0,0,0)' }})));
cs.setMarkers(D.cases.map((c, i) => ({{
  time: times[idxOf(floorT(c.peak_t))],
  position: c.dir > 0 ? 'aboveBar' : 'belowBar',
  color: c.dir > 0 ? css('--up') : css('--down'),
  shape: c.dir > 0 ? 'arrowUp' : 'arrowDown',
  text: '№' + (i + 1) + ' ' + c.peak.toFixed(1),
}})));
const fmt = t => {{ const d = new Date(t * 1000); const p = n => String(n).padStart(2, '0');
  return p(d.getUTCDate()) + '.' + p(d.getUTCMonth() + 1) + '.' + String(d.getUTCFullYear()).slice(2) + ' ' + p(d.getUTCHours()) + ':' + p(d.getUTCMinutes()); }};
const rows = document.getElementById('rows');
rows.innerHTML = D.cases.map((c, i) => `<tr class="c" data-i="${{i}}"><td>${{i + 1}}</td><td>${{fmt(c.peak_t)}}${{c.cross ? ' <span class="n">ночь</span>' : ''}}</td>` +
  `<td class="${{c.dir > 0 ? 'u' : 'd'}}">${{c.dir > 0 ? '▲' : '▼'}}</td><td>${{c.peak.toFixed(1)}}</td>` +
  `<td class="${{c.dir > 0 ? 'u' : 'd'}}">${{c.move_pct > 0 ? '+' : ''}}${{c.move_pct.toFixed(2)}}%</td></tr>`).join('');
let cur = -1;
const legend = document.getElementById('legend');
function show(i) {{
  if (!D.cases.length) return;
  cur = (i + D.cases.length) % D.cases.length;
  const c = D.cases[cur];
  const a = idxOf(floorT(c.base)), b = idxOf(floorT(c.peak_t));
  chart.timeScale().setVisibleLogicalRange({{ from: a - D.meta.ctxBefore, to: b + D.meta.ctxAfter }});
  document.querySelectorAll('tr.on').forEach(r => r.classList.remove('on'));
  const r = rows.querySelector(`tr[data-i="${{cur}}"]`); r.classList.add('on'); r.scrollIntoView({{ block: 'nearest' }});
  document.getElementById('pos').textContent = (cur + 1) + ' из ' + D.cases.length;
  legend.innerHTML = `№${{cur + 1}}: ${{c.dir > 0 ? 'рост' : 'падение'}} ${{c.move_pct > 0 ? '+' : ''}}${{c.move_pct.toFixed(2)}}% = ${{c.peak.toFixed(1)}} ATR за ${{D.meta.w}} мин` +
    ` · начало ${{fmt(c.start)}}, пик ${{fmt(c.peak_t)}}${{c.cross ? ' · через ночной разрыв' : ''}}`;
}}
rows.addEventListener('click', e => {{ const tr = e.target.closest('tr.c'); if (tr) show(+tr.dataset.i); }});
document.getElementById('prev').onclick = () => show(cur - 1);
document.getElementById('next').onclick = () => show(cur + 1);
document.getElementById('all').onclick = () => {{ chart.timeScale().fitContent(); legend.textContent = ''; }};
document.addEventListener('keydown', e => {{ if (e.key === 'ArrowRight') show(cur + 1); if (e.key === 'ArrowLeft') show(cur - 1); }});
show(0);
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", default="LKOH")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--since", default="2025-08-01")
    a = ap.parse_args()
    since = pd.Timestamp(a.since)
    d = load(a.tk, since)
    atr = ev.atr_hourly(d[["high", "low", "close"]])
    eps = [e for e in ev.episodes(a.tk, d[["high", "low", "close"]], atr, "час", W)
           if e["peak"] >= a.k and e["start"] >= since]
    eps.sort(key=lambda e: e["peak_t"])
    c = d["close"]
    cases = []
    for e in eps:
        pos = c.index.searchsorted(e["peak_t"] - pd.Timedelta(minutes=W), side="right") - 1
        base = c.index[max(pos, 0)]                                   # минута, от которой считан ход до пика
        cases.append({"start": epoch(e["start"]), "peak_t": epoch(e["peak_t"]), "base": epoch(base),
                      "dir": int(e["dir"]), "peak": round(e["peak"], 2), "move_pct": round(e["move_pct"], 3),
                      "cross": bool(e["cross"])})
    d = d[d.index >= since]
    last = d.index.max()
    OUT.mkdir(parents=True, exist_ok=True)
    for tf, rule, before, after in (("часовые свечи", "1h", 60, 25), ("5-минутные свечи", "5min", 150, 60)):
        b = bars(d, rule)
        html = page(a.tk, tf, b, cases, a.k, f"{since:%d.%m.%Y}", f"{last:%d.%m.%Y}", rule, before, after)
        p = OUT / f"{a.tk}_{a.k:g}ATR_{'1h' if rule == '1h' else '5m'}.html"
        p.write_text(html, encoding="utf-8")
        print(f"{p.relative_to(ROOT)}: свечей {len(b)}, случаев {len(cases)}, {p.stat().st_size / 1e6:.1f} МБ")
    up = sum(x["dir"] == 1 for x in cases)
    print(f"{a.tk}: {len(cases)} случаев (рост {up}, падение {len(cases) - up}, через ночь {sum(x['cross'] for x in cases)}), "
          f"{since:%d.%m.%Y}–{last:%d.%m.%Y %H:%M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
