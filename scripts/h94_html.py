"""H94, просьба Тимура (исследовательский чат, 21.11, сообщения 409–418): все случаи «ход на 3 часовых ATR за 60 минут»
на одной бумаге на сплошном графике — два HTML: часовые свечи и 5-минутные, с августа 2025 по последнюю минуту.

Случаи — ровно как в H94 (scripts/h94_events.py): ход за 60 минут = закрытие минуты − последнее закрытие не позже чем
60 минут назад; ATR — средний истинный размах 14 прошлых закрытых часов; эпизод — подряд идущие минуты с ходом
≥ 1.5 ATR в одну сторону; в файл попадают эпизоды, у которых пик ≥ 3 ATR. Минутки — горячий файл контура
(data/raw/live/<бумага>_1m_live.csv), поэтому случаи — до последнего дня, а не до июня 2026, как в архиве H94.

На графике: полоса — 60 минут, за которые цена прошла пик хода; стрелка с номером — на свече пика. Справа — список
случаев: щелчок (или ← / →) — график переходит к случаю с запасом истории до него.

Коридор и вертикальный объём (просьба Тимура 23.11, сообщение 445): кнопка «Выделить отрезок» (или Shift + протянуть
мышью) — выделяешь произвольный отрезок по оси времени, и у правой шкалы рисуется профиль объёма по ценам только за
этот отрезок: шаг 1 / 2 / 5 / 10 ₽, уровень наибольшего объёма (POC) и зона 70% объёма. Объём — из минутных свечей:
объём минуты делится поровну по рублёвым уровням от её минимума до максимума (покупки/продажи — нет: секундные данные
с разбивкой кончаются 06.04.2026).

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


def minutes(d: pd.DataFrame) -> dict:
    """Минутки для профиля объёма, компактно: время (разность в минутах), нижний рублёвый уровень (разность), ширина в
    уровнях, объём. Объём минуты в браузере делится поровну по уровням [lo, lo + w]."""
    import numpy as np

    t = np.array([epoch(x) for x in d.index], dtype=np.int64) // 60
    lo = np.floor(d["low"].to_numpy()).astype(np.int64)
    hi = np.floor(d["high"].to_numpy()).astype(np.int64)
    return {"t0": int(t[0]), "dt": np.diff(t, prepend=t[0]).tolist(), "lo0": int(lo[0]),
            "dlo": np.diff(lo, prepend=lo[0]).tolist(), "w": (hi - lo).tolist(), "v": d["volume"].astype(np.int64).tolist()}


def page(tk: str, tf: str, b: pd.DataFrame, cases: list[dict], k: float, since: str, last: str, rule: str,
         ctx_before: int, ctx_after: int, vm: dict | None = None) -> str:
    candles = [[epoch(t), round(r.open, 4), round(r.high, 4), round(r.low, 4), round(r.close, 4)]
               for t, r in b.iterrows()]
    meta = {"tk": tk, "tf": tf, "k": k, "w": W, "ctxBefore": ctx_before, "ctxAfter": ctx_after,
            "step": int(pd.Timedelta(rule).total_seconds())}
    up = sum(c["dir"] == 1 for c in cases)
    night = sum(c["cross"] for c in cases)
    title = f"{tk}: ходы на {k:g} часовых ATR за {W} минут — {tf}"
    data = json.dumps({"candles": candles, "cases": cases, "meta": meta, "vm": vm}, ensure_ascii=False, separators=(",", ":"))
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
#ov {{ position:absolute; inset:0; z-index:2; pointer-events:none; }}
#chart.sel {{ cursor:col-resize; }}
.vp {{ padding:8px 10px; border-bottom:1px solid var(--line); font-size:12.5px; }}
.vp .row {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; }}
.vp button, .vp select {{ border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:6px; padding:3px 8px; cursor:pointer; font-size:12.5px; }}
.vp button.on {{ background:var(--sel); border-color:#2a6bd8; font-weight:600; }}
.vp .info {{ color:var(--muted); margin-top:6px; line-height:1.5; }}
.vp .info b {{ color:var(--ink); }}
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
  <div id="chart"><div class="legend" id="legend"></div><canvas id="ov"></canvas></div>
  <aside>
    <div class="vp">
      <div class="row"><button id="selbtn" title="или Shift + протянуть мышью">Выделить отрезок</button>
        <select id="vstep" title="шаг цены профиля"><option value="1">1 ₽</option><option value="2">2 ₽</option><option value="5">5 ₽</option><option value="10">10 ₽</option></select>
        <button id="selclr">Сбросить</button></div>
      <div class="info" id="vinfo">Вертикальный объём за выбранный отрезок: нажмите «Выделить отрезок» (или держите Shift) и протяните мышью по графику.</div>
    </div>
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
// ---- коридор по времени и вертикальный объём за него
const VM = D.vm;
let mt = null, mlo = null, mw = null, mv = null;
if (VM) {{
  const n = VM.v.length; mt = new Float64Array(n); mlo = new Int32Array(n); mw = VM.w; mv = VM.v;
  let t = VM.t0, l = VM.lo0;
  for (let i = 0; i < n; i++) {{ t += VM.dt[i]; l += VM.dlo[i]; mt[i] = t * 60; mlo[i] = l; }}
}}
const lb = (arr, x) => {{ let lo = 0, hi = arr.length; while (lo < hi) {{ const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }} return lo; }};
const ov = document.getElementById('ov'), ctx = ov.getContext('2d');
const selbtn = document.getElementById('selbtn'), vinfo = document.getElementById('vinfo'), vstep = document.getElementById('vstep');
let selMode = false, drag = null, sel = null, prof = null, lastKey = '';
const nf = x => Math.round(x).toLocaleString('ru-RU');
function profile(a, b) {{
  const step = +vstep.value, t1 = times[a], t2 = times[b] + D.meta.step;
  const i1 = lb(mt, t1), i2 = lb(mt, t2);
  const lv = new Map(); let tot = 0, rub = 0;
  for (let i = i1; i < i2; i++) {{
    const w = mw[i] + 1, per = mv[i] / w;
    for (let j = 0; j < w; j++) {{ const L = Math.floor((mlo[i] + j) / step) * step; lv.set(L, (lv.get(L) || 0) + per); rub += per * (mlo[i] + j + 0.5); }}
    tot += mv[i];
  }}
  const lev = [...lv.entries()].sort((x, y) => x[0] - y[0]);
  if (!lev.length) return null;
  let pi = 0; lev.forEach((e, i) => {{ if (e[1] > lev[pi][1]) pi = i; }});
  let lo = pi, hi = pi, acc = lev[pi][1];                     // зона 70%: расширяем от POC в сторону большего объёма
  while (acc < 0.7 * tot && (lo > 0 || hi < lev.length - 1)) {{
    const dn = lo > 0 ? lev[lo - 1][1] : -1, up = hi < lev.length - 1 ? lev[hi + 1][1] : -1;
    if (up >= dn) acc += lev[++hi][1]; else acc += lev[--lo][1];
  }}
  return {{ step, lev, tot, rub, poc: lev[pi][0], vaLo: lev[lo][0], vaHi: lev[hi][0] + step, pi, lo, hi, minutes: i2 - i1 }};
}}
function draw() {{
  const W = el.clientWidth, H = el.clientHeight, dpr = window.devicePixelRatio || 1;
  if (ov.width !== W * dpr || ov.height !== H * dpr) {{ ov.width = W * dpr; ov.height = H * dpr; ov.style.width = W + 'px'; ov.style.height = H + 'px'; }}
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const r = drag || sel; if (!r) return;
  const ts = chart.timeScale(), plotW = W - chart.priceScale('right').width();
  const tsH = ts.height ? ts.height() : 26;
  const bs = ts.options().barSpacing / 2;                     // logicalToCoordinate берёт только целые индексы
  const c1 = ts.logicalToCoordinate(Math.min(r.a, r.b)), c2 = ts.logicalToCoordinate(Math.max(r.a, r.b));
  if (c1 !== null && c2 !== null) {{
    const x1 = Math.max(0, c1 - bs), x2 = Math.min(plotW, c2 + bs);
    ctx.fillStyle = 'rgba(240,140,0,.10)'; ctx.fillRect(x1, 0, x2 - x1, H - tsH);
    ctx.strokeStyle = 'rgba(240,140,0,.9)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, H - tsH); ctx.moveTo(x2, 0); ctx.lineTo(x2, H - tsH); ctx.stroke(); ctx.setLineDash([]);
  }}
  if (!prof || drag) return;
  const maxV = prof.lev[prof.pi][1], maxW = plotW * 0.3;
  prof.lev.forEach((e, i) => {{
    const y1 = cs.priceToCoordinate(e[0]), y2 = cs.priceToCoordinate(e[0] + prof.step);
    if (y1 === null || y2 === null) return;
    const top = Math.min(y1, y2), h = Math.max(1, Math.abs(y1 - y2) - (Math.abs(y1 - y2) > 3 ? 1 : 0));
    const w = e[1] / maxV * maxW;
    ctx.fillStyle = i === prof.pi ? 'rgba(240,140,0,.85)' : (i >= prof.lo && i <= prof.hi ? 'rgba(42,107,216,.55)' : 'rgba(42,107,216,.25)');
    ctx.fillRect(plotW - w, top, w, h);
  }});
  ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = css('--ink'); ctx.textAlign = 'right';
  [[prof.vaHi, 'VAH'], [prof.poc + prof.step / 2, 'POC'], [prof.vaLo, 'VAL']].forEach(([p, t]) => {{
    const y = cs.priceToCoordinate(p); if (y === null) return;
    const txt = t + ' ' + p.toLocaleString('ru-RU'), tw = ctx.measureText(txt).width, x = plotW - maxW - 6;
    ctx.fillStyle = css('--bg'); ctx.globalAlpha = 0.85; ctx.fillRect(x - tw - 3, y - 8, tw + 6, 15); ctx.globalAlpha = 1;
    ctx.fillStyle = t === 'POC' ? '#f08c00' : css('--ink'); ctx.fillText(txt, x, y + 4);
    ctx.strokeStyle = t === 'POC' ? 'rgba(240,140,0,.7)' : 'rgba(42,107,216,.6)'; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x + 2, y); ctx.lineTo(plotW, y); ctx.stroke(); ctx.setLineDash([]);
  }});
}}
function info() {{
  if (!sel || !prof) {{ vinfo.innerHTML = sel ? 'За отрезок минуток нет.' : 'Вертикальный объём за выбранный отрезок: нажмите «Выделить отрезок» (или держите Shift) и протяните мышью по графику.'; return; }}
  const a = Math.min(sel.a, sel.b), b = Math.max(sel.a, sel.b);
  vinfo.innerHTML = `Отрезок <b>${{fmt(times[a])}} – ${{fmt(times[b] + D.meta.step)}}</b> (${{b - a + 1}} св., ${{nf(prof.minutes)}} мин.)<br>` +
    `Объём <b>${{nf(prof.tot)}}</b> шт. ≈ <b>${{(prof.rub / 1e9).toFixed(2)}}</b> млрд ₽<br>` +
    `<span style="color:#f08c00">■</span> POC (больше всего объёма) <b>${{prof.poc.toLocaleString('ru-RU')}}–${{(prof.poc + prof.step).toLocaleString('ru-RU')}}</b> ₽<br>` +
    `<span style="color:#2a6bd8">■</span> 70% объёма: <b>${{prof.vaLo.toLocaleString('ru-RU')}} – ${{prof.vaHi.toLocaleString('ru-RU')}}</b> ₽`;
}}
function recompute() {{ prof = sel && VM ? profile(Math.min(sel.a, sel.b), Math.max(sel.a, sel.b)) : null; info(); lastKey = ''; }}
const logicalAt = e => {{ const r = el.getBoundingClientRect(); const L = chart.timeScale().coordinateToLogical(e.clientX - r.left);
  return L === null ? null : Math.max(0, Math.min(times.length - 1, Math.round(L))); }};
const lock = on => chart.applyOptions({{ handleScroll: !on, handleScale: !on }});   // пока выделяем — график не двигается
const down = e => {{
  if (!(selMode || e.shiftKey) || e.button !== 0) return;
  e.stopPropagation(); e.preventDefault();
  if (drag) return;
  const L = logicalAt(e); if (L === null) return;
  lock(true); drag = {{ a: L, b: L }}; lastKey = '';
}};
el.addEventListener('pointerdown', down, true);
el.addEventListener('mousedown', down, true);
window.addEventListener('keydown', e => {{ if (e.key === 'Shift') lock(true); }});
window.addEventListener('keyup', e => {{ if (e.key === 'Shift' && !selMode && !drag) lock(false); }});
window.addEventListener('pointermove', e => {{ if (!drag) return; const L = logicalAt(e); if (L !== null) {{ drag.b = L; lastKey = ''; }} }}, true);
window.addEventListener('pointerup', e => {{
  if (!drag) return; sel = drag; drag = null; recompute();
  if (selMode) {{ selMode = false; selbtn.classList.remove('on'); el.classList.remove('sel'); }}
  if (!e.shiftKey) lock(false);
}}, true);
selbtn.onclick = () => {{ selMode = !selMode; selbtn.classList.toggle('on', selMode); el.classList.toggle('sel', selMode); lock(selMode); }};
document.getElementById('selclr').onclick = () => {{ sel = null; drag = null; recompute(); }};
vstep.onchange = recompute;
(function loop() {{                                            // перерисовка, когда двигают или масштабируют график
  const r = drag || sel;
  const key = r ? [el.clientWidth, el.clientHeight, chart.timeScale().logicalToCoordinate(r.a), chart.timeScale().logicalToCoordinate(r.b),
    prof ? cs.priceToCoordinate(prof.poc) : 0, prof ? cs.priceToCoordinate(prof.poc + 100) : 0].join() : 'none';
  if (key !== lastKey) {{ lastKey = key; draw(); }}
  requestAnimationFrame(loop);
}})();
if (!VM) {{ selbtn.disabled = true; vinfo.textContent = 'Минуток для профиля в файле нет.'; }}
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
    vm = minutes(d)
    OUT.mkdir(parents=True, exist_ok=True)
    for tf, rule, before, after in (("часовые свечи", "1h", 60, 25), ("5-минутные свечи", "5min", 150, 60)):
        b = bars(d, rule)
        html = page(a.tk, tf, b, cases, a.k, f"{since:%d.%m.%Y}", f"{last:%d.%m.%Y}", rule, before, after, vm)
        p = OUT / f"{a.tk}_{a.k:g}ATR_{'1h' if rule == '1h' else '5m'}.html"
        p.write_text(html, encoding="utf-8")
        print(f"{p.relative_to(ROOT)}: свечей {len(b)}, случаев {len(cases)}, {p.stat().st_size / 1e6:.1f} МБ")
    up = sum(x["dir"] == 1 for x in cases)
    print(f"{a.tk}: {len(cases)} случаев (рост {up}, падение {len(cases) - up}, через ночь {sum(x['cross'] for x in cases)}), "
          f"{since:%d.%m.%Y}–{last:%d.%m.%Y %H:%M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
