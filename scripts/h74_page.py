"""Страница для оценки разметки фигур на глаз (H74, просьба Тагира: «лучше создать html с размеченными фигурами»).

Самодостаточный файл: дневные свечи вокруг каждой найденной фигуры и её разметка рисуются на canvas, библиотек и
интернета не нужно. Оценки («фигура / не фигура / не знаю») копятся в браузере и выгружаются одним файлом, чтобы
Тимур и Тагир могли прислать их обратно.

    .venv/bin/py scripts/h74_page.py        # docs/results/h74_patterns/patterns.html
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import h74_patterns as H

REP, OUT = H.REP, H.OUT
PER_PAT = 40          # сколько примеров каждой фигуры класть на страницу (случайно, но одинаково при каждом сборе)
PAD_L, PAD_R = 25, 35  # дней слева от начала фигуры и справа от пробоя


def sample(evs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(74)
    parts = []
    for pat, g in evs.groupby("pat"):
        for _, gg in g.groupby("part"):                      # поровну из выбора и проверки
            k = min(PER_PAT // 2, len(gg))
            parts.append(gg.iloc[rng.choice(len(gg), k, replace=False)])
    return pd.concat(parts).sort_values(["pat", "date"]).reset_index(drop=True)


def payload(evs: pd.DataFrame, bars: dict[str, H.Bars], meta: dict) -> dict:
    items = []
    for n, e in enumerate(evs.itertuples()):
        b = bars[e.ticker]
        a, z = max(0, e.i0 - PAD_L), min(b.n, e.brk + PAD_R)
        items.append({
            "id": f"{e.ticker}-{e.pat}-{e.date:%Y%m%d}",
            "pat": e.pat, "ticker": e.ticker, "part": e.part, "note": e.note,
            "date": f"{e.date:%d.%m.%Y}", "tier": meta["chosen"][e.pat]["tier"],
            "a": int(a), "brk": int(e.brk),
            "d": [f"{pd.Timestamp(b.dt[i]):%d.%m.%y}" for i in range(a, z)],
            "o": [round(float(b.o[i]), 4) for i in range(a, z)],
            "h": [round(float(b.h[i]), 4) for i in range(a, z)],
            "l": [round(float(b.l[i]), 4) for i in range(a, z)],
            "c": [round(float(b.c[i]), 4) for i in range(a, z)],
            "pts": e.pts, "lines": e.lines,
        })
        if n % 50 == 0:
            print(f"  {n} / {len(evs)}", flush=True)
    return {"pats": H.PATS, "items": items, "meta": {k: meta[k] for k in ("tickers", "split", "min_events")},
            "chosen": {p: {"tier": c["tier"], "n_sel": c["n_sel"], "n_all": c["n_all"]} for p, c in meta["chosen"].items()}}


PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка фигур теханализа (H74)</title>
<style>
:root {
  --paper:#f6f4ef; --panel:#fffefb; --ink:#15233b; --ink2:#4a5568; --ink3:#7a8391; --rule:#e2ddd2;
  --up:#2a78d6; --down:#eb6834; --ok:#2f8f5b; --no:#d1433d;
  --f:"IBM Plex Sans","Segoe UI",system-ui,-apple-system,sans-serif;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font-family:var(--f); font-size:15px; line-height:1.5; }
header { padding:22px 20px 10px; max-width:1180px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 6px; }
p { margin:6px 0; color:var(--ink2); max-width:80ch; }
main { max-width:1180px; margin:0 auto; padding:0 20px 60px; }
.bar { position:sticky; top:0; z-index:5; background:var(--paper); border-bottom:1px solid var(--rule);
       padding:10px 0; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
select, button { font:inherit; color:var(--ink); background:var(--panel); border:1px solid var(--rule);
                 border-radius:7px; padding:6px 10px; cursor:pointer; }
button:hover, select:hover { border-color:var(--ink3); }
button.ok { border-color:var(--ok); color:var(--ok); }
button.no { border-color:var(--no); color:var(--no); }
.count { color:var(--ink3); margin-left:auto; font-size:14px; }
.card { background:var(--panel); border:1px solid var(--rule); border-radius:12px; padding:14px; margin:16px 0; }
.head { display:flex; gap:12px; flex-wrap:wrap; align-items:baseline; margin-bottom:8px; }
.head b { font-size:16px; }
.tag { font-size:13px; color:var(--ink3); }
canvas { width:100%; height:320px; display:block; }
.votes { display:flex; gap:8px; margin-top:10px; align-items:center; flex-wrap:wrap; }
.votes .mark { font-size:13px; color:var(--ink3); }
footer { max-width:1180px; margin:0 auto; padding:0 20px 50px; color:var(--ink3); font-size:13px; }
</style>
</head>
<body>
<header>
<h1>Разметка фигур теханализа — на оценку (гипотеза H74)</h1>
<p>Каждая картинка — дневные свечи одной бумаги и фигура, которую нашёл алгоритм: кружки — опорные точки,
пунктир — уровень или линия, по которой считается пробой, синяя вертикаль — день пробоя (закрытие выше уровня).
Нужно просто сказать по каждой картинке: <b>это фигура</b> или <b>не фигура</b>. Критерий гипотезы — не меньше 80 %
подтверждений.</p>
<p>Оценки сохраняются прямо в браузере, можно закрыть и вернуться. Когда закончите — нажмите
«скачать оценки» и пришлите файл в группу.</p>
</header>
<main>
<div class="bar">
  <select id="pat"></select>
  <select id="part"><option value="">выбор и проверка</option><option>выбор</option><option>проверка</option></select>
  <select id="todo"><option value="">все</option><option value="1">только неоценённые</option></select>
  <button id="dl">скачать оценки</button>
  <span class="count" id="cnt"></span>
</div>
<div id="list"></div>
</main>
<footer id="foot"></footer>
<script>
const DATA = /*DATA*/;
const KEY = "h74-votes";
const votes = JSON.parse(localStorage.getItem(KEY) || "{}");
const up = "#2a78d6", down = "#eb6834", ink = "#15233b", grid = "#ebe7de";

function draw(cv, it) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const x = cv.getContext("2d"); x.scale(dpr, dpr); x.clearRect(0, 0, w, h);
  const n = it.c.length, L = 46, R = 8, T = 10, B = 22;
  let lo = Math.min(...it.l), hi = Math.max(...it.h);
  for (const p of it.pts) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
  for (const l of it.lines) { lo = Math.min(lo, l[1], l[3]); hi = Math.max(hi, l[1], l[3]); }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const px = i => L + (i + 0.5) * (w - L - R) / n;
  const py = v => T + (hi - v) * (h - T - B) / (hi - lo);
  x.strokeStyle = grid; x.fillStyle = "#7a8391"; x.font = "11px system-ui"; x.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Math.round(py(v)) + 0.5;
    x.beginPath(); x.moveTo(L, y); x.lineTo(w - R, y); x.stroke();
    x.fillText(v.toFixed(v > 500 ? 0 : 2), 4, y + 4);
  }
  const step = Math.max(1, Math.round(n / 8));
  for (let i = 0; i < n; i += step) x.fillText(it.d[i], px(i) - 16, h - 6);
  const bw = Math.max(1.2, (w - L - R) / n * 0.7);
  for (let i = 0; i < n; i++) {
    const c = it.c[i] >= it.o[i] ? up : down;
    x.strokeStyle = c; x.fillStyle = c;
    x.beginPath(); x.moveTo(px(i), py(it.h[i])); x.lineTo(px(i), py(it.l[i])); x.stroke();
    const y0 = py(Math.max(it.o[i], it.c[i])), y1 = py(Math.min(it.o[i], it.c[i]));
    x.fillRect(px(i) - bw / 2, y0, bw, Math.max(1, y1 - y0));
  }
  x.strokeStyle = ink; x.lineWidth = 1.2; x.setLineDash([5, 4]);
  for (const l of it.lines) {
    x.beginPath(); x.moveTo(px(l[0] - it.a), py(l[1])); x.lineTo(px(l[2] - it.a), py(l[3])); x.stroke();
  }
  x.setLineDash([]);
  for (const p of it.pts) {
    x.beginPath(); x.arc(px(p[0] - it.a), py(p[1]), 4.5, 0, 7); x.stroke();
  }
  x.strokeStyle = up; x.lineWidth = 1.5;
  x.beginPath(); x.moveTo(px(it.brk - it.a), T); x.lineTo(px(it.brk - it.a), h - B); x.stroke();
  x.fillStyle = up; x.font = "12px system-ui"; x.fillText("пробой", px(it.brk - it.a) + 4, T + 12);
}

function card(it) {
  const d = document.createElement("div"); d.className = "card";
  const ch = DATA.chosen[it.pat];
  d.innerHTML = `<div class="head"><b>${DATA.pats[it.pat]}</b>
    <span class="tag">${it.ticker} · пробой ${it.date} · ${it.part}${it.note ? " · " + it.note : ""}</span>
    <span class="tag">пороги: ${ch.tier}</span></div><canvas></canvas>
    <div class="votes"><button data-v="1">это фигура</button><button data-v="0">не фигура</button>
    <button data-v="?">не знаю</button><span class="mark"></span></div>`;
  const cv = d.querySelector("canvas");
  requestAnimationFrame(() => draw(cv, it));
  const mark = d.querySelector(".mark");
  const show = () => {
    const v = votes[it.id];
    mark.textContent = v === undefined ? "" : (v === "1" ? "отмечено: фигура" : v === "0" ? "отмечено: не фигура" : "отмечено: не знаю");
    d.querySelectorAll("button").forEach(b => {
      b.className = votes[it.id] === b.dataset.v ? (b.dataset.v === "1" ? "ok" : b.dataset.v === "0" ? "no" : "") : "";
    });
  };
  d.querySelectorAll(".votes button").forEach(b => b.onclick = () => {
    votes[it.id] = b.dataset.v;
    localStorage.setItem(KEY, JSON.stringify(votes));
    show(); counter();
  });
  show();
  return d;
}

function render() {
  const pat = document.getElementById("pat").value;
  const part = document.getElementById("part").value;
  const todo = document.getElementById("todo").value;
  const list = document.getElementById("list"); list.innerHTML = "";
  const items = DATA.items.filter(i => (!pat || i.pat === pat) && (!part || i.part === part)
                                        && (!todo || votes[i.id] === undefined));
  items.forEach(i => list.appendChild(card(i)));
  counter(items.length);
}

function counter(shown) {
  const done = DATA.items.filter(i => votes[i.id] !== undefined).length;
  const yes = DATA.items.filter(i => votes[i.id] === "1").length;
  const no = DATA.items.filter(i => votes[i.id] === "0").length;
  const share = (yes + no) ? Math.round(yes * 100 / (yes + no)) : 0;
  document.getElementById("cnt").textContent =
    `оценено ${done} из ${DATA.items.length}` + ((yes + no) ? ` · «фигура» ${share}% (${yes} из ${yes + no})` : "");
  if (shown !== undefined) document.getElementById("foot").textContent =
    `Показано ${shown} картинок. Всего на странице ${DATA.items.length} примеров; бумаг ${DATA.meta.tickers.length}, ` +
    `дневные свечи 2015–2026, граница выбора и проверки — ${DATA.meta.split}.`;
}

const sel = document.getElementById("pat");
sel.innerHTML = '<option value="">все фигуры</option>' +
  Object.entries(DATA.pats).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
["pat", "part", "todo"].forEach(id => document.getElementById(id).onchange = render);
document.getElementById("dl").onclick = () => {
  const rows = [["id", "фигура", "бумага", "дата пробоя", "часть", "оценка"]].concat(
    DATA.items.filter(i => votes[i.id] !== undefined)
      .map(i => [i.id, DATA.pats[i.pat], i.ticker, i.date, i.part,
                 votes[i.id] === "1" ? "фигура" : votes[i.id] === "0" ? "не фигура" : "не знаю"]));
  const csv = "\\ufeff" + rows.map(r => r.join(";")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  a.download = "h74-ocenki.csv"; a.click();
};
let rt = 0;
window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(render, 250); });
render();
</script>
</body>
</html>
"""


def main() -> None:
    evs = pd.read_pickle(REP / "events.pkl")
    meta = json.loads((REP / "meta.json").read_text(encoding="utf-8"))
    bars = {t: H.Bars(t, H.to_daily(H.hourly(t))) for t in meta["tickers"]}
    pick = sample(evs)
    print(f"на страницу: {len(pick)} примеров", flush=True)
    data = payload(pick, bars, meta)
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "patterns.html"
    p.write_text(PAGE.replace("/*DATA*/", js, 1), encoding="utf-8")
    print(f"{p} — {p.stat().st_size / 1e6:.1f} МБ", flush=True)


if __name__ == "__main__":
    main()
