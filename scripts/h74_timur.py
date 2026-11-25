"""H74, продолжение: разметка фигур кодом Тимура (docs/timur/2026-11-15/chart_patterns.py, прислан 15.11) против
моей разметки (scripts/h74_patterns.py) на тех же данных и по тому же протоколу.

Данные — дневные свечи 19 акций из часового кеша (как в h74_patterns.py), 2015–2026. Код автора — как есть, настройки
по умолчанию (Config(): ZigZag 1.5 / 2.5 / 4 / 6 ATR, min_score 0.55, tol 1.0).

Что считается:
  1. сколько фигур каждого типа находит код автора, по годам и в половинах «выбор» (до 2021) / «проверка» (с 2021);
  2. совпадение с моей разметкой: та же бумага, тот же тип, пробой в пределах ±5 торговых дней;
  3. заглядывание вперёд: пробой раньше, чем подтверждены все экстремумы фигуры (known_at) — в торговле такой пробой
     увидеть нельзя; честный вход — max(пробой, known_at);
  4. ход цены после честного входа в ATR часа против обычного часа (как в моём H74), со знаком направления фигуры;
  5. исходы «цель / стоп» функцией автора evaluate (без комиссий).
    python scripts/h74_timur.py
    python scripts/h74_timur.py --page     # только страница lkoh_timur.html

Страница lkoh_timur.html (разметка автора на LKOH, последние 750 дней) + коридор и вертикальный объём (просьба Тимура
23.11, сообщение 445 в исследовательском чате): «Выделить отрезок» (или Shift + протянуть мышью) — справа на той же
шкале цены профиль объёма только за выбранные дни: шаг 1–50 ₽ (по умолчанию 10), POC и зона 70% объёма. Объём — минутки
основной сессии 10–18 ч (как дневные свечи страницы); объём минуты делится поровну по рублёвым уровням от её
минимума до максимума.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "docs" / "timur" / "2026-11-15"))
import chart_patterns as cp  # noqa: E402
import h74_patterns as mine  # noqa: E402

OUT = ROOT / "docs" / "results" / "h74_timur"
OUT.mkdir(parents=True, exist_ok=True)
REP = ROOT / "reports" / "h74"
SPLIT = mine.SPLIT
# мои фигуры → имена автора (у меня только бычьи версии; bump-and-run у автора нет)
MAP = {"db": "double_bottom", "tb": "triple_bottom", "ihs": "inv_head_shoulders", "htf": "high_tight_flag",
       "tri": "asc_triangle"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#888888"


def md(df: pd.DataFrame, index: bool = False) -> str:
    df = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                              (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    hours, rows = {}, []
    for t in mine.TICKERS:
        h = mine.hourly(t)
        hours[t] = h
        d = mine.to_daily(h)
        pats = cp.detect_patterns(d, cp.Config())
        stats = cp.evaluate(d, pats, cp.Config())  # заполняет outcome / ret у подтверждённых
        del stats
        idx = d.index
        for p in pats:
            bo = p.breakout_idx
            ka = p.known_at
            sig = None if bo is None else (max(bo, ka) if ka is not None else None)
            rows.append({"ticker": t, "pattern": p.name, "ru": p.name_ru, "dir": p.direction, "score": p.score,
                         "scale": p.scale, "start": idx[p.start_idx], "end": idx[p.end_idx],
                         "known_at": idx[ka] if ka is not None else pd.NaT, "status": p.status,
                         "breakout": idx[bo] if bo is not None else pd.NaT,
                         "peek": bo is not None and (ka is None or bo < ka),
                         "peek_bars": (ka - bo) if (bo is not None and ka is not None and bo < ka) else 0,
                         "signal": idx[sig] if sig is not None else pd.NaT,
                         "outcome": p.outcome, "ret": p.ret})
        print(t, len(pats), flush=True)
    a = pd.DataFrame(rows)
    a["year"] = a["start"].dt.year
    a["part"] = np.where(a["breakout"].fillna(a["end"]) < SPLIT, "выбор", "проверка")
    a.to_csv(OUT / "timur_patterns.csv", index=False)
    lines = ["## Разметка кодом автора", ""]

    # 1. сколько и каких
    cnt = a.pivot_table(index="ru", columns="part", values="ticker", aggfunc="size", fill_value=0)
    conf = a[a["status"] == "confirmed"].pivot_table(index="ru", columns="part", values="ticker", aggfunc="size",
                                                     fill_value=0).add_prefix("с пробоем, ")
    peek = a[a["status"] == "confirmed"].groupby("ru")["peek"].mean().mul(100).round(0).rename("пробой раньше known_at, %")
    t1 = cnt.join(conf).join(peek).fillna(0)
    t1.to_csv(OUT / "counts.csv")
    lines += [f"Всего фигур: {len(a)} на {a['ticker'].nunique()} бумагах; с пробоем — {(a['status'] == 'confirmed').sum()}.",
              "", md(t1, True), ""]

    # 2. совпадение с моей разметкой
    my = pd.read_csv(REP / "events.csv", parse_dates=["date"])
    my = my[my["pat"].isin(MAP)]
    cf = a[a["status"] == "confirmed"]
    mrows = []
    for code, name in MAP.items():
        m = my[my["pat"] == code]
        t_ = cf[cf["pattern"] == name]
        hit_m = 0
        for e in m.itertuples():
            tt = t_[t_["ticker"] == e.ticker]["breakout"]
            if len(tt) and (tt - e.date).abs().min() <= pd.Timedelta(days=7):
                hit_m += 1
        hit_t = 0
        for e in t_.itertuples():
            mm = m[m["ticker"] == e.ticker]["date"]
            if len(mm) and (mm - e.breakout).abs().min() <= pd.Timedelta(days=7):
                hit_t += 1
        mrows.append({"фигура": name, "моих": len(m), "у автора (с пробоем)": len(t_),
                      "моих нашёл автор, %": round(100 * hit_m / max(len(m), 1)),
                      "его нашёл я, %": round(100 * hit_t / max(len(t_), 1))})
    t2 = pd.DataFrame(mrows)
    t2.to_csv(OUT / "overlap.csv", index=False)
    lines += ["Совпадение с моей разметкой (та же бумага и тип, пробой ±7 календарных дней):", "", md(t2), ""]

    # 3–4. ход после честного входа
    ev = cf.dropna(subset=["signal"]).rename(columns={"pattern": "pat"}).copy()
    ev["date"] = ev["signal"]
    ev["part"] = np.where(ev["date"] < SPLIT, "выбор", "проверка")
    ev["year"] = ev["date"].dt.year
    fw = mine.forward(ev[["ticker", "pat", "date", "part", "year"]], hours)
    fw = fw.merge(ev[["ticker", "pat", "date", "dir"]], on=["ticker", "pat", "date"], how="left")
    for q in (6, 24, 72):
        fw[f"h{q}"] = fw[f"h{q}"] * np.where(fw["dir"] == 0, 1, fw["dir"])
    base = mine.baseline(hours).groupby(["part", "h"]).apply(lambda g: np.average(g["mean"], weights=g["n"]))
    fw["half"] = np.where(fw["date"] < SPLIT, "выбор 2015–2020",
                          np.where(fw["date"].dt.year <= 2023, "проверка 2021–2023", "проверка 2024–2026"))
    fw.to_csv(OUT / "forward.csv", index=False)
    rows = []
    for (pat, half), g in fw.groupby(["pat", "half"]):
        part = "выбор" if half.startswith("выбор") else "проверка"
        sgn = int(np.sign(g["dir"].iloc[0])) or 1
        r = {"фигура": cp.NAMES_RU.get(pat, pat), "период": half, "n": len(g)}
        for q in (24, 72):
            b = sgn * base[(part, q)]
            r[f"ход {q} ч, ATR"] = round(g[f"h{q}"].mean(), 2)
            r[f"обычный час {q} ч"] = round(b, 2)
            r[f"t {q} ч"] = round((g[f"h{q}"].mean() - b) / (g[f"h{q}"].std(ddof=1) / np.sqrt(len(g))), 1) \
                if len(g) > 2 else np.nan
        rows.append(r)
    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "forward_summary.csv", index=False)
    lines += ["Ход цены после честного входа (max(пробой, known_at), первый час следующего дня) в сторону фигуры, ATR "
              "часа, против обычного часа (для медвежьих фигур — со знаком минус):", "", md(t3), ""]

    # 5. цель / стоп автора
    ev2 = cf.dropna(subset=["outcome"])
    rows = []
    for (ru, part), g in ev2.groupby(["ru", "part"]):
        rows.append({"фигура": ru, "период": part, "сделок": len(g),
                     "цель, %": round(100 * (g["outcome"] == "target").mean()),
                     "стоп, %": round(100 * (g["outcome"] == "stop").mean()),
                     "средняя, %": round(100 * g["ret"].mean(), 2),
                     "с заглядыванием, %": round(100 * g["peek"].mean())})
    t4 = pd.DataFrame(rows)
    t4.to_csv(OUT / "evaluate.csv", index=False)
    lines += ["Исходы «цель или стоп» функцией автора (вход на закрытии пробоя, без комиссий):", "", md(t4), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    charts(a, t3)
    page(hours)


def charts(a: pd.DataFrame, t3: pd.DataFrame) -> None:
    # число фигур по годам у автора
    y = a[a["status"] == "confirmed"].groupby(["year", "ru"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    y.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_title("H74: фигуры с пробоем по годам — код Тимура, 19 акций, дневные свечи")
    ax.set_xlabel("")
    ax.set_ylabel("фигур")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "counts_years.png", dpi=120)
    plt.close(fig)
    # ход за 72 ч по фигурам и периодам
    t3 = t3[t3.groupby("фигура")["n"].transform("min") >= 5]  # фигуры, где случаев меньше 5, не рисуем
    halves = ["выбор 2015–2020", "проверка 2021–2023", "проверка 2024–2026"]
    figs = sorted(t3["фигура"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, h in zip(axes, halves):
        g = t3[t3["период"] == h].set_index("фигура").reindex(figs)
        x = np.arange(len(figs))
        ax.bar(x - 0.2, g["ход 72 ч, ATR"], 0.4, color=C1, label="после фигуры")
        ax.bar(x + 0.2, g["обычный час 72 ч"], 0.4, color=C3, label="обычный час")
        for i, n in enumerate(g["n"]):
            if np.isfinite(n):
                ax.text(i, 0, f"{int(n)}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(figs, rotation=60, ha="right", fontsize=8)
        ax.set_title(h, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("ход в сторону фигуры за 72 ч, ATR часа")
    axes[0].legend(fontsize=8)
    fig.suptitle("H74, код Тимура: ход цены после честного входа (пробой и все экстремумы подтверждены)")
    fig.tight_layout()
    fig.savefig(OUT / "forward72.png", dpi=120)
    plt.close(fig)


def day_levels(tk: str, days: pd.DatetimeIndex) -> dict[str, list[int]]:
    """Объём по рублёвым уровням за каждый день: {"2024-05-17": [нижний уровень, объём уровня, следующего, …]}.
    Минутки основной сессии (10–18 ч, как дневные свечи страницы); объём минуты — поровну по уровням [low, high]."""
    src = mine.SRC.get(tk, mine.RAW / "moex" / f"{tk}_1m.csv")
    first = days.min()
    parts = []
    for ch in pd.read_csv(src, usecols=["high", "low", "volume", "begin"], chunksize=500_000):
        ch["begin"] = pd.to_datetime(ch["begin"])
        ch = ch[(ch["begin"] >= first) & ch["begin"].dt.hour.between(10, 18) & (ch["volume"] > 0)]
        if len(ch):
            parts.append(ch)
    m = pd.concat(parts)
    lo = np.floor(m["low"].to_numpy()).astype(np.int64)
    w = np.floor(m["high"].to_numpy()).astype(np.int64) - lo + 1
    rep = np.repeat(np.arange(len(m)), w)
    lev = lo[rep] + (np.arange(len(rep)) - np.repeat(np.cumsum(w) - w, w))
    vol = (m["volume"].to_numpy() / w)[rep]
    day = m["begin"].dt.normalize().to_numpy()[rep]
    g = pd.DataFrame({"day": day, "lev": lev, "v": vol}).groupby(["day", "lev"])["v"].sum()
    out = {}
    for dday, s in g.groupby(level=0):
        lv = s.index.get_level_values(1)
        full = s.droplevel(0).reindex(range(lv.min(), lv.max() + 1), fill_value=0)
        out[pd.Timestamp(dday).strftime("%Y-%m-%d")] = [int(lv.min())] + np.rint(full.to_numpy()).astype(int).tolist()
    return out


VP_JS = r"""
(function () {
  const gd = document.getElementById('{plot_id}');
  const VL = __VL__;
  const bar = document.createElement('div');
  bar.style.cssText = 'font:13px system-ui,sans-serif;padding:8px 12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #e4e7ec';
  bar.innerHTML = '<button id="vpSel" title="или Shift + протянуть мышью">Выделить отрезок</button>' +
    '<label>шаг <select id="vpStep"><option>1</option><option>2</option><option>5</option><option selected>10</option><option>20</option><option>50</option></select> ₽</label>' +
    '<button id="vpClr">Сбросить</button><span id="vpInfo" style="color:#475467">Вертикальный объём за выбранный отрезок: «Выделить отрезок» (или Shift) и протяните мышью по графику.</span>';
  gd.parentNode.insertBefore(bar, gd);
  bar.querySelectorAll('button,select').forEach(b => b.style.cssText = 'border:1px solid #d0d5dd;background:#fff;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:13px');
  const selBtn = document.getElementById('vpSel'), info = document.getElementById('vpInfo'), stepEl = document.getElementById('vpStep');
  const days = Object.keys(VL).sort();
  const nf = x => Math.round(x).toLocaleString('ru-RU');
  let mode = false, drag = null, sel = null, prevDrag = null;
  const xa = () => gd._fullLayout.xaxis;
  const dayAt = e => {
    const r = gd.getBoundingClientRect(), a = xa();
    const px = e.clientX - r.left - a._offset; if (px < 0 || px > a._length) return null;
    const ms = a.p2c(px); if (ms === undefined || isNaN(ms)) return null;
    return new Date(ms).toISOString().slice(0, 10);
  };
  const setMode = on => { mode = on; selBtn.style.background = on ? '#eef4ff' : '#fff'; selBtn.style.fontWeight = on ? '600' : '400';
    if (on) { prevDrag = gd._fullLayout.dragmode; Plotly.relayout(gd, { dragmode: false }); }
    else if (prevDrag !== null) { Plotly.relayout(gd, { dragmode: prevDrag }); prevDrag = null; } };
  const shapes = () => (gd.layout.shapes || []).filter(s => s.name !== 'vpCorr');
  function corridor(r) {
    const a = r.a < r.b ? r.a : r.b, b = r.a < r.b ? r.b : r.a;
    Plotly.relayout(gd, { shapes: shapes().concat([{ name: 'vpCorr', type: 'rect', xref: 'x', yref: 'paper', x0: a + ' 00:00', x1: b + ' 23:59',
      y0: 0, y1: 1, fillcolor: 'rgba(240,140,0,.12)', line: { color: 'rgba(240,140,0,.9)', width: 1, dash: 'dot' }, layer: 'below' }]) });
  }
  function profile() {
    const step = +stepEl.value, i = gd.data.findIndex(t => t.name === 'vp');
    if (!sel) { Plotly.restyle(gd, { x: [[]], y: [[]] }, [i]); Plotly.relayout(gd, { shapes: shapes() });
      info.textContent = 'Вертикальный объём за выбранный отрезок: «Выделить отрезок» (или Shift) и протяните мышью по графику.'; return; }
    const a = sel.a < sel.b ? sel.a : sel.b, b = sel.a < sel.b ? sel.b : sel.a;
    const lv = new Map(); let tot = 0, rub = 0, nd = 0;
    days.forEach(d => { if (d < a || d > b) return; nd++; const r = VL[d];
      for (let j = 1; j < r.length; j++) { const L = r[0] + j - 1, K = Math.floor(L / step) * step; lv.set(K, (lv.get(K) || 0) + r[j]); tot += r[j]; rub += r[j] * (L + 0.5); } });
    const lev = [...lv.entries()].sort((x, y) => x[0] - y[0]);
    if (!lev.length) { info.textContent = 'За отрезок минуток нет.'; return; }
    let pi = 0; lev.forEach((e, k) => { if (e[1] > lev[pi][1]) pi = k; });
    let lo = pi, hi = pi, acc = lev[pi][1];
    while (acc < 0.7 * tot && (lo > 0 || hi < lev.length - 1)) {
      const dn = lo > 0 ? lev[lo - 1][1] : -1, up = hi < lev.length - 1 ? lev[hi + 1][1] : -1;
      if (up >= dn) acc += lev[++hi][1]; else acc += lev[--lo][1];
    }
    const colors = lev.map((e, k) => k === pi ? 'rgba(240,140,0,.9)' : (k >= lo && k <= hi ? 'rgba(42,107,216,.6)' : 'rgba(42,107,216,.28)'));
    Plotly.restyle(gd, { x: [lev.map(e => e[1])], y: [lev.map(e => e[0] + step / 2)], width: [step * 0.92], 'marker.color': [colors] }, [i]);
    Plotly.relayout(gd, { 'xaxis2.range': [lev[pi][1] / 0.3, 0] });
    const f = d => d.split('-').reverse().join('.');
    info.innerHTML = `Отрезок <b>${f(a)} – ${f(b)}</b> (${nd} дн.) · объём <b>${nf(tot)}</b> шт. ≈ <b>${(rub / 1e9).toFixed(1)}</b> млрд ₽ · ` +
      `<span style="color:#f08c00">■</span> POC <b>${lev[pi][0].toLocaleString('ru-RU')}–${(lev[pi][0] + step).toLocaleString('ru-RU')}</b> ₽ · ` +
      `<span style="color:#2a6bd8">■</span> 70% объёма <b>${lev[lo][0].toLocaleString('ru-RU')} – ${(lev[hi][0] + step).toLocaleString('ru-RU')}</b> ₽`;
  }
  const down = e => {
    if (!(mode || e.shiftKey) || e.button !== 0) return;
    const d = dayAt(e); if (!d) return;
    e.stopPropagation(); e.preventDefault(); drag = { a: d, b: d }; corridor(drag);
  };
  gd.addEventListener('pointerdown', down, true); gd.addEventListener('mousedown', down, true);
  window.addEventListener('pointermove', e => { if (!drag) return; const d = dayAt(e); if (d && d !== drag.b) { drag.b = d; corridor(drag); } }, true);
  window.addEventListener('pointerup', () => { if (!drag) return; sel = drag; drag = null; profile(); if (mode) setMode(false); }, true);
  selBtn.onclick = () => setMode(!mode);
  document.getElementById('vpClr').onclick = () => { sel = null; profile(); };
  stepEl.onchange = profile;
})();
"""


def page(hours: dict) -> None:
    """Интерактивная разметка автора на LKOH (последние 3 года) — для глаза; плюс коридор и вертикальный объём."""
    import json

    import plotly.graph_objects as go

    d = mine.to_daily(hours["LKOH"])
    pats = cp.detect_patterns(d, cp.Config())
    cp.evaluate(d, pats, cp.Config())
    out = OUT / "lkoh_timur.html"
    fig = cp.plot_patterns(d, pats, str(out), last=750, no_gaps=True, plotlyjs=True,
                           title="LKOH, дневные свечи — разметка фигур кодом Тимура (наведите на фигуру — оценки условий)")
    vl = day_levels("LKOH", d.index[-750:])
    fig.add_trace(go.Bar(name="vp", x=[], y=[], orientation="h", xaxis="x2", yaxis="y", showlegend=False,
                         marker=dict(line=dict(width=0)), hovertemplate="%{y:.0f} ₽: %{x:,.0f} шт.<extra>объём</extra>"))
    fig.update_layout(xaxis2=dict(overlaying="x", side="top", range=[1, 0], showgrid=False, showticklabels=False,
                                  zeroline=False, fixedrange=True, visible=False))
    fig.write_html(str(out), include_plotlyjs=True, div_id="chart",
                   post_script=VP_JS.replace("__VL__", json.dumps(vl, separators=(",", ":"))))
    print(f"{out.relative_to(ROOT)}: {len(vl)} дней с объёмом, {out.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    if "--page" in sys.argv:
        page({"LKOH": mine.hourly("LKOH")})
    else:
        main()
