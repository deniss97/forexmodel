"""H94, вторая часть плана: самые сильные случаи без известной причины (список дат автору «глянуть глазами»)
и доля покупок/продаж по ленте обезличенных сделок во время резкого хода (с 2022 года, как в плане).

План H94, пункт «объём»: «С 2022 года — ещё доля покупок и продаж по ленте». Лента — архив обезличенных сделок на
Яндекс Диске (~/yadisk/обезличенные сделки/<год>/<дата>.csv.gz, 185 МБ в день, вся биржа). Читаем её потоком и только
за те дни, где случились самые сильные «взрывы» (дневной ATR): по одному проходу на день сразу по всем нужным бумагам.
Ответ на вопрос 5 плана: по случаям без календарного события — список дат, чтобы автор посмотрел, что тогда было.

    python scripts/h94_tape.py --limit 16      # сколько сильнейших случаев с 2022 года брать для ленты
    python scripts/h94_tape.py --no-tape       # только список дат (без чтения ленты)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "h94"
TAPE = Path.home() / "yadisk" / "обезличенные сделки"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"
# как бумага называется в ленте (TICKER_CC): акции — <код>_TQBR; фьючерсы — контракты серии (GDM4_SPBFUT и т. п.),
# у золота серия GD, у серебра SV (GL* и GLDRUBF — другие инструменты, мини и вечный, их не берём)
EXACT = {"SBER": "SBER_TQBR", "GAZP": "GAZP_TQBR", "LKOH": "LKOH_TQBR", "ROSN": "ROSN_TQBR", "MOEX": "MOEX_TQBR"}
PREFIX = {"GOLD": "GD", "SILV": "SV"}
# оборот = цена × количество: у акций это рубли, у фьючерсов металлов цена в долларах за унцию — поэтому только для
# сравнения с тем же днём той же бумаги (столбец «доля дневного оборота»)
VOL = "оборот в ходе, млн (цена × объём)"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def strongest(cases: pd.DataFrame) -> pd.DataFrame:
    """Случаи «взрыва» по дневному ATR (окно 60 минут, не открытие), от сильного к слабому, в дневных ATR."""
    d = cases[(cases["kind"] == "день") & (cases["w"] == 60) & (~cases["opening"].astype(bool))].copy()
    # move_pct у обвала отрицательный: сила хода — его размер в дневных ATR, всегда положительная
    d["atr"] = d["move_pct"] * d["dir"] / d["atr_d_pct"]
    d["day"] = d["t"].dt.date
    # один случай = бумага, сторона и день; из нескольких порогов оставляем строку с посчитанными признаками
    d["has"] = d["vol_ratio"].notna()
    d = d.sort_values(["has", "atr"], ascending=False).drop_duplicates(subset=["tk", "dir", "day"])
    return d.sort_values("atr", ascending=False)


def tape_day(day: pd.Timestamp, tickers: list[str]) -> pd.DataFrame:
    """Лента за день по нужным бумагам → объём всех сделок и покупок (в рублях) по минутам московского времени."""
    f = TAPE / f"{day:%Y}" / f"{day:%Y-%m-%d}.csv.gz"
    if not f.exists():
        return pd.DataFrame()
    want = sorted(EXACT[t] for t in tickers if t in EXACT)
    pref = sorted(PREFIX[t] for t in tickers if t in PREFIX)
    if not want and not pref:
        return pd.DataFrame()
    # фильтр лентой: awk отбрасывает 99% строк (вся биржа) до того, как они попадут в python
    cond = " || ".join((["($2 in W)"] if want else [])
                       + [f'(index($2, "{p}") == 1 && index($2, "_SPBFUT") > 0)' for p in pref])
    awk = ('BEGIN{FS=","; split("' + " ".join(want) + '", a, " "); for (i in a) W[a[i]] = 1}'
           f'NR > 1 && ({cond}) {{print $1 "," $2 "," $3 "," $4 "," $5}}')
    p = subprocess.Popen(f"zcat {f.as_posix()!r} | awk {awk!r}", shell=True, stdout=subprocess.PIPE, text=True)
    rows = []
    for ch in pd.read_csv(p.stdout, names=["ts", "cc", "dir", "price", "qty"], chunksize=500_000):
        ch = ch[ch["qty"] > 0]
        # контракты серии складываем вместе (у нас склейка по контрактам): GDM4_SPBFUT → GOLD, SVZ4_SPBFUT → SILV
        ch["tk"] = ch["cc"].str.split("_").str[0]
        for tk, pr in PREFIX.items():
            ch.loc[ch["cc"].str.startswith(pr) & ch["cc"].str.endswith("_SPBFUT"), "tk"] = tk
        m = pd.to_datetime(ch["ts"], format="ISO8601", utc=True).dt.tz_convert(None) + pd.Timedelta(hours=3)
        ch["m"] = m.dt.floor("min")
        ch["val"] = ch["price"] * ch["qty"]
        ch["buy"] = ch["val"].where(ch["dir"] == "BUY", 0.0)
        rows.append(ch.groupby(["tk", "m"])[["val", "buy"]].sum())
    p.wait()
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).groupby(level=[0, 1]).sum().reset_index()


def share(tape: pd.DataFrame, tk: str, a: pd.Timestamp, b: pd.Timestamp) -> tuple[float, float]:
    """Доля покупок по обороту и сам оборот (млн ₽) за период [a, b) по бумаге."""
    d = tape[(tape["tk"] == tk) & (tape["m"] >= a) & (tape["m"] < b)]
    v = d["val"].sum()
    return (d["buy"].sum() / v if v > 0 else float("nan"), v / 1e6)


def pct(x: float) -> float | None:
    return round(x * 100, 1) if x == x else None


def md_table(t: pd.DataFrame) -> list[str]:
    """Табличка в markdown без лишней зависимости (tabulate в окружении нет)."""
    cols = list(t.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in t.iterrows():
        rows.append("| " + " | ".join("—" if pd.isna(r[c]) else f"{r[c]}" for c in cols) + " |")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=16, help="сильнейших случаев с 2022 года для ленты")
    ap.add_argument("--top", type=int, default=30, help="сколько случаев без известной причины в список")
    ap.add_argument("--no-tape", action="store_true")
    ap.add_argument("--from-csv", action="store_true", help="не читать ленту заново: таблица и график из tape_share.csv")
    a = ap.parse_args(argv)
    cases = pd.read_parquet(ROOT / "data" / "derived" / "h94_cases.parquet")
    d = strongest(cases)
    unknown = d[(d["news"].isna()) | (d["news"] == "нет")]
    # половину списка — на взлёты, половину на обвалы: иначе весь список займёт отскок 24.02.2022
    half = max(1, a.top // 2)
    top = pd.concat([unknown[unknown["dir"] > 0].head(a.top - half),
                     unknown[unknown["dir"] < 0].head(half)]).sort_values("atr", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    if not a.no_tape and not a.from_csv:
        pick = top[top["t"] >= "2022-01-01"].head(a.limit)
        for day, grp in pick.groupby(pick["t"].dt.normalize()):
            tape = tape_day(day, sorted(grp["tk"].unique()))
            if tape.empty:
                print(f"{day:%F}: ленты нет", flush=True)
                continue
            # «за день» — все сделки этого дня по бумаге (у акций 10:00–18:50, у фьючерсов с вечерней сессией)
            d0, d1 = day - pd.Timedelta(days=1), day + pd.Timedelta(days=2)
            for _, c in grp.iterrows():
                t = c["t"]
                win, vol = share(tape, c["tk"], t - pd.Timedelta(minutes=60), t)
                bef, _ = share(tape, c["tk"], t - pd.Timedelta(minutes=120), t - pd.Timedelta(minutes=60))
                aft, _ = share(tape, c["tk"], t, t + pd.Timedelta(minutes=60))
                day_sh, day_vol = share(tape, c["tk"], d0, d1)
                rows.append({"дата": f"{t:%F %H:%M}", "бумага": c["tk"], "ход": "вверх" if c["dir"] > 0 else "вниз",
                             "ход %": round(c["move_pct"], 2), "в дневных ATR": round(c["atr"], 2),
                             "покупок в ходе %": pct(win), "за час до %": pct(bef), "час после %": pct(aft),
                             "за день %": pct(day_sh), VOL: round(vol, 1),
                             "доля дневного оборота %": round(vol / day_vol * 100, 1) if day_vol > 0 else None,
                             "продолжение до конца дня %": round(c["eod"] * c["dir"], 2)})
                print(f"{t:%F %H:%M} {c['tk']}: покупок в ходе {pct(win)}%, за день {pct(day_sh)}%, "
                      f"оборот {vol:.0f} млн ₽", flush=True)
    tape_df = (pd.read_csv(OUT / "tape_share.csv").rename(columns={"оборот в ходе, млн": VOL})
               if a.from_csv else pd.DataFrame(rows))
    if not tape_df.empty:
        tape_df.to_csv(OUT / "tape_share.csv", index=False)
        chart(tape_df)

    lines = ["# H94: самые сильные случаи без календарного события", "",
             "Взрывы по дневному ATR (окно 60 минут, открытие сессии исключено), от сильного к слабому; «причина "
             "неизвестна» — на эту минуту нет решения ЦБ, отчёта EIA и дивидендной отсечки. Автору — посмотреть "
             "глазами, что было в эти дни: архива новостной ленты у нас нет (вопрос 5 плана).", "",
             "Столбцы «продолжение» — ход после случая в сторону случая: плюс — цена пошла дальше, минус — откат.", "",
             "| дата и время (МСК) | бумага | ход | ход % | в дневных ATR | объём к обычному | пробой дня | "
             "продолжение через час % | до конца дня % | след. день % |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, c in top.iterrows():
        vr = f"{c['vol_ratio']:.1f}×" if pd.notna(c["vol_ratio"]) else "—"
        brk = "—" if c["break_pd"] is None else ("да" if c["break_pd"] in (True, "да") else "нет")
        lines.append(f"| {c['t']:%F %H:%M} | {c['tk']} | {'вверх' if c['dir'] > 0 else 'вниз'} | "
                     f"{c['move_pct']:+.2f} | {c['atr']:.2f} | {vr} | "
                     f"{brk} | {c['h1'] * c['dir']:+.2f} | "
                     f"{c['eod'] * c['dir']:+.2f} | {c['nd'] * c['dir']:+.2f} |")
    if not tape_df.empty:
        lines += ["", "## Лента обезличенных сделок: покупки и продажи во время хода", "",
                  "Доля покупок в обороте: 50% — покупок столько же, сколько продаж. «За день» — та же бумага за весь "
                  "день, для сравнения.", ""] + md_table(tape_df)
    (OUT / "top_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"готово: {OUT / 'top_cases.md'}, случаев в списке {len(top)}, по ленте {len(tape_df)}")
    return 0


def chart(t: pd.DataFrame) -> None:
    up, dn = t[t["ход"] == "вверх"], t[t["ход"] == "вниз"]
    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.42 * len(t) + 3.2)), facecolor=SURF)
    style(ax)
    y = list(range(len(t)))
    ax.barh(y, t["покупок в ходе %"], color=[C1 if s == "вверх" else C2 for s in t["ход"]], height=0.62,
            label="доля покупок во время хода (синий — взлёт, оранжевый — обвал)")
    ax.plot(t["за день %"], y, "d", color=INK2, ms=6, label="та же бумага за весь день")
    ax.plot(t["час после %"], y, "o", color=INK, ms=5, mfc="none", label="та же бумага в следующий час после хода")
    ax.axvline(50, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['дата'][:16]} {r['бумага']} {r['ход %']:+.1f}%" for _, r in t.iterrows()])
    ax.invert_yaxis()
    ax.set_xlim(25, 70)
    ax.set_xlabel("доля покупок в обороте, %")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), facecolor=SURF, edgecolor=GRID, fontsize=9)
    med_u = up["покупок в ходе %"].median() if len(up) else float("nan")
    med_d = dn["покупок в ходе %"].median() if len(dn) else float("nan")
    med_a = dn["час после %"].median() if len(dn) else float("nan")
    med_ua = up["час после %"].median() if len(up) else float("nan")
    ax.set_title(f"Во время обвала в ленте продают (покупок {med_d:.0f}%), в следующий час — покупают ({med_a:.0f}%)\n"
                 f"при взлёте {med_u:.0f}% и {med_ua:.0f}%; обезличенные сделки, сильнейшие случаи с 2022 года\n"
                 f"черта 50% — покупок и продаж поровну", color=INK, fontsize=11.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "tape_buy_share.png", dpi=130, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
