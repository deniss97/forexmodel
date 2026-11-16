"""H40 (план 14.11): сделки наших стратегий, открытые под большим свежим уровнем объёма, — хуже остальных?

Проверка не «уровень как сигнал» (это было 08.11, docs/results/h40_profile.md), а разбор сделок журнала: цена упирается
в уровень, поэтому сделка, открытая прямо под ним, должна идти хуже.

Сделки (журнал экспериментов, издержки биржи, дивиденды учтены; цены переведены в масштаб сырых свечей):
  * «S42 импульс» — S42 (SBER, GAZP, LKOH) и S51 (16 акций индекса);
  * «S46 пробой 55 дней» — S46 и S53; «S48 пробой 20 дней» — S48 и S55;
  * «модель LKOH» — L10 (CatBoost на 3 класса, seed 42), сделки в обе стороны (по просьбе Дениса, ответ в плане).
Уровни — профиль объёма по ценам с шагом 0.1% (уровень = floor(log(цена) / log(1.001))):
  * `--source min` — из минуток (объём минуты в уровень её средней цены), вся история 2016–2026;
  * `--source tape` — из ленты обезличенных сделок (scripts/h40_levels.py), с 2022 по мере выгрузки, с разбивкой на
    покупки и продажи.
Профиль считается только по прошлым N торговым дням и часам до часа входа (без заглядывания вперёд). «Большой уровень» —
локальный максимум сглаженного профиля, где объём в X раз выше среднего по соседним ±W уровням и не ниже 25% от самого
объёмного уровня. «Свежесть» — по правилу Дениса (ответ в плане): уровень живёт, пока цена не вернулась к нему с
расстояния k·ATR; после возврата объём на уровне начинает копиться заново (`--reset 1 2 3`, 0 — весь период окна).

Выход: docs/results/h40_trades.md, графики docs/results/h40_trades/*.png, таблицы reports/h40_trades/*.csv.

    python scripts/h40_trades.py --source min
    python scripts/h40_trades.py --source tape --start 2024-01-01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments import raw_prices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOURLY = ROOT / "data" / "derived" / "hourly"
LVL_MIN = ROOT / "data" / "derived" / "h40_levels_min"
LVL_TAPE = ROOT / "data" / "derived" / "h40_levels"
EXP = ROOT / "docs" / "experiments"
REP = ROOT / "reports" / "h40_trades"
DOC = ROOT / "docs" / "results" / "h40_trades.md"
VIZ = ROOT / "docs" / "results" / "h40_trades"
STEP = np.log(1.001)
MARK = ("<!-- вывод -->", "<!-- /вывод -->")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"

NEW16 = ["tatn", "nvtk", "gmkn", "rosn", "vtbr", "plzl", "sngsp", "irao", "chmf", "nlmk", "magn", "rtkm", "trnfp",
         "aflt", "alrs", "afks"]
BIG3 = ["sber", "gazp", "lkoh"]
# стратегия -> (эксперимент на 3 акциях, эксперимент на 16 акциях)
STRATS = {"S42 импульс": ("S42", "S51"), "S46 пробой 55 дней": ("S46", "S53"), "S48 пробой 20 дней": ("S48", "S55")}
MODEL = ("L10", "lkoh_h63_multiclass_s42")


# ---------------------------------------------------------------- данные

def bars(tk: str) -> pd.DataFrame:
    """Часовые свечи биржи (сырые цены) + ATR(14 ч)."""
    d = pd.read_parquet(HOURLY / f"{tk}.parquet").sort_index()
    c = d["close"]
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(1)
    d["atr"] = tr.rolling(14, min_periods=7).mean()
    d["date"] = d.index.normalize()
    return d


def levels_min(tk: str) -> pd.DataFrame:
    return pd.read_parquet(LVL_MIN / f"{tk}.parquet")


def levels_lkoh(side: str = "both") -> dict[str, pd.DataFrame]:
    """Профиль LKOH из секундных агрегатов ленты (scripts/h40_levels_lkoh.py), 01.2024 — 04.2026."""
    d = pd.read_parquet(ROOT / "data" / "derived" / "h40_levels_lkoh.parquet")
    d["vol"] = {"both": d["buy"] + d["sell"], "buy": d["buy"], "sell": d["sell"]}[side]
    return {"LKOH": d[["hour", "b", "vol", "n"]]}


def levels_tape(side: str = "both") -> dict[str, pd.DataFrame]:
    """Профиль из ленты по всем дням, что уже выгружены: {бумага: (hour, b, vol)}."""
    files = sorted(LVL_TAPE.glob("20*.parquet"))
    if not files:
        raise FileNotFoundError("нет выгрузки ленты — сначала scripts/h40_levels.py")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["vol"] = {"both": d["buy"] + d["sell"], "buy": d["buy"], "sell": d["sell"]}[side]
    d["n"] = d["nbuy"] + d["nsell"]
    return {tk: g[["hour", "b", "vol", "n"]].reset_index(drop=True) for tk, g in d.groupby("tk")}


def trades() -> pd.DataFrame:
    """Сделки журнала: стратегии на акциях + сделки модели по LKOH, цены — в масштабе сырых свечей."""
    parts = []
    for name, (e3, e16) in STRATS.items():
        for exp, keep in ((e3, BIG3), (e16, NEW16)):
            t = pd.read_csv(EXP / exp / "trades.csv.gz", parse_dates=["open_dt", "close_dt"])
            t = t[t["instrument"].isin(keep)].copy()
            t["strategy"], t["exp"] = name, exp
            parts.append(t)
    t = pd.read_csv(EXP / MODEL[0] / "trades.csv.gz", parse_dates=["open_dt", "close_dt"])
    t = t[(t["instrument"] == "lkoh") & (t["run"] == MODEL[1])].copy()
    t["strategy"], t["exp"] = "модель LKOH", MODEL[0]
    parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    d = raw_prices(d)
    d["tk"] = d["instrument"].str.upper()
    d["side"] = d["side"].astype(int)
    d["year"] = d["open_dt"].dt.year
    return d.sort_values("open_dt").reset_index(drop=True)


# ---------------------------------------------------------------- профиль и уровни

class Store:
    """Профиль объёма одной бумаги: быстрый доступ к строкам (час, уровень, объём) в диапазоне часов."""

    def __init__(self, d: pd.DataFrame, hours: pd.DatetimeIndex):
        pos = pd.Series(np.arange(len(hours)), index=hours)
        d = d[d["hour"].isin(pos.index)]
        self.t = pos.loc[d["hour"]].to_numpy()
        self.b = d["b"].to_numpy(np.int64)
        self.v = d["vol"].to_numpy(np.float64)
        order = np.argsort(self.t, kind="stable")
        self.t, self.b, self.v = self.t[order], self.b[order], self.v[order]
        self.start = np.searchsorted(self.t, np.arange(len(hours) + 1))

    def slice(self, i0: int, i1: int):
        a, z = self.start[i0], self.start[i1]
        return self.t[a:z], self.b[a:z], self.v[a:z]


def big_levels(prof: np.ndarray, x: float, w: int, hvn: float) -> np.ndarray:
    """Локальные максимумы сглаженного профиля, которые в x раз выше среднего по соседним ±w уровням."""
    s = np.convolve(prof, np.ones(3) / 3, mode="same")
    if s.max() <= 0:
        return np.zeros(len(s), bool)
    k = np.ones(2 * w + 1)
    near = np.convolve(s, k, mode="same") / np.convolve(np.ones(len(s)), k, mode="same")
    peak = np.r_[False, (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]), False]
    return peak & (s >= x * np.maximum(near, 1e-9)) & (s >= hvn * s.max())


def profile_for(st: Store, d: pd.DataFrame, i_entry: int, i_from: int, b_lo: int, b_hi: int,
                reset_k: float, atr: float, price_of) -> np.ndarray:
    """Профиль уровней [b_lo, b_hi] по часам [i_from, i_entry). reset_k > 0 — объём только после возврата цены."""
    nb = b_hi - b_lo + 1
    t, b, v = st.slice(i_from, i_entry)
    keep = (b >= b_lo) & (b <= b_hi)
    t, b, v = t[keep] - i_from, b[keep] - b_lo, v[keep]
    nt = i_entry - i_from
    if reset_k <= 0:
        prof = np.zeros(nb)
        np.add.at(prof, b, v)
        return prof
    M = np.zeros((nt, nb))
    np.add.at(M, (t, b), v)
    lo = d["low"].to_numpy()[i_from:i_entry]
    hi = d["high"].to_numpy()[i_from:i_entry]
    cl = d["close"].to_numpy()[i_from:i_entry]
    bl = np.floor(np.log(lo) / STEP).astype(np.int64) - b_lo
    bh = np.floor(np.log(hi) / STEP).astype(np.int64) - b_lo
    pb = price_of(np.arange(b_lo, b_hi + 1))
    ib = np.arange(nb)
    far_all = np.abs(cl[:, None] - pb[None, :]) >= reset_k * atr
    last = np.zeros(nb, np.int64)
    seen = np.zeros(nb, bool)
    for i in range(nt):
        touch = (ib >= bl[i]) & (ib <= bh[i])
        back = touch & seen
        last[back] = i
        seen[back] = False
        seen |= far_all[i]
    mask = np.arange(nt)[:, None] >= last[None, :]
    return (M * mask).sum(0)


def features(tr: pd.DataFrame, lv: dict[str, pd.DataFrame], days: int, x: float, w: int, hvn: float,
             reset_k: float, span: int = 60) -> pd.DataFrame:
    """На каждую сделку: расстояние до ближайшего большого уровня сверху и снизу (в ATR) и «застрял у уровня»."""
    out = []
    for tk, g in tr.groupby("tk"):
        if tk not in lv:
            continue
        d = bars(tk)
        st = Store(lv[tk], d.index)
        idx = d.index
        dates = np.unique(d["date"].to_numpy())
        price_of = lambda bb: np.exp((bb + 0.5) * STEP)  # noqa: E731
        hh, ll, atr = d["high"].to_numpy(), d["low"].to_numpy(), d["atr"].to_numpy()
        for r in g.itertuples():
            # час входа: профиль считается по часам строго до него (в самом часе входа объём ещё не известен)
            i_e = int(idx.searchsorted(pd.Timestamp(r.open_dt).floor("h"), "left"))
            if i_e < 2 or i_e >= len(idx):
                continue
            a = atr[i_e - 1]
            if not np.isfinite(a) or a <= 0:
                continue
            k = dates.searchsorted(np.datetime64(r.open_dt.normalize()))
            if k < days:
                continue
            i_f = int(idx.searchsorted(dates[k - days], "left"))
            if i_e - i_f < 20:
                continue
            p = float(r.open_price)
            b_e = int(np.floor(np.log(p) / STEP))
            prof = profile_for(st, d, i_e, i_f, b_e - span, b_e + span, reset_k, a, price_of)
            big = big_levels(prof, x, w, hvn)
            bb = np.arange(b_e - span, b_e + span + 1)
            pb = price_of(bb)
            up = np.flatnonzero(big & (pb > p * 1.0005))
            dn = np.flatnonzero(big & (pb < p * 0.9995))
            d_up = (pb[up[0]] - p) / a if len(up) else np.nan
            d_dn = (p - pb[dn[-1]]) / a if len(dn) else np.nan
            i_x = int(idx.searchsorted(r.close_dt, "right"))
            seg_h = hh[i_e:max(i_x, i_e + 1)]
            seg_l = ll[i_e:max(i_x, i_e + 1)]
            mfe_p = float(seg_h.max()) if len(seg_h) else p
            mae_p = float(seg_l.min()) if len(seg_l) else p
            stall_up = abs(mfe_p - pb[up[0]]) <= 0.2 * a if len(up) else np.nan
            stall_dn = abs(mae_p - pb[dn[-1]]) <= 0.2 * a if len(dn) else np.nan
            out.append({"tk": tk, "open_dt": r.open_dt, "close_dt": r.close_dt, "strategy": r.strategy,
                        "side": r.side, "profit_pct": r.profit_pct, "year": r.year, "price": p, "atr": a,
                        "d_up": d_up, "d_dn": d_dn, "stall_up": stall_up, "stall_dn": stall_dn,
                        "lvl_up": float(pb[up[0]]) if len(up) else np.nan,
                        "lvl_dn": float(pb[dn[-1]]) if len(dn) else np.nan,
                        "mfe_atr": (mfe_p - p) / a, "mae_atr": (p - mae_p) / a, "nbig": int(big.sum())})
    f = pd.DataFrame(out)
    if f.empty:
        return f
    # препятствие по ходу сделки и опора за спиной: для лонга — сверху и снизу, для шорта — наоборот
    long = f["side"] > 0
    f["d_block"] = np.where(long, f["d_up"], f["d_dn"])
    f["d_supp"] = np.where(long, f["d_dn"], f["d_up"])
    f["stall"] = np.where(long, f["stall_up"], f["stall_dn"])
    f["half"] = np.where(f["year"] <= 2024, "2022–2024", "2025–2026")
    return f


# ---------------------------------------------------------------- метрики

def pf(x: np.ndarray) -> float:
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return float(pos / neg) if neg > 0 else np.nan


def stats(x: pd.Series) -> dict:
    v = x.to_numpy(float)
    if len(v) == 0:
        return {"сделок": 0, "PF": np.nan, "выигрышей": np.nan, "средняя": np.nan, "итог": np.nan, "без_2_лучших": np.nan}
    srt = np.sort(v)
    return {"сделок": len(v), "PF": pf(v), "выигрышей": 100 * float((v > 0).mean()), "средняя": float(v.mean()),
            "итог": float(v.sum()), "без_2_лучших": float(srt[:-2].sum()) if len(v) > 2 else np.nan}


def groups(f: pd.DataFrame, col: str = "d_block", thr: float = 1.0) -> pd.DataFrame:
    """Сделки «уровень ближе thr ATR» против остальных по стратегиям и половинам истории."""
    rows = []
    f = f.copy()
    f["grp"] = np.where(f[col] <= thr, f"уровень ближе {thr:g} ATR", "остальные")
    for keys, g in f.groupby(["strategy", "half", "grp"], dropna=False):
        rows.append({"стратегия": keys[0], "период": keys[1], "группа": keys[2], **stats(g["profit_pct"])})
    for keys, g in f.groupby(["strategy", "grp"], dropna=False):
        rows.append({"стратегия": keys[0], "период": "2022–2026", "группа": keys[1], **stats(g["profit_pct"])})
    for keys, g in f.groupby(["half", "grp"], dropna=False):
        rows.append({"стратегия": "все", "период": keys[0], "группа": keys[1], **stats(g["profit_pct"])})
    for keys, g in f.groupby("grp", dropna=False):
        rows.append({"стратегия": "все", "период": "2022–2026", "группа": keys, **stats(g["profit_pct"])})
    return pd.DataFrame(rows)


def criterion(gr: pd.DataFrame, thr: float = 1.0) -> tuple[bool, str, pd.DataFrame]:
    """План: разница PF между группами одного знака в обеих половинах и хотя бы у 2 стратегий из 3; в меньшей группе ≥ 30."""
    near = f"уровень ближе {thr:g} ATR"
    rows = []
    for s in STRATS:
        r = {"стратегия": s}
        ok_n = True
        for h in ("2022–2024", "2025–2026"):
            sel = gr[(gr["стратегия"] == s) & (gr["период"] == h)]
            a = sel[sel["группа"] == near]
            b = sel[sel["группа"] == "остальные"]
            pa = float(a["PF"].iloc[0]) if len(a) else np.nan
            pb = float(b["PF"].iloc[0]) if len(b) else np.nan
            na = int(a["сделок"].iloc[0]) if len(a) else 0
            r[f"PF_рядом_{h}"], r[f"PF_ост_{h}"], r[f"n_рядом_{h}"] = pa, pb, na
            r[f"разница_{h}"] = pa - pb
            ok_n &= na >= 30
        d1, d2 = r["разница_2022–2024"], r["разница_2025–2026"]
        r["знак_один"] = bool(np.isfinite(d1) and np.isfinite(d2) and d1 * d2 > 0)
        r["сделок_хватает"] = ok_n
        r["прошла"] = r["знак_один"] and ok_n
        rows.append(r)
    t = pd.DataFrame(rows)
    n_ok = int(t["прошла"].sum())
    ok = n_ok >= 2
    return ok, f"стратегий с устойчивой разницей {n_ok} из 3 (нужно ≥ 2)", t


# ---------------------------------------------------------------- графики

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


SHORT = {"S42 импульс": "S42\nимпульс", "S46 пробой 55 дней": "S46\nпробой 55", "S48 пробой 20 дней": "S48\nпробой 20",
         "модель LKOH": "модель\nLKOH", "все": "все\nвместе"}
TOP = 2.2


def chart_groups(gr: pd.DataFrame, thr: float, title: str, path: Path) -> Path:
    near = f"уровень ближе {thr:g} ATR"
    order = [*STRATS, "модель LKOH", "все"]
    halves = ["2022–2024", "2025–2026"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURF)
    for ax, h in zip(axes, halves):
        xs = np.arange(len(order))
        for j, (grp, c) in enumerate(((near, C2), ("остальные", C1))):
            vals, ns = [], []
            for s in order:
                sel = gr[(gr["стратегия"] == s) & (gr["период"] == h) & (gr["группа"] == grp)]
                vals.append(float(sel["PF"].iloc[0]) if len(sel) else np.nan)
                ns.append(int(sel["сделок"].iloc[0]) if len(sel) else 0)
            ax.bar(xs + (j - 0.5) * 0.4, np.minimum(vals, TOP), width=0.38, color=c, label=grp)
            for xi, v, n in zip(xs + (j - 0.5) * 0.4, vals, ns):
                if np.isfinite(v):
                    ax.text(xi, min(v, TOP) + 0.03, f"{v:.2f}\n{n}", ha="center", va="bottom", color=INK2, fontsize=7.5)
        style(ax)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_ylim(0, TOP + 0.35)
        ax.set_xticks(xs, [SHORT[s] for s in order], fontsize=8.5)
        ax.set_title(h, color=INK, fontsize=10, loc="center")
    axes[0].set_ylabel("PF (прибыль / убыток); под столбиком — число сделок", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


def chart_deciles(f: pd.DataFrame, title: str, path: Path) -> Path:
    """«Признак»: 10 корзин расстояния до уровня против результата сделки, по периодам."""
    d = f[np.isfinite(f["d_block"])].copy()
    periods = [("2022–2024", 2022, 2024), ("2025–2026", 2025, 2026)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, facecolor=SURF)
    for ax, (lab, a, b) in zip(axes, periods):
        g = d[(d["year"] >= a) & (d["year"] <= b)].copy()
        if len(g) < 50:
            style(ax)
            continue
        g["q"] = pd.qcut(g["d_block"].rank(method="first"), 10, labels=False) + 1
        s = g.groupby("q")["profit_pct"].agg(["mean", "std", "count"])
        se = s["std"] / np.sqrt(s["count"])
        ax.bar(s.index, s["mean"], width=0.8, color=C1)
        ax.errorbar(s.index, s["mean"], yerr=se, fmt="none", ecolor=INK2, lw=1)
        ax.axhline(0, color=INK2, lw=0.8)
        style(ax)
        ax.set_xticks([1, 5, 10], ["1\nуровень рядом", "5", "10\nуровень далеко"], fontsize=8.5)
        ax.set_title(f"{lab}, сделок {len(g)}", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("результат сделки, %", color=INK2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


def chart_years(f: pd.DataFrame, thr: float, title: str, path: Path) -> Path:
    near = f"уровень ближе {thr:g} ATR"
    d = f.copy()
    d["grp"] = np.where(d["d_block"] <= thr, near, "остальные")
    years = sorted(d["year"].unique())
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor=SURF)
    xs = np.arange(len(years))
    for j, (grp, c) in enumerate(((near, C2), ("остальные", C1))):
        vals = [d[(d["year"] == y) & (d["grp"] == grp)]["profit_pct"].mean() for y in years]
        ax.bar(xs + (j - 0.5) * 0.4, vals, width=0.38, color=c, label=grp)
    style(ax)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xs, [str(y) for y in years])
    ax.set_ylabel("средний результат сделки, %", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


def candles(ax, d: pd.DataFrame, i0: int, i1: int):
    o, h, l, c = (d[k].to_numpy()[i0:i1] for k in ("open", "high", "low", "close"))
    x = np.arange(i1 - i0)
    up = c >= o
    ax.vlines(x, l, h, color=INK2, lw=0.6)
    ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.7, color=C1, lw=0)
    ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=0.7, color=C2, lw=0)


def panel(axc, axp, d: pd.DataFrame, i_e: int, i_x: int, prof: np.ndarray, b_lo: int, big: np.ndarray,
          p: float, atr: float, title: str, back: int = 80):
    i0 = max(0, i_e - back)
    i1 = min(len(d), max(i_x + 5, i_e + 10))
    pb = np.exp((np.arange(b_lo, b_lo + len(prof)) + 0.5) * STEP)
    candles(axc, d, i0, i1)
    for lp in pb[big]:
        axc.axhline(lp, color=C2, lw=0.9, alpha=0.65)
    axc.axhline(p, color=INK, lw=0.8, ls=(0, (4, 3)))
    axc.plot([i_e - i0], [p], marker="^" if title.startswith("лонг") else "v", color=INK, ms=6)
    axc.plot([min(i_x, i1 - 1) - i0], [d["close"].to_numpy()[min(i_x, i1 - 1)]], marker="x", color=INK, ms=6)
    style(axc)
    axc.grid(False)
    axc.set_xticks([])
    axc.set_title(title, color=INK, fontsize=8, loc="left")
    lo, hi = d["low"].to_numpy()[i0:i1].min(), d["high"].to_numpy()[i0:i1].max()
    pad = 0.05 * (hi - lo)
    axc.set_ylim(lo - pad, hi + pad)
    sel = (pb >= lo - pad) & (pb <= hi + pad)
    axp.barh(pb[sel], prof[sel], height=pb[sel] * STEP, color=np.where(big[sel], C2, "#c9d6ea"), lw=0)
    axp.set_ylim(*axc.get_ylim())
    style(axp)
    axp.grid(False)
    axp.set_xticks([])
    axp.set_yticks([])
    axp.tick_params(labelleft=False)


def one_profile(d: pd.DataFrame, st: Store, r, days: int, x: float, w: int, hvn: float, reset: float, span: int = 60):
    """Профиль и большие уровни для одной сделки — как в features(), но с сохранением массивов для графика."""
    idx = d.index
    dates = np.unique(d["date"].to_numpy())
    i_e = int(idx.searchsorted(pd.Timestamp(r.open_dt).floor("h"), "left"))
    k = dates.searchsorted(np.datetime64(pd.Timestamp(r.open_dt).normalize()))
    if i_e < 2 or i_e >= len(idx) or k < days:
        return None
    i_f = int(idx.searchsorted(dates[k - days], "left"))
    a = d["atr"].to_numpy()[i_e - 1]
    if i_e - i_f < 20 or not np.isfinite(a) or a <= 0:
        return None
    p = float(r.price if hasattr(r, "price") else r.open_price)
    b_e = int(np.floor(np.log(p) / STEP))
    prof = profile_for(st, d, i_e, i_f, b_e - span, b_e + span, reset, a, lambda bb: np.exp((bb + 0.5) * STEP))
    return i_e, int(idx.searchsorted(r.close_dt, "right")), prof, b_e - span, big_levels(prof, x, w, hvn), p, a


def entry_sheets(f: pd.DataFrame, lv: dict, days: int, x: float, w: int, hvn: float, reset: float,
                 tickers: list[str], years: tuple[int, int], out: Path, tag: str, per: int = 12) -> list[Path]:
    """Все ситуации входов (замечание Дениса к плану): свечи + гистограмма объёма по уровням на момент входа."""
    paths = []
    for tk in tickers:
        g = f[(f["tk"] == tk) & (f["year"] >= years[0]) & (f["year"] <= years[1])].sort_values("open_dt")
        if g.empty or tk not in lv:
            continue
        d = bars(tk)
        st = Store(lv[tk], d.index)
        items = []
        for r in g.itertuples():
            res = one_profile(d, st, r, days, x, w, hvn, reset)
            if res:
                items.append((r, res))
        for s in range(0, len(items), per):
            chunk = items[s:s + per]
            rows = int(np.ceil(len(chunk) / 3))
            fig = plt.figure(figsize=(12, 2.6 * rows), facecolor=SURF)
            gs = fig.add_gridspec(rows, 6, width_ratios=[3, 1, 3, 1, 3, 1], hspace=0.45, wspace=0.12)
            for j, (r, res) in enumerate(chunk):
                i_e, i_x, prof, b_lo, big, p, a = res
                axc = fig.add_subplot(gs[j // 3, 2 * (j % 3)])
                axp = fig.add_subplot(gs[j // 3, 2 * (j % 3) + 1])
                dist = r.d_block
                lab = ("лонг" if r.side > 0 else "шорт")
                title = (f"{lab} {pd.Timestamp(r.open_dt):%d.%m.%y} {r.strategy.split()[0]} · "
                         f"{r.profit_pct:+.1f}% · уровень {'—' if not np.isfinite(dist) else f'{dist:.1f} ATR'}")
                panel(axc, axp, d, i_e, i_x, prof, b_lo, big, p, a, title)
            fig.suptitle(f"{tk}: входы {years[0]}–{years[1]} и объём по уровням на момент входа "
                         f"(оранжевые линии — большие уровни, пунктир — цена входа)", color=INK, fontsize=11, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            p_out = out / f"entries_{tk}_{tag}_{s // per + 1}.png"
            fig.savefig(p_out, dpi=120, facecolor=SURF)
            plt.close(fig)
            paths.append(p_out)
    return paths


def examples_chart(f: pd.DataFrame, lv: dict, days: int, x: float, w: int, hvn: float, reset: float,
                   thr: float, path: Path) -> Path | None:
    """6 сделок у уровня: 3 «упёрлись» (максимум встал у уровня) и 3 «прошли»."""
    near = f[(f["d_block"] <= thr) & f["stall"].notna()].sort_values("open_dt", ascending=False)
    pick = pd.concat([near[near["stall"] == bool(s)].drop_duplicates("tk").head(3) for s in (1, 0)])
    if pick.empty:
        return None
    fig = plt.figure(figsize=(12, 2.6 * 2), facecolor=SURF)
    gs = fig.add_gridspec(2, 6, width_ratios=[3, 1, 3, 1, 3, 1], hspace=0.45, wspace=0.12)
    j = 0
    for tk, g in pick.groupby("tk"):
        d = bars(tk)
        st = Store(lv[tk], d.index)
        for r in g.itertuples():
            res = one_profile(d, st, r, days, x, w, hvn, reset)
            if not res or j >= 6:
                continue
            i_e, i_x, prof, b_lo, big, p, a = res
            axc = fig.add_subplot(gs[j // 3, 2 * (j % 3)])
            axp = fig.add_subplot(gs[j // 3, 2 * (j % 3) + 1])
            mark = "упёрлась в уровень" if r.stall else "прошла уровень"
            panel(axc, axp, d, i_e, i_x, prof, b_lo, big, p, a,
                  f"{'лонг' if r.side > 0 else 'шорт'} {tk} {pd.Timestamp(r.open_dt):%d.%m.%y} · {mark} · "
                  f"итог {r.profit_pct:+.1f}%")
            j += 1
    fig.suptitle("Сделки, открытые под большим уровнем: что было дальше (справа — объём по уровням на момент входа)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["min", "tape", "lkoh"], default="min")
    ap.add_argument("--side", choices=["both", "buy", "sell"], default="both")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--x", type=float, default=2.0)
    ap.add_argument("--w", type=int, default=20)
    ap.add_argument("--hvn", type=float, default=0.0)
    ap.add_argument("--reset", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=1.0)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--grid", action="store_true", help="сетка устойчивости: X, окно, правило свежести, порог группы")
    ap.add_argument("--charts", action="store_true")
    ap.add_argument("--sheets", action="store_true", help="листы «все ситуации входов» (замечание Дениса к плану)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    REP.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.source}_{a.side}_d{a.days}_x{a.x:g}_r{a.reset:g}"

    tr = trades()
    if a.source == "tape":                       # окно профиля должно целиком попадать в выгруженные дни ленты
        got = pd.to_datetime(sorted(f.name[:-8] for f in LVL_TAPE.glob("20*.parquet")))
        brk = np.flatnonzero(np.diff(got.to_numpy()) > np.timedelta64(7, "D"))   # куски выгрузки без дыр
        bounds = np.r_[0, brk + 1, len(got)]
        i = int(np.argmax(np.diff(bounds)))                                      # самый длинный кусок
        first, last = got[bounds[i]], got[bounds[i + 1] - 1]
        lo = (first + pd.Timedelta(days=int(a.days * 1.6) + 5)).strftime("%Y-%m-%d")
        a.start, a.end = max(a.start, lo), min(a.end, last.strftime("%Y-%m-%d"))
        print(f"лента: дней {len(got)}, сплошной кусок {first:%Y-%m-%d} — {last:%Y-%m-%d}; "
              f"сделки {a.start} — {a.end}", flush=True)
    if a.source == "lkoh":                       # лента только по LKOH, зато 01.2024 — 04.2026 целиком
        tr = tr[tr["tk"] == "LKOH"]
        h = pd.read_parquet(ROOT / "data" / "derived" / "h40_levels_lkoh.parquet", columns=["hour"])["hour"]
        lo = (h.min() + pd.Timedelta(days=int(a.days * 1.6) + 5)).strftime("%Y-%m-%d")
        a.start, a.end = max(a.start, lo), min(a.end, h.max().strftime("%Y-%m-%d"))
        print(f"лента LKOH: {h.min():%Y-%m-%d} — {h.max():%Y-%m-%d}; сделки {a.start} — {a.end}", flush=True)
    tr = tr[(tr["open_dt"] >= a.start) & (tr["open_dt"] <= a.end)]
    lv = {"tape": levels_tape, "lkoh": levels_lkoh}[a.source](a.side) if a.source != "min" else \
        {t: levels_min(t) for t in sorted(tr["tk"].unique())}
    print(f"сделок {len(tr)}, бумаг {tr['tk'].nunique()}, профиль — {a.source}", flush=True)
    f = features(tr, lv, a.days, a.x, a.w, a.hvn, a.reset)
    f.to_csv(REP / f"trades_{tag}.csv", index=False)
    gr = groups(f, "d_block", a.thr)
    gr.to_csv(REP / f"groups_{tag}.csv", index=False)
    ok, why, tbl = criterion(gr, a.thr)
    tbl.to_csv(REP / f"criterion_{tag}.csv", index=False)
    print(f"сделок с признаком {len(f)}; {why}; критерий {'да' if ok else 'нет'}", flush=True)
    print(gr[gr["стратегия"] == "все"].to_string(index=False), flush=True)
    (REP / f"summary_{tag}.json").write_text(json.dumps(
        {"tag": tag, "n": len(f), "ok": bool(ok), "why": why}, ensure_ascii=False, indent=1), encoding="utf-8")

    if a.grid:
        rows = []
        variants = [(x, days, 0.0) for x in (1.5, 2.0, 3.0) for days in (10, 20, 40)]
        variants += [(a.x, a.days, k) for k in (1.0, 2.0, 3.0)]          # правило свежести Дениса
        for x, days, reset in variants:
            g = f if (x, days, reset) == (a.x, a.days, a.reset) else features(tr, lv, days, x, a.w, a.hvn, reset)
            for thr in (0.5, 1.0):
                gg = groups(g, "d_block", thr)
                o, _w, _t = criterion(gg, thr)
                s = gg[(gg["стратегия"] == "все") & (gg["период"] == "2022–2026")]
                near = s[s["группа"] == f"уровень ближе {thr:g} ATR"]
                rest = s[s["группа"] == "остальные"]
                rows.append({"X": x, "окно_дней": days, "свежесть_ATR": reset, "порог_ATR": thr,
                             "сделок_рядом": int(near["сделок"].sum()), "PF_рядом": float(near["PF"].mean()),
                             "PF_остальные": float(rest["PF"].mean()), "критерий": "да" if o else "нет"})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(REP / f"grid_{tag}.csv", index=False)

    if a.charts:
        chart_groups(gr, a.thr, "H40: сделки под большим уровнем объёма против остальных — PF по стратегиям", VIZ / f"groups_{tag}.png")
        chart_deciles(f, "H40: расстояние до большого уровня по ходу сделки (10 корзин) и результат сделки", VIZ / f"deciles_{tag}.png")
        chart_years(f, a.thr, "H40: средний результат сделки по годам — у уровня и остальные", VIZ / f"years_{tag}.png")
        examples_chart(f, lv, a.days, a.x, a.w, a.hvn, a.reset, a.thr, VIZ / f"examples_{tag}.png")
    if a.sheets:
        for p in entry_sheets(f, lv, a.days, a.x, a.w, a.hvn, a.reset, ["LKOH", "SBER", "GAZP", "TATN"],
                              (2025, 2026), VIZ, tag):
            print(p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
