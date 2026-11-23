"""H94 «Анализ ситуаций взрывного роста» (Тимур, 18.11; план подтверждён 19.11).

Ответы автора на вопросы плана: ATR — оба варианта (дневной и часовой) и ещё фиксированный ход 0.5% и 1% за час;
бумаги — SBER, GAZP, LKOH, ROSN, MOEX, золото и серебро (фьючерсы Мосбиржи GOLD, SILV); ход на открытии сессии —
отдельно; новости — только календарь (ЦБ, EIA, дивидендная отсечка); горизонты — час, до конца дня, следующий день.

Случай (для каждого определения и окна 60 / 120 минут): первая минута, когда цена за окно ушла на порог
(k дневных ATR, k часовых ATR или p%); следующий случай в ту же сторону — не раньше чем через окно. Цены акций
скорректированы на дивиденды (гэп отсечки — не «взрыв»). «Открытие» — окно начинается в прошлом дне (ночной разрыв)
или случай в первые 30 минут дня: считается отдельно, в основные таблицы не входит.

Продолжение: d · ход после случая (d = +1 вверх, −1 вниз) за час (не дальше конца дня), до закрытия дня и до закрытия
следующего дня. Сравнение — со случайными минутами той же бумаги в тот же час суток и ту же половину истории:
«превышение» = доля продолжений у случаев − доля «в ту же сторону» у случайных минут.

Признаки случая (что смотрим по плану):
  * форма на 5 минутах: доля 5-минутных свечей окна в сторону хода (ровный ход ≥ 0.7) и доля хода за последние 15 минут
    (рывок ≥ 0.6);
  * объём окна против обычного объёма этого часа суток за 20 прошлых дней (высокий ≥ 2 раз) и объём часа до окна;
  * уровни: профиль объёма 20 прошлых дней по уровням 0.1% (как в H40); крупный уровень — объём ≥ 2 средних по ±2 ATR;
    пробил ли ход крупный уровень, стоит ли крупный уровень впереди ближе 0.5 ATR; пробой максимума/минимума прошлого дня;
  * календарь: решение ЦБ (день решения, с 13:20 до конца дня), отчёт EIA (−10…+120 мин, для нефтегаза), день отсечки.

Половины: закономерности ищем на 2016–2020, проверяем на 2021–2026. Критерий плана: признак, при котором превышение
≥ 10 п.п. в обеих половинах, 30+ случаев в группе и средний ход после случая больше издержек 0.1%.

    python scripts/h94_analysis.py            # по всем бумагам → data/derived/h94/<бумага>.parquet, затем сводка
    python scripts/h94_analysis.py --report   # только сводка и графики из готовых таблиц
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forexmodel.data.dividends import adjust_for_dividends, load_dividends  # noqa: E402

RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived" / "h94"
OUT = ROOT / "docs" / "results" / "h94"
CAL = ROOT / "data" / "calendar"
SRC = {"SBER": RAW / "SBER_1m_15-26.csv", "GAZP": RAW / "GAZP_1m_15-26_upd.csv", "LKOH": RAW / "LKOH_1m_15-26.csv",
       "ROSN": RAW / "moex" / "ROSN_1m.csv", "MOEX": RAW / "MOEX_1m_15-26.csv",
       "GOLD": RAW / "moex" / "GOLD_1m_fut.csv", "SILV": RAW / "moex" / "SILV_1m_fut.csv"}
SHARES = {"SBER", "GAZP", "LKOH", "ROSN", "MOEX"}
OIL = {"LKOH", "ROSN", "GAZP"}
FEAT = {("день", 2.0), ("час", 2.0), ("час", 3.0), ("%", 1.0)}   # признаки считаются для них (окно 60 мин)
DEFS = [("день", 1.5), ("день", 2.0), ("день", 3.0), ("час", 1.5), ("час", 2.0), ("час", 3.0), ("%", 0.5), ("%", 1.0)]
WINDOWS = (60, 120)
HOR = ("h1", "eod", "nd")
HOR_NAME = {"h1": "час", "eod": "до конца дня", "nd": "следующий день"}
COST = 0.10
STEP = np.log(1.001)


def half(y) -> str:
    return "2016–2020" if y <= 2020 else "2021–2026"


# ---------------------------------------------------------------- данные

def load(tk: str) -> pd.DataFrame:
    src = SRC[tk]
    tcol = "begin" if "begin" in pd.read_csv(src, nrows=1).columns else "time"
    d = pd.read_csv(src, usecols=[tcol, "open", "high", "low", "close", "volume"])
    d = d.rename(columns={tcol: "time"})
    d["time"] = pd.to_datetime(d["time"])
    d = d.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    d = d[d["time"] >= "2015-09-01"].reset_index(drop=True)
    if tk in SHARES:
        d = adjust_for_dividends(d, load_dividends(ROOT / "data" / "dividends" / "moex_dividends.csv", tk), time_col="time")
    return d


def atrs(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = d.set_index("time")
    out = []
    for rule, shift in (("1D", "day"), ("1h", "hour")):
        b = s.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
        pc = b["close"].shift()
        tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
        a = tr.rolling(14).mean()
        if shift == "day":
            a = a.shift()                                   # только прошлые дни
            out.append(a.reindex(s.index.normalize()).to_numpy())
        else:
            a.index = a.index + pd.Timedelta(hours=1)        # ATR часа известен после его закрытия
            out.append(a.reindex(s.index, method="ffill").to_numpy())
    return out[0], out[1]


def forward(d: pd.DataFrame) -> dict:
    """Ход вперёд от закрытия каждой минуты, %: час (не дальше конца дня), до закрытия дня, до закрытия следующего дня."""
    t = d["time"].to_numpy()
    c = d["close"].to_numpy()
    day = d["time"].dt.normalize().to_numpy()
    last_idx = pd.Series(np.arange(len(d))).groupby(day).transform("max").to_numpy()
    j = np.searchsorted(t, t + np.timedelta64(60, "m"), side="right") - 1
    j = np.minimum(j, last_idx)
    days = np.unique(day)
    day_close = pd.Series(c[last_idx], index=day).groupby(level=0).last()
    nxt = pd.Series(day_close.to_numpy()[1:], index=day_close.index[:-1])
    nd = pd.Series(day).map(nxt).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"h1": (c[j] / c - 1) * 100, "eod": (c[last_idx] / c - 1) * 100, "nd": (nd / c - 1) * 100,
                "first": np.r_[True, day[1:] != day[:-1]], "days": days}


# ---------------------------------------------------------------- признаки

def hour_volume(d: pd.DataFrame) -> pd.Series:
    """Обычный объём часа суток: среднее за 20 прошлых дней (без сегодняшнего), на минуту."""
    s = d.set_index("time")["volume"]
    hv = s.groupby([s.index.normalize(), s.index.hour]).mean()
    hv.index.names = ["day", "hour"]
    w = hv.unstack("hour")
    base = w.rolling(20, min_periods=10).mean().shift()
    return base.stack()


def profiles(d: pd.DataFrame) -> pd.DataFrame:
    """Объём по уровням 0.1% за каждый день: day, b, vol."""
    b = np.floor(np.log(d["close"].clip(lower=1e-9)) / STEP).astype("int64")
    g = pd.DataFrame({"day": d["time"].dt.normalize(), "b": b, "vol": d["volume"]})
    return g.groupby(["day", "b"], sort=True)["vol"].sum()


class Profile:
    """Объём по уровням 0.1% по дням — массивами, чтобы профиль 20 прошлых дней собирался быстро."""

    def __init__(self, prof: pd.Series):
        self.day = prof.index.get_level_values(0).to_numpy()
        self.b = prof.index.get_level_values(1).to_numpy()
        self.vol = prof.to_numpy().astype("float64")

    def feats(self, day, p0: float, p1: float, atr_d: float, sgn: int) -> str:
        """Пробил ли ход крупный уровень; стоит ли крупный уровень впереди ближе 0.5 ATR (профиль 28 прошлых дней)."""
        if not np.isfinite(atr_d) or atr_d <= 0:
            return "нет данных"
        lo = self.day.searchsorted(np.datetime64(day - pd.Timedelta(days=28)))
        hi = self.day.searchsorted(np.datetime64(day))
        if hi - lo < 200:
            return "нет данных"
        b, vol = self.b[lo:hi], self.vol[lo:hi]
        width = max(1, int(np.ceil(np.log(1 + 2 * atr_d / p0) / STEP)))
        b0, b1 = int(np.floor(np.log(p0) / STEP)), int(np.floor(np.log(p1) / STEP))
        zlo, zhi = min(b0, b1) - width, max(b0, b1) + width
        m = (b >= zlo) & (b <= zhi)
        if not m.any():
            return "нет данных"
        prof = np.bincount(b[m] - zlo, weights=vol[m], minlength=zhi - zlo + 1)
        lv = np.flatnonzero(prof >= 2 * prof.mean()) + zlo
        crossed = lv[(lv > min(b0, b1)) & (lv < max(b0, b1))]
        half_atr = max(1, int(np.ceil(np.log(1 + 0.5 * atr_d / p1) / STEP)))
        ahead = lv[((lv - b1) * sgn > 0) & (np.abs(lv - b1) <= half_atr)]
        return "пробил крупный уровень" if len(crossed) else ("крупный уровень впереди" if len(ahead) else "крупных уровней рядом нет")


def calendar() -> tuple[set, dict, set]:
    cb = pd.read_csv(CAL / "cbr_key_rate_meetings.csv", comment="#")
    dcol = [c for c in cb.columns if "date" in c.lower() or "дата" in c.lower()][0]
    cbr = set(pd.to_datetime(cb[dcol]).dt.normalize())
    eia = pd.read_csv(CAL / "eia_wpsr_releases.csv", parse_dates=["release"])["release"]
    eia_by_day = {r.normalize(): r for r in eia}
    dv = pd.read_csv(ROOT / "data" / "dividends" / "moex_dividends.csv", parse_dates=["ex_time"])
    exd = set(zip(dv["ticker"].str.upper(), dv["ex_time"].dt.normalize()))
    return cbr, eia_by_day, exd


# ---------------------------------------------------------------- случаи

def events_for(tk: str) -> pd.DataFrame:
    d = load(tk)
    ad, ah = atrs(d)
    fw = forward(d)
    t = d["time"]
    tn = t.to_numpy()
    c = d["close"].to_numpy()
    v = d["volume"].to_numpy().astype("float64")
    day = t.dt.normalize().to_numpy()
    hv = hour_volume(d)
    prof = Profile(profiles(d))
    dsum = d.set_index("time").resample("1D").agg({"high": "max", "low": "min"}).dropna()
    pdh = dsum["high"].shift().reindex(t.dt.normalize()).to_numpy()
    pdl = dsum["low"].shift().reindex(t.dt.normalize()).to_numpy()
    cbr, eia, exd = calendar()
    first_min = pd.Series(np.arange(len(d))).groupby(day).transform("min").to_numpy()
    # базовая линия: все минуты, по часу суток и половине
    hh = t.dt.hour.to_numpy()
    yy = t.dt.year.to_numpy()
    base = []
    for h_ in HOR:
        x = pd.DataFrame({"hour": hh, "half": np.where(yy <= 2020, "2016–2020", "2021–2026"),
                          "f": fw[h_]}).dropna()
        x = x[(x["f"] != 0)]
        g = x.groupby(["half", "hour"])["f"].agg(up=lambda s: (s > 0).mean(), mean="mean", n="size")
        g["hor"] = h_
        base.append(g.reset_index())
    pd.concat(base).assign(tk=tk).to_parquet(DER / f"{tk}_base.parquet", index=False)
    rows = []
    for w in WINDOWS:
        pos = np.searchsorted(tn, tn - np.timedelta64(w, "m"), side="right") - 1
        ok = pos >= 0
        pos = np.clip(pos, 0, None)
        p0 = np.where(ok, c[pos], np.nan)
        move = c - p0
        mpct = move / p0 * 100
        cross = day[pos] != day
        for kind, k in DEFS:
            den = {"день": ad, "час": ah, "%": None}[kind]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = mpct if den is None else np.where(den > 0, move / den, np.nan)
            for sgn in (1, -1):
                hit = np.flatnonzero(np.nan_to_num(r * sgn, nan=-1) >= k)
                if not len(hit):
                    continue
                gap = np.diff(tn[hit]).astype("timedelta64[m]").astype(np.int64)
                keep = hit[np.r_[True, gap >= w]]
                full = (kind, k) in FEAT and w == 60
                for i in keep:
                    i0 = pos[i]
                    dts = pd.Timestamp(day[i])
                    opening = bool(cross[i] or (tn[i] - tn[first_min[i]]) < np.timedelta64(30, "m"))
                    rec = {"tk": tk, "w": w, "kind": kind, "k": k, "dir": sgn, "t": pd.Timestamp(tn[i]),
                           "year": int(yy[i]), "hour": int(hh[i]), "move_pct": float(mpct[i]), "atr_d_pct": float(ad[i] / c[i] * 100),
                           "h1": fw["h1"][i], "eod": fw["eod"][i], "nd": fw["nd"][i], "opening": opening}
                    if not full:
                        rows.append(rec)
                        continue
                    seg = slice(i0, i + 1)
                    cl = c[seg]
                    # форма на 5 минутах
                    s5 = pd.Series(cl, index=tn[seg]).resample("5min").last().dropna().to_numpy()
                    st = np.diff(np.r_[c[i0], s5]) * sgn
                    smooth = float((st > 0).mean()) if len(st) else np.nan
                    j15 = np.searchsorted(tn, tn[i] - np.timedelta64(15, "m"), side="right") - 1
                    last15 = float((c[i] - c[max(j15, i0)]) / move[i]) if move[i] else np.nan
                    # объём: окно и час до него против обычного объёма этого часа
                    hour_key = (pd.Timestamp(day[i]), int(hh[i]))
                    exp = hv.get(hour_key, np.nan)
                    vr = float(v[seg].mean() / exp) if exp and np.isfinite(exp) and exp > 0 else np.nan
                    jb = np.searchsorted(tn, tn[i0] - np.timedelta64(60, "m"), side="left")
                    expb = hv.get((pd.Timestamp(day[i0]), int(hh[i0])), np.nan)
                    vb = float(v[jb:i0].mean() / expb) if i0 > jb and expb and np.isfinite(expb) and expb > 0 else np.nan
                    lvl = prof.feats(pd.Timestamp(day[i]), float(c[i0]), float(c[i]), float(ad[i]), sgn)
                    brk_pd = bool((sgn > 0 and c[i0] <= pdh[i] < c[i]) or (sgn < 0 and c[i0] >= pdl[i] > c[i]))
                    news = []
                    if dts in cbr and hh[i] >= 13:
                        news.append("ЦБ")
                    if tk in OIL and dts in eia and -10 <= (pd.Timestamp(tn[i]) - eia[dts]).total_seconds() / 60 <= 120:
                        news.append("EIA")
                    if (tk, dts) in exd:
                        news.append("отсечка")
                    rows.append({**rec, "smooth": smooth, "last15": last15, "vol_ratio": vr, "vol_before": vb, "level": lvl,
                                 "break_pd": brk_pd, "news": "+".join(news) or "нет"})
        print(tk, "окно", w, "случаев", len(rows), flush=True)
    ev = pd.DataFrame(rows)
    ev = ev[ev["year"] >= 2016]
    ev.to_parquet(DER / f"{tk}.parquet", index=False)
    return ev


# ---------------------------------------------------------------- сводка

def with_base(ev: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["half"] = np.where(ev["year"] <= 2020, "2016–2020", "2021–2026")
    for h_ in HOR:
        b = base[base["hor"] == h_].set_index(["tk", "half", "hour"])
        key = pd.MultiIndex.from_arrays([ev["tk"], ev["half"], ev["hour"]])
        up = b["up"].reindex(key).to_numpy()
        mean = b["mean"].reindex(key).to_numpy()
        ev[f"c_{h_}"] = (ev[h_] * ev["dir"] > 0).astype(float).where(ev[h_].notna() & (ev[h_] != 0))
        ev[f"b_{h_}"] = np.where(ev["dir"] > 0, up, 1 - up)          # доля «в ту же сторону» у случайных минут
        ev[f"r_{h_}"] = ev[h_] * ev["dir"]                           # ход после случая в сторону случая, %
        ev[f"rb_{h_}"] = ev[f"r_{h_}"] - mean * ev["dir"]            # то же сверх обычного хода этого часа
    return ev


def summarize(g: pd.DataFrame, h_: str) -> dict:
    x = g.dropna(subset=[f"c_{h_}"])
    n = len(x)
    r = x[f"rb_{h_}"]
    return {"случаев": n, "продолжений %": round(x[f"c_{h_}"].mean() * 100, 1) if n else np.nan,
            "у случайных %": round(x[f"b_{h_}"].mean() * 100, 1) if n else np.nan,
            "превышение п.п.": round((x[f"c_{h_}"].mean() - x[f"b_{h_}"].mean()) * 100, 1) if n else np.nan,
            "ход после, %": round(x[f"r_{h_}"].mean(), 3) if n else np.nan,
            "t": round(r.mean() / r.std() * np.sqrt(n), 2) if n > 2 and r.std() > 0 else np.nan}


def groups(ev: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "все (без открытия)": pd.Series(True, index=ev.index),
        "форма: ровный ход (≥70% 5-мин свечей в сторону хода)": ev["smooth"] >= 0.7,
        "форма: рывок (≥60% хода за последние 15 мин)": ev["last15"] >= 0.6,
        "объём окна ≥ 2 обычных": ev["vol_ratio"] >= 2,
        "объём окна < 1.2 обычного": ev["vol_ratio"] < 1.2,
        "объём за час до окна ≥ 1.5 обычного": ev["vol_before"] >= 1.5,
        "пробил крупный уровень объёма": ev["level"] == "пробил крупный уровень",
        "крупный уровень впереди ближе 0.5 ATR": ev["level"] == "крупный уровень впереди",
        "крупных уровней рядом нет": ev["level"] == "крупных уровней рядом нет",
        "пробил максимум/минимум прошлого дня": ev["break_pd"],
        "календарь: ЦБ / EIA / отсечка": ev["news"] != "нет",
        "вверх": ev["dir"] > 0,
        "вниз": ev["dir"] < 0,
    }


def report() -> None:
    ev = pd.concat([pd.read_parquet(p) for p in sorted(DER.glob("*.parquet")) if not p.stem.endswith("_base")], ignore_index=True)
    base = pd.concat([pd.read_parquet(p) for p in DER.glob("*_base.parquet")], ignore_index=True)
    ev = with_base(ev, base)
    ev.to_parquet(DER.parent / "h94_cases.parquet", index=False)
    lines = ["# H94: числа", ""]
    # 1. все определения: продолжение по горизонтам, отдельно открытие
    rows = []
    for (kind, k, w), g in ev.groupby(["kind", "k", "w"]):
        for op, gg in g.groupby("opening"):
            for hf, x in gg.groupby("half"):
                for h_ in HOR:
                    rows.append({"порог": f"{k:g} {'%' if kind == '%' else 'ATR ' + kind}", "окно": w,
                                 "открытие": "да" if op else "нет", "половина": hf, "горизонт": HOR_NAME[h_], **summarize(x, h_)})
    allt = pd.DataFrame(rows)
    allt.to_csv(OUT / "continuation_all.csv", index=False)
    main = allt[(allt["окно"] == 60) & (allt["открытие"] == "нет")]
    lines += ["## Продолжение после случая (окно 60 минут, без открытия сессии)", "",
              "Превышение — доля продолжений минус доля «в ту же сторону» у случайных минут той же бумаги в тот же час.", "",
              md(main.drop(columns=["окно", "открытие"])), ""]
    op = allt[(allt["окно"] == 60) & (allt["открытие"] == "да") & (allt["горизонт"] == "до конца дня")]
    lines += ["## Открытие сессии отдельно (окно 60 минут, до конца дня)", "", md(op.drop(columns=["окно", "открытие", "горизонт"])), ""]
    # 2. признаки: по основным определениям, до конца дня
    rows = []
    for (kind, k), g in ev[(ev["w"] == 60) & (~ev["opening"])].groupby(["kind", "k"]):
        if (kind, k) not in {("день", 2.0), ("час", 2.0), ("час", 3.0), ("%", 1.0)}:
            continue
        for name, m in groups(g).items():
            for h_ in ("h1", "eod"):
                r = {"порог": f"{k:g} {'%' if kind == '%' else 'ATR ' + kind}", "признак": name, "горизонт": HOR_NAME[h_]}
                for hf in ("2016–2020", "2021–2026"):
                    s = summarize(g[m & (g["half"] == hf)], h_)
                    r[f"n {hf}"] = s["случаев"]
                    r[f"превыш. {hf}"] = s["превышение п.п."]
                    r[f"ход {hf}"] = s["ход после, %"]
                rows.append(r)
    ft = pd.DataFrame(rows)
    ft["критерий"] = ((ft["превыш. 2016–2020"] >= 10) & (ft["превыш. 2021–2026"] >= 10) & (ft["n 2016–2020"] >= 30) &
                      (ft["n 2021–2026"] >= 30) & (ft["ход 2016–2020"] > COST) & (ft["ход 2021–2026"] > COST))
    ft.to_csv(OUT / "features.csv", index=False)
    lines += ["## Признаки (окно 60 минут, без открытия)", "", md(ft), ""]
    hit = ft[ft["критерий"]]
    lines += ["## Прошли критерий плана", "", md(hit) if len(hit) else "Ни один признак.", ""]
    # 3. по бумагам
    rows = []
    for (tk, hf), g in ev[(ev["w"] == 60) & (~ev["opening"]) & (ev["kind"] == "час") & (ev["k"] == 2.0)].groupby(["tk", "half"]):
        rows.append({"бумага": tk, "половина": hf, **summarize(g, "eod")})
    lines += ["## По бумагам (2 часовых ATR за 60 минут, до конца дня)", "", md(pd.DataFrame(rows)), ""]
    # 4. отскок по дням: после резкого хода вниз / вверх (дневной ATR) — цена к закрытию дня против хода;
    #    несколько бумаг в один день — одно наблюдение (среднее), чтобы кризисные дни не весили больше
    rows = []
    for k in (1.5, 2.0, 3.0):
        x = ev[(ev["kind"] == "день") & (ev["k"] == k) & (ev["w"] == 60) & (~ev["opening"])]
        for dr, lab in ((-1, "падение"), (1, "взлёт")):
            y = x[x["dir"] == dr].assign(day=lambda z: z["t"].dt.normalize())
            g = y.groupby("day")["eod"].mean() * -dr          # > 0 — отскок против хода
            for hf, m in (("2016–2020", g.index.year <= 2020), ("2021–2026", g.index.year >= 2021),
                          ("2021–2026 без 2022", (g.index.year >= 2021) & (g.index.year != 2022))):
                z = g[m]
                rows.append({"порог": f"{k:g} ATR день", "ход": lab, "период": hf, "дней": len(z),
                             "доля дней с отскоком %": round((z > 0).mean() * 100) if len(z) else np.nan,
                             "отскок к закрытию, ср. %": round(z.mean(), 2) if len(z) else np.nan,
                             "медиана %": round(z.median(), 2) if len(z) else np.nan})
    rb = pd.DataFrame(rows)
    rb.to_csv(OUT / "rebound_days.csv", index=False)
    lines += ["## Отскок после резкого хода (дневной ATR, по дням)", "",
              "Отскок — ход от момента случая до закрытия дня против направления случая; несколько бумаг в один день — "
              "одно наблюдение.", "", md(rb), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    charts(ev, ft)
    chart_rebound(rb)
    print("\n".join(lines[:60]))


def charts(ev: pd.DataFrame, ft: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C3 = "#2459b3", "#9aa3ae"
    for thr in ("2 ATR час", "2 ATR день", "1 %"):
        x = ft[(ft["порог"] == thr) & (ft["горизонт"] == "до конца дня")]
        if x.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 0.45 * len(x) + 1.5))
        y = np.arange(len(x))
        ax.barh(y + 0.2, x["превыш. 2016–2020"], 0.4, color=C3, label="2016–2020 (выбор)")
        ax.barh(y - 0.2, x["превыш. 2021–2026"], 0.4, color=C1, label="2021–2026 (проверка)")
        for yi, a, b in zip(y, x["n 2016–2020"], x["n 2021–2026"]):
            ax.text(ax.get_xlim()[1] if False else 0, yi, f"  n {a} / {b}", va="center", fontsize=7, color="#555")
        ax.set_yticks(y)
        ax.set_yticklabels(x["признак"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        for v_ in (-10, 10):
            ax.axvline(v_, color="#c8641e", lw=0.8, ls="--")
        ax.set_xlabel("доля продолжений минус доля у случайных минут, п.п.; пунктир — критерий ±10 п.п.\n"
                      "меньше нуля — после резкого хода цена чаще разворачивается")
        ax.set_title(f"H94: продолжается ли ход после резкого часа\n({thr} за 60 минут, без открытия сессии, до конца дня)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(OUT / f"features_{thr.replace(' ', '_').replace('%', 'pct')}.png", dpi=110)
        plt.close(fig)


def chart_rebound(rb: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cols = {"2016–2020": "#9aa3ae", "2021–2026": "#2459b3", "2021–2026 без 2022": "#8db3f2"}
    for ax, lab in zip(axes, ("падение", "взлёт")):
        x = rb[rb["ход"] == lab]
        thr = list(dict.fromkeys(x["порог"]))
        for j, (per, col) in enumerate(cols.items()):
            y = x[x["период"] == per].set_index("порог").reindex(thr)
            pos = np.arange(len(thr)) + (j - 1) * 0.27
            ax.bar(pos, y["доля дней с отскоком %"], 0.27, color=col, label=per)
            for p_, v_, n_, m_ in zip(pos, y["доля дней с отскоком %"], y["дней"], y["отскок к закрытию, ср. %"]):
                if np.isfinite(v_):
                    ax.text(p_, v_ + 1, f"{n_:.0f} дн.\n{m_:+.1f}%", ha="center", fontsize=6.5)
        ax.axhline(50, color="k", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(thr)))
        ax.set_xticklabels(thr)
        ax.set_ylim(0, 118)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(f"после резкого {'падения' if lab == 'падение' else 'взлёта'} за час: к закрытию дня цена {'выше' if lab == 'падение' else 'ниже'}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("доля дней с отскоком, %")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("H94: после резкого падения цена чаще отскакивает к закрытию дня; после взлёта устойчивого рисунка нет\n"
                 "(подписи: число дней и средний отскок; пунктир — 50%, как у монетки)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "rebound.png", dpi=115)
    plt.close(fig)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    DER.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if argv[:1] != ["--report"]:
        for tk in (argv or SRC):
            if (DER / f"{tk}.parquet").exists() and not argv:
                continue
            events_for(tk)
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
