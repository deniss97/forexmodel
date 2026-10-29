"""Картинка «состояние контура» для группы Telegram (29.10): позиции на графиках цены с уровнями, результат счёта,
бумажные стратегии. Каждый день в 23 ч по Москве (scripts/queues/live_hourly.sh) и по команде бота /chart.

  * шапка — время, стоп-кран, позиции счёта и слоты, свежесть данных;
  * до шести мини-графиков: позиции счёта, затем бумажные S42 и S46/S48 — часовая цена за неделю, вход, стоп и, у S42,
    уровень включения трейлинга (свечи — кэш контура reports/live/bars, без чтения файлов минуток);
  * результат сделок счёта по дням с порогами просадки «внимание» и «стоп»;
  * бумажные стратегии: результат закрытых сделок и открытых позиций с 24.10.

    python scripts/live_charts.py           # reports/live/charts/dashboard.png
    python scripts/live_charts.py --send    # и отправить в группу
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import notify  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "live"
CHARTS = OUT / "charts"
BG, INK, INK2, GRID = "#f6f4ef", "#15233b", "#4a5568", "#dcd7cc"
BLUE, RED, AQUA, GRAY, ORANGE = "#2459b3", "#b3261e", "#199e70", "#9aa3ae", "#c8641e"   # палитра проекта, валидатор пройден


def load(name: str) -> dict:
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def positions(acc: dict, s42: dict, it: dict) -> list[dict]:
    """Что рисовать: позиции счёта, затем бумажные S42 и S46/S48 — не больше шести, свежие раньше."""
    out = []
    for p in acc.get("позиции", []):
        out.append({"бумага": p["бумага"], "метка": f"{p['стратегия']} · счёт, {p.get('статус')}", "вход": p.get("цена_входа") or p["цена"],
                    "стоп": max(float(p["стоп"]), float(p.get("стоп стратегии") or 0)), "активация": p.get("активация"),
                    "с": pd.Timestamp(p["вход_в"]), "счёт": True})
    for inst, m in (s42.get("решения") or {}).items():
        if m and m.get("вход") and m.get("решение") in ("держать", "открыть"):
            out.append({"бумага": inst.upper(), "метка": "S42 · бумага" + (", трейлинг" if m.get("трейлинг_вкл") else ""),
                        "вход": m["цена_входа"], "стоп": m["стоп"], "активация": None if m.get("трейлинг_вкл") else m.get("активация"),
                        "с": pd.Timestamp(m["вход"]), "счёт": False})
    for p in it.get("позиции", []):
        out.append({"бумага": p["бумага"], "метка": f"{p['стратегия']} · бумага", "вход": p["open_price"], "стоп": p["стоп"],
                    "активация": None, "с": pd.Timestamp(p["open_dt"]), "счёт": False})
    out.sort(key=lambda p: (0 if p["счёт"] else 1 if p["метка"].startswith("S42") else 2, -p["с"].value))   # счёт, S42, остальные
    return out[:6]


def bars(inst: str, since: pd.Timestamp) -> pd.DataFrame:
    p = OUT / "bars" / f"{inst.lower()}.pkl"
    if not p.exists():
        return pd.DataFrame(columns=["time", "close"])
    b = pd.read_pickle(p)["bars"]
    return b[b["time"] >= since][["time", "close"]]


def axes(ax) -> None:
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=7.5)


def draw_position(ax, p: dict, now: pd.Timestamp) -> None:
    """Часовая цена за неделю по торговым часам (ночи и выходные сжаты), вход, стоп, включение трейлинга."""
    axes(ax)
    b = bars(p["бумага"], min(p["с"] - pd.Timedelta(days=2), now - pd.Timedelta(days=7))).reset_index(drop=True)
    if b.empty:
        ax.text(0.5, 0.5, "нет свечей", ha="center", va="center", color=INK2, transform=ax.transAxes)
        return
    last = float(b["close"].iloc[-1])
    ax.plot(b.index, b["close"], color=BLUE, linewidth=1.8)
    ax.axhline(p["вход"], color=GRAY, linestyle="--", linewidth=1.2)
    ax.axhline(p["стоп"], color=RED, linewidth=1.4)
    if p.get("активация"):
        ax.axhline(float(p["активация"]), color=AQUA, linestyle=":", linewidth=1.6)
    after = b.index[b["time"] >= p["с"]]
    if len(after):
        ax.axvline(after[0], color=GRAY, linewidth=0.8, alpha=0.6)
    ax.plot([b.index[-1]], [last], "o", color=BLUE, markersize=5, markeredgecolor=BG, markeredgewidth=1.5)
    day = b["time"].dt.normalize()
    starts = list(b.index[day != day.shift()])
    starts = [i for i, j in zip(starts, starts[1:] + [len(b)], strict=True) if j - i >= 5]   # неполный первый день — без подписи
    step = max(1, len(starts) // 4)
    ax.set_xticks(starts[::step], [f"{b['time'][i]:%d.%m}" for i in starts[::step]])
    ax.set_xlim(-1, len(b) + 1)
    pct, room = (last / float(p["вход"]) - 1) * 100, (last / float(p["стоп"]) - 1) * 100
    ax.set_title(f"{p['бумага']} · {p['метка']}", loc="left", fontsize=9, color=INK, pad=14)
    ax.text(0, 1.02, f"{last:g} ({pct:+.1f}% от входа {float(p['вход']):g}), до стопа {room:.1f}%".replace("-", "−"), transform=ax.transAxes,
            fontsize=7.5, color=(AQUA if pct >= 0 else RED) if abs(pct) >= 0.05 else INK2)


def draw_account(ax, acc: dict, cfg: dict) -> None:
    axes(ax)
    t = pd.read_csv(OUT / "account_trades.csv") if (OUT / "account_trades.csv").exists() else pd.DataFrame()
    ax.set_title("Результат сделок счёта, ₽", loc="left", fontsize=9.5, color=INK)
    if t.empty or "цена выхода" not in t:
        ax.text(0.5, 0.55, "Сделок на счёте ещё не было", ha="center", va="center", fontsize=10, color=INK, transform=ax.transAxes)
        ax.text(0.5, 0.38, f"пороги просадки: −{cfg.get('dd_warn_pct', 8)}% и −{cfg.get('dd_stop_pct', 12)}% депозита "
                f"({float(cfg['capital']) * cfg.get('dd_warn_pct', 8) / 100:,.0f} и {float(cfg['capital']) * cfg.get('dd_stop_pct', 12) / 100:,.0f} ₽)".replace(",", " "),
                ha="center", va="center", fontsize=8, color=INK2, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    t = t[t["цена выхода"].notna()].copy()
    entry = t["цена_входа"].fillna(t["цена"]) if "цена_входа" in t else t["цена"]
    t["pnl"] = (t["цена выхода"].astype(float) - entry.astype(float)) * t["штук"].astype(float)
    t["когда"] = pd.to_datetime(t["закрыто"])
    eq = t.sort_values("когда").set_index("когда")["pnl"].cumsum()
    eq.loc[pd.Timestamp(acc.get("запуск") or eq.index[-1])] = float(acc.get("результат", eq.iloc[-1]))
    ax.step(eq.index, eq.values, where="post", color=BLUE, linewidth=2)
    peak = max(float(acc.get("пик_результата", 0.0)), float(eq.max()), 0.0)
    for key, color, label in (("dd_warn_pct", ORANGE, "внимание"), ("dd_stop_pct", RED, "стоп")):
        lvl = peak - float(cfg["capital"]) * float(cfg.get(key, 8 if key == "dd_warn_pct" else 12)) / 100
        ax.axhline(lvl, color=color, linestyle="--", linewidth=1)
        ax.text(eq.index[0], lvl, f" {label}", va="bottom", fontsize=7.5, color=color)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))


def draw_paper(ax, s42: dict, it: dict) -> None:
    """Бумажные стратегии с 24.10 табличкой: открытые позиции (в среднем %) и закрытые сделки (сумма %)."""
    ax.axis("off")
    ax.set_title("Бумажные стратегии с 24.10", loc="left", fontsize=9.5, color=INK)
    rows = []
    open42 = [(m["цена"] / m["цена_входа"] - 1) * 100 for m in (s42.get("решения") or {}).values()
              if m and m.get("вход") and m.get("решение") in ("держать", "открыть")]
    rows.append(("S42", open42, None))
    j = pd.read_csv(OUT / "iterator_journal.csv") if (OUT / "iterator_journal.csv").exists() else pd.DataFrame()
    last = {k: v.get("цена") for k, v in (it.get("данные") or {}).items()}
    for code in ("S46", "S48"):
        x = j[(j["стратегия"] == code) & j["цена выхода"].notna()] if len(j) else j
        op = [(last[p["бумага"]] / p["open_price"] - 1) * 100 for p in it.get("позиции", []) if p["стратегия"] == code and last.get(p["бумага"])]
        rows.append((code, op, x["результат %"].astype(float).tolist() if len(x) else []))
    ax.text(0.0, 0.86, "стратегия", fontsize=7.5, color=INK2, transform=ax.transAxes)
    ax.text(0.28, 0.86, "открыто, в среднем", fontsize=7.5, color=INK2, transform=ax.transAxes)
    ax.text(0.70, 0.86, "закрыто, сумма", fontsize=7.5, color=INK2, transform=ax.transAxes)
    for i, (code, op, closed) in enumerate(rows):
        y = 0.68 - i * 0.2
        ax.text(0.0, y, code, fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
        avg = sum(op) / len(op) if op else 0.0
        ax.text(0.28, y, f"{len(op)} · {avg:+.1f}%".replace("-", "−") if op else "0", fontsize=10, transform=ax.transAxes,
                color=(AQUA if avg >= 0 else RED) if op else INK2)
        tot = sum(closed) if closed else 0.0
        ax.text(0.70, y, "—" if closed is None else (f"{len(closed)} · {tot:+.1f}%".replace("-", "−") if closed else "0"), fontsize=10,
                transform=ax.transAxes, color=(AQUA if tot >= 0 else RED) if closed else INK2)
    ax.text(0.0, 0.0, "S42 закрытые бумажные сделки — в отчётах контура", fontsize=7, color=INK2, transform=ax.transAxes)


def dashboard(path: Path | None = None) -> tuple[Path, str]:
    """Собрать картинку; вернуть путь и короткую подпись для Telegram."""
    now = pd.Timestamp.now("Europe/Moscow").tz_localize(None).floor("min")
    cfg = yaml.safe_load((ROOT / "configs" / "account.yaml").read_text(encoding="utf-8"))
    acc, s42, it = load("account_state.json"), load("s42_state.json"), load("iterator_state.json")
    stop = bool(cfg.get("stop")) or (OUT / "STOP").exists()
    pos = positions(acc, s42, it)
    data = [pd.Timestamp(m["данные_до"]) for m in (s42.get("решения") or {}).values() if m and m.get("данные_до")]
    fig = plt.figure(figsize=(11, 12.6), facecolor=BG)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.62, wspace=0.28, top=0.86, bottom=0.05, left=0.07, right=0.97)
    fig.text(0.07, 0.955, f"Контур forexmodel · {now:%d.%m.%Y %H:%M} МСК", fontsize=16, fontweight="bold", color=INK)
    n_acc = len(acc.get("позиции", []))
    fig.text(0.07, 0.928, f"{'Стоп-кран включён — сделки на счёт не берутся' if stop else 'Стоп-кран выключен — торгуем'} · на счёте позиций {n_acc} из {cfg['max_positions']} · "
             f"капитал {float(cfg['capital']):,.0f} ₽".replace(",", " ") + (f" · данные до {max(data):%d.%m %H:%M}" if data else ""),
             fontsize=10, color=INK2)
    fig.text(0.07, 0.903, "на графиках: синяя — цена по часам, серая пунктирная — вход, красная — стоп, зелёная точками — включение трейлинга",
             fontsize=8, color=INK2)
    for k in range(6):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        if k < len(pos):
            draw_position(ax, pos[k], now)
        else:
            ax.axis("off")
    if not pos:
        fig.text(0.5, 0.68, "Открытых позиций нет — ни на счёте, ни на бумаге", ha="center", fontsize=11, color=INK2)
    draw_account(fig.add_subplot(gs[2, :2]), acc, cfg)
    draw_paper(fig.add_subplot(gs[2, 2]), s42, it)
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = path or CHARTS / "dashboard.png"
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    caption = (f"📊 Контур на {now:%d.%m %H:%M}: {'стоп-кран включён' if stop else 'торгуем'}, позиций на счёте {n_acc}, "
               f"на бумаге {len([p for p in pos if not p['счёт']])}, результат счёта " + f"{float(acc.get('результат', 0)):+,.0f} ₽".replace(",", " "))
    return path, caption


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--send", action="store_true", help="отправить в группу")
    a = ap.parse_args(argv)
    path, caption = dashboard()
    print(path, caption)
    if a.send:
        print("доставлено" if notify.photo(path, caption) else "не доставлено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
