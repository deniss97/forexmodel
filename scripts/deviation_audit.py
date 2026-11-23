"""Проверка сверки (scripts/live_deviation.py) на S42 — просьба Тимура на созвоне 23.11: «проверку перед запуском
надо сделать на нашем коде, смотреть, когда он выключает — есть сомнения, что он корректно посчитал».

Сверка 21.11 остановила S42: 5 живых сделок, средняя −3.19% против +0.40% в бэктесте 2021–2026, p = 0.003. Здесь:
  1. живые сделки S42 и то же p тремя способами:
     а) как в сверке — средняя живых против средних из 5 случайных независимых сделок бэктеста;
     б) подряд идущие 5 сделок бэктеста (по времени входа) — сохраняет то, что убытки идут сериями;
     в) по дням входа — сделки, открытые в один день, считаются одной (средняя за день), дни — случайные;
  2. ложные остановки: правило сверки запускается с каждой 3-й сделки бэктеста, как будто там начался живой режим, и
     идёт 5…12 сделок; доля запусков, где сработал бы «стоп» (по замыслу — около 1%);
  3. график: средняя 5 подряд сделок бэктеста по времени, уровень живой средней и случаи срабатывания.

    python scripts/deviation_audit.py      → docs/results/s42_check_audit.md, s42_check_audit.png
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import live_deviation as ld  # noqa: E402

OUT = ld.ROOT / "docs" / "results"
DRAWS = 5000


def p_iid(u: np.ndarray, x: np.ndarray, rng) -> float:
    d = u[rng.integers(0, len(u), (DRAWS, len(x)))]
    return float((d.mean(1) <= x.mean()).mean())


def by_day(t: pd.DataFrame, time_col: str) -> np.ndarray:
    return t.groupby(t[time_col].dt.normalize())["unit"].mean().to_numpy()


def main() -> None:
    rng = np.random.default_rng(7)
    ref = ld.reference()["S42"].sort_values("open_dt").reset_index(drop=True)
    lv = ld.live()["S42"].sort_values("вход")
    u, x = ref["unit"].to_numpy(), lv["unit"].to_numpy()
    n = len(x)
    roll = pd.Series(u).rolling(n).mean()
    p_a = p_iid(u, x, rng)
    p_b = float((roll <= x.mean()).mean())
    ud, xd = by_day(ref, "open_dt"), by_day(lv, "вход")
    p_c = p_iid(ud, xd, rng)

    # ложные остановки правилом сверки на самом бэктесте
    fires, starts = [], 0
    for s in range(0, len(u) - 12, 3):
        starts += 1
        for k in range(ld.MIN_TRADES, 13):
            if p_iid(u, u[s:s + k], rng) < 0.01:
                fires.append((s, s + k - 1))
                break
    rate = len(fires) / starts
    same = ref[ref["open_dt"].dt.normalize().duplicated(keep=False)]
    pairs = same.groupby(same["open_dt"].dt.normalize())["unit"].apply(list)
    pairs = pairs[pairs.str.len() == 2]
    corr = float(np.corrcoef([v[0] for v in pairs], [v[1] for v in pairs])[0, 1])

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(ref["open_dt"], roll, color="#2a6bd8", lw=1, label=f"средняя {n} подряд сделок бэктеста S42 (после издержек)")
    ax.axhline(x.mean(), color="#e03131", ls="--", lw=1.2, label=f"живые {n} сделок: {x.mean():+.2f}%")
    ax.axhline(u.mean(), color="#868e96", ls=":", lw=1, label=f"средняя сделка бэктеста {u.mean():+.2f}%")
    for s, e in fires:
        ax.axvspan(ref["open_dt"][s], ref["open_dt"][e], color="#f08c00", alpha=0.25, lw=0)
    ax.plot([], [], color="#f08c00", alpha=0.4, lw=6, label=f"правило «стоп» сработало бы ({len(fires)} из {starts} запусков)")
    ax.set_title("S42: как часто сверка выключила бы стратегию на её же бэктесте 2021–2026")
    ax.set_ylabel("% на сделку")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "s42_check_audit.png", dpi=115)

    rows = "\n".join(f"| {r['бумага']} | {r['вход']:%d.%m %H:%M} | {r['выход']:%d.%m %H:%M} | {r['unit']:+.2f}% |"
                     for _, r in lv.iterrows())
    md = f"""# Сверка S42: правильно ли она остановила стратегию (23.11, по просьбе Тимура)

`python scripts/deviation_audit.py` · сверка — `scripts/live_deviation.py` · график — `s42_check_audit.png`

## Живые сделки S42 (журнал стратегии, после издержек 0.2%)

| бумага | вход | выход | результат |
|---|---|---|---|
{rows}

Средняя **{x.mean():+.2f}%** против **{u.mean():+.2f}%** в бэктесте 2021–2026 ({len(u)} сделок, убыточных {(u <= 0).mean():.0%}).
Сделки шли парами: 30.10 LKOH и GAZP, 11.11 GAZP и SBER — по сути {len(xd)} дня входа, а не {n} независимых сделок.

## Насколько это редко — три способа

| способ | p | решение по правилу (стоп < 1%, внимание < 5%) |
|---|---|---|
| а) как в сверке: {n} случайных независимых сделок бэктеста | {p_a:.3f} | {"стоп" if p_a < 0.01 else "внимание" if p_a < 0.05 else "в норме"} |
| б) {n} подряд идущих сделок бэктеста (серии убытков сохраняются) | {p_b:.3f} | {"стоп" if p_b < 0.01 else "внимание" if p_b < 0.05 else "в норме"} |
| в) по дням входа ({len(xd)} дня против случайных дней бэктеста) | {p_c:.3f} | {"мало сделок (< 5 дней)" if len(xd) < ld.MIN_TRADES else ("стоп" if p_c < 0.01 else "внимание" if p_c < 0.05 else "в норме")} |

## Ложные остановки на самом бэктесте

Правило сверки запущено с каждой 3-й сделки бэктеста, как будто там начался живой режим (5…12 сделок): «стоп»
сработал бы в **{len(fires)} из {starts}** запусков (**{rate:.1%}**) при задуманном ~1%. Причина — сделки одного дня связаны
(корреляция результатов пар одного дня {corr:.2f}, таких сделок {len(same)} из {len(u)}), а сверка считает их независимыми.

## Вывод

* Сверка считала **правильно по своему правилу**, но правило **строже, чем задумано**: примерно в {rate / 0.01:.0f} раз чаще
  ложно выключает. С учётом серий (способ б) живые 5 сделок — «внимание», а не «стоп»; по дням входа сделок пока мало.
* Плохая полоса на живых данных всё же реальна: такие 5 подряд в бэктесте — {p_b:.1%} окон, худшие окна — 2022 и 2024.
* Что можно поменять в сверке (решение — за командой; сейчас она не менялась, S42 стоит в `strategy_stop.json`, пока
  её не уберут руками): считать по дням входа или подряд идущими окнами; «стоп» — от 8–10 дней входа, до этого
  только «внимание».
"""
    (OUT / "s42_check_audit.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
