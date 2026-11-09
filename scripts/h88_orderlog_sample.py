"""H88: что видно в «типе А» Мосбиржи (Full Orders Log) — разбор бесплатного примера за 01.10.2024 по SBER, GAZP, LKOH.

Пример (1.65 ГБ zip, все акции за день) читается потоком и фильтруется на лету:
    curl -skL https://ftp.moex.com/pub/info/data/Equities/OrderLog20241001_A.zip | funzip \
      | grep -E '^[0-9]+,(SBER|GAZP|LKOH),' > ord3.csv
Колонки: NO, SECCODE, BUYSELL, TIME (ЧЧММССмммммм), ORDERNO, ACTION (0 снятие, 1 постановка, 2 сделка), PRICE (0 —
рыночная), VOLUME, TRADENO, TRADEPRICE. Считаем то, чего нет в ленте сделок: заявки и их судьба, айсберги, стакан.

    python -I scripts/h88_orderlog_sample.py <ord3.csv>   # → docs/results/h88_orderlog_sample.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "h88_orderlog_sample.json"


def tsec(t: pd.Series) -> pd.Series:
    """ЧЧММССмммммм → секунды от полуночи."""
    t = t.astype(np.int64)
    return (t // 10**10) * 3600 + (t // 10**8 % 100) * 60 + (t // 10**6 % 100) + (t % 10**6) / 1e6


def book_imbalance(g: pd.DataFrame, step: float = 300.0, levels: int = 10) -> pd.DataFrame:
    """Восстановить стакан по журналу заявок и раз в `step` секунд снять дисбаланс объёма в `levels` лучших уровнях."""
    resting: dict[int, list] = {}          # заявка → [сторона, цена, остаток]
    rows, next_t = [], None
    for r in g.itertuples(index=False):
        if next_t is None:
            next_t = (r.t // step + 1) * step
        while r.t >= next_t:
            bids = [(p, v) for s, p, v in resting.values() if s == "B" and v > 0 and p > 0]
            asks = [(p, v) for s, p, v in resting.values() if s == "S" and v > 0 and p > 0]
            if bids and asks:
                bp = pd.DataFrame(bids, columns=["p", "v"]).groupby("p")["v"].sum().sort_index(ascending=False).head(levels)
                ap = pd.DataFrame(asks, columns=["p", "v"]).groupby("p")["v"].sum().sort_index().head(levels)
                if bp.index[0] < ap.index[0]:
                    rows.append({"t": next_t, "bid": bp.index[0], "ask": ap.index[0],
                                 "imb": (bp.sum() - ap.sum()) / (bp.sum() + ap.sum())})
            next_t += step
        if r.ACTION == 1:
            resting[r.ORDERNO] = [r.BUYSELL, r.PRICE, r.VOLUME]
        elif r.ACTION == 0:
            resting.pop(r.ORDERNO, None)
        elif r.ACTION == 2 and r.ORDERNO in resting:
            resting[r.ORDERNO][2] -= r.VOLUME
            if resting[r.ORDERNO][2] <= 0:
                resting.pop(r.ORDERNO)
    return pd.DataFrame(rows)


def main(path: str) -> int:
    d = pd.read_csv(path, header=None, names=["NO", "SECCODE", "BUYSELL", "TIME", "ORDERNO", "ACTION", "PRICE", "VOLUME",
                                              "TRADENO", "TRADEPRICE"],
                    dtype={"SECCODE": "category", "BUYSELL": "category", "ACTION": "int8", "VOLUME": "int64"})
    d["t"] = tsec(d["TIME"])
    out = {}
    for sec, g in d.groupby("SECCODE", observed=True):
        g = g.sort_values("NO")
        add, can, trd = g[g.ACTION == 1], g[g.ACTION == 0], g[g.ACTION == 2]
        first = add.groupby("ORDERNO")["t"].first()
        life = (can.set_index("ORDERNO")["t"] - first.reindex(can["ORDERNO"]).values).dropna()
        filled = trd.groupby("ORDERNO")["VOLUME"].sum()
        shown = add.groupby("ORDERNO")["VOLUME"].max()
        both = pd.concat([filled, shown], axis=1, keys=["filled", "shown"]).dropna()
        ice = both[both["filled"] > both["shown"]]
        # агрессор сделки — заявка, поставленная позже (больший номер); заявка = номер, а не метка времени
        t2 = trd.sort_values(["TRADENO", "ORDERNO"]).groupby("TRADENO").last()
        aggr = t2.groupby("ORDERNO").agg(vol=("VOLUME", "sum"), side=("BUYSELL", "first"), t=("t", "first"))
        proxy = t2.groupby(["TIME", "BUYSELL"], observed=True)["VOLUME"].sum()   # как в H83: метка времени + сторона
        q99 = aggr["vol"].quantile(0.99)
        imb = book_imbalance(g)
        if len(imb) > 3:
            mid = (imb["bid"] + imb["ask"]) / 2
            imb["fwd"] = mid.shift(-1) / mid - 1
            ic = float(imb[["imb", "fwd"]].corr(method="spearman").iloc[0, 1])
        else:
            ic = float("nan")
        out[str(sec)] = {
            "событий": int(len(g)), "постановок": int(len(add)), "снятий": int(len(can)), "сделок": int(trd["TRADENO"].nunique()),
            "снято заявок, % от поставленных": round(100 * can["ORDERNO"].nunique() / max(1, add["ORDERNO"].nunique()), 1),
            "медиана жизни снятой заявки, с": round(float(life.median()), 2) if len(life) else None,
            "снято быстрее 1 с, %": round(100 * float((life < 1).mean()), 1) if len(life) else None,
            "рыночных заявок, % постановок": round(100 * float((add["PRICE"] == 0).mean()), 1),
            "айсбергов (исполнено больше видимого)": int(len(ice)),
            "объём айсбергов, % объёма сделок": round(100 * float((ice["filled"] - ice["shown"]).sum() / max(1, trd["VOLUME"].sum() / 2)), 1),
            "агрессивных заявок по номеру": int(len(aggr)),
            "«заявок» по метке времени (как в H83)": int(len(proxy)),
            "99-й перцентиль агрессивной заявки, лотов": float(q99),
            "срезов стакана по 5 мин": int(len(imb)),
            "IC дисбаланса 10 уровней с ходом за 5 мин (один день)": round(ic, 3),
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
