"""Снимок банка гипотез из общей страницы в репозиторий.

Банк живёт в базе страницы https://claude.ai/artifact/QkKUwLx2zAQ8X5CZypbU6f. Claude выгружает коллекции
`hypotheses` и `notes` (ArtifactData list с out_dir) в папку — по JSON-файлу на документ:
<папка>/hypotheses/H01.json, <папка>/notes/T01.json. Скрипт пишет:

  docs/hypothesis_bank/hypotheses.json — название, описание, критерий, статус гипотез; из него
      scripts/experiments.py --meta берёт описание гипотезы в карточку эксперимента журнала;
  docs/hypotheses.md — читаемый снимок: гипотезы по разделам, техдолг и уточнения.

Запуск: python scripts/bank_snapshot.py <папка выгрузки>
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "hypothesis_bank" / "hypotheses.json"
OUT_MD = ROOT / "docs" / "hypotheses.md"
BANK = "https://claude.ai/artifact/QkKUwLx2zAQ8X5CZypbU6f"
JOURNAL = "https://claude.ai/artifact/E4teCbCVvojCf96CJy9UQU"

# те же справочники, что на странице банка
AREAS = {"model": "Модель", "impulse": "Импульсная стратегия", "search": "Итератор параметров", "trend": "Тренд-фильтр"}
KINDS = {"check": "проверка достоверности", "entry": "вход и сигнал", "exit": "выход и сопровождение",
         "filter": "фильтр входа и направления", "sizing": "размер позиции и риск", "costs": "издержки и исполнение",
         "markets": "рынки и инструменты", "portfolio": "сочетание стратегий", "feature": "признаки модели",
         "approach": "постановка задачи", "data": "источник данных", "viz": "визуализация", "process": "процесс"}
STATUSES = {"new": "🆕 новая", "work": "🔄 в работе", "yes": "✅ подтвердилась", "no": "❌ не подтвердилась",
            "hold": "⏸ отложена", "ask": "❓ уточнить"}
OPEN = ("new", "work", "ask", "hold")
NKINDS = {"clarify": "уточнение", "debt": "техдолг", "task": "задача"}
NSTAT = {"open": "открыто", "work": "в работе", "done": "сделано", "drop": "отменено"}


def load(folder: Path, name: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / name).glob("*.json"))]


def scores(h: dict) -> tuple[float, float, float] | None:
    """Средняя оценка участников, иначе черновая — как на странице."""
    votes = [v for v in (h.get("votes") or {}).values() if v and all(v.get(k) for k in "ice")]
    if votes:
        return tuple(sum(v[k] for v in votes) / len(votes) for k in "ice")
    d = h.get("draft") or {}
    return (d["i"], d["c"], d["e"]) if all(d.get(k) for k in "ice") else None


def cell(s: str) -> str:
    return " ".join(str(s or "").split()).replace("|", "/")


def num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}".replace(".", ",")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 1
    folder = Path(argv[0])
    hyps = sorted(load(folder, "hypotheses"), key=lambda h: h.get("num") or 0)
    notes = sorted(load(folder, "notes"), key=lambda n: n.get("num") or 0)
    if not hyps:
        print(f"нет гипотез в {folder}/hypotheses")
        return 1

    OUT_JSON.write_text(json.dumps({"date": date.today().isoformat(), "hypotheses": {
        h["code"]: {k: h.get(k, "") for k in ("title", "text", "criterion", "status", "area", "kind", "result")} | {"exps": h.get("exps") or []}
        for h in hyps}}, ensure_ascii=False, indent=1), encoding="utf-8")

    today = date.today().strftime("%d.%m.%Y")
    md = [
        "# Банк гипотез", "",
        f"> **Живая версия — общая страница [{BANK}]({BANK})**: там добавляют гипотезы, ставят оценки, отмечают, что "
        f"взято на неделю, и ведут техдолг. Этот файл — снимок на {today}, собран `scripts/bank_snapshot.py`.", "",
        f"> Результаты проверок на едином протоколе — **журнал экспериментов** {JOURNAL} "
        "(в репозитории — [experiments/README.md](experiments/README.md)).", "",
        "Правила (созвон 16.10): в проверку берётся только гипотеза, описанная в банке — что, инструменты, параметры и как "
        "они сочетаются, периоды, критерий. Приоритет = В × У × С (влияние × уверенность × дешевизна проверки, каждая 1–5, "
        "до 125). Новая стратегия не обязана быть лучше B01: цель — портфель из 8–10 стратегий, важны самостоятельная "
        "прибыльность и вклад в портфель. Результат проверки — эксперимент журнала, привязанный к гипотезе.", "",
    ]
    for area, title in [*AREAS.items(), ("", "Без раздела")]:
        part = [h for h in hyps if (h.get("area") or "") == area]
        if not part:
            continue
        md += [f"## {title}", ""]
        opened = sorted((h for h in part if h.get("status") in OPEN),
                        key=lambda h: -(lambda s: s[0] * s[1] * s[2] if s else -1)(scores(h)))
        closed = [h for h in part if h.get("status") not in OPEN]
        if opened:
            md += ["| ID | гипотеза | вид | В | У | С | приоритет | статус |", "|---|---|---|---|---|---|---|---|"]
            for h in opened:
                s = scores(h)
                md.append(f"| {h['code']} | **{cell(h.get('title'))}** — {cell(h.get('text'))} | {KINDS.get(h.get('kind'), '')} | "
                          + (" | ".join(num(x) for x in s) + f" | {num(s[0] * s[1] * s[2])}" if s else " |  |  | ")
                          + f" | {STATUSES.get(h.get('status'), '')} |")
            md.append("")
        if closed:
            md += ["Закрытые — чтобы не проверять повторно:", "", "| ID | гипотеза | итог | эксперименты |", "|---|---|---|---|"]
            for h in closed:
                exps = ", ".join(f"[{x}]({JOURNAL}#{x})" for x in h.get("exps") or [])
                md.append(f"| {h['code']} | **{cell(h.get('title'))}** | {STATUSES.get(h.get('status'), '')} {cell(h.get('result'))} | {exps} |")
            md.append("")
    if notes:
        md += ["## Техдолг и уточнения", "", "Не гипотезы: что выяснить или сделать.", "",
               "| ID | что | вид | кто | когда | статус | ответ |", "|---|---|---|---|---|---|---|"]
        for n in sorted(notes, key=lambda n: (n.get("status") not in ("open", "work"), n.get("num") or 0)):
            md.append(f"| {n['code']} | **{cell(n.get('title'))}** {cell(n.get('text'))} | {NKINDS.get(n.get('kind'), '')} | "
                      f"{cell(n.get('who'))} | {cell(n.get('due'))} | {NSTAT.get(n.get('status'), '')} | {cell(n.get('answer'))} |")
        md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"гипотез: {len(hyps)}, записей техдолга: {len(notes)} → {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
