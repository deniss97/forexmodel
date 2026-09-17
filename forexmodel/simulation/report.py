"""Сводный отчёт по сделкам.

PnL считается в процентах ЗА СДЕЛКУ и суммируется — как в исходном коде, чтобы
цифры оставались сопоставимыми со старыми прогонами. Дополнительно считается
equity по сложному проценту: при 200+ сделках разница между суммой и
произведением уже существенная.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = ["build_report", "format_report"]

EMPTY_REPORT: Dict[str, float] = {
    "total_trades": 0,
    "winrate": 0.0,
    "total_pnl_pct": 0.0,
    "avg_profit_pct": 0.0,
    "profit_factor": 0.0,
    "expectancy_pct": 0.0,
    "max_drawdown_pct": 0.0,
    "equity_multiple": 1.0,
    "avg_minutes": 0.0,
}


def build_report(trades_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df is None or trades_df.empty:
        return dict(EMPTY_REPORT)

    pnl = trades_df["profit_pct"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = (1 + pnl / 100).cumprod()
    drawdown = equity / equity.cummax() - 1

    report = {
        "total_trades": int(len(trades_df)),
        "winrate": round(float((pnl > 0).mean() * 100), 2),
        "total_pnl_pct": round(float(pnl.sum()), 2),
        "avg_profit_pct": round(float(pnl.mean()), 3),
        "median_profit_pct": round(float(pnl.median()), 3),
        "profit_factor": round(float(profit_factor), 3),
        "expectancy_pct": round(float(pnl.mean()), 3),
        "avg_win_pct": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_pct": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "equity_multiple": round(float(equity.iloc[-1]), 4),
        "avg_minutes": round(float(trades_df["minutes_in_trade"].mean()), 1),
        "long_share": round(float((trades_df["side"] == "buy").mean() * 100), 1),
    }

    for reason, count in trades_df["exit_reason"].value_counts().items():
        report[f"exit_{reason}"] = int(count)

    return report


#: как называть метрики в человекочитаемом отчёте и в каком порядке их выводить
_LABELS = [
    ("Сделки", [
        ("total_trades", "всего сделок", "{}"),
        ("winrate", "winrate", "{}%"),
        ("long_share", "из них в лонг", "{}%"),
        ("avg_minutes", "средняя длительность", "{} мин"),
    ]),
    ("Доходность", [
        ("total_pnl_pct", "сумма PnL", "{:+}%"),
        ("equity_multiple", "капитал (сложный процент)", "x{}"),
        ("max_drawdown_pct", "макс. просадка", "{}%"),
        ("profit_factor", "profit factor", "{}"),
    ]),
    ("На одну сделку", [
        ("avg_profit_pct", "средний результат", "{:+}%"),
        ("median_profit_pct", "медиана", "{:+}%"),
        ("avg_win_pct", "средняя прибыльная", "{:+}%"),
        ("avg_loss_pct", "средняя убыточная", "{:+}%"),
    ]),
]

_EXIT_NAMES = {
    "take_profit": "по тейк-профиту",
    "stop_loss": "по стопу",
    "trailing_stop": "по трейлинг-стопу",
    "timeout": "по истечении горизонта",
    "trend_flip": "по развороту тренда",
}


def format_report(report: Dict[str, float], title: str = "РЕЗУЛЬТАТ СИМУЛЯЦИИ") -> str:
    """Отчёт для человека: сгруппированный, с единицами и расшифровкой выходов.

    Плоский дамп `key: value` читался тяжело — в логе прогона нужно видеть
    сразу, заработал пайплайн или нет, и чем закрывались сделки.
    """
    width = 60
    lines = ["", "=" * width, f"  {title}", "=" * width]

    if not report.get("total_trades"):
        lines += ["  Сделок нет — проверьте пороги сигнала и фильтры", "=" * width]
        return "\n".join(lines)

    for group, items in _LABELS:
        lines.append(f"  {group}:")
        for key, label, fmt in items:
            if key in report:
                lines.append(f"    {label:<28} {fmt.format(report[key])}")
        lines.append("")

    exits = {k[len("exit_"):]: v for k, v in report.items() if k.startswith("exit_")}
    if exits:
        total = sum(exits.values()) or 1
        lines.append("  Чем закрывались сделки:")
        for reason, count in sorted(exits.items(), key=lambda kv: -kv[1]):
            name = _EXIT_NAMES.get(reason, reason)
            lines.append(f"    {name:<28} {count} ({count / total * 100:.0f}%)")
        lines.append("")

    pf = report.get("profit_factor", 0)
    verdict = "прибыльно" if report.get("total_pnl_pct", 0) > 0 else "убыточно"
    lines.append(f"  Вывод: {verdict}; profit factor {pf} ({'>1 — выигрыши перекрывают потери' if pf and pf > 1 else '<=1 — нет'})")
    lines.append("=" * width)
    return "\n".join(lines)
