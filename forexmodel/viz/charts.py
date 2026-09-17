"""Интерактивный график: свечи + фон тренда + сделки (plotly -> HTML).

Перенесено из ноутбука с двумя изменениями: имена колонок больше не берутся из
глобальных CANDLE_COLS/TRADE_COLS, а передаются аргументами, и функция не
падает, если колонки тренда/EMA нет.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["create_trading_chart", "create_equity_chart"]


def create_equity_chart(
    df_trades: pd.DataFrame,
    output_file: Path | str = "reports/equity.html",
    title: str = "Кривая капитала",
):
    """Кривая капитала + просадка + распределение PnL + PnL по причинам выхода.

    Свечной график отвечает на вопрос «где мы входили», а этот — на вопрос
    «что из этого вышло»: ровно ли растёт кривая, где сидят просадки и на чём
    теряются деньги.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df_trades is None or df_trades.empty:
        log.warning("Нет сделок — график капитала не строится")
        return None

    t = df_trades.copy()
    t["close_dt"] = pd.to_datetime(t["close_dt"])
    t = t.sort_values("close_dt").reset_index(drop=True)

    equity = (1 + t["profit_pct"] / 100).cumprod()
    drawdown = (equity / equity.cummax() - 1) * 100

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "Капитал (сложный процент, старт = 1.0)",
            "Просадка от максимума, %",
            "Распределение PnL по сделкам, %",
            "Суммарный PnL по причинам выхода, %",
        ),
        row_heights=[0.42, 0.22, 0.36],
        vertical_spacing=0.09,
    )

    fig.add_trace(
        go.Scatter(x=t["close_dt"], y=equity, name="Капитал", mode="lines",
                   line=dict(color="#26A69A", width=2), fill="tozeroy", fillcolor="rgba(38,166,154,0.12)"),
        row=1, col=1,
    )
    fig.add_hline(y=1.0, line=dict(color="gray", width=1, dash="dash"), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=t["close_dt"], y=drawdown, name="Просадка", mode="lines",
                   line=dict(color="#EF5350", width=1.5), fill="tozeroy", fillcolor="rgba(239,83,80,0.20)"),
        row=2, col=1,
    )

    fig.add_trace(
        go.Histogram(x=t["profit_pct"], name="PnL сделки", nbinsx=60, marker_color="#42A5F5"),
        row=3, col=1,
    )

    by_reason = t.groupby("exit_reason")["profit_pct"].sum().sort_values()
    fig.add_trace(
        go.Bar(
            x=by_reason.values,
            y=by_reason.index,
            orientation="h",
            name="PnL по выходам",
            marker_color=["#EF5350" if v < 0 else "#26A69A" for v in by_reason.values],
        ),
        row=3, col=2,
    )

    total = t["profit_pct"].sum()
    fig.update_layout(
        title=f"{title} — сделок {len(t)}, сумма PnL {total:+.2f}%, капитал x{equity.iloc[-1]:.3f}, "
              f"макс. просадка {drawdown.min():.2f}%",
        template="plotly_dark",
        height=1000,
        showlegend=False,
        margin=dict(t=90, b=50, l=60, r=30),
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file, include_plotlyjs="cdn", full_html=True, config={"responsive": True})
    log.info("График капитала сохранён: %s", output_file)
    return fig


def create_trading_chart(
    df_candles: pd.DataFrame,
    df_trades: pd.DataFrame,
    output_file: Path | str = "reports/trading_chart.html",
    title: str = "Свечи + тренд + сделки",
    trend_col: Optional[str] = "trend_4h",
    ema_col: Optional[str] = None,
    max_trades: int = 500,
    max_points: int = 5000,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    candles = df_candles.copy()
    trades = df_trades.copy()

    candles["time"] = pd.to_datetime(candles["time"])
    candles = candles.sort_values("time").reset_index(drop=True)

    if len(candles) > max_points:
        step = max(len(candles) // max_points, 1)
        log.info("Свечи прорежены с %d до ~%d точек (шаг %d)", len(candles), max_points, step)
        candles = candles.iloc[::step].reset_index(drop=True)

    for col in ("open_dt", "close_dt"):
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col])
    if len(trades) > max_trades:
        log.info("Показаны последние %d сделок из %d", max_trades, len(trades))
        trades = trades.iloc[-max_trades:]

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    if trend_col and trend_col in candles.columns:
        fig.update_layout(shapes=_trend_shapes(candles, trend_col))

    fig.add_trace(
        go.Candlestick(
            x=candles["time"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="Свечи",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        )
    )

    if ema_col and ema_col in candles.columns:
        fig.add_trace(
            go.Scatter(x=candles["time"], y=candles[ema_col], name=ema_col, line=dict(color="orange", width=2, dash="dash"))
        )

    if not trades.empty:
        is_long = trades["side"].isin(["buy", "long"])
        fig.add_trace(
            go.Scatter(
                x=trades["open_dt"],
                y=trades["open_price"],
                mode="markers+text",
                name="Входы",
                marker=dict(
                    symbol=["triangle-up" if v else "triangle-down" for v in is_long],
                    size=13,
                    color=["#00FF88" if v else "#FF3366" for v in is_long],
                    line=dict(width=1.5, color="black"),
                ),
                text=["LONG" if v else "SHORT" for v in is_long],
                textposition="top center",
                hovertemplate="%{text}<br>%{x}<br>%{y}<extra></extra>",
            )
        )

        line_x, line_y = [], []
        for t in trades.itertuples(index=False):
            if pd.notna(getattr(t, "close_dt", None)):
                line_x += [t.open_dt, t.close_dt, None]
                line_y += [t.open_price, t.exit_price, None]
        if line_x:
            fig.add_trace(
                go.Scatter(x=line_x, y=line_y, mode="lines", name="Путь сделки",
                           line=dict(width=1.5, dash="dot", color="white"), hoverinfo="skip")
            )

    fig.update_layout(
        title=title,
        xaxis_title="Время",
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=50, l=50, r=30),
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_file,
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "responsive": True},
    )
    log.info("График сохранён: %s", output_file)
    return fig


def _trend_shapes(candles: pd.DataFrame, trend_col: str) -> list[dict]:
    """Фоновые прямоугольники по блокам одинакового тренда (одним пакетом —
    поштучный add_vrect в связке с make_subplots иногда теряется при сериализации)."""
    trend = pd.to_numeric(candles[trend_col], errors="coerce").round()
    change = trend.diff().fillna(1) != 0
    edges = sorted(set([0] + candles.index[change].tolist() + [len(candles) - 1]))

    colors = {1: "rgba(0, 255, 120, 0.18)", -1: "rgba(255, 60, 60, 0.18)"}
    shapes = []
    for start, end in zip(edges, edges[1:]):
        if start == end:
            continue
        value = trend.iloc[start]
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y domain",
                x0=candles["time"].iloc[start],
                x1=candles["time"].iloc[end],
                y0=0,
                y1=1,
                fillcolor=colors.get(value, "rgba(150, 150, 150, 0.10)"),
                opacity=1,
                layer="below",
                line_width=0,
            )
        )
    return shapes
