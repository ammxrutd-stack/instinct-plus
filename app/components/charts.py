"""Plotly figures for the app: score gauges and driver bars."""

from typing import Dict

import plotly.graph_objects as go

GREEN = "#2e9e5b"
RED = "#d64545"
GREY = "#8a8f98"


def gauge(score: float, title: str) -> go.Figure:
    colour = RED if score < 35 else ("#e3a008" if score < 65 else GREEN)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score),
            title={"text": title, "font": {"size": 16}},
            number={"font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 10}},
                "bar": {"color": colour, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(214,69,69,0.12)"},
                    {"range": [35, 65], "color": "rgba(227,160,8,0.12)"},
                    {"range": [65, 100], "color": "rgba(46,158,91,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(height=210, margin={"l": 30, "r": 30, "t": 45, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def driver_bars(drivers: Dict[str, float], labels: Dict[str, str], title: str = "") -> go.Figure:
    items = sorted(drivers.items(), key=lambda kv: kv[1])
    names = [labels.get(k, k.replace("_", " ")) for k, _ in items]
    values = [v for _, v in items]
    colours = [GREEN if v > 0 else RED for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colours,
            text=[f"{v:+.0f}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    layout = {"title": title} if title else {}
    fig.update_layout(
        height=max(180, 32 * len(items) + 60),
        margin={"l": 10, "r": 40, "t": 30 if title else 10, "b": 10},
        **layout,
        xaxis_title="points",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"zeroline": True, "zerolinecolor": GREY, "gridcolor": "rgba(128,128,128,0.15)"},
    )
    return fig


def importance_bars(importance: Dict[str, float], labels: Dict[str, str], top: int = 15) -> go.Figure:
    items = [(k, v) for k, v in importance.items() if not k.startswith("ct_")][:top][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[labels.get(k, k) for k, _ in items], orientation="h", marker_color="#4a7bd0"))
    fig.update_layout(height=max(200, 26 * len(items) + 40), margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis_title="share of split gain", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
