"""
Plotly helpers — keep visual style consistent across pages.

Conventions:
  - All charts use hovermode='x unified' so values show as soon as the
    mouse enters the chart area, without needing to hover on the line.
  - Display window is constant DISPLAY_START. Percentile calculations use
    the full series so the historical context is preserved.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .stats import percentile_of, regime_label


# Where charts START displaying. Change this constant in one place.
DISPLAY_START = pd.Timestamp("2025-01-01")


def _clip(series_or_df):
    """Return a copy starting at DISPLAY_START."""
    if series_or_df is None:
        return series_or_df
    try:
        return series_or_df.loc[series_or_df.index >= DISPLAY_START]
    except Exception:
        return series_or_df


def apply_hover_style(fig: go.Figure) -> go.Figure:
    """Apply unified hover + consistent margins to any figure.

    Important: in unified mode the hovertemplate should NOT use
    `<extra>name</extra>` because that hides the y value. Use a clean
    `%{y:.2f}` (no <extra>) or `%{y:.2f}<extra></extra>` to suppress the
    extra box entirely.
    """
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#888",
            font=dict(size=12, color="#000"),
            namelength=-1,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikedash="dot", spikecolor="#666", spikethickness=1,
    )
    # Force every existing trace to use a clean unified-friendly template.
    # Skip candlestick/ohlc — they need OHLC hover, not y-only.
    for tr in fig.data:
        ttype = getattr(tr, "type", "") or ""
        if ttype in ("candlestick", "ohlc"):
            tr.hovertemplate = (
                "Open: %{open:.2f}<br>"
                "High: %{high:.2f}<br>"
                "Low: %{low:.2f}<br>"
                "Close: %{close:.2f}<extra></extra>"
            )
            continue
        if hasattr(tr, "hovertemplate"):
            tr.hovertemplate = "%{y:.2f}<extra></extra>"
    return fig


def indicator_card(
    series: pd.Series,
    *,
    title: str = "",
    unit: str = "",
    invert: bool = False,
    description: str = "",
) -> tuple[go.Figure, dict]:
    """A compact line chart + the metadata used by st.metric.

    Percentile is computed on the FULL series. Chart shows only the
    display window (from DISPLAY_START).
    `invert=True` means "low values are bullish".
    """
    s_full = series.dropna()
    if s_full.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        apply_hover_style(fig)
        return fig, {
            "value": None, "unit": unit, "pct": float("nan"),
            "label": "no data", "color": "gray", "description": description,
        }

    cur = float(s_full.iloc[-1])
    pct = percentile_of(s_full)
    display_pct = (100 - pct) if invert else pct
    label, color = regime_label(display_pct)

    s_disp = _clip(s_full)
    if s_disp.empty:
        s_disp = s_full.tail(60)  # fallback if no data in display window

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_disp.index,
            y=s_disp.values,
            mode="lines",
            line=dict(width=1.5, color="#3498db"),
            hovertemplate="%{y:.2f}<extra></extra>",
            name=title or "value",
        )
    )
    # Reference bands from FULL history
    for q, dash in [(0.95, "dash"), (0.05, "dash"), (0.5, "dot")]:
        fig.add_hline(y=s_full.quantile(q), line=dict(color="gray", dash=dash, width=1))
    fig.add_hline(y=cur, line=dict(color=color, width=2))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        showlegend=False,
        height=240,
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    apply_hover_style(fig)

    meta = {
        "value": cur,
        "unit": unit,
        "pct": display_pct,
        "label": label,
        "color": color,
        "description": description,
    }
    return fig, meta


def price_with_ma(price, mas=(50, 200)):
    """Price + moving averages. MAs computed on FULL series, then clipped."""
    fig = go.Figure()
    price_disp = _clip(price)
    fig.add_trace(go.Scatter(
        x=price_disp.index, y=price_disp.values,
        name="Price",
        line=dict(color="#2c3e50", width=1.5),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    palette = ["#e67e22", "#c0392b", "#16a085"]
    for i, w in enumerate(mas):
        ma_full = price.rolling(w).mean()
        ma_disp = _clip(ma_full)
        fig.add_trace(go.Scatter(
            x=ma_disp.index, y=ma_disp.values,
            name=f"{w}d MA",
            line=dict(width=1.2, color=palette[i % len(palette)]),
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        height=380,
        legend=dict(orientation="h", y=1.1),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    apply_hover_style(fig)
    return fig
