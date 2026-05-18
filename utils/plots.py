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


# ---------------------------------------------------------------------------
# Historical bear comparison overlay
# ---------------------------------------------------------------------------

BEAR_PERIODS = {
    "2018 Q4 (Powell pivot)": ("2018-09-15", "2019-01-15"),
    "2022 升息熊市": ("2022-01-01", "2022-10-31"),
    "2024 AI 修正": ("2024-07-01", "2024-09-15"),
    "現在 (近 6 個月)": None,  # special — use last 6 months of data
}

BEAR_COLORS = {
    "2018 Q4 (Powell pivot)": "#8e44ad",
    "2022 升息熊市": "#c0392b",
    "2024 AI 修正": "#e67e22",
    "現在 (近 6 個月)": "#16a085",
}


def compare_historical_bears(series, title="歷史熊市對照", value_label="value"):
    """Return a 4-panel subplot comparing the series across 3 bears + current.

    series: pd.Series indexed by date (full history)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    s = series.dropna().sort_index()
    if s.empty:
        return None

    periods = list(BEAR_PERIODS.items())
    fig = make_subplots(
        rows=1, cols=len(periods),
        subplot_titles=[name for name, _ in periods],
        horizontal_spacing=0.04,
    )
    for i, (name, rng) in enumerate(periods, start=1):
        if rng is None:
            # current — last 6 months
            slice_ = s.tail(126)
        else:
            slice_ = s.loc[(s.index >= rng[0]) & (s.index <= rng[1])]
        if slice_.empty:
            continue
        color = BEAR_COLORS.get(name, "#3498db")
        # X-axis: trading days from period start (0, 1, 2, ...)
        x = list(range(len(slice_)))
        fig.add_trace(
            go.Scatter(
                x=x, y=slice_.values, mode="lines",
                line=dict(color=color, width=2),
                name=name,
                hovertemplate=(
                    "day %{x}<br>"
                    "<b>" + value_label + "</b>: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=i,
        )
        # Add date range annotation
        date_range = f"{slice_.index[0].strftime('%Y-%m-%d')} → {slice_.index[-1].strftime('%Y-%m-%d')}"
        fig.add_annotation(
            text=date_range, xref=f"x{i if i>1 else ''} domain", yref=f"y{i if i>1 else ''} domain",
            x=0.5, y=-0.18, showarrow=False, font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        height=320,
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=10, r=10, t=60, b=40),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(title_text="自起始日的交易日數", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig
