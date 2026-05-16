"""
Statistical helpers — percentile, forward-return distributions, etc.
The whole point of this dashboard is to put current readings in historical context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_of(series: pd.Series, value: float | None = None) -> float:
    """Where does `value` (default: most recent) sit in the historical distribution?
    Returns a value in 0..100.
    """
    s = series.dropna()
    if s.empty:
        return float("nan")
    if value is None:
        value = float(s.iloc[-1])
    return float((s <= value).mean() * 100)


def rolling_percentile(series: pd.Series, window: int = 252 * 5) -> pd.Series:
    """Rolling percentile — useful for indicators where the long-term mean
    drifts (e.g. margin debt grows nominally over time)."""
    return series.rolling(window).apply(
        lambda x: (x <= x.iloc[-1]).mean() * 100, raw=False
    )


def forward_returns(
    price: pd.Series,
    horizon_days: int = 30,
) -> pd.Series:
    """N-day forward return for each date in the series."""
    return price.pct_change(horizon_days).shift(-horizon_days)


def conditional_forward_stats(
    price: pd.Series,
    condition: pd.Series,
    horizon_days: int = 30,
) -> dict:
    """Given a boolean condition series aligned to `price`, return summary
    stats of forward returns when the condition is True.

    Returns dict with: n, mean, median, win_rate, p10, p90.
    """
    fr = forward_returns(price, horizon_days)
    aligned = pd.concat([fr, condition.astype(bool)], axis=1, join="inner").dropna()
    aligned.columns = ["fr", "cond"]
    sample = aligned.loc[aligned["cond"], "fr"]
    if sample.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win_rate": np.nan, "p10": np.nan, "p90": np.nan}
    return {
        "n": int(len(sample)),
        "mean": float(sample.mean()),
        "median": float(sample.median()),
        "win_rate": float((sample > 0).mean()),
        "p10": float(sample.quantile(0.1)),
        "p90": float(sample.quantile(0.9)),
    }


def regime_label(pct: float) -> tuple[str, str]:
    """Map a 0..100 percentile to a (label, color) pair.
    Colors are CSS-compatible — used by the gauge widget.
    """
    if pd.isna(pct):
        return ("no data", "gray")
    if pct >= 95:
        return ("extreme high", "#c0392b")
    if pct >= 80:
        return ("elevated", "#e67e22")
    if pct >= 60:
        return ("above avg", "#f1c40f")
    if pct >= 40:
        return ("typical", "#27ae60")
    if pct >= 20:
        return ("below avg", "#3498db")
    if pct >= 5:
        return ("depressed", "#2980b9")
    return ("extreme low", "#8e44ad")
