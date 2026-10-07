"""Causal technical indicators. Every value at row t uses only rows <= t."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    return true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * std, "lower": mid - n_std * std})


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset at each UTC day. Uses tick volume if present, else equal weights."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vol = vol.where(vol > 0, 1.0)
    day = df.index.normalize()
    pv = (typical * vol).groupby(day).cumsum()
    v = vol.groupby(day).cumsum()
    return pv / v


def align_higher_timeframe(
    htf: pd.Series | pd.DataFrame,
    htf_freq: str | pd.Timedelta,
    ltf_index: pd.DatetimeIndex,
    ltf_freq: str | pd.Timedelta,
) -> pd.Series | pd.DataFrame:
    """Map higher-timeframe values onto lower-timeframe bars without lookahead.

    Both indexes are bar *open* times. A higher-timeframe bar becomes known only
    when it closes (open + htf_freq). A lower-timeframe bar is evaluated at its
    close (open + ltf_freq), so it may see every htf bar whose close <= its close.
    """
    known_at = htf.index + pd.Timedelta(htf_freq)
    shifted = htf.copy()
    shifted.index = known_at
    shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()
    ltf_close = ltf_index + pd.Timedelta(ltf_freq)
    aligned = shifted.reindex(shifted.index.union(ltf_close)).ffill().reindex(ltf_close)
    aligned.index = ltf_index
    return aligned
