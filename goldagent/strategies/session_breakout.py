"""Asian-range breakout at the London / New York opens, filtered by the H1 trend.

Gold usually consolidates during the Asian session and makes its directional
move when European and US liquidity arrives. We only take a breakout in the
direction of the higher-timeframe trend and only on the first close beyond the
range (no chasing).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from goldagent.data.resample import resample_ohlc
from goldagent.indicators import align_higher_timeframe, atr, ema
from goldagent.strategies.base import Signal, Strategy


class SessionBreakout(Strategy):
    name = "session_breakout"
    defaults: dict[str, Any] = {
        "asia_start_hour": 0,
        "asia_end_hour": 7,
        "entry_windows": [[7, 10], [12, 15]],
        "breakout_buffer_atr": 0.10,
        "atr_period": 14,
        "trend_fast": 50,
        "trend_slow": 200,
        "sl_atr_mult": 1.5,
        "tp_r": 2.0,
        "breakeven_r": 1.0,
        "max_hold_bars": 36,
        "max_trades_per_day": 2,
        "min_range_atr": 2.0,
        "max_range_atr": 15.0,
        "atr_regime_lookback": 2880,
        "atr_regime_min": 0.5,
        "atr_regime_max": 2.5,
        "max_spread": 0.60,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        p = self.params
        if any(start < p["asia_end_hour"] for start, _ in p["entry_windows"]):
            raise ValueError("entry windows must start after the Asian range is complete")
        self._day = None
        self._trades_today = 0
        self._sides_today: set[int] = set()

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        freq = pd.Series(bars.index[:500]).diff().median()
        f = pd.DataFrame(index=bars.index)
        f["close"] = bars["close"]
        f["prev_close"] = bars["close"].shift(1)
        f["spread"] = bars["spread"] if "spread" in bars.columns else np.nan
        f["atr"] = atr(bars, p["atr_period"])
        lookback = p["atr_regime_lookback"]
        f["atr_ratio"] = f["atr"] / f["atr"].rolling(lookback, min_periods=lookback // 4).median()

        hours = bars.index.hour
        day = bars.index.normalize()
        asia = (hours >= p["asia_start_hour"]) & (hours < p["asia_end_hour"])
        after_asia = hours >= p["asia_end_hour"]
        f["range_high"] = bars["high"].where(asia).groupby(day).transform("max").where(after_asia)
        f["range_low"] = bars["low"].where(asia).groupby(day).transform("min").where(after_asia)

        h1 = resample_ohlc(bars, "1h")
        fast, slow = ema(h1["close"], p["trend_fast"]), ema(h1["close"], p["trend_slow"])
        trend = pd.Series(0, index=h1.index, dtype=float)
        trend[(h1["close"] > fast) & (fast > slow)] = 1
        trend[(h1["close"] < fast) & (fast < slow)] = -1
        trend[slow.isna()] = np.nan
        f["trend"] = align_higher_timeframe(trend, "1h", bars.index, freq)

        in_window = np.zeros(len(bars), dtype=bool)
        for start, end in p["entry_windows"]:
            in_window |= (hours >= start) & (hours < end)
        f["in_window"] = in_window
        return f

    def on_bar(self, ts: pd.Timestamp, row: dict[str, Any]) -> Signal | None:
        p = self.params
        day = ts.date()
        if day != self._day:
            self._day, self._trades_today, self._sides_today = day, 0, set()
        if not row["in_window"] or self._trades_today >= p["max_trades_per_day"]:
            return None
        a, hi, lo, trend = row["atr"], row["range_high"], row["range_low"], row["trend"]
        if any(pd.isna(x) for x in (a, hi, lo, trend, row["atr_ratio"], row["prev_close"])) or a <= 0:
            return None
        if not p["atr_regime_min"] <= row["atr_ratio"] <= p["atr_regime_max"]:
            return None
        if not pd.isna(row["spread"]) and row["spread"] > p["max_spread"]:
            return None
        width = (hi - lo) / a
        if not p["min_range_atr"] <= width <= p["max_range_atr"]:
            return None

        buf = p["breakout_buffer_atr"] * a
        close, prev = row["close"], row["prev_close"]
        side = 0
        if trend > 0 and close > hi + buf and prev <= hi + buf:
            side = 1
        elif trend < 0 and close < lo - buf and prev >= lo - buf:
            side = -1
        if side == 0 or side in self._sides_today:
            return None

        stop = p["sl_atr_mult"] * a
        return Signal(
            side=side,
            stop_distance=stop,
            tp_distance=p["tp_r"] * stop,
            breakeven_r=p["breakeven_r"],
            max_hold_bars=p["max_hold_bars"],
            reason=f"{'long' if side > 0 else 'short'} breakout of Asian range with H1 trend",
            context={
                "hour": ts.hour,
                "atr": round(a, 3),
                "atr_ratio": round(row["atr_ratio"], 3),
                "range_width_atr": round(width, 2),
                "breakout_atr": round(((close - hi) if side > 0 else (lo - close)) / a, 2),
                "trend": int(trend),
                "spread": None if pd.isna(row["spread"]) else round(row["spread"], 3),
            },
        )

    def on_entry(self, ts: pd.Timestamp, signal: Signal) -> None:
        self._trades_today += 1
        self._sides_today.add(signal.side)
