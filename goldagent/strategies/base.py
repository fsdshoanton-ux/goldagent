"""Strategy interface shared by backtest, paper and live trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """Intent to enter at the next bar's open. Distances are in price units (USD)."""

    side: int  # +1 long, -1 short
    stop_distance: float
    tp_distance: float
    breakeven_r: float | None = None  # move stop to entry once price reaches this many R
    max_hold_bars: int | None = None
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    name: str = "base"
    defaults: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None):
        unknown = set(params or {}) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        self.params = {**self.defaults, **(params or {})}

    @abstractmethod
    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Vectorised, causal feature computation. Row t may only use bars <= t."""

    @abstractmethod
    def on_bar(self, ts: pd.Timestamp, row: dict[str, Any]) -> Signal | None:
        """Called at the close of each bar while flat and allowed to trade."""

    def on_entry(self, ts: pd.Timestamp, signal: Signal) -> None:
        """Called when a signal is actually filled."""
