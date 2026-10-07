"""Position sizing and account-level loss limits.

This module is the last line of defence and is deliberately simple. Limits come
from the human-owned ``risk`` config section; nothing else may change them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from goldagent.config import InstrumentConfig, RiskConfig


@dataclass
class RiskState:
    peak_equity: float
    day: object = None
    day_start_balance: float = 0.0
    week: object = None
    week_start_balance: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    events: list[tuple[pd.Timestamp, str]] = field(default_factory=list)


class RiskManager:
    def __init__(self, risk: RiskConfig, instrument: InstrumentConfig, initial_balance: float):
        self.risk = risk
        self.inst = instrument
        self.state = RiskState(peak_equity=initial_balance)
        self._balance = initial_balance

    def position_size(self, balance: float, stop_distance: float) -> float:
        """Lots such that hitting the stop loses at most risk_per_trade of balance.

        Rounds *down* to the lot step; returns 0.0 if even the minimum lot is too big.
        """
        if stop_distance <= 0:
            return 0.0
        risk_usd = balance * self.risk.risk_per_trade
        raw = risk_usd / (stop_distance * self.inst.contract_size)
        steps = math.floor(raw / self.inst.lot_step + 1e-9)
        lots = round(steps * self.inst.lot_step, 8)
        if lots < self.inst.min_lot:
            return 0.0
        return min(lots, self.inst.max_lot)

    def _roll_periods(self, ts: pd.Timestamp) -> None:
        s = self.state
        day = ts.date()
        week = ts.isocalendar()[:2]
        if s.day != day:
            s.day = day
            s.day_start_balance = self._balance
            s.consecutive_losses = 0
        if s.week != week:
            s.week = week
            s.week_start_balance = self._balance

    def update_equity(self, ts: pd.Timestamp, equity: float) -> None:
        s = self.state
        self._roll_periods(ts)
        s.peak_equity = max(s.peak_equity, equity)
        if not s.halted and equity <= s.peak_equity * (1 - self.risk.kill_switch_drawdown):
            s.halted = True
            s.halt_reason = f"kill-switch: equity {equity:.2f} is {self.risk.kill_switch_drawdown:.0%} below peak {s.peak_equity:.2f}"
            s.events.append((ts, s.halt_reason))

    def on_trade_closed(self, ts: pd.Timestamp, pnl: float, balance: float) -> None:
        self._roll_periods(ts)
        self._balance = balance
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl < 0 else 0

    def can_open(self, ts: pd.Timestamp, open_positions: int) -> tuple[bool, str]:
        s = self.state
        self._roll_periods(ts)
        if s.halted:
            return False, s.halt_reason
        if open_positions >= self.risk.max_open_positions:
            return False, "max open positions"
        if s.consecutive_losses >= self.risk.max_consecutive_losses:
            return False, f"{s.consecutive_losses} consecutive losses today"
        if self._balance <= s.day_start_balance * (1 - self.risk.daily_loss_limit):
            return False, "daily loss limit"
        if self._balance <= s.week_start_balance * (1 - self.risk.weekly_loss_limit):
            return False, "weekly loss limit"
        return True, ""
