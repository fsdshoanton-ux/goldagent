"""Bar-level execution simulator with spread, commission and slippage.

Bars are BID prices; the ask is bid + spread. Longs enter at the ask and exit at
the bid, shorts the reverse. When a bar touches both stop and target we assume
the stop was hit first (pessimistic), and gaps through the stop fill at the open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from goldagent.config import CostsConfig, InstrumentConfig
from goldagent.strategies.base import Signal


@dataclass
class Bar:
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    spread: float


@dataclass
class Position:
    side: int
    lots: float
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    take_profit: float
    risk_distance: float
    signal: Signal
    entry_spread: float
    signal_time: pd.Timestamp | None = None
    bars_held: int = 0
    breakeven_done: bool = False
    mfe: float = 0.0  # max favourable excursion, price units
    mae: float = 0.0  # max adverse excursion, price units


@dataclass
class Trade:
    side: int
    lots: float
    signal_time: pd.Timestamp | None
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    initial_stop: float
    take_profit: float
    gross_pnl: float
    commission: float
    pnl: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    mfe_r: float
    mae_r: float
    entry_spread: float
    reason: str
    context: dict[str, Any] = field(default_factory=dict)


class SimBroker:
    def __init__(self, instrument: InstrumentConfig, costs: CostsConfig):
        self.inst = instrument
        self.costs = costs

    def open(self, bar: Bar, signal: Signal, lots: float, signal_time: pd.Timestamp | None = None) -> Position:
        slip = self.costs.slippage
        if signal.side > 0:
            price = bar.open + bar.spread + slip
        else:
            price = bar.open - slip
        return Position(
            side=signal.side,
            lots=lots,
            entry_time=bar.time,
            entry_price=price,
            stop=price - signal.side * signal.stop_distance,
            take_profit=price + signal.side * signal.tp_distance,
            risk_distance=signal.stop_distance,
            signal=signal,
            entry_spread=bar.spread,
            signal_time=signal_time,
        )

    def _exit_quotes(self, pos: Position, bar: Bar) -> tuple[float, float, float, float]:
        """OHLC of the price we would exit at (bid for longs, ask for shorts)."""
        add = 0.0 if pos.side > 0 else bar.spread
        return bar.open + add, bar.high + add, bar.low + add, bar.close + add

    def check_exit(self, pos: Position, bar: Bar) -> Trade | None:
        """Process one bar for an open position. Returns a Trade if it closed."""
        o, h, l, c = self._exit_quotes(pos, bar)
        slip = self.costs.slippage
        s = pos.side
        pos.bars_held += 1
        fav = (h - pos.entry_price) if s > 0 else (pos.entry_price - l)
        adv = (pos.entry_price - l) if s > 0 else (h - pos.entry_price)
        pos.mfe, pos.mae = max(pos.mfe, fav), max(pos.mae, adv)

        if s > 0:
            if o <= pos.stop:
                return self._close(pos, bar.time, o - slip, "stop_gap")
            if l <= pos.stop:
                return self._close(pos, bar.time, pos.stop - slip, "stop" if not pos.breakeven_done else "breakeven")
            if o >= pos.take_profit:
                return self._close(pos, bar.time, o, "target_gap")
            if h >= pos.take_profit:
                return self._close(pos, bar.time, pos.take_profit, "target")
        else:
            if o >= pos.stop:
                return self._close(pos, bar.time, o + slip, "stop_gap")
            if h >= pos.stop:
                return self._close(pos, bar.time, pos.stop + slip, "stop" if not pos.breakeven_done else "breakeven")
            if o <= pos.take_profit:
                return self._close(pos, bar.time, o, "target_gap")
            if l <= pos.take_profit:
                return self._close(pos, bar.time, pos.take_profit, "target")

        sig = pos.signal
        if sig.max_hold_bars and pos.bars_held >= sig.max_hold_bars:
            return self._close(pos, bar.time, c - s * slip, "time")
        # Breakeven is applied after this bar's exit checks: it protects later bars only.
        if sig.breakeven_r and not pos.breakeven_done and fav >= sig.breakeven_r * pos.risk_distance:
            pos.stop = pos.entry_price
            pos.breakeven_done = True
        return None

    def close_at_market(self, pos: Position, bar: Bar, reason: str) -> Trade:
        _, _, _, c = self._exit_quotes(pos, bar)
        return self._close(pos, bar.time, c - pos.side * self.costs.slippage, reason)

    def unrealized(self, pos: Position, bar: Bar) -> float:
        _, _, _, c = self._exit_quotes(pos, bar)
        return pos.side * (c - pos.entry_price) * pos.lots * self.inst.contract_size

    def _close(self, pos: Position, ts: pd.Timestamp, price: float, reason: str) -> Trade:
        size = pos.lots * self.inst.contract_size
        gross = pos.side * (price - pos.entry_price) * size
        commission = self.costs.commission_per_lot * pos.lots
        pnl = gross - commission
        risk_usd = pos.risk_distance * size
        return Trade(
            side=pos.side,
            lots=pos.lots,
            signal_time=pos.signal_time,
            entry_time=pos.entry_time,
            exit_time=ts,
            entry_price=pos.entry_price,
            exit_price=price,
            initial_stop=pos.entry_price - pos.side * pos.risk_distance,
            take_profit=pos.take_profit,
            gross_pnl=gross,
            commission=commission,
            pnl=pnl,
            r_multiple=pnl / risk_usd if risk_usd else 0.0,
            exit_reason=reason,
            bars_held=pos.bars_held,
            mfe_r=pos.mfe / pos.risk_distance if pos.risk_distance else 0.0,
            mae_r=pos.mae / pos.risk_distance if pos.risk_distance else 0.0,
            entry_spread=pos.entry_spread,
            reason=pos.signal.reason,
            context=dict(pos.signal.context),
        )
